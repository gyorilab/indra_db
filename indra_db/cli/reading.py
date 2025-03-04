import click
import logging
from os import path
from functools import wraps
from datetime import datetime, timedelta

from sqlalchemy import func


# Note that imports from indra_reading and indra_db.reading are burried in
# functions to avoid imports of complex dependencies for simple CLI access.

from .util import format_date

logger = logging.getLogger(__name__)
THIS_DIR = path.dirname(path.abspath(__file__))


class ReadingUpdateError(Exception):
    pass


class ReadingManager(object):
    """Abstract class for managing the readings of the database.

    Parameters
    ----------
    reader_names : lsit [str]
        A list of the names of the readers to be used in a given run of
        reading.
    buffer_days : int
        The number of days before the previous update/initial upload to look for
        "new" content to be read. This prevents any issues with overlaps between
        the content upload pipeline and the reading pipeline.
    only_unread : bool
        Only read papers that have not been read (making the determination can
        be expensive).
    """
    def __init__(self, reader_names, buffer_days=1, only_unread=False):
        self.reader_names = reader_names
        self.buffer = timedelta(days=buffer_days)
        self.run_datetime = None
        self.begin_datetime = None
        self.end_datetime = None
        self.only_unread = only_unread
        return

    @classmethod
    def _run_all_readers(cls, func):
        @wraps(func)
        def run_and_record_update(self, db, *args, **kwargs):
            all_completed = False
            for reader_name in self.reader_names:
                self.run_datetime = datetime.utcnow()
                done = func(self, db, reader_name, *args, **kwargs)
                all_completed &= done
                logger.info("%s is%s done" % (reader_name,
                                              '' if done else ' not'))
                if done:
                    is_read_all = (func.__name__ == 'read_all')
                    reader_version = self.get_version(reader_name)
                    if reader_version is None:
                        # This effectively indicates no jobs ran.
                        logger.info("It appears no %s jobs ran. No update "
                                    "will be logged." % reader_name)
                        continue
                    logger.info("Recording this reading in reading_updates: "
                                "%s version %s running at %s reading content "
                                "between %s and %s."
                                % (reader_name, reader_version,
                                   self.run_datetime, self.begin_datetime,
                                   self.end_datetime))
                    db.insert('reading_updates', complete_read=is_read_all,
                              reader=reader_name,
                              reader_version=reader_version,
                              run_datetime=self.run_datetime,
                              earliest_datetime=self.begin_datetime,
                              latest_datetime=self.end_datetime)
            return all_completed
        return run_and_record_update

    def get_version(self, reader_name):
        from indra_reading.readers import get_reader_class
        return get_reader_class(reader_name).get_version()

    @staticmethod
    def get_latest_updates(db):
        """Get the date of the latest update."""
        res = (db.session.query(db.ReadingUpdates.reader,
                                func.max(db.ReadingUpdates.latest_datetime))
               .group_by(db.ReadingUpdates.reader))
        return {reader: last_updated for reader, last_updated in res}

    @classmethod
    def _get_latest_updatetime(cls, db, reader_name):
        latest_updates = cls.get_latest_updates(db)
        if reader_name not in latest_updates:
            logger.warning("The database has not had an initial upload "
                           "for %s, or else the updates table has not "
                           "been populated." % reader_name)
            return None
        return latest_updates[reader_name]

    def read_all(self, db, reader_name):
        """Perform an initial reading all content in the database (populate).

        This must be defined in a child class.
        """
        raise NotImplementedError

    def read_new(self, db, reader_name):
        """Read only new content (update).

        This must be defined in a child class.
        """
        raise NotImplementedError


class BulkReadingManager(ReadingManager):
    """An abstract class which defines methods required for reading in bulk.

    This takes exactly the parameters used by :py:class:`ReadingManager`.
    """
    def _run_reading(self, db, tcids, reader_name):
        raise NotImplementedError("_run_reading must be defined in child.")

    def _get_constraints(self, db, reader_name):
        # Ignore xDD placeholders.
        constrains = [db.TextContent.format != 'xdd']

        # Only read titles for TRIPS.
        if reader_name.lower() == 'trips':
            constrains.append(db.TextContent.text_type == "title")
        elif reader_name.lower() == 'mti':
            constrains.append(
                db.TextContent.text_type.in_(['title', 'abstract'])
            )
        return constrains

    @ReadingManager._run_all_readers
    def read_all(self, db, reader_name):
        """Read everything available on the database."""
        self.end_datetime = self.run_datetime

        constraints = self._get_constraints(db, reader_name)

        tcid_q = db.filter_query(db.TextContent.id, *constraints)
        if self.only_unread:
            tcid_q = tcid_q.except_(db.filter_query(db.Reading.text_content_id))
        tcids = {tcid for tcid, in tcid_q.all()}
        if not tcids:
            logger.info("Nothing found to read with %s." % reader_name)
            return False
        self._run_reading(db, tcids, reader_name)
        return True

    @ReadingManager._run_all_readers
    def read_new(self, db, reader_name):
        """Update the readings and raw statements in the database."""
        from indra_reading.readers import get_reader_class

        self.end_datetime = self.run_datetime
        latest_updatetime = self._get_latest_updatetime(db, reader_name)
        if latest_updatetime is not None:
            self.begin_datetime = latest_updatetime - self.buffer
        else:
            raise ReadingUpdateError("There are no previous updates for %s. "
                                     "Please run_all." % reader_name)

        constraints = self._get_constraints(db, reader_name)

        tcid_q = db.filter_query(
            db.TextContent.id,
            db.TextContent.insert_date > self.begin_datetime,
            *constraints
            )
        if self.only_unread:
            reader_class = get_reader_class(reader_name)
            reader_version = reader_class.get_version()
            tcid_q = tcid_q.except_(
                db.filter_query(db.Reading.text_content_id,
                                db.Reading.reader == reader_name,
                                db.Reading.reader_version == reader_version)
            )
        tcids = {tcid for tcid, in tcid_q.all()}
        if not tcids:
            logger.info("Nothing new to read with %s." % reader_name)
            return False
        self._run_reading(db, tcids, reader_name)
        return True


class BulkAwsReadingManager(BulkReadingManager):
    """This is the reading manager when updating using AWS Batch.

    This takes all the parameters used by :py:class:`BulkReadingManager`, and
    in addition:

    Parameters
    ----------
    project_name : str
        You can select a name for the project for which this reading is being
        run. This name has a default value set in your config file. The batch
        jobs used in reading will be tagged with this project name, for
        accounting purposes.
    """
    timeouts = {
        'reach': 1200,
        'sparser': 600,
        'isi': 5400,
        'trips': 1200,
        'eidos': 2400,
        'mti': 5400,
    }

    ids_per_job = {
        'reach': 5000,
        'sparser': 5000,
        'isi': 5000,
        'trips': 500,
        'eidos': 1000,  # 5000 id jobs were often terminated by BidEvictionEvent
        'mti': None,  # meaning all content run in a single job.
    }

    batch_batch = {
        'mti': 100
    }

    def __init__(self, *args, **kwargs):
        self.project_name = kwargs.pop('project_name', None)
        super(BulkAwsReadingManager, self).__init__(*args, **kwargs)
        self.reader_versions = {}
        return

    def get_version(self, reader_name):
        if reader_name not in self.reader_versions.keys():
            logger.error("Expected to find %s in %s."
                         % (reader_name, self.reader_versions))
            raise ReadingUpdateError("Tried to access reader version before "
                                     "reading started.")
        elif self.reader_versions[reader_name] is None:
            logger.warning("Reader version was never written to s3.")
            return None
        return self.reader_versions[reader_name]

    def _run_reading(self, db, tcids, reader_name):
        from indra_db.reading.submitter import DbReadingSubmitter

        ids_per_job = self.ids_per_job[reader_name.lower()]
        if ids_per_job is not None and len(tcids)/ids_per_job >= 1000:
            raise ReadingUpdateError("Too many id's for one submission. "
                                     "Break it up and do it manually.")

        logger.info("Producing readings on aws for %d text refs with new "
                    "content not read by %s."
                    % (len(tcids), reader_name))
        group_name = '%s_reading' % reader_name.lower()
        basename = self.run_datetime.strftime('%Y%m%d_%H%M%S')
        file_name = '{group_name}_{basename}.txt'.format(group_name=group_name,
                                                         basename=basename)
        with open(file_name, 'w') as f:
            f.write('\n'.join(['%s' % tcid for tcid in tcids]))
        logger.info("Submitting jobs...")
        sub = DbReadingSubmitter(basename, [reader_name.lower()],
                                 project_name=self.project_name,
                                 group_name=group_name,
                                 batch_batch=self.batch_batch.get(reader_name))
        sub.submit_reading(file_name, 0, None,
                           self.ids_per_job[reader_name.lower()])

        logger.info("Waiting for complete...")
        sub.watch_and_wait(poll_interval=30,
                           idle_log_timeout=self.timeouts[reader_name.lower()],
                           kill_on_timeout=True, stash_log_method='s3')

        # Get the versions of the reader reader used in all the jobs, check for
        # consistancy and record the result (at least one result).
        rv_dict = sub.poll_reader_versions()
        for job_name, rvs in rv_dict.items():
            # Sometimes the job hasn't started yet, or else the job has crashed
            # instantly, before the reader version can be written. If the
            # latter, we shouldn't crash the reading monitor as a result.
            if rvs is None:
                logger.warning("Reader version was not yet available.")
                self.reader_versions[reader_name] = None
                continue

            # There should only be one reader per job.
            assert len(rvs) == 1 and reader_name in rvs.keys(), \
                "There should be only one reader: %s, but got %s." \
                % (reader_name, str(rvs))
            if reader_name not in self.reader_versions.keys():
                self.reader_versions[reader_name] = rvs[reader_name]
            elif self.reader_versions[reader_name] is None:
                logger.info("Found the reader version.")
                self.reader_versions[reader_name] = rvs[reader_name]
            elif self.reader_versions[reader_name] != rvs[reader_name]:
                logger.warning("Different jobs used different reader "
                               "versions: %s vs. %s"
                               % (self.reader_versions[reader_name],
                                  rvs[reader_name]))
        return


class BulkLocalReadingManager(BulkReadingManager):
    """This is the reading manager to be used when running reading locally.

    This takes all the parameters used by :py:class:`BulkReadingManager`, and
    in addition:

    Parameters
    ----------
    n_proc : int
        The number of processed to dedicate to reading. Note the some of the
        readers (e.g. REACH) do not always obey these restrictions.
    verbose : bool
        If True, more detailed logs will be printed. Default is False.
    """
    def __init__(self, *args, **kwargs):
        self.n_proc = kwargs.pop('n_proc', 1)
        self.verbose = kwargs.pop('verbose', False)
        super(BulkLocalReadingManager, self).__init__(*args, **kwargs)
        return

    def _run_reading(self, db, tcids, reader_name):
        from indra_db.reading import read_db as rdb
        ids_per_job = 5000
        if len(tcids) > ids_per_job:
            raise ReadingUpdateError("Too many id's to run locally. Try "
                                     "running on batch (use_batch).")
        logger.info("Producing readings locally for %d new text refs."
                    % len(tcids))
        base_dir = path.join(THIS_DIR, 'read_all_%s' % reader_name)
        readers = rdb.construct_readers([reader_name], base_dir=base_dir,
                                        n_proc=self.n_proc)

        rdb.run_reading(readers, tcids, db=db, batch_size=ids_per_job,
                        verbose=self.verbose)
        return


@click.group()
def reading():
    """Manage the reading jobs."""


@reading.command()
@click.argument('task', type=click.Choice(["all", "new"]))
@click.option('-b', '--buffer', type=int, default=1,
              help='Set the number of buffer days to read prior to the most '
                   'recent update. The default is 1 day.')
@click.option('--project-name', type=str,
              help="Set the project name to be different from the config "
                   "default.")
def run(task, buffer, project_name):
    """Manage the the reading of text content on AWS.

    \b
    Tasks:
    - "all": Read all the content available.
    - "new": Read only the new content that has not been read.
    """
    from indra_db.util import get_db
    db = get_db('primary')
    #readers = ['SPARSER', 'REACH', 'EIDOS', 'TRIPS', 'ISI', 'MTI']
    readers = ['SPARSER', 'REACH', 'EIDOS', 'TRIPS']
    bulk_manager = BulkAwsReadingManager(readers,
                                         buffer_days=buffer,
                                         project_name=project_name)
    if task == 'all':
        bulk_manager.read_all(db)
    elif task == 'new':
        bulk_manager.read_new(db)


@reading.command()
@click.argument('task', type=click.Choice(["all", "new"]))
@click.option('-b', '--buffer', type=int, default=1,
              help='Set the number of buffer days to read prior to the most '
                   'recent update. The default is 1 day.')
@click.option('-n', '--num-procs', type=int,
              help="Select the number of processors to use.")
def run_local(task, buffer, num_procs):
    """Run reading locally, save the results on the database.

    \b
    Tasks:
    - "all": Read all the content available.
    - "new": Read only the new content that has not been read.
    """
    from indra_db.util import get_db
    db = get_db('primary')

    #readers = ['SPARSER', 'REACH', 'TRIPS', 'ISI', 'EIDOS', 'MTI']
    readers = ['SPARSER', 'REACH', 'EIDOS', 'TRIPS']
    bulk_manager = BulkLocalReadingManager(readers,
                                           buffer_days=buffer,
                                           n_proc=num_procs)
    if task == 'all':
        bulk_manager.read_all(db)
    elif task == 'new':
        bulk_manager.read_new(db)


@reading.command('list')
def show_list():
    """List the readers and their most recent runs."""
    import tabulate
    from indra_db.util import get_db

    db = get_db('primary')
    rows = [(rn, format_date(lu))
            for rn, lu in ReadingManager.get_latest_updates(db).items()]
    headers = ('Reader', 'Last Updated')
    print(tabulate.tabulate(rows, headers))


