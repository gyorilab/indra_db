import logging
from os import path
from functools import wraps
from datetime import datetime, timedelta
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from indra.tools.reading.readers import get_reader_class
from indra_db.reading import read_db as rdb
from indra_db.util import get_primary_db, get_db
from indra_db.reading.submit_reading_pipeline import DbReadingSubmitter

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
    """
    def __init__(self, reader_names, buffer_days=1):
        self.reader_names = reader_names
        self.buffer = timedelta(days=buffer_days)
        self.run_datetime = None
        self.begin_datetime = None
        self.end_datetime = None
        return

    @classmethod
    def _run_all_readers(cls, func):
        @wraps(func)
        def run_and_record_update(self, db, *args, **kwargs):
            for reader_name in self.reader_names:
                self.run_datetime = datetime.utcnow()
                completed = func(self, db, reader_name, *args, **kwargs)
                if completed:
                    is_read_all = (func.__name__ == 'read_all')
                    reader_version = get_reader_class(reader_name).get_version()
                    db.insert('reading_updates', complete_read=is_read_all,
                              reader=reader_name,
                              reader_version=reader_version,
                              run_datetime=self.run_datetime,
                              earliest_datetime=self.begin_datetime,
                              latest_datetime=self.end_datetime)
            return completed
        return run_and_record_update

    def _get_latest_updatetime(self, db, reader_name):
        """Get the date of the latest update."""
        update_list = db.select_all(
            db.ReadingUpdates,
            db.ReadingUpdates.reader == reader_name
            )
        if not len(update_list):
            logger.warning("The database has not had an initial upload, or "
                           "else the updates table has not been populated.")
            return None

        return max([u.latest_datetime for u in update_list])

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
    def _run_reading(self, db, tcids, reader_name, ids_per_job=5000):
        raise NotImplementedError("_run_reading must be defined in child.")

    @ReadingManager._run_all_readers
    def read_all(self, db, reader_name):
        """Read everything available on the database."""
        self.end_datetime = self.run_datetime
        tcids = {tcid for tcid, in db.select_all(db.TextContent.id)}
        if not tcids:
            logger.info("Nothing found to read with %s." % reader_name)
            return False
        self._run_reading(db, tcids, reader_name)
        return True

    @ReadingManager._run_all_readers
    def read_new(self, db, reader_name):
        """Update the readings and raw statements in the database."""
        self.end_datetime = self.run_datetime
        latest_updatetime = self._get_latest_updatetime(db, reader_name)
        if latest_updatetime is not None:
            self.begin_datetime = latest_updatetime - self.buffer
        else:
            raise ReadingUpdateError("There are no previous updates. "
                                     "Please run_all.")
        tcid_q = db.filter_query(
            db.TextContent.id,
            db.TextContent.insert_date > self.begin_datetime
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
    def __init__(self, *args, **kwargs):
        self.project_name = kwargs.pop('project_name', None)
        super(BulkAwsReadingManager, self).__init__(*args, **kwargs)
        return

    def _run_reading(self, db, tcids, reader_name, ids_per_job=5000):
        if len(tcids)/ids_per_job >= 1000:
            raise ReadingUpdateError("Too many id's for one submission. "
                                     "Break it up and do it manually.")

        logger.info("Producing readings on aws for %d text refs with new "
                    "content not read by %s."
                    % (len(tcids), reader_name))
        job_prefix = ('%s_reading_%s'
                      % (reader_name.lower(),
                         self.run_datetime.strftime('%Y%m%d_%H%M%S')))
        with open(job_prefix + '.txt', 'w') as f:
            f.write('\n'.join(['%s' % tcid for tcid in tcids]))
        logger.info("Submitting jobs...")
        sub = DbReadingSubmitter(job_prefix, [reader_name.lower()],
                                 self.project_name)
        sub.submit_reading(job_prefix + '.txt', 0, None, ids_per_job)

        logger.info("Waiting for complete...")
        sub.watch_and_wait(idle_log_timeout=1200, kill_on_timeout=True,
                           stash_log_method='s3')
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

    def _run_reading(self, db, tcids, reader_name, ids_per_job=5000):
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


def get_parser():
    parser = ArgumentParser(
        description='Manage content on INDRA\'s database.'
    )
    parent_read_parser = ArgumentParser(add_help=False)
    parent_read_parser.add_argument(
        choices=['read_all', 'read_new'],
        dest='task',
        help=('Choose whether you want to try to read everything, or only '
              'read the content added since the last update. Note that '
              'content from one day before the latests update will also be '
              'checked, to avoid content update overlap errors. Note also '
              'that no content will be re-read in either case.')
    )
    parent_read_parser.add_argument(
        '-t', '--test',
        action='store_true',
        help='Run tests using one of the designated test databases.'
    )
    parent_read_parser.add_argument(
        '-b', '--buffer',
        type=int,
        default=1,
        help=('Set the number number of buffer days read prior to the most '
              'recent update. The default is 1 day.')
    )
    local_read_parser = ArgumentParser(add_help=False)
    local_read_parser.add_argument(
        '-n', '--num_procs',
        dest='num_procs',
        type=int,
        default=1,
        help=('Select the number of processors to use during this operation. '
              'Default is 1.')
    )
    parser.add_argument(
        '--database',
        default='primary',
        help=('Choose a database from the names given in the config or '
              'environment, for example primary is INDRA_DB_PRIMAY in the '
              'config file and INDRADBPRIMARY in the environment. The default '
              'is \'primary\'. Note that this is overwridden by use of the '
              '--test flag if \'test\' is not a part of the name given.')
    )
    aws_read_parser = ArgumentParser(add_help=False)
    aws_read_parser.add_argument(
        '--project_name',
        help=('For use with --use_batch. Set the name of the project for '
              'which this reading is being done. This is used to label jobs '
              'on aws batch for monitoring and accounting purposes.')
    )
    subparsers = parser.add_subparsers(title='Method')
    subparsers.required = True
    subparsers.dest = 'method'

    local_desc = 'Run the reading for the update locally.'
    subparsers.add_parser(
        'local',
        parents=[parent_read_parser, local_read_parser],
        help=local_desc,
        description=local_desc,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    aws_desc = 'Run the reading for the update on amazon batch.'
    subparsers.add_parser(
        'aws',
        parents=[parent_read_parser, aws_read_parser],
        help=aws_desc,
        description=aws_desc,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.test:
        if 'test' not in args.database:
            from indra_db.tests.util import get_temp_db
            db = get_temp_db()
        else:
            db = get_db(args.database)
    elif args.database == 'primary':
        db = get_primary_db()
    else:
        db = get_db(args.database)

    readers = ['SPARSER', 'REACH', 'ISI']
    if args.method == 'local':
        bulk_manager = BulkLocalReadingManager(readers,
                                               buffer_days=args.buffer,
                                               n_procs=args.num_procs)
    elif args.method == 'aws':
        bulk_manager = BulkAwsReadingManager(readers,
                                             buffer_days=args.buffer,
                                             project_name=args.project_name)
    else:
        assert False, "This shouldn't be allowed."

    if args.task == 'read_all':
        bulk_manager.read_all(db)
    elif args.task == 'read_new':
        bulk_manager.read_new(db)
    return


if __name__ == '__main__':
    main()
