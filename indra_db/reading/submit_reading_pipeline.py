""""This file acts as a script to run large batch jobs on AWS.

The key components are the DbReadingSubmitter class, and the submit_db_reading
function. The function is provided as a shallow wrapper for backwards
compatibility, and may eventually be removed. The preferred method for running
large batches via the ipython, or from a python environment, is the following:

>> sub = DbReadingSubmitter('name_for_run', ['reach', 'sparser'])
>> sub.set_options(prioritize=True)
>> sub.submit_reading('file/location/of/ids_to_read.txt', 0, None, ids_per_job=1000)
>> sub.watch_and_wait(idle_log_timeout=100, kill_on_timeout=True)

Additionally, this file may be run as a script. For details, run

bash$ python submit_reading_pipeline.py --help

In your favorite command line.
"""
import json
from asyncio import sleep

import boto3
import logging
import botocore
from datetime import datetime
from functools import partial

from indra_reading.util import get_s3_log_prefix
from indra_reading.scripts.submit_reading_pipeline import create_submit_parser,\
    create_read_parser, Submitter

from indra_db.reading.read_db_aws import get_s3_reader_version_loc, bucket_name

logger = logging.getLogger('indra_db_submitter')


class DbReadingSubmitter(Submitter):
    """A class for the management of a batch of reading jobs on AWS.

    Parameters
    ----------
    basename : str
        The name of this batch of readings. This will be used to distinguish
        the jobs on AWS batch, and the logs on S3.
    readers : list[str]
        A list of the names of readers to use in this job.
    project_name : str
        Optional. Used for record-keeping on AWS.
    group_name : Optional[str]
        Indicate the name of this group of readings.

    Other keyword parameters go to the `get_options` method.
    """
    _s3_input_name = 'id_list'
    _purpose = 'db_reading'
    _job_queue_dict = {'run_db_reading_queue': ['reach', 'sparser', 'isi'],
                       'run_db_trips_queue': ['trips']}
    _job_def_dict = {'run_db_reading_jobdef': ['reach', 'sparser'],
                     'run_db_reading_isi_jobdef': ['isi'],
                     'run_db_reading_trips_jobdef': ['trips']}

    def __init__(self, *args, **kwargs):
        super(DbReadingSubmitter, self).__init__(*args, **kwargs)
        self.s3_prefix = get_s3_log_prefix(self.s3_base)
        self.time_tag = datetime.now().strftime('%Y%m%d_%H%M')
        self.run_record = {}
        self.start_time = None
        self.end_time = None
        return

    def submit_reading(self, *args, **kwargs):
        self.start_time = datetime.utcnow()
        super(DbReadingSubmitter, self).submit_reading(*args, **kwargs)
        return

    def _get_base(self, job_name, start_ix, end_ix):
        read_mode = self.options.pop('read_mode', 'unread')
        stmt_mode = self.options.pop('stmt_mode', 'all')

        base = ['python3', '-m', 'indra_db.reading.read_db_aws',
                self.job_base, job_name, self.s3_base]
        base += ['/sw/tmp', read_mode, stmt_mode, '32', str(start_ix),
                 str(end_ix)]
        return base

    def _get_extensions(self):
        extensions = []
        for key, val in self.options.items():
            if val is not None:
                extensions.extend(['--' + key, val])
        return extensions

    def set_options(self, stmt_mode='all', read_mode='unread',
                    max_reach_input_len=None, max_reach_space_ratio=None):
        """Set the options for this reading job.

        Parameters
        ----------
        stmt_mode : bool
            Optional, default 'all' - If 'all', produce statements for all
            content for all readers. If the readings were already produced,
            they will be retrieved from the database if `read_mode` is 'none'
            or 'unread'. If this option is 'unread', only the newly produced
            readings will be processed. If 'none', no statements will be
            produced.
        read_mode : str : 'all', 'unread', or 'none'
            Optional, default 'unread' - If 'all', read everything (generally
            slow); if 'unread', only read things that were unread, (the cache
            of old readings may still be used if `stmt_mode='all'` to get
            everything); if 'none', don't read, and only retrieve existing
            readings.
        max_reach_input_len : int
            The maximum number of characters to all for inputs to REACH. The
            reader tends to hang up on larger papers, and beyond a certain
            threshold, greater length tends to imply errors in formatting or
            other quirks.
        max_reach_space_ratio : float in [0,1]
            Some content erroneously has spaces between all characters. The
            fraction of characters that are spaces is a fast and simple way to
            catch and avoid such problems. Recommend a value of 0.5.
        """
        self.options['stmt_mode'] = stmt_mode
        self.options['read_mode'] = read_mode
        self.options['max_reach_input_len'] = max_reach_input_len
        self.options['max_reach_space_ratio'] = max_reach_space_ratio
        return

    def poll_reader_versions(self):
        """Look up the self-reported reader versions for all the jobs."""
        ret = {}
        s3 = boto3.client('s3')
        for job_list in self.job_lists.values():
            for job_d in job_list:
                job_name = job_d['jobName']
                s3_key = get_s3_reader_version_loc(self.s3_base, job_name)
                try:
                    res = s3.get_object(Bucket=bucket_name, Key=s3_key)
                except botocore.exceptions.ClientError as e:
                    # Handle a missing object gracefully
                    if e.response['Error']['Code'] == 'NoSuchKey':
                        logger.info('Could not find reader version json at %s.' %
                                    s3_key)
                        ret[job_name] = None
                    # If there was some other kind of problem, log an error
                    else:
                        logger.error("Encountered unexpected error accessing "
                                     "reader version json: " + str(e))
                    continue
                rv_json = json.loads(res['Body'].read())
                ret[job_name] = rv_json
        return ret

    def watch_and_wait(self, *args, **kwargs):
        """Watch the logs of the batch jobs and wait for all jobs to complete.

        Logs are monitored, and jobs may be killed if no output is seen for a
        given amount of time. Essential if jobs are left un-monitored (by
        humans) for any length of time.

        Parameters
        ----------
        poll_interval: int
            Default 10. The number of seconds to wait between examining logs and
            the states of the jobs.
        idle_log_timeout : int or None,
            Default is None. If an int, sets the number of seconds to wait
            before declaring a job timed out. This parameter alone does not lead
            to the deletion/stopping of stalled jobs.
        kill_on_timeout : bool
            Default is False. If true, and a job is deemed to have timed out,
            kill the job.
        stash_log_method : str or None
            Default is None. If a string is given, logs will be saved in the
            indicated location. Value choices are 's3' or 'local'.
        tag_instances : bool
            Default is False. In the past, it was necessary to tag instances
            from the outside. THey should now be tagging themselves, however if
            for any reason external measures are needed, this option may be set
            to True.
        """
        kwargs['result_record'] = self.run_record
        super(DbReadingSubmitter, self).watch_and_wait(*args, **kwargs)
        self.end_time = datetime.utcnow()


if __name__ == '__main__':
    import argparse

    parent_submit_parser = create_submit_parser()
    parent_read_parser = create_read_parser()

    parser = argparse.ArgumentParser(
        'indra_db.reading.submig_reading_pipeline.py',
        parents=[parent_submit_parser, parent_read_parser],
        description=('Run reading with content on the db and submit results. '
                     'In this option, ids in \'input_file\' are given in the '
                     'as one text content id per line.'),
        )
    parser.add_argument(
        '-S', '--stmt_mode',
        choices=['all', 'unread', 'none'],
        default='all',
        help='Choose the subset of statements on which to run reading.'
    )
    parser.add_argument(
        '-R', '--read_mode',
        choices=['all', 'unread', 'none'],
        default='unread',
        help=('Choose whether you want to read everything, nothing, or only '
              'the content that hasn\'t been read.')
    )
    parser.add_argument(
        '--max_reach_space_ratio',
        type=float,
        help='Set the maximum ratio of spaces to non-spaces for REACH input.',
        default=None
    )
    parser.add_argument(
        '--max_reach_input_len',
        type=int,
        help='Set the maximum length of content that REACH will read.',
        default=None
    )
    parser.add_argument(
        '--no_wait',
        action='store_true',
        help=('Don\'t run wait_for_complete at the end of the script. '
              'NOTE: wait_for_complete should always be run, so if it is not '
              'run here, it should be run manually.')
    )
    parser.add_argument(
        '--idle_log_timeout',
        type=int,
        default=600,
        help=("Set the time to wait for any given reading job to continue "
              "without any updates to the logs (at what point do you want it "
              "assumed dead).")
    )
    parser.add_argument(
        '--no_kill_on_timeout',
        action='store_true',
        help="If set, do not kill processes that have timed out."
    )
    args = parser.parse_args()

    logger.info("Reading for %s" % args.readers)
    sub = DbReadingSubmitter(args.basename, args.readers, args.project)
    sub.set_options(args.stmt_mode, args.read_mode,
                    args.max_reach_input_len, args.max_reach_space_ratio)

    submit = partial(sub.submit_reading, args.input_file, args.start_ix,
                     args.end_ix, args.ids_per_job, stagger=args.stagger)
    if args.stagger > 0:
        from threading import Thread
        th = Thread(target=submit)
        th.start()
        while not sub.monitors:
            sleep(1)
            logger.info("Waiting for monitors...")
    else:
        submit()

    if not args.no_wait:
        sub.watch_and_wait(idle_log_timeout=args.idle_log_timeout,
                           kill_on_timeout=not args.no_kill_on_timeout)

    if args.stagger > 0:
        th.join()
