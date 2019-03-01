"""This script is intended to be run on an Amazon ECS container, so information
for the job either needs to be provided in environment variables (e.g., the
REACH version and path) or loaded from S3 (e.g., the list of PMIDs).
"""
import os
import sys
import boto3
import botocore
import logging
import random
from datetime import datetime

from indra_db.reading.read_db import run_reading, construct_readers
from indra_db.reading.report_db_aws import DbAwsStatReporter


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description=__doc__
        )
    parser.add_argument(
        dest='basename',
        help='The name of this run.'
        )
    parser.add_argument(
        dest='job_name',
        help='The name of this job.'
        )
    parser.add_argument(
        dest='out_dir',
        help='The name of the temporary output directory'
        )
    parser.add_argument(
        dest='read_mode',
        choices=['all', 'unread', 'none'],
        help=("Set the reading mode. If 'all', read everything, if "
              "'unread', only read content that does not have pre-existing "
              "readings of the same reader and version, if 'none', only "
              "use pre-existing readings. Default is 'unread'.")
        )
    parser.add_argument(
        dest='stmt_mode',
        choices=['all', 'unread', 'none'],
        help=("Choose which readings should produce statements. If 'all', all "
              "readings that are produced or retrieved will be used to produce "
              "statements. If 'unread', only produce statements from "
              "previously unread content. If 'none', do not produce any "
              "statements (only readings will be produced).")
        )
    parser.add_argument(
        dest='num_cores',
        help='Select the number of cores on which to run.',
        type=int
        )
    parser.add_argument(
        dest='start_index',
        help='Select the index of the first pmid in the list to read.',
        type=int
        )
    parser.add_argument(
        dest='end_index',
        help='Select the index of the last pmid in the list to read.',
        type=int
        )
    parser.add_argument(
        '-r', '--readers',
        dest='readers',
        choices=['reach', 'sparser'],
        nargs='+',
        help='Choose which reader(s) to use.'
        )
    parser.add_argument(
        '--test',
        action='store_true',
        help="Use the test database."
        )
    args = parser.parse_args()

    logger = logging.getLogger('read_db_aws')
    logger.setLevel(logging.DEBUG)

    client = boto3.client('s3')
    bucket_name = 'bigmech'
    id_list_key = 'reading_results/%s/id_list' % args.basename
    try:
        id_list_obj = client.get_object(
            Bucket=bucket_name,
            Key=id_list_key
            )
    # Handle a missing object gracefully
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.info('Could not find PMID list file at %s, exiting' %
                        id_list_key)
            sys.exit(1)
        # If there was some other kind of problem, re-raise the exception
        else:
            raise e

    # Get the reader objects
    kwargs = {'base_dir': args.basename, 'n_proc': args.num_cores}
    readers = construct_readers(args.readers, **kwargs)

    # Get the content from the object
    id_list_str = id_list_obj['Body'].read().decode('utf8').strip()
    id_str_list = id_list_str.splitlines()[args.start_index:args.end_index]
    random.shuffle(id_str_list)
    tcids = [int(line.strip()) for line in id_str_list]

    # Some combinations of options don't make sense:
    forbidden_combos = [('all', 'unread'), ('none', 'unread'), ('none', 'none')]
    assert (args.read_mode, args.stmt_mode) not in forbidden_combos, \
        ("The combination of reading mode %s and statement mode %s is not "
         "allowed." % (args.reading_mode, args.stmt_mode))

    # Init some timing dicts
    starts = {}
    ends = {}

    # Get a handle for the database
    if args.test:
        from indra_db import util as dbu
        db = dbu.get_test_db()
    else:
        db = None

    s3_log_prefix = ('reading_results/%s/logs/run_db_reading_queue/%s/'
                     % (args.basename, args.job_name))

    # Read everything ========================================
    starts['reading'] = datetime.now()
    workers = run_reading(readers, tcids, verbose=True, db=db,
                          reading_mode=args.read_mode,
                          stmt_mode=args.stmt_mode)
    ends['reading'] = datetime.now()

    # Preserve the sparser logs
    contents = os.listdir('.')
    sparser_logs = [fname for fname in contents
                    if fname.startswith('sparser') and fname.endswith('log')]
    sparser_log_dir = s3_log_prefix + 'sparser_logs/'
    for fname in sparser_logs:
        s3_key = sparser_log_dir + fname
        logger.info("Saving sparser logs to %s on s3 in %s."
                    % (s3_key, bucket_name))
        with open(fname, 'r') as f:
            client.put_object(Key=s3_key, Body=f.read(),
                              Bucket=bucket_name)

    # Create a summary report.
    rep = DbAwsStatReporter(args.job_name, s3_log_prefix, client, bucket_name)
    reading_outputs = [rd for worker in workers
                       for rd in worker.extant_readings + worker.new_readings]
    stmt_outputs = [s for worker in workers for s in worker.statement_outputs]
    rep.report_statistics(workers)
