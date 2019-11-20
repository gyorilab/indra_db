"""This script is intended to be run on an Amazon ECS container, so information
for the job either needs to be provided in environment variables (e.g., the
REACH version and path) or loaded from S3 (e.g., the list of PMIDs).
"""
import json
import os
import sys
import boto3
import botocore
import logging
import random
from argparse import ArgumentParser

from indra.tools.reading.readers import get_reader_classes

from indra_db.reading.read_db import run_reading, construct_readers
from indra_db.reading.report_db_aws import DbAwsStatReporter


logger = logging.getLogger('read_db_aws')
logger.setLevel(logging.DEBUG)


bucket_name = 'bigmech'


def get_parser():
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
        choices=[rc.name.lower() for rc in get_reader_classes()],
        nargs='+',
        help='Choose which reader(s) to use.'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help="Use the test database."
    )
    return parser


def get_s3_root(basename, job_queue="run_db_reading_queue"):
    return 'reading_results/%s/logs/%s/' % (basename, job_queue)


def get_s3_job_prefix(basename, job_name, job_queue="run_db_reading_queue"):
    s3_root = get_s3_root(basename, job_queue)
    return s3_root + '%s/' % job_name


def get_s3_list_loc(basename):
    return 'reading_results/%s/id_list' % basename


def get_s3_reader_version_loc(*args, **kwargs):
    prefix = get_s3_job_prefix(*args, **kwargs)
    return prefix + 'reader_versions.json'


def main():
    arg_parser = get_parser()
    args = arg_parser.parse_args()

    s3 = boto3.client('s3')
    s3_log_prefix = get_s3_job_prefix(args.basename, args.job_name)
    id_list_key = get_s3_list_loc(args.basename)
    try:
        id_list_obj = s3.get_object(Bucket=bucket_name, Key=id_list_key)
    except botocore.exceptions.ClientError as e:
        # Handle a missing object gracefully
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.info('Could not find PMID list file at %s, exiting' %
                        id_list_key)
            sys.exit(1)
        # If there was some other kind of problem, re-raise the exception
        else:
            raise e

    # Get the content from the object
    id_list_str = id_list_obj['Body'].read().decode('utf8').strip()
    id_str_list = id_list_str.splitlines()[args.start_index:args.end_index]
    random.shuffle(id_str_list)
    tcids = [int(line.strip()) for line in id_str_list]

    # Get the reader objects
    kwargs = {'base_dir': args.basename, 'n_proc': args.num_cores}
    readers = construct_readers(args.readers, **kwargs)

    # Record the reader versions used in this run.
    reader_versions = {}
    for reader in readers:
        reader_versions[reader.name] = reader.get_version()
    s3.put_object(Bucket=bucket_name,
                  Key=get_s3_reader_version_loc(args.basename, args.job_name),
                  Body=json.dumps(reader_versions))

    # Some combinations of options don't make sense:
    forbidden_combos = [('all', 'unread'), ('none', 'unread'), ('none', 'none')]
    assert (args.read_mode, args.stmt_mode) not in forbidden_combos, \
        ("The combination of reading mode %s and statement mode %s is not "
         "allowed." % (args.reading_mode, args.stmt_mode))

    # Get a handle for the database
    if args.test:
        from indra_db.tests.util import get_temp_db
        db = get_temp_db(clear=True)
    else:
        db = None

    # Read everything ========================================
    workers = run_reading(readers, tcids, verbose=True, db=db,
                          reading_mode=args.read_mode,
                          stmt_mode=args.stmt_mode)

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
            s3.put_object(Key=s3_key, Body=f.read(),
                          Bucket=bucket_name)

    # Create a summary report.
    rep = DbAwsStatReporter(args.job_name, s3_log_prefix, s3, bucket_name)
    rep.report_statistics(workers)


if __name__ == '__main__':
    main()
