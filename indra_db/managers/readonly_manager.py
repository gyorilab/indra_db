"""Manage Materialized Views
---------------------------------------------------------------------
We use materialized views to allow fast and efficient load of data,
and to add a layer of separation between the processes of updating
the content of the database and accessing the content of the
database. Note that the materialized views are not created using SQL Alchemy.

The following views must be built in this specific order:
  1. fast_raw_pa_link
  2. evidence_counts
  3. pa_meta
  4. raw_stmt_src
  5. pa_stmt_src
The following can be built at any time and in any order:
  - reading_ref_link
Note that the order of views below is determined not by the above
order but by constraints imposed by use-case.
"""
import logging
from datetime import datetime
from subprocess import check_call
from argparse import ArgumentParser

from indra_db.util import get_db
from indra_db.config import S3_DUMP
from indra_db.exceptions import IndraDbException

logger = logging.getLogger(__name__)


def _form_db_args(url):
    """Arrange the url elements into a list of arguments for pg calls."""
    return ['-h', url.host,
            '-U', url.username,
            '-W', url.password,
            '-d', url.database]


def dump(principal_db, dump_file=None):
    """Dump the readonly schema to s3."""
    # Form the name of the s3 file, if not given.
    if dump_file is None:
        dump_file = 's3://{bucket}/{prefix}'.format(**S3_DUMP)
        if not dump_file.endswith('/'):
            dump_file += '/'
        now_str = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
        dump_file += 'readonly-%s.dump' % now_str

    # Dump the database onto s3, piping through this machine.
    check_call(["pg_dump",
                *_form_db_args(principal_db.url),
                '-n', 'readonly',
                '|', 'aws', 's3', 'cp', '-', dump_file])
    return dump_file


def push(readonly_db, dump_file, force_clear=True):
    """Load the data from a dump onto a database."""
    # Make sure the database is clear.
    if readonly_db.get_active_tables():
        if force_clear:
            readonly_db.drop_tables()
        else:
            raise IndraDbException("Tables already exist and force_clear "
                                   "is False.")

    # Pipe the database dump from s3 through this machine into the database.
    check_call(['aws', 's3', 'cp', dump_file, '-', '|',
                'psql', *_form_db_args(readonly_db.url)])

    return


def transfer_readonly(principal_db, readonly_db):
    """Move the contents of the new schema from the old DB to the new db."""
    logger.info("Beginning dump of database (est. 1 + epsilon hours)")
    dump_file = dump(principal_db)

    logger.info("Beginning upload of content (est. ~30 minutes)")
    push(readonly_db, dump_file)
    return


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Manage the materialized views.'
        )
    parser.add_argument(
        choices=['create', 'update'],
        dest='task',
        help=('Choose whether you want to create the materialized views for '
              'the first time, or simply update existing views. Create is '
              'necessary if the definition of the view changes.')
        )
    parser.add_argument(
        '-D', '--database',
        default='primary',
        help=('Choose a database from the names given in the config or '
              'environment, for example primary is INDRA_DB_PRIMAY in the '
              'config file and INDRADBPRIMARY in the environment. The default '
              'is \'primary\'.')
        )
    parser.add_argument(
        '-m', '--m_views',
        default='all',
        nargs='+',
        help='Specify certain views to create or refresh.'
        )

    args = parser.parse_args()

    if args.m_views == 'all':
        views = None
    else:
        views = args.m_views

    db = get_db(args.database)
    db.generate_readonly(args.task, view_list=views)
