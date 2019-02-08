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
from argparse import ArgumentParser

from indra_db.util import get_db

logger = logging.getLogger("materialized_views")

ORDERED_VIEWS = ['fast_raw_pa_link', 'evidence_counts', 'pa_meta',
                 'raw_stmt_src', 'pa_stmt_src']
OTHER_VIEWS = {'reading_ref_link'}


def create_views(db):
    """Create the materialized views."""
    for i, view in enumerate(ORDERED_VIEWS):
        logger.info('%d. Creating %s view...' % (i, view))
        db.create_materialized_view(view)
    for view in ORDERED_VIEWS:
        logger.info('- Creating %s view...' % view)
        db.create_materialized_view(view)
    return


def refresh_views(db):
    """Update the materialized views."""
    for i, view in enumerate(ORDERED_VIEWS):
        logger.info('%d. Refreshing %s view...' % (i, view))
        db.refresh_materialized_view(view)
    for view in OTHER_VIEWS:
        logger.info('- Refreshing %s view...' % view)
        db.refresh_materialized_view(view)
    return


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Manage the materialized views.'
        )
    parser.add_argument(
        choices=['create', 'refresh'],
        dest='task',
        help=('Choose whether you want to create the materialized views for '
              'the first time, or simply refresh existing views. Create is '
              'necessary if the definition of the view changes.')
        )
    parser.add_argument(
        '-D', '--database',
        default='primary',
        help=('Select a database from the names given in the config or '
              'environment, for example primary is INDRA_DB_PRIMAY in the '
              'config file and INDRADBPRIMARY in the environment. The default '
              'is \'primary\'.')
        )

    args = parser.parse_args()

    db = get_db(args.database)
    if args.task == 'create':
        create_views(db)
    else:
        refresh_views(db)

