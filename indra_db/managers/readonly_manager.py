import logging
from datetime import datetime
from argparse import ArgumentParser

from indra_db.util import get_db, get_ro

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    if args.m_views == 'all':
        ro_names = None
    else:
        ro_names = args.m_views

    principal_db = get_db(args.database)
    readonly_db = get_ro(args.readonly)

    logger.info("%s - Generating readonly schema (est. a long time)"
                % datetime.now())
    principal_db.generate_readonly(ro_list=ro_names)

    logger.info("%s - Beginning dump of database (est. 1 + epsilon hours)"
                % datetime.now())
    dump_file = principal_db.dump_readonly()

    logger.info("%s - Beginning upload of content (est. ~30 minutes)"
                % datetime.now())
    readonly_db.load_dump(dump_file)
    return


def parse_args():
    parser = ArgumentParser(
        description='Manage the materialized views.'
    )
    parser.add_argument(
        '-D', '--database',
        default='primary',
        help=('Choose a database from the names given in the config or '
              'environment, for example primary is [primary] in the '
              'config file and INDRADBPRIMARY in the environment. The default '
              'is \'primary\'.')
    )
    parser.add_argument(
        '-R', '--readonly',
        default='primary',
        help=('Choose a readonly database from the names given in the config '
              'file, or INDRARO... in the env (e.g. INDRAROPRIMARY for the '
              '"primary" database.')
    )
    parser.add_argument(
        '-m', '--m_views',
        default='all',
        nargs='+',
        help='Specify certain views to create or refresh.'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
