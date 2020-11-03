from argparse import ArgumentParser
from datetime import datetime

from indra_db import get_db
from indra_db.preassembly.preassembly_submitter import VALID_STATEMENTS, \
    PreassemblySubmitter


def filter_updates(stmt_type, pa_updates):
    return {u.run_datetime for u in pa_updates if u.stmt_type == stmt_type}


def main(project_name):
    db = get_db('primary')
    pa_updates = db.select_all(db.PreassemblyUpdates)
    last_full_update = max(filter_updates(None, pa_updates))
    last_updates = {st: max(filter_updates(st, pa_updates) | {last_full_update})
                    for st in VALID_STATEMENTS}

    need_to_update = []
    for stmt_type, last_update in last_updates.items():
        res = db.select_one(db.RawStatements,
                            db.RawStatements.type == stmt_type,
                            db.RawStatements.create_date > last_update)
        if res:
            need_to_update.append(stmt_type)

    basename = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    ps = PreassemblySubmitter(basename, 'update', project_name=project_name)
    ps.run(need_to_update, 100000, True, stagger=600, poll_interval=120)


def get_parser():
    parser = ArgumentParser('Manage Database preassembly jobs.')
    parser.add_argument(
        '--project-name',
        help=("The name of the project tag to be applied to the various "
              "resources used.")
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args.project_name)
