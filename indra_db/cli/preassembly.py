import click
from datetime import datetime

from indra_db import get_db
from indra_db.exceptions import IndraDbException

from .util import format_date

def filter_updates(stmt_type, pa_updates):
    return {u.run_datetime for u in pa_updates if u.stmt_type == stmt_type}


def list_last_updates(db):
    """Return a dict of the most recent updates for each statement type."""
    from indra_db.preassembly.submitter import VALID_STATEMENTS
    pa_updates = db.select_all(db.PreassemblyUpdates)
    last_full_update = max(filter_updates(None, pa_updates))
    last_updates = {st: max(filter_updates(st, pa_updates)
                            | {last_full_update})
                    for st in VALID_STATEMENTS}
    return last_updates


def list_latest_raw_stmts(db):
    """Return a dict of the most recent new raw statement for each type."""
    from sqlalchemy import func
    res = (db.session.query(db.RawStatements.type,
                            func.max(db.RawStatements.create_date))
                     .group_by(db.RawStatements.type)
                     .all())
    return {k: v for k, v in res}


def run_preassembly(mode, project_name):
    """Construct a submitter and begin submitting jobs to Batch for preassembly.

    This function will determine which statement types need to be updated and
    how far back they go, and will create the appropriate
    :class:`PreassemblySubmitter
    <indra_db.preassembly.submitter.PreassemblySubmitter>`
    instance, and run the jobs with pre-set parameters on statement types that
    need updating.

    Parameters
    ----------
    project_name : str
        This name is used to gag the various AWS resources used for accounting
        purposes.
    """
    from indra_db.preassembly.submitter import VALID_STATEMENTS, \
        PreassemblySubmitter
    db = get_db('primary')
    if mode == 'update':
        # Find the latest update for each statement type.
        last_updates = list_last_updates(db)

        # Get the most recent raw statement datetimes
        latest_raw_stmts = list_latest_raw_stmts(db)

        # Only include statements types that have new raw statements.
        need_to_update = [s_type for s_type, last_upd in last_updates.items()
                          if s_type in latest_raw_stmts.keys()
                          and latest_raw_stmts[s_type] > last_upd]
    else:
        # Make sure the pa_statements table is truly empty.
        if db.select_one(db.PAStatements):
            raise IndraDbException("Please clear the pa_statements table "
                                   "before running create. If you want to run "
                                   "an incremental update, please run with "
                                   "mode 'update'.")

        # Just run them all.
        need_to_update = VALID_STATEMENTS[:]

    # Create the submitter, and run it.
    basename = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    ps = PreassemblySubmitter(basename, mode, project_name=project_name)
    ps.set_max_jobs(4)
    ps.run(need_to_update, 100000, True, stagger=600, poll_interval=120)


@click.group()
def pa():
    """Manage the preassembly pipeline."""


@pa.command()
@click.argument('task', type=click.Choice(['create', 'update']),
                required=True)
@click.argument('project-name', required=False)
def run(task, project_name):
    """Manage the indra_db preassembly.

    \b
    Tasks:
     - "create": populate the pa_statements table for the first time (this
       requires that the table be empty).
     - "update": update the existing content in pa_statements with the latest
       from raw statements.

    A project name is required to tag the AWS instances with a "project" tag.
    """
    run_preassembly(task, project_name)


@pa.command('list')
@click.option('-r', '--with-raw', is_flag=True,
              help="Include the latest datetimes for raw statements of each "
                   "type. This will take much longer.")
def show_list(with_raw):
    """List the latest updates for each type of Statement."""
    import tabulate

    db = get_db('primary')
    rows = [(st, lu) for st, lu in list_last_updates(db).items()]
    header = ('Statement Type', 'Last Update')
    if with_raw:
        print("This may take a while...", end='', flush=True)
        raw_stmt_dates = list_latest_raw_stmts(db)
        print("\r", end='')
        new_rows = []
        for st, lu in rows:
            raw_date = raw_stmt_dates.get(st)
            if raw_date is None:
                new_rows.append((st, format_date(lu), "[None]", "No"))
            else:
                new_rows.append((st, format_date(lu), format_date(raw_date),
                                 "Yes" if raw_date > lu else "No"))
        rows = new_rows
        header += ('Latest Raw Stmt', 'Needs Update?')
    else:
        rows = [(st, format_date(lu)) for st, lu in rows]
    rows.sort()
    print(tabulate.tabulate(rows, header))


