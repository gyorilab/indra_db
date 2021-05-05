from argparse import ArgumentParser
from datetime import datetime

from indra_db import get_db
from indra_db.exceptions import IndraDbException
from indra_db.preassembly.submitter import VALID_STATEMENTS, \
    PreassemblySubmitter


def filter_updates(stmt_type, pa_updates):
    return {u.run_datetime for u in pa_updates if u.stmt_type == stmt_type}


def list_last_updates(db):
    """Return a dict of the most recent updates for each statement type."""
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
