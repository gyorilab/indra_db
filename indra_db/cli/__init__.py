import click
from datetime import datetime

from indra_db import get_db


@click.group()
def main():
    """INDRA Database Infrastructure CLI

    The INDRA Database is both a physical database and an infrastructure for
    managing and updating the content of that physical database. This CLI
    is used for executing these management commands.
    """


def format_date(dt):
    if not isinstance(dt, datetime):
        return dt
    return dt.strftime("%Y %b %d %I:%M%p")


@main.command()
@click.argument("task", type=click.Choice(["upload", "update"]))
@click.argument("sources", nargs=-1, type=click.STRING, required=False)
def kb_run(task, sources):
    """Upload/update the knowledge bases used by the database.

    \b
    Usage tasks are:
     - upload: use if the knowledge bases have not yet been added.
     - update: if they have been added, but need to be updated.

    Specify which knowledge base sources to update by their name, e.g. "Pathway
    Commons" or "pc". If not specified, all sources will be updated.
    """
    from .knowledgebase import KnowledgebaseManager
    db = get_db('primary')

    # Determine which sources we are working with
    source_set = None
    if sources:
        source_set = {s.lower() for s in sources}
    selected_kbs = (M for M in KnowledgebaseManager.__subclasses__()
                    if not source_set or M.name.lower() in source_set
                    or M.short_name in source_set)

    # Handle the list option.
    if task == 'list':
        return

    # Handle the other tasks.
    for Manager in selected_kbs:
        kbm = Manager()

        # Perform the requested action.
        if task == 'upload':
            print(f'Uploading {kbm.name}...')
            kbm.upload(db)
        elif task == 'update':
            print(f'Updating {kbm.name}...')
            kbm.update(db)


@main.command()
def kb_list():
    """List the knowledge sources and their status."""
    import tabulate
    from .knowledgebase import KnowledgebaseManager
    db = get_db('primary')
    rows = [(M.name, M.short_name, format_date(M.get_last_update(db)))
            for M in KnowledgebaseManager.__subclasses__()]
    print(tabulate.tabulate(rows, ('Name', 'Short Name', 'Last Updated'),
                            tablefmt='simple'))


@main.command()
@click.argument("task", type=click.Choice(["gather"]))
def pipeline_stats(task):
    """Manage the pipeline stats gathered on s3.

    All major upload and update pipelines have basic timeing and success-failure
    stats gather on them using the
    :class:`DataGatherer <indra_db.util.data_gatherer.DataGatherer>` class
    wrapper.

    These stats are displayed on the ``/monitor`` endpoint of the database
    service.

    \b
    Tasks are:
     - gather: gather the individual job JSONs into an aggregated file.
    """
    if task == "gather":
        from indra_db.util.data_gatherer import digest_s3_files
        digest_s3_files()


@main.command()
@click.argument("task", type=click.Choice(["upload", "update"]))
@click.argument("sources", nargs=-1,
                type=click.Choice(["pubmed", "pmc_oa", "manuscripts"]),
                required=False)
@click.option('-c', '--continuing', is_flag=True,
              help=('Continue uploading or updating, picking up where you left '
                    'off.'))
@click.option('-d', '--debug', is_flag=True,
              help='Run with debugging level output.')
def content_run(task, sources, continuing, debug):
    """Upload/update text refs and content on the database.

    \b
    Usage tasks are:
     - upload: use if the knowledge bases have not yet been added.
     - update: if they have been added, but need to be updated.

    The currently available sources are "pubmed", "pmc_oa", and "manuscripts".
    """
    # Import what is needed.
    from .content import Pubmed, PmcOA, Manuscripts
    content_managers = [Pubmed, PmcOA, Manuscripts]
    db = get_db('primary')

    # Set the debug level.
    if debug:
        import logging
        from indra_db.databases import logger as db_logger
        from indra_db.cli.content import logger as content_logger
        content_logger.setLevel(logging.DEBUG)
        db_logger.setLevel(logging.DEBUG)

    # Define which sources we touch, just once
    if not sources:
        sources = {cm.my_source for cm in content_managers}
    else:
        sources = set(sources)
    selected_managers = (CM for CM in content_managers
                         if CM.my_source in sources)

    # Perform the task.
    for ContentManager in selected_managers:
        if task == 'upload':
            print(f"Uploading {ContentManager.my_source}.")
            ContentManager().populate(db, continuing)
        elif task == 'update':
            print(f"Updating {ContentManager.my_source}")
            ContentManager().update(db)


@main.command()
@click.option('-l', '--long', is_flag=True,
              help="Include a list of the most recently added content for all "
                   "source types.")
def content_list(long):
    """List the current knowledge sources and their status."""
    # Import what is needed.
    import tabulate
    from sqlalchemy import func
    from .content import Pubmed, PmcOA, Manuscripts

    content_managers = [Pubmed, PmcOA, Manuscripts]
    db = get_db('primary')

    # Generate the rows.
    source_updates = {cm.my_source: format_date(cm.get_latest_update(db))
                      for cm in content_managers}
    if long:
        print("This may take a while...", end='', flush=True)
        q = (db.session.query(db.TextContent.source,
                              func.max(db.TextContent.insert_date))
                       .group_by(db.TextContent.source))
        rows = [(src, source_updates.get(src, '-'), format_date(last_insert))
                for src, last_insert in q.all()]
        headers = ('Source', 'Last Updated', 'Latest Insert')
        print("\r", end='')
    else:
        rows = [(k, v) for k, v in source_updates.items()]
        headers = ('Source', 'Last Updated')
    print(tabulate.tabulate(rows, headers, tablefmt='simple'))


@main.command()
@click.option('-P', '--principal', default="primary",
              help="Specify which principal database to use.")
@click.option('-R', '--readonly', default="primary",
              help="Specify which readonly database to use.")
@click.option('-a', '--allow-continue', is_flag=True,
              help="Indicate whether you want the job to continue building an "
                   "existing dump corpus, or if you want to start a new one.")
@click.option('-d', '--delete-existing', is_flag=True,
              help="Delete and restart an existing readonly schema in "
                   "principal.")
@click.option('-u', '--dump-only', is_flag=True,
              help='Only generate the dumps on s3.')
@click.option('-l', '--load-only', is_flag=True,
              help='Only load a readonly dump from s3 into the given readonly '
                   'database.')
def dump_run(principal, readonly, allow_continue, delete_existing, load_only,
             dump_only):
    """Generate new dumps and list existing dumps."""
    from indra_db import get_ro
    from indra_db.cli.dump import dump
    dump(get_db(principal, protected=False),
         get_ro(readonly, protected=False), delete_existing,
         allow_continue, load_only, dump_only)


@main.command()
@click.argument("state", type=click.Choice(["started", "done", "unfinished"]),
                required=False)
def dump_list(state):
    """List existing dumps and their s3 paths.

    \b
    State options:
     - "started": get all dumps that have started (have "start.json" in them).
     - "done": get all dumps that have finished (have "end.json" in them).
     - "unfinished": get all dumps that have started but not finished.

    If no option is given, all dumps will be listed.
    """
    import boto3
    from indra_db.cli.dump import list_dumps
    s3 = boto3.client('s3')

    # Set the parameters of the list_dumps function.
    if state == 'started':
        s = True
        e = None
    elif state == 'done':
        s = True
        e = True
    elif state == 'unfinished':
        s = True
        e = False
    else:
        s = None
        e = None

    # List the dump paths and their contents.
    for s3_path in list_dumps(s, e):
        print()
        print(s3_path)
        for el in s3_path.list_objects(s3):
            print('   ', str(el).replace(str(s3_path), ''))


@main.command()
@click.argument('task', type=click.Choice(['create', 'update']),
                required=True)
@click.argument('project-name', required=False)
def pa_run(task, project_name):
    """Manage the indra_db preassembly.

    \b
    Tasks:
     - "create": populate the pa_statements table for the first time (this
       requires that the table be empty).
     - "update": update the existing content in pa_statements with the latest
       from raw statements.

    A project name is required to tag the AWS instances with a "project" tag.
    """
    from indra_db.cli.preassembly import run_preassembly
    run_preassembly(task, project_name)


@main.command()
@click.option('-r', '--with-raw', is_flag=True,
              help="Include the latest datetimes for raw statements of each "
                   "type. This will take much longer.")
def pa_list(with_raw):
    """List the latest updates for each type of Statement."""
    import tabulate
    from indra_db.cli.preassembly import list_last_updates, \
        list_latest_raw_stmts

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


@main.command()
@click.argument('task', type=click.Choice(["all", "new"]))
@click.option('-b', '--buffer', type=int, default=1,
              help='Set the number of buffer days to read prior to the most '
                   'recent update. The default is 1 day.')
@click.option('--project-name', type=str,
              help="Set the project name to be different from the config "
                   "default.")
def reading_run(task, buffer, project_name):
    """Manage the the reading of text content on AWS.

    \b
    Tasks:
    - "all": Read all the content available.
    - "new": Read only the new content that has not been read.
    """
    from indra_db.cli.reading import BulkAwsReadingManager
    db = get_db('primary')
    readers = ['SPARSER', 'REACH', 'TRIPS', 'ISI', 'EIDOS', 'MTI']
    bulk_manager = BulkAwsReadingManager(readers,
                                         buffer_days=buffer,
                                         project_name=project_name)
    if task == 'all':
        bulk_manager.read_all(db)
    elif task == 'new':
        bulk_manager.read_new(db)


@main.command()
@click.argument('task', type=click.Choice(["all", "new"]))
@click.option('-b', '--buffer', type=int, default=1,
              help='Set the number of buffer days to read prior to the most '
                   'recent update. The default is 1 day.')
@click.option('-n', '--num-procs', type=int,
              help="Select the number of processors to use.")
def reading_run_local(task, buffer, num_procs):
    """Run reading locally, save the results on the database.

    \b
    Tasks:
    - "all": Read all the content available.
    - "new": Read only the new content that has not been read.
    """
    from indra_db.cli.reading import BulkLocalReadingManager
    db = get_db('primary')

    readers = ['SPARSER', 'REACH', 'TRIPS', 'ISI', 'EIDOS', 'MTI']
    bulk_manager = BulkLocalReadingManager(readers,
                                           buffer_days=buffer,
                                           n_procs=num_procs)
    if task == 'all':
        bulk_manager.read_all(db)
    elif task == 'new':
        bulk_manager.read_new(db)


@main.command()
def reading_list():
    """List the readers and their most recent runs."""
    import tabulate
    from indra_db.cli.reading import ReadingManager

    db = get_db('primary')
    rows = [(rn, format_date(lu))
            for rn, lu in ReadingManager.get_latest_updates(db).items()]
    headers = ('Reader', 'Last Updated')
    print(tabulate.tabulate(rows, headers))


@main.command()
def xdd_run():
    """Process the latest outputs from xDD."""
    from indra_db.cli.xdd import XddManager
    db = get_db('primary')
    XddManager().run(db)
