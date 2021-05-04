import click

from indra_db import get_db


@click.group()
def main():
    """INDRA Database Infrastructure CLI

    The INDRA Database is both a physical database and an infrastructure for
    managing and updating the content of that physical database. This CLI
    is used for executing these management commands.
    """


@main.command()
@click.argument("task", type=click.Choice(["upload", "update", "list"]))
@click.argument("sources", nargs=-1, type=click.STRING, required=False)
def knowledgebase(task, sources):
    """Upload/update the knowledge bases used by the database.

    \b
    Usage tasks are:
     - upload: use if the knowledge bases have not yet been added.
     - update: if they have been added, but need to be updated.
     - list: list the current knowledge sources that have been implemented.

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
        import tabulate
        rows = [(M.name, M.short_name, M.get_last_update(db))
                for M in selected_kbs]
        print(tabulate.tabulate(rows, ('Name', 'Short Name', 'Last Updated'),
                                tablefmt='simple'))
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
@click.argument("task", type=click.Choice(["upload", "update", "list"]))
@click.argument("sources", nargs=-1,
                type=click.Choice(["pubmed", "pmc_oa", "manuscripts"]),
                required=False)
@click.option("-n", "--num_procs", type=int, default=1,
              help=('Select the number of processors to use during this '
                    'operation. Default is 1.'))
@click.option('-c', '--continuing', is_flag=True,
              help=('Continue uploading or updating, picking up where you left '
                    'off.'))
@click.option('-d', '--debug', is_flag=True,
              help='Run with debugging level output.')
def content(task, sources, num_procs, continuing, debug):
    """Upload/update text refs and content on the database.

    \b
    Usage tasks are:
     - upload: use if the knowledge bases have not yet been added.
     - update: if they have been added, but need to be updated.
     - list: list the current knowledge sources that have been implemented.

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

    # Handle the list option.
    if task == 'list':
        import tabulate
        rows = []
        for cm in selected_managers:
            rows.append((cm.my_source, cm.get_latest_update(db)))
        print(tabulate.tabulate(rows, ('Source', 'Last Updated'),
                                tablefmt='simple'))
        return

    # Perform the task.
    for ContentManager in selected_managers:
        if task == 'upload':
            print(f"Uploading {ContentManager.my_source}.")
            ContentManager().populate(db, num_procs, continuing)
        elif task == 'update':
            print(f"Updating {ContentManager.my_source}")
            ContentManager().update(db, num_procs)


@main.command('dump-build')
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
def build_dumps(principal, readonly, allow_continue, delete_existing, load_only,
                dump_only):
    """Generate new dumps and list existing dumps."""
    from indra_db import get_ro
    from indra_db.cli.dump import dump
    dump(get_db(principal, protected=False),
         get_ro(readonly, protected=False), delete_existing,
         allow_continue, load_only, dump_only)


@main.command('dump-list')
@click.argument("state", type=click.Choice(["started", "done", "unfinished"]),
                required=False)
def list_dumps_cli(state):
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
