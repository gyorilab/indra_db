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
@click.argument("mode", type=click.Choice(["upload", "update", "list"]))
@click.argument("sources", nargs=-1, type=click.STRING, required=False)
def knowledgebase(mode, sources):
    """Upload or update the knowledge bases used by the database.

    \b
    Usage modes are:
     - upload: use if the knowledge bases have not yet been added.
     - update: if they have been added, but need to be updated.
     - list: list the current knowledge sources that have been implemented.

    Specify which knowledge base sources to update by their name, e.g. "Pathway
    Commons" or "pc". If not specified, all sources will be updated.
    """
    from .knowledgebase import KnowledgebaseManager
    if mode == 'list':
        import tabulate
        rows = []

    source_set = None
    if sources:
        source_set = {s.lower() for s in sources}

    db = get_db('primary')
    for Manager in KnowledgebaseManager.__subclasses__():
        kbm = Manager()

        # Skip entries that were not listed.
        if source_set and kbm.name.lower() not in source_set \
                and kbm.short_name not in source_set:
            continue

        # Perform the requested action.
        if mode == 'upload':
            print(f'Uploading {kbm.name}...')
            kbm.upload(db)
        elif mode == 'update':
            print(f'Updating {kbm.name}...')
            kbm.update(db)
        elif mode == 'list':
            rows.append([kbm.name, kbm.short_name, kbm.get_last_update(db)])

    if mode == 'list':
        tbl = tabulate.tabulate(rows,
                                headers=("Name", "Short Name", "Last Updated"),
                                tablefmt="simple")
        print(tbl)
