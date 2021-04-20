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
@click.argument("mode", type=click.Choice(["upload", "update"]))
def knowledgebase(mode):
    """Upload or update the knowledge bases used by the database.

    Use "upload" if the knowledge bases have not yet been added.
    Use "update" if they have been added, but need to be updated.
    """
    from .knowledgebase import KnowledgebaseManager

    db = get_db('primary')
    for Manager in KnowledgebaseManager.__subclasses__():
        kbm = Manager()
        print(kbm.name, '...')
        if mode == 'upload':
            kbm.upload(db)
        elif mode == 'update':
            kbm.update(db)
