import click

from .knowledgebase import kb
from .content import content
from .dump import dump_cli
from .preassembly import pa
from .reading import reading
from .xdd import xdd


@click.group()
def main():
    """INDRA Database Infrastructure CLI

    The INDRA Database is both a physical database and an infrastructure for
    managing and updating the content of that physical database. This CLI
    is used for executing these management commands.
    """


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


main.add_command(kb)
main.add_command(content)
main.add_command(dump_cli)
main.add_command(pa)
main.add_command(reading)
main.add_command(xdd)
