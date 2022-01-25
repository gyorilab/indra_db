import os

import click



@click.group()
def main():
    """Run the indra db rest service CLI."""


@main.command()
@click.argument('deployment', nargs=1)
def push(deployment):
    """Push a new deployment to the remote lambdas using zappa."""
    from indra_db_service.cli.zappa_tools import fix_permissions, ZAPPA_CONFIG
    click.echo(f"Updating {deployment} deployment.")
    if ZAPPA_CONFIG not in os.listdir('.'):
        click.echo(f"Please run in directory with {ZAPPA_CONFIG}.")
        return
    os.system(f'zappa update {deployment}')
    fix_permissions(deployment)


@main.command()
@click.option('-p', '--port', type=click.INT,
              help="Override the default port number.")
@click.option('-h', '--host', default='0.0.0.0',
              help="Override the default host.")
def test_service(port, host):
    """Run the service in test mode locally."""
    from indra_db_service.config import TESTING
    TESTING['status'] = True

    from indra_db_service.api import app
    app.run(host=host, port=port, debug=True)
