import os

import click

from .zappa_tools import fix_permissions, ZAPPA_CONFIG


@click.group()
def main():
    """Run the indra db CLI."""


@main.command()
@click.argument('deployment', nargs=1)
def push_service(deployment):
    """Push a new deployment to the remote lambdas using zappa."""
    click.echo(f"Updating {deployment} deployment.")
    if ZAPPA_CONFIG not in os.listdir('.'):
        click.echo(f"Please run in directory with {ZAPPA_CONFIG}.")
        return
    os.system(f'zappa update {deployment}')
    fix_permissions(deployment)
