import os
from pathlib import Path

import click


@click.group()
def main():
    """Run the indra db rest service CLI."""


@main.command()
@click.argument('deployment', nargs=1)
@click.option('-z', '--zappa-settings',
              help="Path to the zappa settings file.")
def push(deployment, zappa_settings):
    """Push a new deployment to the remote lambdas using zappa."""
    from indra_db_service.cli.zappa_tools import fix_permissions, ZAPPA_CONFIG
    click.echo(f"Updating {deployment} deployment.")
    zappa_settings_path = Path(zappa_settings or ZAPPA_CONFIG)
    if not zappa_settings_path.exists():
        click.echo(f"Please run in directory with {ZAPPA_CONFIG} or specify "
                   f"the path to the zappa settings file with the "
                   f"'--zappa-settings' option.")
        return
    click.echo(f"Fixing permissions with "
               f"{zappa_settings_path.resolve().as_posix()}")
    os.system(f'zappa update {deployment}')
    fix_permissions(deployment, zappa_settings_path)


@main.command()
@click.option('-p', '--port', type=click.INT,
              help="Override the default port number.")
@click.option('-h', '--host', default='0.0.0.0',
              help="Override the default host.")
@click.option('--deployment',
              help="Load the vue package from this S3 deployment instead of "
                   "a local directory.")
def test_service(port, host, deployment=None):
    """Run the service in test mode locally."""
    from indra_db_service.config import TESTING
    TESTING['status'] = True
    if deployment is not None:
        TESTING['deployment'] = deployment
        TESTING['vue-root'] = (
            f'https://bigmech.s3.amazonaws.com/indra-db/indralabvue-'
            f'{deployment}'
        )
        click.echo(f'Using deployment {deployment} from S3 at {TESTING["vue-root"]}')

    from indra_db_service.api import app
    app.run(host=host, port=port, debug=True)


if __name__ == '__main__':
    main()
