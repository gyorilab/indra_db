import os

import click


@click.group()
def main():
    """Run the indra db rest service CLI."""


@main.command()
@click.argument('deployment', nargs=1)
@click.option('-s', '--settings', 'zappa_settings_file',
              default='zappa_settings.json',
              help="Specify the zappa settings file to use. Default is "
                   "'zappa_settings.json'.")
def push(deployment, zappa_settings_file):
    """Push a new deployment to the remote lambdas using zappa."""
    import json
    from pathlib import Path
    from indra_db_service.cli.zappa_tools import fix_permissions
    click.echo(f"Updating {deployment} deployment.")
    if not Path(zappa_settings_file).exists():
        click.echo(f"Zappa settings file not found: {zappa_settings_file}")
        return
    zappa_settings = json.load(open(zappa_settings_file, 'r'))
    os.system(f'zappa update {deployment}')
    fix_permissions(deployment, zappa_settings=zappa_settings)


@main.command()
@click.option('-p', '--port', type=click.INT,
              help="Override the default port number.")
@click.option('-h', '--host', default='0.0.0.0',
              help="Override the default host.")
@click.option('-vd', '--vue-deployment',
              type=click.Choice(['stable', 'dev', 'latest', 'test']),
              help="Load the vue package from this S3 deployment instead of "
                   "a local directory.",
              required=False)
def test_service(port, host, vue_deployment):
    """Run the service in test mode locally."""
    from indra_db_service.config import TESTING
    TESTING['status'] = True
    if vue_deployment is not None:
        TESTING['deployment'] = vue_deployment
        TESTING['vue-root'] = (
            f'https://bigmech.s3.amazonaws.com/indra-db/indralabvue-'
            f'{vue_deployment}'
        )
        click.echo(f'Using deployment {vue_deployment} from S3 at {TESTING["vue-root"]}')

    from indra_db_service.api import app
    app.run(host=host, port=port, debug=True)


if __name__ == '__main__':
    main()
