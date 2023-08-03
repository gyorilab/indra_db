import json
from pathlib import Path

import boto3

from indra_db.config import CONFIG
from indra_db.util.aws import get_role_kwargs


# Lambda CONFIG parameters
aws_role = CONFIG['lambda']['role']
aws_primary_function = 'indra-db-api-ROOT'

# Load the Zappa config file.
ZAPPA_CONFIG = 'zappa_settings.json'


def load_zappa_settings(zappa_config: Path = None) -> dict:
    """Load the zappa settings file."""
    if zappa_config is None:
        zappa_config = Path(ZAPPA_CONFIG)
    if not zappa_config.exists():
        raise FileNotFoundError(
            f"No valid zappa config file present at {zappa_config}"
        )
    with zappa_config.open('r') as f:
        zappa_settings = json.load(f)
    return zappa_settings


def fix_permissions(deployment, zappa_settings_path: Path = None):
    """Add permissions to the lambda function to allow access from API Gateway.

    When Zappa runs, it removes permission for the primary endpoint to call
    the lambda functions it creates. This function goes in and fixes those
    permissions, and is intended to be run after a zappa update.
    """
    # Get relevant settings from the zappa config.
    zappa_settings = load_zappa_settings(zappa_settings_path)
    project_name = zappa_settings[deployment]['project_name']
    region = zappa_settings[deployment]['aws_region']
    if zappa_settings[deployment]['profile_name'].lower() != aws_role.lower():
        raise Exception("Required roles do not match!")

    # Get the ID for the API on API Gateway
    kwargs, identity = get_role_kwargs(aws_role)
    if 'region_name' not in kwargs:
        kwargs['region_name'] = region
    api_gateway = boto3.client('apigateway', **kwargs)
    api_data = api_gateway.get_rest_apis()
    for item in api_data['items']:
        if item['name'] == aws_primary_function:
            break
    else:
        raise Exception(f"Could not find api matching name: "
                        f"{aws_primary_function}")

    # Give the API Gateway access to the lambda functions.
    account_id = identity['Account']
    lambda_client = boto3.client('lambda', **kwargs)
    for label, endpoint in [('root', ''), ('leafs', '/*')]:
        source_arn = (f"arn:aws:execute-api:{region}:{account_id}:{item['id']}"
                      f"/*/*/{deployment}{endpoint}")
        statement_id = f'{aws_primary_function}-access-to-{deployment}-{label}'
        lambda_client.add_permission(FunctionName=f'{project_name}-{deployment}',
                                     Action='lambda:InvokeFunction',
                                     Principal='apigateway.amazonaws.com',
                                     SourceArn=source_arn,
                                     StatementId=statement_id)
    return
