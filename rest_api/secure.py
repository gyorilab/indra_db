import sys
import json
from argparse import ArgumentParser

import boto3
import shutil

from os.path import dirname, abspath, join, pardir
from zipfile import ZipFile


HERE = dirname(abspath(__file__))


def get_gateway_client(role='SUDO'):
    """Get a boto3 client to the gateway with SUDO role permissions.

    It is assumed the base user is able to access this role. If not, some boto
    error will be raised.
    """
    sts = boto3.client('sts')
    resp = sts.get_caller_identity()
    acct_id = resp['Account']
    aro = sts.assume_role(RoleArn='arn:aws:iam::%s:role/%s' % (acct_id, role),
                          RoleSessionName='Assuming%s' % role)
    creds = aro['Credentials']
    agc = boto3.client('apigateway', aws_access_key_id=creds['AccessKeyId'],
                       aws_secret_access_key=creds['SecretAccessKey'],
                       aws_session_token=creds['SessionToken'])
    return agc


class SecurityManager(object):
    """Object to manage the security of the REST API."""

    def __init__(self, stage):
        with open(join(HERE, pardir, 'zappa_settings.json'), 'r') as f:
            info = json.load(f)
        deployment_info = info[stage]
        self.function_name = deployment_info['project_name'] + '-' + stage
        return

    def update_lambdas(self):
        """Update the verification and api key creation lambdas.

        It is assumed that the current virtual environment is the one to be
        packaged. The env should be minimal because lambdas have a pretty strict
        size limit.
        """
        # Package up the env
        zip_path = shutil.make_archive(join(HERE, 'lambda'), 'zip', sys.prefix)

        # Add the relevant files from indra_db.
        idbr_dir = join(HERE, pardir, 'indra_db')
        with ZipFile(zip_path, 'a') as zf:
            zf.write(join(idbr_dir, 'managers', 'database_manager.py'),
                     'indra_db/database_manager.py')
            zf.write(join(idbr_dir, 'util', '__init__.py'),
                     'indra_db/util/__init__.py')
            zf.write(join(idbr_dir, '__init__.py'),
                     'indra_db/__init__.py')
            zf.write(join(idbr_dir, 'exceptions.py'),
                     'indra_db/exceptions.py')
            zf.write(join(HERE, 'security_lambdas', 'verify_key_script.py'),
                     'verify_key_script.py')

        # Update the lambda.
        lamb = boto3.client('lambda')
        with open(join(HERE, 'lambda.zip'), 'rb') as zf:
            ret = lamb.update_function_code(ZipFile=zf.read(),
                                            FunctionName=self.function_name)
            print(ret)


def get_parser():
    parser = ArgumentParser(description='Apply and update the security to '
                                        'the database REST API.')
    parser.add_argument('action',
                        choices=['update-lambdas'],
                        help='Select which action to perform.')
    parser.add_argument('stage', help='Select which stage to operate on.')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    sec_man = SecurityManager(args.stage)

    if args.action == 'update-lambdas':
        sec_man.update_lambdas()
    return


if __name__ == '__main__':
    main()
