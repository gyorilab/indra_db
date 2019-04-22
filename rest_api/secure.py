import sys
import json
from argparse import ArgumentParser

import boto3
import shutil

from os.path import dirname, abspath, join, pardir, exists
from os import remove
from zipfile import ZipFile


HERE = dirname(abspath(__file__))


class SecurityManager(object):
    """Object to manage the security of the REST API."""

    def __init__(self, stage):
        with open(join(HERE, pardir, 'zappa_settings.json'), 'r') as f:
            zappa_info = json.load(f)
        self.info = zappa_info[stage]
        self.function_name = self.info['project_name'] + '-' + stage
        self._zip_files = []
        self._creds = None
        return

    def _sudoify(self, role='SUDO'):
        if self._creds is not None:
            return

        sts = boto3.client('sts')
        resp = sts.get_caller_identity()
        acct_id = resp['Account']
        aro = sts.assume_role(RoleArn='arn:aws:iam::%s:role/%s'
                                      % (acct_id, role),
                              RoleSessionName='Assuming%s' % role)
        self._creds = {}
        for key in ['access_key_id', 'secret_access_key', 'session_token']:
            cred_key = ''.join([s.capitalize() for s in key.split('_')])
            self._creds['aws_' + key] = aro['Credentials'][cred_key]
        return

    def get_zappa_role(self):
        self._sudoify()
        iam = boto3.client('iam', **self._creds)
        resp = iam.list_roles()
        expected_name = self.function_name + '-ZappaLambdaExecutionRole'
        arn = None
        for role_info in resp['Roles']:
            if role_info['RoleName'] == expected_name:
                arn = role_info['Arn']
                break
        return arn

    def package_lambdas(self):
        """Create a zip file for the lambdas."""
        print("Packaging the environment...", end='')
        # Find the site packages
        for sp in sys.path:
            if sp.startswith(sys.prefix) and sp.endswith('site-packages'):
                break
        else:
            raise EnvironmentError("Cannot find site packages.")

        # Package up the env
        zip_path = shutil.make_archive(join(HERE, 'lambda'), 'zip', sp)
        self._zip_files.append(zip_path)

        # Add the relevant files from indra_db.
        idbr_dir = join(HERE, pardir, 'indra_db')
        with ZipFile(zip_path, 'a') as zf:
            zf.write(join(idbr_dir, 'managers', 'database_manager.py'),
                     'indra_db/managers/database_manager.py')
            zf.write(join(idbr_dir, 'util', 'constructors.py'),
                     'indra_db/util/constructors.py')
            zf.write(join(idbr_dir, 'config.py'),
                     'indra_db/config.py')
            zf.write(join(idbr_dir, 'exceptions.py'),
                     'indra_db/exceptions.py')
            zf.write(join(HERE, 'security_lambdas', 'create_account_script.py'),
                     'create_account_script.py')
            size = sum([f.file_size for f in zf.filelist])/1e6  # MB
        print(size, 'MB')
        return zip_path

    def _clear_packages(self):
        """Remove any zip files that were created."""
        for zip_path in self._zip_files:
            if exists(zip_path):
                remove(zip_path)
        return

    def create_lambdas(self):
        """Create the necessary lambda functions."""
        try:
            # Package the environment and code.
            zip_path = self.package_lambdas()

            print("Creating the lambda function...")
            self._sudoify()
            lamb = boto3.client('lambda', **self._creds)
            with open(zip_path, 'rb') as zf:
                fname = self.function_name + '-auth'
                env = {'Variables': self.info['environment_variables']}
                lamb.create_function(
                    FunctionName=fname, Runtime=self.info['runtime'],
                    Role=self.get_zappa_role(), Code={'ZipFile': zf.read()},
                    VpcConfig=self.info['vpc_config'], Environment=env,
                    Handler='create_account_script.lambda_handler',
                    Tags={'project': 'cwc'}
                    )
        finally:
            self._clear_packages()

    def update_lambdas(self):
        """Update the verification and api key creation lambdas.

        It is assumed that the current virtual environment is the one to be
        packaged. The env should be minimal because lambdas have a pretty strict
        size limit.
        """
        try:
            # Package the lambda
            zip_path = self.package_lambdas()

            # Update the lambda.
            print("Updating the lambda function...")
            lamb = boto3.client('lambda')
            with open(zip_path, 'rb') as zf:
                fname = self.function_name + '-auth'
                ret = lamb.update_function_code(ZipFile=zf.read(),
                                                FunctionName=fname)
                print(ret)
        finally:
            self._clear_packages()


def get_parser():
    parser = ArgumentParser(description='Apply and update the security to '
                                        'the database REST API.')
    parser.add_argument('action',
                        choices=['update-lambdas', 'create-lambdas',
                                 'package-lambdas'],
                        help='Select which action to perform.')
    parser.add_argument('stage', help='Select which stage to operate on.')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    sec_man = SecurityManager(args.stage)

    if args.action == 'update-lambdas':
        sec_man.update_lambdas()
    elif args.action == 'create-lambdas':
        sec_man.create_lambdas()
    elif args.action == 'package-lambdas':
        sec_man.package_lambdas()

    return


if __name__ == '__main__':
    main()
