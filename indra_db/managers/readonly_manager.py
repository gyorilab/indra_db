import re
import boto3
import logging
from datetime import datetime
from argparse import ArgumentParser

from indra_db.util import get_db, get_ro
from indra_db.config import CONFIG

logger = logging.getLogger(__name__)


# READONLY UPDATE CONFIG
aws_role = CONFIG['lambda']['role']
aws_lambda_function = CONFIG['lambda']['function']


def uncamel(word):
    return re.sub(r'([a-z])([A-Z])', '\g<1>_\g<2>', word).lower()


def get_lambda_client():
    sts = boto3.client('sts')

    # Check the current role
    kwargs = {}
    ident = sts.get_caller_identity()
    if aws_role and not ident['Arn'].endswith(aws_role):
        # If the role is not the default, assume that role.
        new_role_arn = "arn:aws:iam::%s:role/%s" % (ident['Account'], aws_role)
        res = sts.assume_role(RoleArn=new_role_arn,
                              RoleSessionName="AssumeRoleReadonlyDBUpdate")
        kwargs = {'aws_' + uncamel(k): v for k, v in res['Credentials'].items()
                  if 'expiration' not in k.lower()}

    # Get a client to Lambda
    return boto3.client('lambda', **kwargs)


class ReadonlyTransferEnv(object):
    def __init__(self, db, ro):
        self.principal = db
        self.readonly = ro

    def _set_lambda_env(self, env_dict):
        lambda_client = get_lambda_client()
        lambda_client.update_function_configuration(
            FunctionName=aws_lambda_function,
            Environment={"Variables": env_dict}
        )

    def __enter__(self):
        self._set_lambda_env({'INDRAROOVERRIDE': str(self.principal.url)})

    def __exit__(self, exc_type, value, traceback):
        # Check for exceptions. Only change back over if there were no
        # exceptions.
        if exc_type is None:
            self._set_lambda_env({})
        else:
            logger.warning("An error %s occurred. Assuming the database is "
                           "not usable, and not transfering the service back "
                           "to Readonly." % exc_type)


def main():
    args = parse_args()
    if args.m_views == 'all':
        ro_names = None
    else:
        ro_names = args.m_views

    # Generate and dump the readonly schema
    principal_db = get_db(args.database)
    if not args.load_only:

        logger.info("%s - Generating readonly schema (est. a long time)"
                    % datetime.now())
        if args.delete_existing and 'readonly' in principal_db.get_schemas():
            principal_db.drop_schema('readonly')
        principal_db.generate_readonly(ro_list=ro_names,
                                       allow_continue=args.allow_continue)

        logger.info("%s - Beginning dump of database (est. 1 + epsilon hours)"
                    % datetime.now())
        dump_file = principal_db.dump_readonly()
    else:
        dump_file = principal_db.get_latest_dump_file()

    # Load the database into the readonly database
    if not args.dump_only:
        readonly_db = get_ro(args.readonly)
        logger.info("Using dump_file = \"%s\"." % dump_file)
        logger.info("%s - Beginning upload of content (est. ~30 minutes)"
                    % datetime.now())
        with ReadonlyTransferEnv(principal_db, readonly_db):
            readonly_db.load_dump(dump_file)

    # Do some cleanup
    if not args.load_only:
        # This database no longer needs this schema (this only executes if
        # the check_call does not error).
        principal_db.session.close()
        principal_db.grab_session()
        principal_db.drop_schema('readonly')
    return


def parse_args():
    parser = ArgumentParser(
        description='Manage the materialized views.'
    )
    parser.add_argument(
        '-D', '--database',
        default='primary',
        help=('Choose a database from the names given in the config or '
              'environment, for example primary is [primary] in the '
              'config file and INDRADBPRIMARY in the environment. The default '
              'is \'primary\'.')
    )
    parser.add_argument(
        '-R', '--readonly',
        default='primary',
        help=('Choose a readonly database from the names given in the config '
              'file, or INDRARO... in the env (e.g. INDRAROPRIMARY for the '
              '"primary" database.')
    )
    parser.add_argument(
        '-m', '--m_views',
        default='all',
        nargs='+',
        help='Specify certain views to create or refresh.'
    )
    parser.add_argument(
        '-a', '--allow_continue',
        action='store_true',
        help=("Indicate whether you want to job to continue building atop an "
              "existing readonly schema, or if you want it to give up if the "
              "schema already exists.")
    )
    parser.add_argument(
        '-d', '--delete_existing',
        action='store_true',
        help=("Add this flag to delete an existing schema if it exists. "
              "Selecting this option makes -a/--allow_continue moot.")
    )
    parser.add_argument(
        '-l', '--load_only',
        action='store_true',
        help=('Use this flag to only load the latest s3 file onto the '
              'readonly database.')
    )
    parser.add_argument(
        '-u', '--dump_only',
        action='store_true',
        help=('Use this flag to only generate and dump the readonly database '
              'image to s3.')
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
