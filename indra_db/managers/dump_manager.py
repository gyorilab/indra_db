import re
import json
import boto3
import logging
from datetime import datetime
from argparse import ArgumentParser

from indra_db.belief import get_belief
from indra_db.config import CONFIG
from indra_db.config import get_s3_dump
from indra_db.util import get_db, get_ro, S3Path
from indra_db.util.dump_sif import dump_sif


logger = logging.getLogger(__name__)


# READONLY UPDATE CONFIG
aws_role = CONFIG['lambda']['role']
aws_lambda_function = CONFIG['lambda']['function']


dump_names = ['sif', 'belief']


class Dumper(object):
    name = NotImplemented
    fmt = NotImplemented

    def __init__(self, db_label='primary'):
        self.db_label = db_label
        self.s3_dump_path = None

    def get_s3_path(self):
        if self.s3_dump_path is None:
            self.s3_dump_path= self._gen_s3_name()
        return self.s3_dump_path

    @classmethod
    def _gen_s3_name(cls):
        s3_base = get_s3_dump()
        dt_ts = datetime.now().strftime('%Y-%m-%d')
        s3_path = s3_base.get_element_path(dt_ts, '%s.%s' % (cls.name, cls.fmt))
        return s3_path

    def dump(self, continuing=False):
        raise NotImplementedError()


class Sif(Dumper):
    name = 'sif'
    fmt = 'pkl'

    def __init__(self, db_label='primary', use_principal=False):
        self.use_principal = use_principal
        super(Sif, self).__init__(db_label)

    def dump(self, continuing=False):
        if self.use_principal:
            ro = get_db(self.db_label)
        else:
            ro = get_ro(self.db_label)
        s3_path = self.get_s3_path()
        dump_sif(s3_path, ro=ro)


class Belief(Dumper):
    name = 'belief'
    fmt = 'pkl'

    def dump(self, continuing=False):
        db = get_db(self.db_label)
        belief_dict = get_belief(db)
        s3 = boto3.client('s3')
        s3.put_object(Body=json.dumps(belief_dict), **self.get_s3_path().kw())


class Readonly(Dumper):
    name = 'readonly'
    fmt = 'dump'

    def dump(self, continuing=False):
        principal_db = get_db(self.db_label)

        logger.info("%s - Generating readonly schema (est. a long time)"
                    % datetime.now())
        principal_db.generate_readonly(allow_continue=continuing)

        logger.info("%s - Beginning dump of database (est. 1 + epsilon hours)"
                    % datetime.now())
        principal_db.dump_readonly(self.get_s3_path())
        return


def load_readonly_dump(db_label, ro_label, dump_file):
    principal_db = get_db(db_label)
    readonly_db = get_ro(ro_label)
    logger.info("Using dump_file = \"%s\"." % dump_file)
    logger.info("%s - Beginning upload of content (est. ~30 minutes)"
                % datetime.now())
    with ReadonlyTransferEnv(principal_db, readonly_db):
        readonly_db.load_dump(dump_file)


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
        logger.info("Redirecting the service to %s." % self.principal.url)
        self._set_lambda_env({'INDRAROOVERRIDE': str(self.principal.url)})

    def __exit__(self, exc_type, value, traceback):
        # Check for exceptions. Only change back over if there were no
        # exceptions.
        if exc_type is None:
            logger.info("Directing the service back to %s."
                        % self.readonly.url)
            self._set_lambda_env({})
        else:
            logger.warning("An error %s occurred. Assuming the database is "
                           "not usable, and not transfering the service back "
                           "to Readonly." % exc_type)


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
        '-u', '--dump_only',
        action='store_true',
        help=('Use this flag to only generate and dump the readonly database '
              'image to s3.')
    )
    parser.add_argument(
        '-l', '--load_only',
        action='store_true',
        help=('Use this flag to only load the latest s3 file onto the '
              'readonly database.')
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    principal_db = get_db(args.database)
    if args.delete_existing and 'readonly' in principal_db.get_schemas():
        principal_db.drop_schema('readonly')

    if not args.load_only:
        logger.info("Generating readonly schema (est. a long time)")
        ro_dumper = Readonly()
        ro_dumper.dump(continuing=args.allow_continue)

        logger.info("Dumping sif from the readonly schema on principal.")
        Sif(use_principal=True).dump(continuing=args.allow_continue)

        logger.info("Dumping belief.")
        Belief().dump(continuing=args.allow_continue)
        dump_file = ro_dumper.get_s3_path()
    else:
        dump_file = principal_db.get_latest_dump_file()

    if not args.dump_only:
        load_readonly_dump(args.database, args.readonly, dump_file)

    if not args.load_only:
        # This database no longer needs this schema (this only executes if
        # the check_call does not error).
        principal_db.session.close()
        principal_db.grab_session()
        principal_db.drop_schema('readonly')


if __name__ == '__main__':
    main()

