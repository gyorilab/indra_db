import json

import boto3
import pickle
import logging
from datetime import datetime
from argparse import ArgumentParser

from indra.statements import get_all_descendants
from indra.statements.io import stmts_from_json
from indra_db.belief import get_belief
from indra_db.config import CONFIG, get_s3_dump, record_in_test
from indra_db.util import get_db, get_ro, S3Path
from indra_db.util.aws import get_role_kwargs
from indra_db.util.dump_sif import dump_sif_from_stmts, get_source_counts


logger = logging.getLogger(__name__)


# READONLY UPDATE CONFIG
aws_role = CONFIG['lambda']['role']
aws_lambda_function = CONFIG['lambda']['function']


def list_dumps(started=None, ended=None):
    """List all dumps, optionally filtered by their status.

    Parameters
    ----------
    started : Optional[bool]
        If True, find dumps that have started. If False, find dumps that have
        NOT been started. If None, do not filter by start status.
    ended : Optional[bool]
        The same as `started`, but checking whether the dump is ended or not.
    """
    # Get all the dump "directories".
    s3_base = get_s3_dump()
    s3 = boto3.client('s3')
    res = s3.list_objects_v2(Delimiter='/', **s3_base.kw(prefix=True))
    if res['KeyCount'] == 0:
        return []
    dumps = [S3Path.from_key_parts(s3_base.bucket, d['Prefix'])
             for d in res['CommonPrefixes']]

    # Filter to those that have "started"
    if started is not None:
        dumps = [p for p in dumps
                 if p.get_element_path(Start.file_name()).exists(s3) == started]

    # Filter to those that have "ended"
    if ended is not None:
        dumps = [p for p in dumps
                 if p.get_element_path(End.file_name()).exists(s3) == ended]

    return dumps


def get_latest_dump_s3_path(dumper_name):
    """Get the latest version of a dump file by the given name.

    `dumper_name` is indexed using the standardized `name` class attribute of
    the dumper object.
    """
    # Get all the dumps that were properly started.
    s3 = boto3.client('s3')
    all_dumps = list_dumps(started=True)

    # Going in reverse order (implicitly by timestamp) and look for the file.
    for s3_path in sorted(all_dumps, reverse=True):
        sought_path = s3_path.get_element_path(dumpers[dumper_name].file_name())
        if sought_path.exists(s3):
            return sought_path

    # If none is found, return None.
    return None


class Dumper(object):
    name = NotImplemented
    fmt = NotImplemented
    db_options = []
    db_required = False

    def __init__(self, date_stamp=None, **kwargs):
        # Get a database handle, if needed.
        self.db = self._choose_db(**kwargs)

        # Get the date stamp.
        self.s3_dump_path = None
        if date_stamp is None:
            self.date_stamp = datetime.now().strftime('%Y-%m-%d')
        else:
            self.date_stamp = date_stamp

    @classmethod
    def _choose_db(cls, **kwargs):
        # If we don't need a database handle, just keep on walking.
        if not cls.db_required:
            return None

        # If a database is required, the type must be specified.
        assert len(cls.db_options), \
            "If db is required, db_options are too."

        # If a handle was given, use it, regardless of other inputs, if it
        # is at all permissible.
        if 'ro' in kwargs and 'db' in kwargs:
            raise ValueError("Only one database handle may be defined at"
                             "a time.")
        if 'ro' in kwargs:
            if 'readonly' in cls.db_options:
                return kwargs['ro']
            else:
                raise ValueError("Cannot use readonly database, but ro "
                                 "handle was given.")
        elif 'db' in kwargs:
            if 'principal' in cls.db_options:
                return kwargs['db']
            else:
                raise ValueError("Cannot use principal database, but db "
                                 "handle was given.")

        # Otherwise, read or guess the database type, and make a new instance.
        if len(cls.db_options) == 1:
            db_opt = cls.db_options[0]
        else:
            if 'use_principal' in kwargs:
                if kwargs['use_principal']:
                    db_opt = 'principal'
                else:
                    db_opt = 'readonly'
            else:
                raise ValueError("No database specified.")
        if db_opt == 'principal':
            return get_db('primary', protected=False)
        else:  # if db_opt == 'readonly'
            return get_ro('primary', protected=False)

    def get_s3_path(self):
        if self.s3_dump_path is None:
            self.s3_dump_path = self._gen_s3_name()
        return self.s3_dump_path

    @classmethod
    def file_name(cls):
        return '%s.%s' % (cls.name, cls.fmt)

    def _gen_s3_name(self):
        s3_base = get_s3_dump()
        s3_path = s3_base.get_element_path(self.date_stamp, self.file_name())
        return s3_path

    @classmethod
    def is_dump_path(cls, s3_path):
        s3_base = get_s3_dump()
        if s3_base.bucket != s3_path.bucket:
            return False
        if s3_base.key not in s3_path.key:
            return False
        if cls.name not in s3_path.key:
            return False
        return True

    @classmethod
    def from_list(cls, s3_path_list):
        for p in s3_path_list:
            if cls.is_dump_path(p):
                return p
        return None

    def dump(self, continuing=False):
        raise NotImplementedError()

    def shallow_mock_dump(self, *args, **kwargs):
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, b'')


class Start(Dumper):
    name = 'start'
    fmt = 'json'
    db_required = False

    def __init__(self, *args, **kwargs):
        super(Start, self).__init__(*args, **kwargs)
        self.manifest = []

    def _mark_start(self, s3):
        s3.put_object(
            Body=json.dumps(
                {'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 'date_stamp': self.date_stamp}
            ),
            **self.get_s3_path().kw()
        )
        self.manifest.append(self.get_s3_path())
        return

    def dump(self, continuing=False):
        s3 = boto3.client('s3')
        if not continuing:
            self._mark_start(s3)
        else:
            dumps = list_dumps()
            if not dumps:
                self._mark_start(s3)
                return

            latest_dump = max(dumps)
            manifest = latest_dump.list_objects(s3)
            start = None
            end = None
            for obj in manifest:
                if Start.name in obj.key:
                    start = obj
                elif End.name in obj.key:
                    end = obj

            if end or not start:
                self._mark_start(s3)
                return

            # Set up to continue where a previous job left off.
            res = start.get(s3)
            start_json = json.loads(res['Body'].read())
            self.date_stamp = start_json['date_stamp']
            self.manifest = manifest
        return


class End(Dumper):
    name = 'end'
    fmt = 'json'
    db_required = False

    def dump(self, continuing=False):
        s3 = boto3.client('s3')
        s3.put_object(
            Body=json.dumps(
                {'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            ),
            **self.get_s3_path().kw()
        )


class Sif(Dumper):
    name = 'sif'
    fmt = 'pkl'
    db_required = False

    def __init__(self, use_principal=False, **kwargs):
        super(Sif, self).__init__(use_principal=use_principal, **kwargs)

    def dump(self, continuing=False):
        latest_full_pa = get_latest_dump_s3_path(FullPaStmts.name)
        if latest_full_pa:
            s3 = boto3.client('s3')
            s3_res = latest_full_pa.get(s3)
            pa_stmts_list = pickle.loads(s3_res['Body'].read())
            s3_outpath = self.get_s3_path()
            dump_sif_from_stmts(stmt_list=pa_stmts_list, output=s3_outpath)


class Belief(Dumper):
    name = 'belief'
    fmt = 'json'
    db_required = True
    db_options = ['principal']

    def dump(self, continuing=False):
        belief_dict = get_belief(self.db, partition=False)
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, json.dumps(belief_dict).encode('utf-8'))


class SourceCount(Dumper):
    name = 'source_count'
    fmt = 'pkl'
    db_required = True
    db_options = ['principal', 'readonly']

    def __init__(self, use_principal=True, **kwargs):
        super(SourceCount, self).__init__(use_principal=use_principal, **kwargs)

    def dump(self, continuing=False):
        get_source_counts(self.get_s3_path(), self.db)


class FullPaJson(Dumper):
    name = 'full_pa_json'
    fmt = 'json'
    db_required = True
    db_options = ['principal', 'readonly']

    def __init__(self, use_principal=False, **kwargs):
        super(FullPaJson, self).__init__(use_principal=use_principal, **kwargs)

    def dump(self, continuing=False):
        query_res = self.db.session.query(self.db.FastRawPaLink.pa_json.distinct())
        json_list = [json.loads(js[0]) for js in query_res.all()]
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, json.dumps(json_list).encode('utf-8'))


class FullPaStmts(Dumper):
    name = 'full_pa_stmts'
    fmt = 'pkl'
    db_required = True
    db_options = ['principal', 'readonly']

    def __init__(self, use_principal=False, **kwargs):
        super(FullPaStmts, self).__init__(use_principal=use_principal, **kwargs)

    def dump(self, continuing=False):
        query_res = self.db.session.query(self.db.FastRawPaLink.pa_json.distinct())
        stmt_list = stmts_from_json([json.loads(js[0]) for js in
                                     query_res.all()])
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, pickle.dumps(stmt_list))


class Readonly(Dumper):
    name = 'readonly'
    fmt = 'dump'
    db_required = True
    db_options = ['principal']

    def dump(self, belief_dump, continuing=False):

        logger.info("%s - Generating readonly schema (est. a long time)"
                    % datetime.now())
        import boto3
        s3 = boto3.client('s3')
        belief_data = belief_dump.get(s3)
        belief_dict = json.loads(belief_data['Body'].read())
        self.db.generate_readonly(belief_dict, allow_continue=continuing)

        logger.info("%s - Beginning dump of database (est. 1 + epsilon hours)"
                    % datetime.now())
        self.db.dump_readonly(self.get_s3_path())
        return


class StatementHashMeshId(Dumper):
    name = 'mti_mesh_ids'
    fmt = 'pkl'
    db_required = True
    db_options = ['principal', 'readonly']

    def __init__(self, use_principal=False, **kwargs):
        super(StatementHashMeshId, self).__init__(use_principal=use_principal,
                                                  **kwargs)

    def dump(self, continuing=False):
        mesh_term_tuples = self.db.select_all([
            self.db.MeshTermMeta.mk_hash,
            self.db.MeshTermMeta.mesh_num])
        mesh_concept_tuples = self.db.select_all([
            self.db.MeshConceptMeta.mk_hash,
            self.db.MeshConceptMeta.mesh_num])
        mesh_data = {'terms': mesh_term_tuples,
                     'concepts': mesh_concept_tuples}

        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, pickle.dumps(mesh_data))


def load_readonly_dump(principal_db, readonly_db, dump_file):
    logger.info("Using dump_file = \"%s\"." % dump_file)
    logger.info("%s - Beginning upload of content (est. ~30 minutes)"
                % datetime.now())
    with ReadonlyTransferEnv(principal_db, readonly_db):
        readonly_db.load_dump(dump_file)


def get_lambda_client():
    kwargs, _ = get_role_kwargs(aws_role)
    return boto3.client('lambda', **kwargs)


class ReadonlyTransferEnv(object):
    def __init__(self, db, ro):
        self.principal = db
        self.readonly = ro

    @record_in_test
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


dumpers = {dumper.name: dumper for dumper in get_all_descendants(Dumper)}


def dump(principal_db, readonly_db, delete_existing=False, allow_continue=True,
         load_only=False, dump_only=False):
    if delete_existing and 'readonly' in principal_db.get_schemas():
        principal_db.drop_schema('readonly')

    if not load_only:
        starter = Start()
        starter.dump(continuing=allow_continue)

        belief_dump = Belief.from_list(starter.manifest)
        if not allow_continue or not belief_dump:
            logger.info("Dumping belief.")
            belief_dumper = Belief(db=principal_db,
                                   date_stamp=starter.date_stamp)
            belief_dumper.dump(continuing=allow_continue)
            belief_dump = belief_dumper.get_s3_path()
        else:
            logger.info("Belief dump exists, skipping.")

        dump_file = Readonly.from_list(starter.manifest)
        if not allow_continue or not dump_file:
            logger.info("Generating readonly schema (est. a long time)")
            ro_dumper = Readonly(db=principal_db,
                                 date_stamp=starter.date_stamp)
            ro_dumper.dump(belief_dump=belief_dump,
                           continuing=allow_continue)
            dump_file = ro_dumper.get_s3_path()
        else:
            logger.info("Readonly dump exists, skipping.")

        if not allow_continue \
                or not FullPaStmts.from_list(starter.manifest):
            logger.info("Dumping all PA Statements as a pickle.")
            FullPaStmts(db=principal_db,
                        date_stamp=starter.date_stamp)\
                .dump(continuing=allow_continue)
        else:
            logger.info("Statement dump exists, skipping.")

        if not allow_continue or not Sif.from_list(starter.manifest):
            logger.info("Dumping sif from the readonly schema on principal.")
            Sif(db=principal_db, date_stamp=starter.date_stamp)\
                .dump(continuing=allow_continue)
        else:
            logger.info("Sif dump exists, skipping.")

        if not allow_continue \
                or not StatementHashMeshId.from_list(starter.manifest):
            logger.info("Dumping hash-mesh tuples.")
            StatementHashMeshId(db=principal_db,
                                date_stamp=starter.date_stamp)\
                .dump(continuing=allow_continue)

        End(date_stamp=starter.date_stamp).dump(continuing=allow_continue)
    else:
        # Find the most recent dump that has a readonly.
        dump_file = get_latest_dump_s3_path(Readonly.name)
        if dump_file is None:
            raise Exception("Could not find any suitable readonly dumps.")

    if not dump_only:
        print("Dump file:", dump_file)
        load_readonly_dump(principal_db, readonly_db, dump_file)

    if not load_only:
        # This database no longer needs this schema (this only executes if
        # the check_call does not error).
        principal_db.session.close()
        principal_db.grab_session()
        principal_db.drop_schema('readonly')


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


if __name__ == '__main__':
    args = parse_args()
    dump(get_db(args.database, protected=False),
         get_ro(args.readonly, protected=False), args.delet_existing,
         args.allow_continue, args.load_only, args.dump_only)
