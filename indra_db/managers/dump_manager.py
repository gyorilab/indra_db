import json

import boto3
import pickle
import logging
from datetime import datetime
from argparse import ArgumentParser

from indra.statements import get_all_descendants
from indra.statements.io import stmts_from_json
from indra_db.belief import get_belief
from indra_db.config import CONFIG, get_s3_dump
from indra_db.util import get_db, get_ro, S3Path
from indra_db.util.aws import get_role_kwargs
from indra_db.util.dump_sif import dump_sif, get_source_counts


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

    def __init__(self, db_label='primary', date_stamp=None):
        self.db_label = db_label
        self.s3_dump_path = None
        if date_stamp is None:
            self.date_stamp = datetime.now().strftime('%Y-%m-%d')
        else:
            self.date_stamp = date_stamp

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

    def __init__(self, db_label='primary', use_principal=False, **kwargs):
        self.use_principal = use_principal
        super(Sif, self).__init__(db_label, **kwargs)

    def dump(self, continuing=False, include_src_counts=True):
        if self.use_principal:
            ro = get_db(self.db_label)
        else:
            ro = get_ro(self.db_label)
        s3_path = self.get_s3_path()
        if include_src_counts:
            srcc = SourceCount()
            dump_sif(s3_path, src_count_file=srcc.get_s3_path(), ro=ro)
        else:
            dump_sif(s3_path, ro=ro)


class Belief(Dumper):
    name = 'belief'
    fmt = 'json'

    def dump(self, continuing=False):
        db = get_db(self.db_label)
        belief_dict = get_belief(db, partition=False)
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, json.dumps(belief_dict))


class SourceCount(Dumper):
    name = 'source_count'
    fmt = 'pkl'

    def __init__(self, db_label='primary', use_principal=True, **kwargs):
        self.use_principal = use_principal
        super(SourceCount, self).__init__(db_label, **kwargs)

    def dump(self, continuing=False):
        if self.use_principal:
            ro = get_db(self.db_label)
        else:
            ro = get_ro(self.db_label)
        get_source_counts(self.get_s3_path(), ro)


class FullPaJson(Dumper):
    name = 'full_pa_json'
    fmt = 'json'

    def __init__(self, db_label='primary', use_principal=False, **kwargs):
        self.use_principal = use_principal
        super(FullPaJson, self).__init__(db_label, **kwargs)

    def dump(self, continuing=False):
        if self.use_principal:
            ro = get_db(self.db_label)
        else:
            ro = get_ro(self.db_label)
        query_res = ro.session.query(ro.FastRawPaLink.pa_json.distinct())
        json_list = [json.loads(js[0]) for js in query_res.all()]
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, json.dumps(json_list))


class FullPaStmts(Dumper):
    name = 'full_pa_stmts'
    fmt = 'pkl'

    def __init__(self, db_label='primary', use_principal=False, **kwargs):
        self.use_principal = use_principal
        super(FullPaStmts, self).__init__(db_label, **kwargs)

    def dump(self, continuing=False):
        if self.use_principal:
            ro = get_db(self.db_label)
        else:
            ro = get_ro(self.db_label)
        query_res = ro.session.query(ro.FastRawPaLink.pa_json.distinct())
        stmt_list = stmts_from_json([json.loads(js[0]) for js in
                                     query_res.all()])
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, pickle.dumps(stmt_list))


class Readonly(Dumper):
    name = 'readonly'
    fmt = 'dump'

    def dump(self, belief_dump, continuing=False):
        principal_db = get_db(self.db_label)

        logger.info("%s - Generating readonly schema (est. a long time)"
                    % datetime.now())
        import boto3
        s3 = boto3.client('s3')
        belief_data = belief_dump.get(s3)
        belief_dict = json.loads(belief_data['Body'].read())
        principal_db.generate_readonly(belief_dict, allow_continue=continuing)

        logger.info("%s - Beginning dump of database (est. 1 + epsilon hours)"
                    % datetime.now())
        principal_db.dump_readonly(self.get_s3_path())
        return


class StatementHashMeshId(Dumper):
    name = 'mti_mesh_ids'
    fmt = 'pkl'

    def __init__(self, db_label='primary', use_principal=False, **kwargs):
        self.use_principal = use_principal
        super(StatementHashMeshId, self).__init__(db_label, **kwargs)

    def dump(self, continuing=False):
        if self.use_principal:
            ro = get_db(self.db_label)
        else:
            ro = get_ro(self.db_label)

        mesh_term_tuples = ro.select_all([ro.MeshTermMeta.mk_hash,
                                          ro.MeshTermMeta.mesh_num])
        mesh_concept_tuples = ro.select_all([ro.MeshConceptMeta.mk_hash,
                                             ro.MeshConceptMeta.mesh_num])
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
        if not allow_continue or not Belief.from_list(starter.manifest):
            logger.info("Dumping belief.")
            belief_dumper = Belief(date_stamp=starter.date_stamp)
            belief_dumper.dump(continuing=allow_continue)
            belief_dump = belief_dumper.get_s3_path()
        else:

            logger.info("Belief dump exists, skipping.")

        dump_file = Readonly.from_list(starter.manifest)
        if not allow_continue or not dump_file:
            logger.info("Generating readonly schema (est. a long time)")
            ro_dumper = Readonly(date_stamp=starter.date_stamp)
            ro_dumper.dump(belief_dump=belief_dump,
                           continuing=allow_continue)
            dump_file = ro_dumper.get_s3_path()
        else:
            logger.info("Readonly dump exists, skipping.")

        if not allow_continue or not Sif.from_list(starter.manifest):
            logger.info("Dumping sif from the readonly schema on principal.")
            Sif(use_principal=True, date_stamp=starter.date_stamp)\
                .dump(continuing=allow_continue)
        else:
            logger.info("Sif dump exists, skipping.")

        if not allow_continue \
                or not FullPaStmts.from_list(starter.manifest):
            logger.info("Dumping all PA Statements as a pickle.")
            FullPaStmts(date_stamp=starter.date_stamp)\
                .dump(continuing=allow_continue)
        else:
            logger.info("Statement dump exists, skipping.")

        if not allow_continue \
                or not StatementHashMeshId.from_list(starter.manifest):
            logger.info("Dumping hash-mesh tuples.")
            StatementHashMeshId(use_principal=True,
                                date_stamp=starter.date_stamp)\
                .dump(continuing=allow_continue)

        End(date_stamp=starter.date_stamp).dump(continuing=allow_continue)
    else:
        dumps = list_dumps()

        # Find the most recent dump that has a readonly.
        s3 = boto3.client('s3')
        for dump in sorted(dumps, reverse=True):
            manifest = dump.list_objects(s3)
            dump_file = Readonly.from_list(manifest)
            if dump_file is not None:
                # dump_file will be the file we want, leave it assigned.
                break
        else:
            raise Exception("Could not find any suitable readonly dumps.")

    if not dump_only:
        print(dump_file)
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
    dump(get_db(args.database), get_ro(args.readonly), args.delet_existing,
         args.allow_continue, args.load_only, args.dump_only)
