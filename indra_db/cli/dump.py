import json
from collections import defaultdict
from typing import Optional, List

import click
import boto3
import pickle
import logging
from datetime import datetime

from indra.statements import get_all_descendants
from indra.statements.io import stmts_from_json
from indra_db.belief import get_belief
from indra_db.config import CONFIG, get_s3_dump, record_in_test
from indra_db.util import get_db, get_ro, S3Path
from indra_db.util.aws import get_role_kwargs
from indra_db.util.dump_sif import dump_sif, get_source_counts, load_res_pos


logger = logging.getLogger(__name__)


@click.group('dump')
def dump_cli():
    """Manage the data dumps from Principal to files and Readonly."""


@click.group('run')
def run_commands():
    """Run dumps."""


def list_dumps(started=None, ended=None):
    """List all dumps, optionally filtered by their status.

    Parameters
    ----------
    started : Optional[bool]
        If True, find dumps that have started. If False, find dumps that have
        NOT been started. If None, do not filter by start status.
    ended : Optional[bool]
        The same as `started`, but checking whether the dump is ended or not.

    Returns
    -------
    list of S3Path objects
        Each S3Path object contains the bucket and key prefix information for
        a set of dump files, e.g.

            [S3Path(bigmech, indra-db/dumps/2020-07-16/),
             S3Path(bigmech, indra-db/dumps/2020-08-28/),
             S3Path(bigmech, indra-db/dumps/2020-09-18/),
             S3Path(bigmech, indra-db/dumps/2020-11-12/),
             S3Path(bigmech, indra-db/dumps/2020-11-13/)]
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

    Searches dumps that have already been *started* and gets the full S3
    file path for the latest version of the dump of that type (e.g. "sif",
    "belief", "source_count", etc.)

    Parameters
    ----------
    dumper_name : str
        The standardized name for the dumper classes defined in this module,
        defined in the `name` class attribute of the dumper object.
        E.g., the standard dumper name "sif" can be obtained from ``Sif.name``.

    Returns
    -------
    Union[S3Path, None]
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


class DumpOrderError(Exception):
    pass


DATE_FMT = '%Y-%m-%d'


class Dumper(object):
    name: str = NotImplemented
    fmt: str = NotImplemented
    db_options: list = []
    db_required: bool = False
    requires: list = NotImplemented
    heavy_compute: bool = True

    def __init__(self, start=None, date_stamp=None, **kwargs):
        # Get a database handle, if needed.
        self.db = self._choose_db(**kwargs)

        # Get s3 paths for required dumps
        self.required_s3_paths = {}
        if self.requires and not start:
            raise DumpOrderError(f"{self.name} has prerequisites, but no start "
                                 f"given.")
        for ReqDump in self.requires:
            dump_path = ReqDump.from_list(start.manifest)
            if dump_path is None:
                raise DumpOrderError(f"{self.name} dump requires "
                                     f"{ReqDump.name} to be completed before "
                                     f"running.")
            self.required_s3_paths[ReqDump.name] = dump_path

        # Get the date stamp.
        self.s3_dump_path = None
        if date_stamp is None:
            if start:
                self.date_stamp = start.date_stamp
            else:
                self.date_stamp = datetime.now().strftime(DATE_FMT)
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
    def from_list(cls, s3_path_list: List[S3Path]) -> Optional[S3Path]:
        for p in s3_path_list:
            if cls.is_dump_path(p):
                return p
        return None

    def dump(self, continuing=False):
        raise NotImplementedError()

    def shallow_mock_dump(self, *args, **kwargs):
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, b'')

    @classmethod
    def register(cls):

        # Define the dump function.
        @click.command(cls.name.replace('_', '-'), help=cls.__doc__)
        @click.option('-c', '--continuing', is_flag=True,
                      help="Continue a partial dump, if applicable.")
        @click.option('-d', '--date-stamp',
                      type=click.DateTime(formats=['%Y-%m-%d']),
                      help="Provide a datestamp with which to mark this dump. "
                           "The default is same as the start dump from which "
                           "this is built.")
        @click.option('-f', '--force', is_flag=True,
                      help="Run the build even if the dump file has already "
                           "been produced.")
        @click.option('--from-dump', type=click.DateTime(formats=[DATE_FMT]),
                      help="Indicate a specific start dump from which to "
                           "build. The default is the most recent.")
        def run_dump(continuing, date_stamp, force, from_dump):
            start = Start.from_date(from_dump)

            if not cls.from_list(start.manifest) or force:
                logger.info(f"Dumping {cls.name} for {start.date_stamp}.")
                cls(start, date_stamp=date_stamp).dump(continuing)
            else:
                logger.info(f"{cls.name} for {date_stamp} exists, nothing to "
                            f"do. To force a re-computation use -f/--force.")

        # Register it with the run commands.
        run_commands.add_command(run_dump)

    @classmethod
    def config_to_json(cls):
        return {'requires': [r.name.replace('_', '-') for r in cls.requires],
                'heavy_compute': cls.heavy_compute}

class Start(Dumper):
    """Initialize the dump on s3, marking the start datetime of the dump."""
    name = 'start'
    fmt = 'json'
    db_required = False
    heavy_compute = False
    requires = []

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
            self.load(latest_dump)
        return

    def load(self, dump_path):
        """Load manifest from the Start of the given dump path."""
        s3 = boto3.client('s3')
        manifest = dump_path.list_objects(s3)
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

    @classmethod
    def from_date(cls, dump_date: datetime):
        """Select a dump based on the given datetime."""
        all_dumps = list_dumps(started=True)
        if dump_date:
            for dump_base in all_dumps:
                if dump_date.strftime(DATE_FMT) in dump_base.prefix:
                    selected_dump = dump_base
                    break
            else:
                raise ValueError(f"Could not find dump from date {dump_date}.")
        else:
            selected_dump = max(all_dumps)
        start = cls()
        start.load(selected_dump)
        return start

    @classmethod
    def register(cls):
        # Define the dump function.
        @click.command(cls.name.replace('_', '-'), help=cls.__doc__)
        @click.option('-c', '--continuing', is_flag=True,
                      help="Add this flag to only create a new start if an "
                           "unfinished start does not already exist.")
        def run_dump(continuing):
            start = Start()
            start.dump(continuing)

        # Register it with the run commands.
        run_commands.add_command(run_dump)



class PrincipalStats(Dumper):
    """Dump a CSV of extensive counts of content in the principal database."""
    name = 'principal-statistics'
    fmt = 'csv'
    db_required = True
    db_options = ['principal']
    requires = [Start]
    heavy_compute = False

    def dump(self, continuing=False):
        import io
        import csv

        # Get the data from the database
        res = self.db.session.execute("""
            SELECT source, text_type, reader, reader_version, raw_statements.type,
                   COUNT(DISTINCT(text_content.id)), COUNT(DISTINCT(reading.id)),
                   COUNT(DISTINCT(raw_statements.id)),
                   COUNT(DISTINCT(pa_statements.mk_hash))
            FROM text_content
             LEFT JOIN reading ON text_content_id = text_content.id
             LEFT JOIN raw_statements ON reading_id = reading.id
             LEFT JOIN raw_unique_links ON raw_statements.id = raw_stmt_id
             LEFT JOIN pa_statements ON pa_statements.mk_hash = pa_stmt_mk_hash
            GROUP BY source, text_type, reader, reader_version, raw_statements.type;
        """)

        # Create the CSV
        str_io = io.StringIO()
        writer = csv.writer(str_io)
        writer.writerow(["source", "text type", "reader", "reader version",
                         "raw statements", "statement type", "content count",
                         "reading count", "raw statement count",
                         "preassembled statement count"])
        writer.writerows(res)

        # Upload a bytes-like object
        csv_bytes = str_io.getvalue().encode('utf-8')
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, csv_bytes)


class Belief(Dumper):
    """Dump a dict of belief scores keyed by hash"""
    name = 'belief'
    fmt = 'json'
    db_required = True
    db_options = ['principal']
    requires = [Start]

    def dump(self, continuing=False):
        belief_dict = get_belief(self.db, partition=False)
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, json.dumps(belief_dict).encode('utf-8'))


class Readonly(Dumper):
    """Generate the readonly schema, and dump it using pgdump."""
    name = 'readonly'
    fmt = 'dump'
    db_required = True
    db_options = ['principal']
    requires = [Belief]
    heavy_compute = True

    def dump(self, continuing=False):

        logger.info("%s - Generating readonly schema (est. a long time)"
                    % datetime.now())
        import boto3
        s3 = boto3.client('s3')
        logger.info("Getting belief data from S3")
        belief_data = self.required_s3_paths[Belief.name].get(s3)
        logger.info("Reading belief data body")
        belief_body = belief_data['Body'].read()
        logger.info("Loading belief dict from string")
        belief_dict = json.loads(belief_body)
        logger.info("Generating readonly schema")
        self.db.generate_readonly(belief_dict, allow_continue=continuing)

        logger.info("%s - Beginning dump of database (est. 1 + epsilon hours)"
                    % datetime.now())
        self.db.dump_readonly(self.get_s3_path())
        return

class SourceCount(Dumper):
    """Dumps a dict of dicts with source counts per source api per statement"""
    name = 'source_count'
    fmt = 'pkl'
    db_required = True
    db_options = ['principal', 'readonly']
    requires = [Readonly]

    def __init__(self, start, use_principal=True, **kwargs):
        super(SourceCount, self).__init__(start, use_principal=use_principal,
                                          **kwargs)

    def dump(self, continuing=False):
        get_source_counts(self.get_s3_path(), self.db)


class ResiduePosition(Dumper):
    """Dumps a dict of dicts with residue/position data from Modifications"""
    name = 'res_pos'
    fmt = 'pkl'
    db_required = True
    db_options = ['readonly', 'principal']
    requires = [Readonly]

    def __init__(self, start, use_principal=True, **kwargs):
        super(ResiduePosition, self).__init__(start, use_principal=use_principal,
                                              **kwargs)

    def dump(self, continuing=False):
        res_pos_dict = load_res_pos(ro=self.db)
        s3 = boto3.client('s3')
        logger.info(f'Uploading residue position dump to '
                    f'{self.get_s3_path().to_string()}')
        self.get_s3_path().upload(s3=s3, body=pickle.dumps(res_pos_dict))


class FullPaStmts(Dumper):
    """Dumps all statements found in FastRawPaLink as a pickle"""
    name = 'full_pa_stmts'
    fmt = 'pkl'
    db_required = True
    db_options = ['principal', 'readonly']
    requires=[Readonly]

    def __init__(self, start, use_principal=False, **kwargs):
        super(FullPaStmts, self).__init__(start, use_principal=use_principal, **kwargs)

    def dump(self, continuing=False):
        logger.info('Querying the database to get FastRawPaLink statements')
        query_res = self.db.session.query(self.db.FastRawPaLink.pa_json.distinct())
        logger.info('Processing query result into jsons')
        stmt_jsons = [json.loads(row[0]) for row in query_res.all()]
        logger.info('Getting statements from json')
        stmt_list = stmts_from_json(stmt_jsons)
        logger.info('Dumping to pickle')
        stmt_obj = pickle.dumps(stmt_list)
        logger.info('Uploading to S3')
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, stmt_obj)


class FullPaJson(Dumper):
    """Dumps all statements found in FastRawPaLink as jsonl"""
    name = 'full_pa_json'
    fmt = 'jsonl'
    db_required = True
    db_options = ['principal', 'readonly']
    requires = [Readonly]

    def __init__(self, start, use_principal=False, **kwargs):
        super(FullPaJson, self).__init__(start, use_principal=use_principal,
                                         **kwargs)

    def dump(self, continuing=False):
        query_res = self.db.session.query(self.db.FastRawPaLink.pa_json.distinct())
        jsonl_str = '\n'.join([js.decode() for js, in query_res.all()])
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, jsonl_str.encode('utf-8'))


class Sif(Dumper):
    """Dumps a pandas dataframe of preassembled statements"""
    name = 'sif'
    fmt = 'pkl'
    db_required = True
    db_options = ['principal', 'readonly']
    requires = [SourceCount, ResiduePosition, Belief]

    def __init__(self, start, use_principal=False, **kwargs):
        super(Sif, self).__init__(start, use_principal=use_principal, **kwargs)

    def dump(self, continuing=False):
        s3_path = self.get_s3_path()
        dump_sif(df_file=s3_path,
                 src_count_file=self.required_s3_paths[SourceCount.name],
                 res_pos_file=self.required_s3_paths[ResiduePosition.name],
                 belief_file=self.required_s3_paths[Belief.name],
                 reload=True,
                 reconvert=True,
                 ro=self.db,
                 normalize_names=True)


class StatementHashMeshId(Dumper):
    """Dump a mapping from Statement hashes to MeSH terms."""
    name = 'mti_mesh_ids'
    fmt = 'pkl'
    db_required = True
    db_options = ['principal', 'readonly']
    requires = [Readonly]

    def __init__(self, start, use_principal=False, **kwargs):
        super(StatementHashMeshId, self).__init__(start,
                                                  use_principal=use_principal,
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


class End(Dumper):
    """Mark the dump as complete."""
    name = 'end'
    fmt = 'json'
    db_required = False
    # We don't need a FullPaStmts as a pickle because we already have the
    # jsonl (keeping the class definition if ever need to save a pickle)
    requires = [dumper for dumper in get_all_descendants(Dumper)
                if dumper.name != 'full_pa_stmts']
    heavy_compute = False

    def dump(self, continuing=False):
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, json.dumps(
            {'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        ).encode('utf-8'))


def load_readonly_dump(principal_db, readonly_db, dump_file,
                       no_redirect_to_principal=True):
    logger.info("Using dump_file = \"%s\"." % dump_file)
    logger.info("%s - Beginning upload of content (est. ~2.5 hours)"
                % datetime.now())
    if no_redirect_to_principal:
        readonly_db.load_dump(dump_file)
    else:
        with ReadonlyTransferEnv(principal_db, readonly_db):
            readonly_db.load_dump(dump_file)


def get_lambda_client():
    aws_role = CONFIG['lambda']['role']
    kwargs, _ = get_role_kwargs(aws_role)
    return boto3.client('lambda', **kwargs)


class ReadonlyTransferEnv(object):
    def __init__(self, db, ro):
        self.principal = db
        self.readonly = ro

    @record_in_test
    def _set_lambda_env(self, env_dict):
        aws_lambda_function = CONFIG['lambda']['function']
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


def dump(principal_db, readonly_db=None, delete_existing=False,
         allow_continue=True, load_only=False, dump_only=False,
         no_redirect_to_principal=True):
    """Run the suite of dumps in the specified order.

    Parameters
    ----------
    principal_db : :class:`indra_db.databases.PrincipalDatabaseManager`
        A handle to the principal database.
    readonly_db : :class:`indra_db.databases.ReadonlyDatabaseManager`
        A handle to the readonly database. Optional when running dump only.
    delete_existing : bool
        If True, clear out the existing readonly build from the principal
        database. Otherwise it will be continued. (Default is False)
    allow_continue : bool
        If True, each step will assume that it may already have been done, and
        where possible the work will be picked up where it was left off.
        (Default is True)
    load_only : bool
        No new dumps will be created, but an existing dump will be used to
        populate the given readonly database. (Default is False)
    dump_only : bool
        Do not load a new readonly database, only produce the dump files on s3.
        (Default is False)
    no_redirect_to_principal : bool
        If False (default), and if we are running without dump_only (i.e.,
        we are also loading a dump into a readonly DB), then we redirect the
        lambda function driving the REST API to the readonly schema in the
        principal DB while the readonly DB is being restored. If True,
        this redirect is not attempted and we assume it is okay if the
        readonly DB being restored is not accessible for the duration
        of the load.
    """
    # Check if readonly is needed:
    if not dump_only and readonly_db is None:
        raise ValueError("readonly_db must be provided if dump_only == False")
    if not load_only:
        # START THE DUMP
        if delete_existing and 'readonly' in principal_db.get_schemas():
            principal_db.drop_schema('readonly')

        start = Start()
        start.dump(continuing=allow_continue)

        # STATS DUMP
        if not allow_continue or not PrincipalStats.from_list(start.manifest):
            logger.info("Dumping principal stats.")
            PrincipalStats(start, db=principal_db)\
                .dump(continuing=allow_continue)
        else:
            logger.info("Stats dump exists, skipping.")

        # BELIEF DUMP
        if not allow_continue or not Belief.from_list(start.manifest):
            logger.info("Dumping belief.")
            Belief(start, db=principal_db).dump(continuing=allow_continue)
        else:
            logger.info("Belief dump exists, skipping.")

        # READONLY DUMP
        dump_file = Readonly.from_list(start.manifest)
        if not allow_continue or not dump_file:
            logger.info("Generating readonly schema (est. a long time)")
            ro_dumper = Readonly(start, db=principal_db)
            ro_dumper.dump(continuing=allow_continue)
            dump_file = ro_dumper.get_s3_path()
        else:
            logger.info("Readonly dump exists, skipping.")

        # RESIDUE POSITION DUMP
        # By now, the readonly schema should exist on principal, so providing
        # the principal manager should be ok for source counts and
        # residue/position
        if not allow_continue or not ResiduePosition.from_list(start.manifest):
            logger.info("Dumping residue and position")
            ResiduePosition(start, db=principal_db)\
                .dump(continuing=allow_continue)
        else:
            logger.info("Residue position dump exists, skipping")

        # SOURCE COUNT DUMP
        if not allow_continue or not SourceCount.from_list(start.manifest):
            logger.info("Dumping source count")
            SourceCount(start, db=principal_db)\
                .dump(continuing=allow_continue)
        else:
            logger.info("Source count dump exists, skipping.")

        # SIF DUMP
        if not allow_continue or not Sif.from_list(start.manifest):
            logger.info("Dumping sif from the readonly schema on principal.")
            Sif(start, db=principal_db).dump(continuing=allow_continue)
        else:
            logger.info("Sif dump exists, skipping.")

        # FULL PA JSON DUMP
        if not allow_continue or not FullPaJson.from_list(start.manifest):
            logger.info("Dumping all PA Statements as jsonl.")
            FullPaJson(start, db=principal_db).dump(continuing=allow_continue)
        else:
            logger.info("Statement dump exists, skipping.")

        # HASH MESH ID DUMP
        if not allow_continue \
                or not StatementHashMeshId.from_list(start.manifest):
            logger.info("Dumping hash-mesh tuples.")
            StatementHashMeshId(start, db=principal_db)\
                .dump(continuing=allow_continue)

        # END DUMP
        End(start).dump(continuing=allow_continue)
    else:
        # Find the most recent dump that has a readonly.
        dump_file = get_latest_dump_s3_path(Readonly.name)
        if dump_file is None:
            raise DumpOrderError("Could not find any suitable readonly dumps.")

    if not dump_only:
        # READONLY LOAD
        print("Dump file:", dump_file)
        load_readonly_dump(principal_db, readonly_db, dump_file,
                           no_redirect_to_principal=no_redirect_to_principal)

    if not load_only:
        # This database no longer needs this schema (this only executes if
        # the check_call does not error).
        principal_db.session.close()
        principal_db.grab_session()
        principal_db.drop_schema('readonly')


@run_commands.command('all')
@click.option('-c', '--continuing', is_flag=True,
              help="Indicate whether you want the job to continue building an "
                   "existing dump corpus, or if you want to start a new one.")
@click.option('-d', '--dump-only', is_flag=True,
              help='Only generate the dumps on s3.')
@click.option('-l', '--load-only', is_flag=True,
              help='Only load a readonly dump from s3 into the given readonly '
                   'database.')
@click.option('--delete-existing', is_flag=True,
              help="Delete and restart an existing readonly schema in "
                   "principal.")
@click.option('--no-redirect-to-principal', is_flag=True,
              help="If given, the lambda function serving the REST API will not"
                   "be modified to redirect from the readonly database to the"
                   "principal database while readonly is being loaded.")
def run_all(continuing, delete_existing, load_only, dump_only,
            no_redirect_to_principal):
    """Generate new dumps and list existing dumps."""
    from indra_db import get_ro

    # Check if the readonly db handle is needed
    if not dump_only:
        ro_manager = get_ro('primary', protected=False)
    else:
        ro_manager = None

    dump(get_db('primary', protected=False),
         ro_manager, delete_existing,
         continuing, load_only, dump_only,
         no_redirect_to_principal=no_redirect_to_principal)


@dump_cli.command()
@click.option('--from-dump', type=click.DateTime(formats=[DATE_FMT]),
              help="Indicate a specific start dump from which to build. "
                   "The default is the most recent.")
@click.option('--no-redirect-to-principal', is_flag=True,
              help="If given, the lambda function serving the REST API will not"
                   "be modified to redirect from the readonly database to the"
                   "principal database while readonly is being loaded.")
def load_readonly(from_dump, no_redirect_to_principal):
    """Load the readonly database with readonly schema dump."""
    start = Start.from_date(from_dump)
    dump_file = Readonly.from_list(start.manifest).get_s3_path()
    if not dump_file:
        print(f"ERROR: No readonly dump for {start.date_stamp}")
        return
    load_readonly_dump(get_db('primary', protected=True),
                       get_ro('primary', protected=False), dump_file,
                       no_redirect_to_principal=no_redirect_to_principal)


@dump_cli.command('list')
@click.argument("state", type=click.Choice(["started", "done", "unfinished"]),
                required=False)
def show_list(state):
    """List existing dumps and their s3 paths.

    \b
    State options:
     - "started": get all dumps that have started (have "start.json" in them).
     - "done": get all dumps that have finished (have "end.json" in them).
     - "unfinished": get all dumps that have started but not finished.

    If no option is given, all dumps will be listed.
    """
    import boto3
    s3 = boto3.client('s3')

    # Set the parameters of the list_dumps function.
    if state == 'started':
        s = True
        e = None
    elif state == 'done':
        s = True
        e = True
    elif state == 'unfinished':
        s = True
        e = False
    else:
        s = None
        e = None

    # List the dump paths and their contents.
    for s3_path in list_dumps(s, e):
        print()
        print(s3_path)
        for el in s3_path.list_objects(s3):
            print('   ', str(el).replace(str(s3_path), ''))


@dump_cli.command()
def print_database_stats():
    """Print the summary counts for the content on the database."""
    from humanize import intword
    from tabulate import tabulate

    ro = get_ro('primary')
    db = get_db('primary')

    # Do source and text-type counts.
    res = db.session.execute("""
    SELECT source, text_type, COUNT(*) FROM text_content GROUP BY source, text_type;
    """)
    print("Source-type Counts:")
    print("-------------------")
    print(tabulate([(src, tt, intword(n)) for src, tt, n in sorted(res, key=lambda t: -t[-1])],
                   headers=["Source", "Text Type", "Content Count"]))
    print()

    # Do reader counts.
    res = db.session.execute("""
    SELECT reader, source, text_type, COUNT(*) 
    FROM text_content LEFT JOIN reading ON text_content_id = text_content.id
    GROUP BY reader, source, text_type;
    """)
    print("Reader and Source Type Counts:")
    print("------------------------------")
    print(tabulate([t[:-1] + (intword(t[-1]),) for t in sorted(res, key=lambda t: -t[-1])],
                   headers=["Reader", "Reader Version", "Source", "Text Type", "Reading Count"]))
    print()

    # Get the list of distinct HGNC IDs (and dump them as JSON).
    resp = ro.session.execute("""
    SELECT DISTINCT db_id FROM readonly.other_meta WHERE db_name = 'HGNC';
    """)
    ids = {db_id for db_id, in resp}
    print("Distinct HGNC Ids:", intword(len(ids)))
    with open('unique_hgnc_ids.json', 'w') as f:
        json.dump(list(ids), f)

    # Count the number of distinct groundings.
    res = ro.session.execute("""
    SELECT DISTINCT db_id, db_name
    FROM readonly.other_meta
    """)
    ids = {tuple(t) for t in res}
    print("Number of all distinct groundings:", intword(len(ids)))

    # Count the number of raw statements.
    (raw_cnt,), = db.session.execute("""
    SELECT count(*) FROM raw_statements
    """)
    print("Raw stmt count:", intword(raw_cnt))

    # Count the number of preassembled statements.
    (pa_cnt,), = db.session.execute("""
    SELECT count(*) FROM pa_statements
    """)
    print("PA stmt count:", intword(pa_cnt))

    # Count the number of links between raw and preassembled statements.
    (raw_used_in_pa_cnt,), = db.session.execute("""
    SELECT count(*) FROM raw_unique_links
    """)
    print("Raw statements used in PA:", intword(raw_used_in_pa_cnt))

    # Get the distinct grounded hashes.
    res = db.session.execute("""
    SELECT DISTINCT stmt_mk_hash FROM pa_agents WHERE db_name NOT IN ('TEXT', 'TEXT_NORM', 'NAME')
    """)
    grounded_hashes = {h for h, in res}
    print("Number of grounded hashes:", intword(len(grounded_hashes)))

    # Get number of hashes with agent counts.
    res = ro.session.execute("""
    SELECT mk_hash, agent_count, array_agg(ag_num), array_agg(db_name)
    FROM readonly.other_meta
    WHERE NOT is_complex_dup
    GROUP BY mk_hash, agent_count
    """)
    stmt_vals = [tuple(t) for t in res]
    print("Number of hashes, should be close to grounded pa count:", intword(len(stmt_vals)))

    # Count up the number of statements with all agents grounded.
    cnt = 0
    for h, n, ag_ids, ag_grnds in stmt_vals:
        ag_dict = defaultdict(set)
        for ag_id, ag_grnd in zip(ag_ids, ag_grnds):
            ag_dict[ag_id].add(ag_grnd)
        if len(ag_dict) == n:
            cnt += 1
    print("Number of pa statements in ro with all agents grounded:", intword(cnt))


@dump_cli.command('hierarchy')
def dump_hierarchy():
    """Dump hierarchy of Dumper classes to S3."""
    hierarchy = {}
    for d in get_all_descendants(Dumper):
        # Skip the FullPaStmts here.
        if d.name == 'full_pa_stmts':
            continue
        command_name = d.name.replace('_', '-')
        hierarchy[command_name] = d.config_to_json()
    s3_base = get_s3_dump()
    s3_path = s3_base.get_element_path('hierarchy.json')
    s3 = boto3.client('s3')
    s3_path.upload(s3, json.dumps(hierarchy).encode('utf-8'))


for DumperChild in get_all_descendants(Dumper):
    DumperChild.register()


dump_cli.add_command(run_commands)