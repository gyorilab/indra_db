import json
from collections import defaultdict

import click
import boto3
import pickle
import logging
from datetime import datetime

from indra.statements import get_all_descendants
from indra.statements.io import stmts_from_json
from indra_db import get_ro, get_db
from indra_db.belief import get_belief
from indra_db.config import CONFIG, get_s3_dump, record_in_test
from indra_db.util import get_db, get_ro, S3Path
from indra_db.util.aws import get_role_kwargs
from indra_db.util.dump_sif import dump_sif, get_source_counts, load_res_pos


logger = logging.getLogger(__name__)


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
    """Dumps a pandas dataframe of preassembled statements"""
    name = 'sif'
    fmt = 'pkl'
    db_required = True
    db_options = ['principal', 'readonly']

    def __init__(self, use_principal=False, **kwargs):
        super(Sif, self).__init__(use_principal=use_principal, **kwargs)

    def dump(self, src_counts_path, res_pos_path, belief_path,
             continuing=False):
        s3_path = self.get_s3_path()
        dump_sif(df_file=s3_path,
                 src_count_file=src_counts_path,
                 res_pos_file=res_pos_path,
                 belief_file=belief_path,
                 reload=True,
                 reconvert=True,
                 ro=self.db,
                 normalize_names=True)


class Belief(Dumper):
    """Dump a dict of belief scores keyed by hash"""
    name = 'belief'
    fmt = 'json'
    db_required = True
    db_options = ['principal']

    def dump(self, continuing=False):
        belief_dict = get_belief(self.db, partition=False)
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, json.dumps(belief_dict).encode('utf-8'))


class SourceCount(Dumper):
    """Dumps a dict of dicts with source counts per source api per statement"""
    name = 'source_count'
    fmt = 'pkl'
    db_required = True
    db_options = ['principal', 'readonly']

    def __init__(self, use_principal=True, **kwargs):
        super(SourceCount, self).__init__(use_principal=use_principal,
                                          **kwargs)

    def dump(self, continuing=False):
        get_source_counts(self.get_s3_path(), self.db)


class ResiduePosition(Dumper):
    """Dumps a dict of dicts with residue/position data from Modifications"""
    name = 'res_pos'
    fmt = 'pkl'
    db_required = True
    db_options = ['readonly', 'principal']

    def __init__(self, use_principal=True, **kwargs):
        super(ResiduePosition, self).__init__(use_principal=use_principal,
                                              **kwargs)

    def dump(self, continuing=False):
        res_pos_dict = load_res_pos(ro=self.db)
        s3 = boto3.client('s3')
        logger.info(f'Uploading residue position dump to '
                    f'{self.get_s3_path().to_string()}')
        self.get_s3_path().upload(s3=s3, body=pickle.dumps(res_pos_dict))


class FullPaJson(Dumper):
    """Dumps all statements found in FastRawPaLink as jsonl"""
    name = 'full_pa_json'
    fmt = 'jsonl'
    db_required = True
    db_options = ['principal', 'readonly']

    def __init__(self, use_principal=False, **kwargs):
        super(FullPaJson, self).__init__(use_principal=use_principal, **kwargs)

    def dump(self, continuing=False):
        query_res = self.db.session.query(self.db.FastRawPaLink.pa_json.distinct())
        jsonl_str = '\n'.join([js.decode() for js, in query_res.all()])
        s3 = boto3.client('s3')
        self.get_s3_path().upload(s3, jsonl_str.encode('utf-8'))


class FullPaStmts(Dumper):
    """Dumps all statements found in FastRawPaLink as a pickle"""
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



class PrincipalStats(Dumper):
    name = 'principal-statistics'
    fmt = 'csv'
    db_required = True
    db_options = ['principal']

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
        self.s3_dump_path().upload(s3, csv_bytes)


def load_readonly_dump(principal_db, readonly_db, dump_file):
    logger.info("Using dump_file = \"%s\"." % dump_file)
    logger.info("%s - Beginning upload of content (est. ~30 minutes)"
                % datetime.now())
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


def dump(principal_db, readonly_db, delete_existing=False, allow_continue=True,
         load_only=False, dump_only=False):
    """Run the suite of dumps in the specified order.

    Parameters
    ----------
    principal_db : :class:`indra_db.databases.PrincipalDatabaseManager`
        A handle to the principal database.
    readonly_db : :class:`indra_db.databases.ReadonlyDatabaseManager`
        A handle to the readonly database.
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
    """
    if not load_only:
        # START THE DUMP
        if delete_existing and 'readonly' in principal_db.get_schemas():
            principal_db.drop_schema('readonly')

        starter = Start()
        starter.dump(continuing=allow_continue)

        # STATS DUMP
        stats_dumper = PrincipalStats.from_list(starter.manifest)
        if not allow_continue or not stats_dumper:
            logger.info("Dumping principal stats.")
            PrincipalStats(db=principal_db, date_stamp=starter.date_stamp)\
                .dump(continuing=allow_continue)
        else:
            logger.info("Stats dump exists, skipping.")

        # BELIEF DUMP
        belief_dump = Belief.from_list(starter.manifest)
        if not allow_continue or not belief_dump:
            logger.info("Dumping belief.")
            belief_dumper = Belief(db=principal_db,
                                   date_stamp=starter.date_stamp)
            belief_dumper.dump(continuing=allow_continue)
            belief_dump = belief_dumper.get_s3_path()
        else:
            logger.info("Belief dump exists, skipping.")

        # READONLY DUMP
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


        # RESIDUE POSITION DUMP
        # By now, the readonly schema should exist on principal, so providing
        # the principal manager should be ok for source counts and
        # residue/position
        res_pos_dump = ResiduePosition.from_list(starter.manifest)
        if not allow_continue or not res_pos_dump:
            logger.info("Dumping residue and position")
            res_pos_dumper = ResiduePosition(db=principal_db,
                                             date_stamp=starter.date_stamp)
            res_pos_dumper.dump(continuing=allow_continue)
            res_pos_dump = res_pos_dumper.get_s3_path()
        else:
            logger.info("Residue position dump exists, skipping")

        # SOURCE COUNT DUMP
        src_count_dump = SourceCount.from_list(starter.manifest)
        if not allow_continue or not src_count_dump:
            logger.info("Dumping source count")
            src_count_dumper = SourceCount(db=principal_db,
                                           date_stamp=starter.date_stamp)
            src_count_dumper.dump(continuing=allow_continue)
            src_count_dump = src_count_dumper.get_s3_path()

        # SIF DUMP
        if not allow_continue or not Sif.from_list(starter.manifest):
            logger.info("Dumping sif from the readonly schema on principal.")
            Sif(db=principal_db, date_stamp=starter.date_stamp)\
                .dump(src_counts_path=src_count_dump,
                      res_pos_path=res_pos_dump,
                      belief_path=belief_dump,
                      continuing=allow_continue)
        else:
            logger.info("Sif dump exists, skipping.")

        # FULL PA JSON DUMP
        if not allow_continue or not FullPaJson.from_list(starter.manifest):
            logger.info("Dumping all PA Statements as jsonl.")
            FullPaJson(db=principal_db, date_stamp=starter.date_stamp)\
                .dump(continuing=allow_continue)
        else:
            logger.info("Statement dump exists, skipping.")

        # HASH MESH ID DUMP
        if not allow_continue \
                or not StatementHashMeshId.from_list(starter.manifest):
            logger.info("Dumping hash-mesh tuples.")
            StatementHashMeshId(db=principal_db,
                                date_stamp=starter.date_stamp)\
                .dump(continuing=allow_continue)

        # END DUMP
        End(date_stamp=starter.date_stamp).dump(continuing=allow_continue)
    else:
        # Find the most recent dump that has a readonly.
        dump_file = get_latest_dump_s3_path(Readonly.name)
        if dump_file is None:
            raise Exception("Could not find any suitable readonly dumps.")

    if not dump_only:
        # READONLY LOAD
        print("Dump file:", dump_file)
        load_readonly_dump(principal_db, readonly_db, dump_file)

    if not load_only:
        # This database no longer needs this schema (this only executes if
        # the check_call does not error).
        principal_db.session.close()
        principal_db.grab_session()
        principal_db.drop_schema('readonly')


@click.group('dump')
def dump_cli():
    """Manage the data dumps from Principal to files and Readonly."""


@dump_cli.command()
@click.option('-P', '--principal', default="primary",
              help="Specify which principal database to use.")
@click.option('-R', '--readonly', default="primary",
              help="Specify which readonly database to use.")
@click.option('-a', '--allow-continue', is_flag=True,
              help="Indicate whether you want the job to continue building an "
                   "existing dump corpus, or if you want to start a new one.")
@click.option('-d', '--delete-existing', is_flag=True,
              help="Delete and restart an existing readonly schema in "
                   "principal.")
@click.option('-u', '--dump-only', is_flag=True,
              help='Only generate the dumps on s3.')
@click.option('-l', '--load-only', is_flag=True,
              help='Only load a readonly dump from s3 into the given readonly '
                   'database.')
def run(principal, readonly, allow_continue, delete_existing, load_only,
        dump_only):
    """Generate new dumps and list existing dumps."""
    from indra_db import get_ro
    dump(get_db(principal, protected=False),
         get_ro(readonly, protected=False), delete_existing,
         allow_continue, load_only, dump_only)


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
def print_summary_counts():
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
