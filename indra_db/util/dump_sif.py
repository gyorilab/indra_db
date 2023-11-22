__all__ = ['load_db_content', 'make_dataframe', 'get_source_counts', 'NS_LIST',
           'dump_sif', 'load_res_pos']

import json
import pickle
import logging
import argparse
from io import StringIO
from datetime import datetime
from itertools import permutations
from collections import OrderedDict, defaultdict
from typing import Tuple, Dict

from tqdm import tqdm

from indra.util.aws import get_s3_client
from indra_db.schemas.readonly_schema import ro_type_map
from indra.statements import get_all_descendants, Modification
from indra.statements.agent import default_ns_order
from indra.statements.validate import assert_valid_db_refs
from indra.databases.identifiers import ensure_prefix_if_needed

try:
    import pandas as pd
    from pandas import DataFrame
except ImportError:
    print("Pandas not available.")
    pd = None
    from typing import Any
    DataFrame = Any

from indra_db.util.s3_path import S3Path
from indra_db.util.constructors import get_ro, get_db

logger = logging.getLogger(__name__)
S3_SIF_BUCKET = 'bigmech'
S3_SUBDIR = 'indra_db_sif_dump'

NS_PRIORITY_LIST = tuple(default_ns_order)
NS_LIST = ('NAME', ) + NS_PRIORITY_LIST


def _pseudo_key(fname, ymd_date):
    return S3Path.from_key_parts(S3_SIF_BUCKET, S3_SUBDIR, ymd_date, fname)


def upload_pickle_to_s3(obj, s3_path):
    """Upload a python object as a pickle to s3"""
    logger.info('Uploading %s as pickle object to bucket %s'
                % (s3_path.key.split('/')[-1], s3_path.bucket))
    s3 = get_s3_client(unsigned=False)
    try:
        s3_path.upload(s3, pickle.dumps(obj))
        logger.info('Finished dumping file to s3')
    except Exception as e:
        logger.error('Failed to upload to s3')
        logger.exception(e)


def load_pickle_from_s3(s3_path):
    logger.info('Loading pickle %s.' % s3_path)
    s3 = get_s3_client(False)
    try:
        res = s3_path.get(s3)
        obj = pickle.loads(res['Body'].read())
        logger.info('Finished loading %s.' % s3_path)
        return obj
    except Exception as e:
        logger.error('Failed to load %s.' % s3_path)
        logger.exception(e)


def load_json_from_s3(s3_path):
    """Helper to load json from s3"""
    logger.info(f'Loading json {s3_path} from s3.')
    s3 = get_s3_client(False)
    try:
        res = s3_path.get(s3)
        obj = json.loads(res['Body'].read().decode())
        logger.info(f'Finished loading {s3_path}.')
        return obj
    except Exception as e:
        logger.error(f'Failed to load {s3_path}.')
        logger.exception(e)


def load_db_content(ns_list, pkl_filename=None, ro=None, reload=False):
    """Get preassembled stmt metadata from the DB for export.

    Queries the NameMeta, TextMeta, and OtherMeta tables as needed to get
    agent/stmt metadata for agents from the given namespaces.

    Parameters
    ----------
    ns_list : list of str
        List of agent namespaces to include in the metadata query.
    pkl_filename : str
        Name of pickle file to save to (if reloading) or load from (if not
        reloading). If an S3 path is given (i.e., pkl_filename starts with
        `s3:`), the file is loaded to/saved from S3. If not given,
        automatically reloads the content (overriding reload).
    ro : ReadonlyDatabaseManager
        Readonly database to load the content from. If not given, calls
        `get_ro('primary')` to get the primary readonly DB.
    reload : bool
        Whether to re-query the database for content or to load the content
        from from `pkl_filename`. Note that even if `reload` is False,
        if no `pkl_filename` is given, data will be reloaded anyway.

    Returns
    -------
    set of tuples
        Set of tuples containing statement information organized
        by agent. Tuples contain (stmt_hash, agent_ns, agent_id, agent_num,
        evidence_count, stmt_type).
    """
    if isinstance(pkl_filename, str) and pkl_filename.startswith('s3:'):
        pkl_filename = S3Path.from_string(pkl_filename)
    # Get the raw data
    if reload or not pkl_filename:
        if not ro:
            ro = get_ro('primary')
        logger.info("Querying the database for statement metadata...")
        results = {}
        for ns in ns_list:
            logger.info("Querying for {ns}".format(ns=ns))
            filters = []
            if ns == 'NAME':
                tbl = ro.NameMeta
            elif ns == 'TEXT':
                tbl = ro.TextMeta
            else:
                tbl = ro.OtherMeta
                filters.append(tbl.db_name.like(ns))
            filters.append(tbl.is_complex_dup == False)
            res = ro.select_all([tbl.mk_hash, tbl.db_id, tbl.ag_num,
                                 tbl.ev_count, tbl.type_num], *filters)
            results[ns] = res
        results = {(h, dbn, dbi, ag_num, ev_cnt, ro_type_map.get_str(tn))
                   for dbn, value_list in results.items()
                   for h, dbi, ag_num, ev_cnt, tn in value_list}
        if pkl_filename:
            if isinstance(pkl_filename, S3Path):
                upload_pickle_to_s3(results, pkl_filename)
            else:
                with open(pkl_filename, 'wb') as f:
                    pickle.dump(results, f)
    # Get a cached pickle
    else:
        logger.info("Loading database content from %s" % pkl_filename)
        if pkl_filename.startswith('s3:'):
            results = load_pickle_from_s3(pkl_filename)
        else:
            with open(pkl_filename, 'rb') as f:
                results = pickle.load(f)
    logger.info("{len} stmts loaded".format(len=len(results)))
    return results


def load_res_pos(ro=None):
    """Return residue/position data keyed by hash"""
    logger.info('Getting residue and position info')
    if ro is None:
        ro = get_ro('primary')
    res = {'residue': {}, 'position': {}}
    for stmt_type in get_all_descendants(Modification):
        stmt_name = stmt_type.__name__
        if stmt_name in ('Modification', 'AddModification',
                         'RemoveModification'):
            continue
        logger.info(f'Getting statements for type {stmt_name}')
        type_num = ro_type_map.get_int(stmt_name)
        query = ro.select_all(ro.FastRawPaLink.pa_json,
                              ro.FastRawPaLink.type_num == type_num)
        for jsb, in query:
            js = json.loads(jsb)
            if 'residue' in js:
                res['residue'][int(js['matches_hash'])] = js['residue']
            if 'position' in js:
                res['position'][int(js['matches_hash'])] = js['position']
    return res


def get_source_counts(pkl_filename=None, ro=None):
    """Returns a dict of dicts with evidence count per source, per statement

    The dictionary is at the top level keyed by statement hash and each
    entry contains a dictionary keyed by the source that support the
    statement where the entries are the evidence count for that source."""
    logger.info('Getting source counts per statement')
    if isinstance(pkl_filename, str) and pkl_filename.startswith('s3:'):
        pkl_filename = S3Path.from_string(pkl_filename)
    if not ro:
        ro = get_ro('primary-ro')
    ev = {h: j for h, j in ro.select_all([ro.SourceMeta.mk_hash,
                                          ro.SourceMeta.src_json])}

    if pkl_filename:
        if isinstance(pkl_filename, S3Path):
            upload_pickle_to_s3(obj=ev, s3_path=pkl_filename)
        else:
            with open(pkl_filename, 'wb') as f:
                pickle.dump(ev, f)
    return ev


def normalize_sif_names(sif_df: DataFrame):
    """Try to normalize names in the sif dump dataframe

    This function tries to normalize the names of the entities in the sif
    dump. The 'bio_ontology' is the arbiter of what constitutes a normalized
    name. If no name exists, no further attempt to change the name is made.

    Parameters
    ----------
    sif_df :
        The sif dataframe
    """
    from indra.ontology.bio import bio_ontology
    bio_ontology.initialize()
    logger.info('Getting ns, id, name tuples')

    # Get the set of grounded entities
    ns_id_name_tups = set(
        zip(sif_df.agA_ns, sif_df.agA_id, sif_df.agA_name)).union(
        set(zip(sif_df.agB_ns, sif_df.agB_id, sif_df.agB_name))
    )

    # Get the ontology name, if it exists, and check if the name in the
    # dataframe needs update
    logger.info('Checking which names need updating')
    inserted_set = set()
    for ns_, id_, cur_name in tqdm(ns_id_name_tups):
        oname = bio_ontology.get_name(ns_, id_)
        # If there is a name in the ontology and it is different than the
        # original, insert it
        if oname and oname != cur_name and (ns_, id_, oname) not in inserted_set:
            inserted_set.add((ns_, id_, oname))

    if len(inserted_set) > 0:
        logger.info(f'Found {len(inserted_set)} names in dataframe that need '
                    f'renaming')

        # Make dataframe of rename dict
        logger.info('Making rename dataframe')
        df_dict = defaultdict(list)
        for ns_, id_, name in inserted_set:
            df_dict['ns'].append(ns_)
            df_dict['id'].append(id_)
            df_dict['name'].append(name)

        rename_df = pd.DataFrame(df_dict)

        # Do merge on with relevant columns from sif for both A and B
        logger.info('Getting temporary dataframes for renaming')

        # Get dataframe with ns, id, new name column
        rename_a = sif_df[['agA_ns', 'agA_id']].merge(
            right=rename_df,
            left_on=['agA_ns', 'agA_id'],
            right_on=['ns', 'id'], how='left'
        ).drop('ns', axis=1).drop('id', axis=1)

        # Check which rows have name entries
        truthy_a = pd.notna(rename_a.name)

        # Rename in sif_df from new names
        sif_df.loc[truthy_a, 'agA_name'] = rename_a.name[truthy_a]

        # Repeat for agB_name
        rename_b = sif_df[['agB_ns', 'agB_id']].merge(
            right=rename_df,
            left_on=['agB_ns', 'agB_id'],
            right_on=['ns', 'id'], how='left'
        ).drop('ns', axis=1).drop('id', axis=1)
        truthy_b = pd.notna(rename_b.name)
        sif_df.loc[truthy_b, 'agB_name'] = rename_b.name[truthy_b]

        # Check that there are no missing names
        logger.info('Performing sanity checks')
        assert sum(pd.isna(sif_df.agA_name)) == 0
        assert sum(pd.isna(sif_df.agB_name)) == 0

        # Get the set of ns, id, name tuples and check diff
        ns_id_name_tups_after = set(
            zip(sif_df.agA_ns, sif_df.agA_id, sif_df.agA_name)).union(
            set(zip(sif_df.agB_ns, sif_df.agB_id, sif_df.agB_name))
        )
        # Check that rename took place
        assert ns_id_name_tups_after != ns_id_name_tups
        # Check that all new names are used
        assert set(rename_df.name).issubset({n for _, _, n in ns_id_name_tups_after})
        logger.info('Sif dataframe renamed successfully')
    else:
        logger.info('No names need renaming')


def make_dataframe(reconvert, db_content, res_pos_dict, src_count_dict,
                   belief_dict, pkl_filename=None,
                   normalize_names: bool = False):
    """Make a pickled DataFrame of the db content, one row per stmt.

    Parameters
    ----------
    reconvert : bool
        Whether to generate a new DataFrame from the database content or
        to load and return a DataFrame from the given pickle file. If False,
        `pkl_filename` must be given.
    db_content : set of tuples
        Set of tuples of agent/stmt data as returned by `load_db_content`.
    res_pos_dict : Dict[str, Dict[str, str]]
        Dict containing residue and position keyed by hash.
    src_count_dict : Dict[str, Dict[str, int]]
        Dict of dicts containing source counts per source api keyed by hash.
    belief_dict : Dict[str, float]
        Dict of belief scores keyed by hash.
    pkl_filename : str
        Name of pickle file to save to (if reconverting) or load from (if not
        reconverting). If an S3 path is given (i.e., pkl_filename starts with
        `s3:`), the file is loaded to/saved from S3. If not given,
        reloads the content (overriding reload).
    normalize_names :
        If True, detect and try to merge name duplicates (same entity with
        different names, e.g. Loratadin vs loratadin). Default: False

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the content, with columns: 'agA_ns', 'agA_id',
        'agA_name', 'agB_ns', 'agB_id', 'agB_name', 'stmt_type',
        'evidence_count', 'stmt_hash'.
    """
    if isinstance(pkl_filename, str) and pkl_filename.startswith('s3:'):
        pkl_filename = S3Path.from_string(pkl_filename)
    if reconvert:
        # Content consists of tuples organized by agent, e.g.
        # (-11421523615931377, 'UP', 'P04792', 1, 1, 'Phosphorylation')
        #
        # First we need to organize by statement, collecting all agents
        # for each statement along with evidence count and type.
        # We also separately store the NAME attribute for each statement
        # agent (indexing by hash/agent_num).
        logger.info("Organizing by statement...")
        stmt_info = {} # Store statement info (agents, ev, type) by hash
        ag_name_by_hash_num = {} # Store name for each stmt agent
        for h, db_nm, db_id, num, n, t in tqdm(db_content):
            db_nmn, db_id = fix_id(db_nm, db_id)
            # Populate the 'NAME' dictionary per agent
            if db_nm == 'NAME':
                ag_name_by_hash_num[(h, num)] = db_id
            if h not in stmt_info.keys():
                stmt_info[h] = {'agents': [], 'ev_count': n, 'type': t}
            stmt_info[h]['agents'].append((num, db_nm, db_id))
        # Turn into dataframe with geneA, geneB, type, indexed by hash;
        # expand out complexes to multiple rows

        # Organize by pairs of genes, counting evidence.
        nkey_errors = 0
        error_keys = []
        rows = []
        logger.info("Converting to pairwise entries...")
        # Iterate over each statement
        for hash, info_dict in tqdm(stmt_info.items()):
            # Get the priority grounding for the agents in each position
            agents_by_num = {}
            for num, db_nm, db_id in info_dict['agents']:
                # Agent name is handled separately so we skip it here
                if db_nm == 'NAME':
                    continue
                # For other namespaces, we get the top-priority namespace
                # given all namespaces for the agent
                else:
                    assert db_nm in NS_PRIORITY_LIST
                    db_rank = NS_PRIORITY_LIST.index(db_nm)
                    # If we don't already have an agent for this num, use the
                    # one we've found
                    if num not in agents_by_num:
                        agents_by_num[num] = (num, db_nm, db_id, db_rank)
                    # Otherwise, take the current agent if the identifier type
                    # has a higher rank
                    else:
                        cur_rank = agents_by_num[num][3]
                        if db_rank < cur_rank:
                            agents_by_num[num] = (num, db_nm, db_id, db_rank)
            # Make ordered list of agents for this statement, picking up
            # the agent name from the ag_name_by_hash_num dict that we
            # built earlier
            agents = []
            for num, db_nm, db_id, _ in sorted(agents_by_num.values()):
                # Try to get the agent name
                ag_name = ag_name_by_hash_num.get((hash, num), None)
                # If the name is not found, log it but allow the agent
                # to be included as None
                if ag_name is None:
                    nkey_errors += 1
                    error_keys.append((hash, num))
                    if nkey_errors < 11:
                        logger.warning('Missing key in agent name dict: '
                                       '(%s, %s)' % (hash, num))
                    elif nkey_errors == 11:
                        logger.warning('Got more than 10 key warnings: '
                                       'muting further warnings.')
                agents.append((db_nm, db_id, ag_name))

            # Need at least two agents.
            if len(agents) < 2:
                continue

            # If this is a complex, or there are more than two agents, permute!
            if info_dict['type'] == 'Complex':
                # Skip complexes with 4 or more members
                if len(agents) > 3:
                    continue
                pairs = permutations(agents, 2)
            else:
                pairs = [agents]

            # Add all the pairs, and count up total evidence.
            for pair in pairs:
                row = OrderedDict([
                    ('agA_ns', pair[0][0]),
                    ('agA_id', pair[0][1]),
                    ('agA_name', pair[0][2]),
                    ('agB_ns', pair[1][0]),
                    ('agB_id', pair[1][1]),
                    ('agB_name', pair[1][2]),
                    ('stmt_type', info_dict['type']),
                    ('evidence_count', info_dict['ev_count']),
                    ('stmt_hash', hash),
                    ('residue', res_pos_dict['residue'].get(hash)),
                    ('position', res_pos_dict['position'].get(hash)),
                    ('source_counts', src_count_dict.get(hash)),
                    ('belief', belief_dict.get(str(hash)))
                ])
                rows.append(row)
        if nkey_errors:
            ef = 'key_errors.csv'
            logger.warning('%d KeyErrors. Offending keys found in %s' %
                           (nkey_errors, ef))
            with open(ef, 'w') as f:
                f.write('hash,PaMeta.ag_num\n')
                for kn in error_keys:
                    f.write('%s,%s\n' % kn)
        df = pd.DataFrame.from_dict(rows)

        if normalize_names:
            normalize_sif_names(sif_df=df)

        if pkl_filename:
            if isinstance(pkl_filename, S3Path):
                upload_pickle_to_s3(obj=df, s3_path=pkl_filename)
            else:
                with open(pkl_filename, 'wb') as f:
                    pickle.dump(df, f)
    else:
        if not pkl_filename:
            logger.error('Have to provide pickle file if not reconverting')
            raise FileNotFoundError(pkl_filename)
        else:
            if isinstance(pkl_filename, S3Path):
                df = load_pickle_from_s3(pkl_filename)
            else:
                with open(pkl_filename, 'rb') as f:
                    df = pickle.load(f)
    return df


def get_parser():
    parser = argparse.ArgumentParser(description='DB sif dumper',
                                     usage=('Usage: dump_sif.py <db_dump_pkl> '
                                            '<dataframe_pkl> <csv_file>'))
    parser.add_argument('--db-dump',
                        help='A pickle of the database dump. If provided '
                             'with --reload, this is the name of a new '
                             'db dump pickle, otherwise this is assumed to '
                             'be a cached pickle that already exists.')
    parser.add_argument('--reload',
                        help='Reload the database content from the database.',
                        action='store_true',
                        default=False)
    parser.add_argument('--dataframe',
                        help='A pickle of the database dump processed '
                             'into a pandas dataframe with pair '
                             'interactions. If provided with the --reconvert '
                             'option, this is the name of a new dataframe '
                             'pickle, otherwise this is assumed to '
                             'be a cached pickle that already exists.')
    parser.add_argument('--reconvert',
                        help='Re-run the dataframe processing on the db-dump',
                        action='store_true',
                        default=False)
    parser.add_argument('--csv-file',
                        help='Dump a csv file with statistics of the database '
                             'dump')
    parser.add_argument('--src-counts',
                        help='If provided, also run and dump a pickled '
                             'dictionary of the stratified evidence count '
                             'per statement from each of the different '
                             'sources.')
    parser.add_argument('--s3',
                        action='store_true',
                        default=False,
                        help='Upload files to the bigmech s3 bucket instead '
                             'of saving them on the local disk.')
    parser.add_argument('--s3-ymd',
                        default=datetime.utcnow().strftime('%Y-%m-%d'),
                        help='Set the dump sub-directory name on s3 '
                             'specifying the date when the file was '
                             'processed. Default: %Y-%m-%d of '
                             'datetime.datetime.utcnow()')
    parser.add_argument('--principal',
                        action='store_true',
                        default=False,
                        help='Use the principal db instead of the readonly')
    return parser


def dump_sif(src_count_file, res_pos_file, belief_file, df_file=None,
             db_res_file=None, csv_file=None, reload=True, reconvert=True,
             ro=None, normalize_names: bool = True):
    """Build and dump a sif dataframe of PA statements with grounded agents

    Parameters
    ----------
    src_count_file : Union[str, S3Path]
        A location to load the source count dict from. Can be local file
        path, an s3 url string or an S3Path instance.
    res_pos_file : Union[str, S3Path]
        A location to load the residue-postion dict from. Can be local file
        path, an s3 url string or an S3Path instance.
    belief_file : Union[str, S3Path]
        A location to load the belief dict from. Can be local file path,
        an s3 url string or an S3Path instance.
    df_file : Optional[Union[str, S3Path]]
        If provided, dump the sif to this location. Can be local file path,
        an s3 url string or an S3Path instance.
    db_res_file : Optional[Union[str, S3Path]]
        If provided, save the db content to this location. Can be local file
        path, an s3 url string or an S3Path instance.
    csv_file : Optional[str, S3Path]
        If provided, calculate dataframe statistics and save to local file
        or s3. Can be local file path, an s3 url string or an S3Path instance.
    reconvert : bool
        Whether to generate a new DataFrame from the database content or
        to load and return a DataFrame from `df_file`. If False, `df_file`
        must be given. Default: True.
    reload : bool
        If True, load new content from the database and make a new
        dataframe. If False, content can be loaded from provided files.
        Default: True.
    ro : Optional[ReadonlyDatabaseManager]
        Provide a DatabaseManager to load database content from. If not
        provided, `get_ro('primary')` will be used.
    normalize_names :
        If True, detect and try to merge name duplicates (same entity with
        different names, e.g. Loratadin vs loratadin). Default: False
    """
    def _load_file(path):
        if isinstance(path, str) and path.startswith('s3:') or \
                isinstance(path, S3Path):
            if isinstance(path, str):
                s3path = S3Path.from_string(path)
            else:
                s3path = path
            if s3path.to_string().endswith('pkl'):
                return load_pickle_from_s3(s3path)
            elif s3path.to_string().endswith('json'):
                return load_json_from_s3(s3path)
            else:
                raise ValueError(f'Unknown file format of {path}')
        else:
            if path.endswith('pkl'):
                with open(path, 'rb') as f:
                    return pickle.load(f)
            elif path.endswith('json'):
                with open(path, 'r') as f:
                    return json.load(f)

    if ro is None:
        ro = get_db('primary')

    # Get the db content from a new DB dump or from file
    db_content = load_db_content(reload=reload, ns_list=NS_LIST,
                                 pkl_filename=db_res_file, ro=ro)

    # Load supporting files
    res_pos = _load_file(res_pos_file)
    src_count = _load_file(src_count_file)
    belief = _load_file(belief_file)

    # Convert the database query result into a set of pairwise relationships
    df = make_dataframe(pkl_filename=df_file, reconvert=reconvert,
                        db_content=db_content, src_count_dict=src_count,
                        res_pos_dict=res_pos, belief_dict=belief,
                        normalize_names=normalize_names)

    if csv_file:
        if isinstance(csv_file, str) and csv_file.startswith('s3:'):
            csv_file = S3Path.from_string(csv_file)
        # Aggregate rows by genes and stmt type
        logger.info("Saving to CSV...")
        filt_df = df.filter(items=['agA_ns', 'agA_id', 'agA_name',
                                   'agB_ns', 'agB_id', 'agB_name',
                                   'stmt_type', 'evidence_count'])
        type_counts = filt_df.groupby(by=['agA_ns', 'agA_id', 'agA_name',
                                          'agB_ns', 'agB_id', 'agB_name',
                                          'stmt_type']).sum()
        # This requires package s3fs under the hood. See:
        # https://pandas.pydata.org/pandas-docs/stable/whatsnew/v0.20.0.html#s3-file-handling
        if isinstance(csv_file, S3Path):
            try:
                type_counts.to_csv(csv_file.to_string())
            except Exception as e:
                try:
                    logger.warning('Failed to upload csv to s3 using direct '
                                   's3 url, trying boto3: %s.' % e)
                    s3 = get_s3_client(unsigned=False)
                    csv_buf = StringIO()
                    type_counts.to_csv(csv_buf)
                    csv_file.upload(s3, csv_buf)
                    logger.info('Uploaded CSV file to s3')
                except Exception as second_e:
                    logger.error('Failed to upload csv file with fallback '
                                 'method')
                    logger.exception(second_e)
        # save locally
        else:
            type_counts.to_csv(csv_file)
    return


def fix_id(db_ns: str, db_id: str) -> Tuple[str, str]:
    """Fix ID issues specific to the SIF dump."""
    if db_ns == "GO":
        if db_id.isnumeric():
            db_id = "0" * (7 - len(db_id)) + db_id
    elif db_ns == "EFO" and db_id.startswith("EFO:"):
        db_id = db_id[4:]
    elif db_ns == "UP" and db_id.startswith("SL"):
        db_ns = "UPLOC"
    elif db_ns == "UP" and "-" in db_id and not db_id.startswith("SL-"):
        db_id = db_id.split("-")[0]
    elif db_ns == 'FPLX' and db_id == 'TCF-LEF':
        db_id = 'TCF_LEF'
    db_id = ensure_prefix_if_needed(db_ns, db_id)
    return db_ns, db_id


def main():
    args = get_parser().parse_args()

    ymd = args.s3_ymd
    if args.s3:
        logger.info('Uploading to %s/%s/%s on s3 instead of saving locally'
                    % (S3_SIF_BUCKET, S3_SUBDIR, ymd))
    db_res_file = _pseudo_key(args.db_dump, ymd) if args.s3 and args.db_dump\
        else args.db_dump
    df_file = _pseudo_key(args.dataframe, ymd) if args.s3 and args.dataframe\
        else args.dataframe
    csv_file = _pseudo_key(args.csv_file, ymd) if args.s3 and args.csv_file\
        else args.csv_file
    src_count_file = _pseudo_key(args.src_counts, ymd) if args.s3 and \
        args.src_counts else args.src_counts

    reload = args.reload
    if reload:
        logger.info('Reloading the database content from the database')
    else:
        logger.info('Loading cached database content from %s' % db_res_file)

    reconvert = args.reconvert
    if reconvert:
        logger.info('Reconverting database content into pandas dataframe')
    else:
        logger.info('Loading cached dataframe from %s' % df_file)

    for f in [db_res_file, df_file, csv_file, src_count_file]:
        if f:
            logger.info('Using file name %s' % f)
        else:
            continue

    dump_sif(df_file, db_res_file, csv_file, src_count_file, reload, reconvert,
             get_db('primary') if args.principal else get_ro('primary'))


if __name__ == '__main__':
    main()
