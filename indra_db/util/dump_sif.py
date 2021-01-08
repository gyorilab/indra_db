__all__ = ['load_db_content', 'make_dataframe', 'get_source_counts', 'NS_LIST',
           'dump_sif', 'dump_sif_from_stmts']

import json
import pickle
import logging
import argparse
from io import StringIO
from datetime import datetime
from itertools import permutations
from collections import OrderedDict

from indra.util.aws import get_s3_client
from indra.assemblers.indranet import IndraNetAssembler
from indra_db.schemas.readonly_schema import ro_type_map
from indra.statements import get_all_descendants, Modification
from indra.statements.agent import default_ns_order

try:
    import pandas as pd
except ImportError:
    print("Pandas not available.")
    pd = None

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
    s3 = get_s3_client(unsigned=False)
    try:
        res = s3.get_object(**s3_path.kw())
        obj = pickle.loads(res['Body'].read())
        logger.info('Finished loading %s.' % s3_path)
        return obj
    except Exception as e:
        logger.error('Failed to load %s.' % s3_path)
        logger.exception(e)


def load_db_content(ns_list, pkl_filename=None, ro=None):
    """Loads database content for grounded agents

    Parameters
    ----------
    ns_list : List[str]
        The list of namespaces to load from the database
    pkl_filename : Optional[Union[str, S3Path]]
        If provided, save the db content to it. Can be local file path, an
        s3 url string or an S3Path instance.
    ro : Optional[PrincipalDatabaseManager]
        A DatabaseManager to use for querying the db. If not provided,
        a new instance will be created from `get_ro('primary')`

    Returns
    -------
    Set[Tuple[str]]
        The database content as a set of tuples.
    """
    if isinstance(pkl_filename, str) and pkl_filename.startswith('s3:'):
        pkl_filename = S3Path.from_string(pkl_filename)

    # Load database content
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
    logger.info(f"{len(results)} stmts loaded")
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
                res['residue'][js['matches_hash']] = js['residue']
            if 'position' in js:
                res['position'][js['matches_hash']] = js['position']
    return res


def get_source_counts(output_pkl_filename=None, ro=None):
    """Returns a dict of dicts with evidence count per source, per statement

    The dictionary is at the top level keyed by statement hash and each
    entry contains a dictionary keyed by the source that support the
    statement where the entries are the evidence count for that source

    Parameters
    ----------
    output_pkl_filename : Union[str, S3Path]
        If provided, dump the source counts dict to the provided location.
    ro : Optional[PrincipalDatabaseManager]
        If provided, used to query the database. If not provided,
        `get_ro('primary')` will be used to provide database access.

    Returns
    -------
    Dict[str, Dict[str, int]]
        A dict of source count dicts, keyed by hash
    -------

    """
    if isinstance(output_pkl_filename, str) and \
            output_pkl_filename.startswith('s3:'):
        pkl_filename = S3Path.from_string(output_pkl_filename)
    if not ro:
        ro = get_ro('primary-ro')
    ev = {h: j for h, j in ro.select_all([ro.SourceMeta.mk_hash,
                                          ro.SourceMeta.src_json])}

    if output_pkl_filename:
        if isinstance(output_pkl_filename, S3Path):
            upload_pickle_to_s3(obj=ev, s3_path=output_pkl_filename)
        else:
            with open(output_pkl_filename, 'wb') as f:
                pickle.dump(ev, f)
    return ev


def make_dataframe(db_content, res_pos_dict, src_count_dict,
                   pkl_filename=None):
    """Load data in db_content into a pandas dataframe and dump it

    Parameters
    ----------
    db_content : Set[Tuple[str]]
        The loaded database content as a set of tuples.
    res_pos_dict : Dict[str, Dict[str, str]]
        Provide a dict of {'residue': {hash: residue}, 'position': {hash:
        position}} to load residue and position into the dataframe with.
    src_count_dict : Dict[str, Dict[str, int]]
        Provide source counts to merge into the dataframe.
    pkl_filename : Optional[Union[str, S3Path]]
        If provided, dump the dataframe to this location. Can be local
        file path, s3 url string, or instance of S3Path.


    Returns
    -------
    pd.DataFrame
        A pandas dataframe of pairwise agent interactions with statement
        information for the interaction
    """
    if isinstance(pkl_filename, str) and pkl_filename.startswith('s3:'):
        pkl_filename = S3Path.from_string(pkl_filename)

    # Organize by statement
    logger.info("Organizing by statement...")
    stmt_info = {}
    ag_name_by_hash_num = {}
    for h, db_nm, db_id, num, n, t in db_content:
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
    for hash, info_dict in stmt_info.items():
        # Find roles with more than one agent
        agents_by_num = {}
        for num, db_nm, db_id in info_dict['agents']:
            if db_nm == 'NAME':
                continue
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

        agents = []
        for num, db_nm, db_id, _ in sorted(agents_by_num.values()):
            try:
                agents.append((db_nm, db_id,
                               ag_name_by_hash_num[(hash, num)]))
            except KeyError:
                nkey_errors += 1
                error_keys.append((hash, num))
                if nkey_errors < 11:
                    logger.warning('Missing key in agent name dict: '
                                   '(%s, %s)' % (hash, num))
                elif nkey_errors == 11:
                    logger.warning('Got more than 10 key warnings: '
                                   'muting further warnings.')
                continue

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
                ('source_count', src_count_dict.get(hash))
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

    if pkl_filename:
        if isinstance(pkl_filename, S3Path):
            upload_pickle_to_s3(obj=df, s3_path=pkl_filename)
        else:
            with open(pkl_filename, 'wb') as f:
                pickle.dump(df, f)
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


def dump_sif_from_stmts(stmt_list, output):
    """Create a dataframe with pairs of agents

    Parameters
    ----------
    stmt_list : Union[str, List[indra.statements.Statement]
        A list of statements to transform to a DataFrame
    output : Union[str, S3Path]
        Where to save the output to. Can be local file path or an S3Path.
    """
    ia = IndraNetAssembler(statements=stmt_list)
    df = ia.make_df()
    if isinstance(output, S3Path):
        s3 = get_s3_client(False)
        output.put(s3=s3, body=pickle.dumps(df))
    else:
        with open(output, 'wb') as f:
            pickle.dump(obj=df, file=f)


def dump_sif(df_file=None, db_res_file=None, csv_file=None,
             src_count_file=None, ro=None):
    """Build and dump a sif dataframe of PA statements with grounded agents

    Parameters
    ----------
    df_file : Optional[Union[str, S3Path]]
        If provided, dump the sif to this location. Can be local file path, an
        s3 url string or an S3Path instance.
    src_count_file : Optional[Union[str, S3Path]]
        A location to dump the source count dict. Can be local file path, an
        s3 url string or an S3Path instance.
    db_res_file : Optional[Union[str, S3Path]]
        If provided, save the db content to this location. Can be local file
        path, an s3 url string or an S3Path instance.
    csv_file : Optional[str]
        If provided, save a CSV file with count statistics. Can be local file
        path, an s3 url string  or an S3Path instance.
    ro : Optional[PrincipalDatabaseManager]
        Provide a DatabaseManager to load database content from. If not
        provided, `get_ro('primary')` will be used.
    """
    if ro is None:
        ro = get_db('primary')

    # Get the db content from a new DB dump or from file
    db_content = load_db_content(ns_list=NS_LIST, pkl_filename=db_res_file,
                                 ro=ro)

    # Get residue and position for the relevant stmt types
    res_pos = load_res_pos(ro=ro)

    # Get source counts: will also dump source counts
    src_counts = get_source_counts(output_pkl_filename=src_count_file, ro=ro)

    # Convert the database query result into a set of pairwise relationships
    df = make_dataframe(pkl_filename=df_file, db_content=db_content,
                        res_pos_dict=res_pos, src_count_dict=src_counts)

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
