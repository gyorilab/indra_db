__all__ = ['load_db_content', 'make_dataframe', 'get_source_counts', 'NS_LIST',
           'dump_sif']

import pickle
import logging
import argparse
from io import StringIO
from datetime import datetime
from itertools import permutations
from collections import OrderedDict

from indra.util.aws import get_s3_client

try:
    import pandas as pd
except ImportError:
    print("Pandas not available.")
    pd = None

from indra_db.util.constructors import get_ro, get_db

logger = logging.getLogger(__name__)
S3_SIF_BUCKET = 'bigmech'
S3_SUBDIR = 'indra_db_sif_dump'

NS_PRIORITY_LIST = (
    'FPLX',
    'MIRBASE',
    'HGNC',
    'GO',
    'MESH',
    'HMDB',
    'CHEBI',
    'PUBCHEM',
)


# All namespaces here (except NAME) should also be included in the
# NS_PRIORITY_LIST above
NS_LIST = ('NAME', 'MIRBASE', 'HGNC', 'FPLX', 'GO', 'MESH', 'HMDB', 'CHEBI',
           'PUBCHEM')


def _pseudo_key(fname, ymd_date):
    return S3Path.from_key_parts(S3_SIF_BUCKET, S3_SUBDIR, ymd_date, fname)


def upload_pickle_to_s3(obj, s3_path):
    """Upload a python object as a pickle to s3"""
    logger.info('Uploading %s as pickle object to bucket %s'
                % (s3_path.key.split('/')[-1], s3_path.bucket))
    s3 = get_s3_client(unsigned=False)
    try:
        s3.put_object(Body=pickle.dumps(obj=obj), *s3_path.kw())
        logger.info('Finished dumping file to s3')
    except Exception as e:
        logger.error('Failed to upload to s3')
        logger.exception(e)


def load_pickle_from_s3(s3_path):
    logger.info('Loading pickle %s.' % s3_path)
    s3 = get_s3_client(unsigned=False)
    try:
        res = s3.get_object(*s3_path.kw())
        obj = pickle.loads(res['Body'].read())
        logger.info('Finished loading %s.' % s3_path)
        return obj
    except Exception as e:
        logger.error('Failed to load %s.' % s3_path)
        logger.exception(e)


def load_db_content(reload, ns_list, pkl_filename=None, ro=None):
    if isinstance(pkl_filename, str) and pkl_filename.startswith('s3:'):
        pkl_filename = S3Path.from_string(pkl_filename)
    # Get the raw data
    if reload:
        if not ro:
            ro = get_ro('primary')
        logger.info("Querying the database for statement metadata...")
        results = []
        for ns in ns_list:
            logger.info("Querying for {ns}".format(ns=ns))
            res = ro.select_all([ro.PaMeta.mk_hash, ro.PaMeta.db_name,
                                 ro.PaMeta.db_id, ro.PaMeta.ag_num,
                                 ro.PaMeta.ev_count, ro.PaMeta.type],
                                ro.PaMeta.db_name.like(ns))
            results.extend(res)
        results = set(results)
        if pkl_filename:
            if isinstance(pkl_filename, S3Path):
                upload_pickle_to_s3(results, pkl_filename)
            else:
                with open(pkl_filename, 'wb') as f:
                    pickle.dump(results, f)
    # Get a cached pickle
    elif pkl_filename:
        logger.info("Loading database content from %s" % pkl_filename)
        if pkl_filename.startswith('s3:'):
            results = load_pickle_from_s3(pkl_filename)
        else:
            with open(pkl_filename, 'rb') as f:
                results = pickle.load(f)
    else:
        raise ValueError("If `reload` is False a `pkl_filename` must be given.")
    logger.info("{len} stmts loaded".format(len=len(results)))
    return results


def get_source_counts(pkl_filename=None, ro=None):
    """Returns a dict of dicts with evidence count per source, per statement

    The dictionary is at the top level keyed by statement hash and each
    entry contains a dictionary keyed by the source that support the
    statement where the entries are the evidence count for that source."""
    if isinstance(pkl_filename, str) and pkl_filename.startswith('s3:'):
        pkl_filename = S3Path.from_string(pkl_filename)
    if not ro:
        ro = get_ro('primary-ro')
    res = ro.select_all(ro.PaStmtSrc)
    ev = {}
    for r in res:
        rd = r.__dict__
        ev[rd['mk_hash']] = {k: v for k, v in rd.items() if
                             k not in ['_sa_instance_state', 'mk_hash'] and
                             rd[k] is not None}

    if pkl_filename:
        if isinstance(pkl_filename, S3Path):
            upload_pickle_to_s3(obj=ev, s3_path=pkl_filename)
        else:
            with open(pkl_filename, 'wb') as f:
                pickle.dump(ev, f)
    return ev


def make_dataframe(reconvert, db_content, pkl_filename=None):
    if isinstance(pkl_filename, str) and pkl_filename.startswith('s3:'):
        pkl_filename = S3Path.from_string(pkl_filename)
    if reconvert:
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
                        ('stmt_hash', hash)])
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
    else:
        if not pkl_filename:
            logger.error('Have to provide pickle file if not reconverting')
            raise FileExistsError
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


def dump_sif(df_file=None, db_res_file=None, csv_file=None, src_count_file=None,
             reload=False, reconvert=False, ro=None):
    if ro is None:
        ro = get_db('primary')

    # Get the db content from a new DB dump or from file
    db_content = load_db_content(reload=reload, ns_list=NS_LIST,
                                 pkl_filename=db_res_file, ro=ro)

    # Convert the database query result into a set of pairwise relationships
    df = make_dataframe(pkl_filename=df_file, reconvert=reconvert,
                        db_content=db_content)

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
                    s3.put_object(Body=csv_buf.getvalue(), **csv_file.kw())
                    logger.info('Uploaded CSV file to s3')
                except Exception as second_e:
                    logger.error('Failed to upload csv file with fallback '
                                 'method')
                    logger.exception(second_e)
        # save locally
        else:
            type_counts.to_csv(csv_file)

    if src_count_file:
        _ = get_source_counts(src_count_file, ro=ro)
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
