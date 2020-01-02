__all__ = ['load_db_content', 'make_dataframe', 'make_ev_strata', 'NS_LIST']

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

from indra_db import util as dbu

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


def upload_pickle_to_s3(obj, key, bucket=S3_SIF_BUCKET):
    """Upload a python object as a pickle to s3"""
    if key.startswith('s3:'):
        key = key[3:]
    s3 = get_s3_client(unsigned=False)
    try:
        s3.put_object(Body=pickle.dumps(obj=obj), Bucket=bucket, Key=key)
    except Exception as e:
        logger.error('Failed to upload to s3')
        logger.exception(e)


def load_db_content(reload, ns_list, pkl_filename=None, ro=None):
    # Get the raw data
    if reload:
        if not ro:
            ro = dbu.get_ro('primary-ro')
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
            with open(pkl_filename, 'wb') as f:
                pickle.dump(results, f)
    elif pkl_filename:
        logger.info("Loading database content from %s" % pkl_filename)
        with open(pkl_filename, 'rb') as f:
            results = pickle.load(f)
    logger.info("{len} stmts loaded".format(len=len(results)))
    return results


def make_ev_strata(pkl_filename=None, ro=None):
    """Returns a dict of dicts with evidence count per source, per statement

    The dictionary is at the top level keyed by statement hash and each
    entry contains a dictionary keyed by the source that support the
    statement where the entries are the evidence count for that source."""
    if not ro:
        ro = dbu.get_ro('primary-ro')
    res = ro.select_all(ro.PaStmtSrc)
    ev = {}
    for r in res:
        rd = r.__dict__
        ev[rd['mk_hash']] = {k: v for k, v in rd.items() if
                             k not in ['_sa_instance_state', 'mk_hash'] and
                             rd[k] is not None}

    if pkl_filename:
        with open(pkl_filename, 'wb') as f:
            pickle.dump(ev, f)
    return ev


def make_dataframe(reconvert, db_content, pkl_filename=None):
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
                    if nkey_errors < 10:
                        logger.warning('Missing key in agent name dict: '
                                       '(%s, %s)' % (hash, num))
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
            with open(pkl_filename, 'wb') as f:
                pickle.dump(df, f)
    else:
        if not pkl_filename:
            logger.error('Have to provide pickle file if not reconverting')
            raise FileExistsError
        else:
            with open(pkl_filename, 'rb') as f:
                df = pickle.load(f)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DB sif dumper',
        usage='Usage: dump_sif.py <db_dump_pkl> <dataframe_pkl> <csv_file>')
    parser.add_argument('--db-dump',
                        help='Dump a pickle of the database dump')
    parser.add_argument('--reload',
                        help='Reload from the database',
                        action='store_true',
                        default=True)
    parser.add_argument('--dataframe',
                        help='Dump a pickle of the database dump processed '
                             'into a pandas dataframe with pair interactions')
    parser.add_argument('--reconvert',
                        help='Re-run the dataframe processing on the db-dump',
                        action='store_true',
                        default=True)
    parser.add_argument('--csv-file',
                        help='Dump a csv file with statistics of the database '
                             'dump')
    parser.add_argument('--strat-ev',
                        help='If provided, also run and dump a pickled '
                             'dictionary of the stratified evidence count '
                             'per statement')
    args = parser.parse_args()

    dump_file = args.db_dump
    df_file = args.dataframe
    reload = args.reload
    csv_file = args.csv_file
    
    # Get the db content from a new DB dump or from file
    db_content = load_db_content(reload=reload, ns_list=NS_LIST,
                                 pkl_filename=dump_file)
    # Convert the database query result into a set of pairwise relationships
    df = make_dataframe(pkl_filename=df_file, reconvert=True,
                        db_content=db_content)

    if csv_file:
        # Aggregate rows by genes and stmt type
        logger.info("Saving to CSV...")
        filt_df = df.filter(items=['agA_ns', 'agA_id', 'agA_name',
                                   'agB_ns', 'agB_id', 'agB_name',
                                   'stmt_type', 'evidence_count'])
        type_counts = filt_df.groupby(by=['agA_ns', 'agA_id', 'agA_name',
                                          'agB_ns', 'agB_id', 'agB_name',
                                          'stmt_type']).sum()
        type_counts.to_csv(csv_file)

    if args.strat_ev:
        _ = make_ev_strata(args.strat_ev)
