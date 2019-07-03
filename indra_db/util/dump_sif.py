__all__ = ['load_db_content', 'make_dataframe']

import sys
import pickle
import logging
import argparse
from itertools import permutations
from collections import OrderedDict

try:
    import pandas as pd
except ImportError:
    print("Pandas not available.")
    pd = None

from indra_db import util as dbu
from indra.databases import hgnc_client, go_client, mesh_client

logger = logging.getLogger(__name__)

NS_PRIORITY_LIST = (
    'FPLX',
    'HGNC',
    'GO',
    'MESH',
    'HMDB',
    'CHEBI',
    'PUBCHEM',
)


# All namespaces here (except NAME) should also be included in the
# NS_PRIORITY_LIST above
NS_LIST = ('NAME', 'HGNC', 'FPLX', 'GO', 'MESH', 'HMDB', 'CHEBI', 'PUBCHEM')

def format_agent(db_nm, db_id):
    if db_nm == 'FPLX':
        return db_id
    elif db_nm == 'HGNC':
        return hgnc_client.get_hgnc_name(db_id)
    elif db_nm == 'GO':
        norm_go_id = db_id[3:] if db_id.startswith('GO:GO:') else db_id
        return go_client.get_go_label(norm_go_id)
    elif db_nm == 'MESH':
        return mesh_client.get_mesh_name(db_id)
    return '{db_nm}:{db_id}'.format(db_nm=db_nm, db_id=db_id)

def load_db_content(reload, ns_list, pkl_filename=None):
    # Get the raw data
    if reload:
        db = dbu.get_primary_db()
        logger.info("Querying the database for statement metadata...")
        results = []
        for ns in ns_list:
            logger.info("Querying for {ns}".format(ns=ns))
            res = db.select_all([db.PaMeta.mk_hash, db.PaMeta.db_name,
                                 db.PaMeta.db_id, db.PaMeta.role,
                                 db.PaMeta.ev_count, db.PaMeta.type],
                                 db.PaMeta.db_name.like(ns))
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


def make_ev_strata(pkl_filename=None):

    """Returns a dict of dicts with evidence count per source, per statement

    The dictionary is at the top level keyed by statement hash and each
    entry contains a dictionary keyed by the source that support the
    statement where the entries are the evidence count for that source."""
    db = dbu.get_primary_db()
    res = db.select_all(db.PaStmtSrc)
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


def make_dataframe(pkl_filename, reconvert, db_content):
    if reconvert:
        # Organize by statement
        logger.info("Organizing by statement...")
        stmt_info = {}
        roles = {'SUBJECT':0, 'OBJECT':1, 'OTHER':2}
        for h, db_nm, db_id, r, n, t in db_content:
            if h not in stmt_info.keys():
                stmt_info[h] = {'agents': [], 'ev_count': n, 'type': t}
            stmt_info[h]['agents'].append((roles[r], db_nm, db_id))
        # Turn into dataframe with geneA, geneB, type, indexed by hash;
        # expand out all complexes
        # to multiple rows

        # Organize by pairs of genes, counting evidence.
        rows = []
        logger.info("Converting to pairwise entries...")
        for hash, info_dict in stmt_info.items():
            # Find roles with more than one agent
            agents_by_role = {}
            for role, db_nm, db_id in info_dict['agents']:
                assert db_nm in ns_priority_list
                db_rank = ns_priority_list.index(db_nm)
                # If we don't already have an agent for this role, use the
                # one we've found
                if role not in agents_by_role:
                    agents_by_role[role] = (role, db_nm, db_id, db_rank)
                # Otherwise, take the current agent if the identifier type
                # has a higher rank
                else:
                    cur_rank = agents_by_role[role][3]
                    if db_rank < cur_rank:
                        agents_by_role[role] = (role, db_nm, db_id, db_rank)

            agents = [(db_nm, db_id, format_agent(db_nm, db_id))
                      for _, db_nm, db_id, _ in sorted(agents_by_role.values())]

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
                        ('hash', hash)])
                rows.append(row)

        df = pd.DataFrame.from_dict(rows)

        with open(pkl_filename, 'wb') as f:
            pickle.dump(df, f)
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
