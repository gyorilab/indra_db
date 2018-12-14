import pickle
from itertools import combinations, permutations
from collections import OrderedDict
import pandas as pd
from sqlalchemy import or_
from indra_db import util as dbu
from indra.databases import hgnc_client, go_client, mesh_client

ns_priority_list = (
    'FPLX',
    'HGNC',
    'GO',
    'MESH',
    'HMDB',
    'CHEBI',
    'PUBCHEM',
)

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
    return f'{db_nm}:{db_id}'


def load_db_content(pkl_filename, reload, ns_list):
    # Get the raw data
    if reload:
        db = dbu.get_primary_db()
        print("Querying the database for statement metadata...")
        results = []
        for ns in ns_list:
            print(f"Querying for {ns}")
            res = db.select_all([db.PaMeta.mk_hash, db.PaMeta.db_name,
                                 db.PaMeta.db_id, db.PaMeta.role,
                                 db.PaMeta.ev_count, db.PaMeta.type],
                                 db.PaMeta.db_name.like(ns))
            results.extend(res)
        results = set(results)
        with open(pkl_filename, 'wb') as f:
            pickle.dump(results, f)
    else:
        print("Loading database content from %s" % pkl_filename)
        with open(pkl_filename, 'rb') as f:
            results = pickle.load(f)
    print(f"{len(results)} stmts loaded")
    return results


def make_dataframe(pkl_filename, reconvert, db_content):
    if reconvert:
        # Organize by statement
        print("Organizing by statement...")
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
        pair_dict = {}
        rows = []
        print("Converting to pairwise entries...")
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
                #pair_key = tuple(pair)
                #if pair_key not in pair_dict.keys():
                #    pair_dict[pair_key] = 0
                #pair_dict[pair_key] += info_dict['ev_count']

        df = pd.DataFrame.from_dict(rows)

        with open(pkl_filename, 'wb') as f:
            pickle.dump(df, f)
    else:
        with open(pkl_filename, 'rb') as f:
            df = pickle.load(f)
    return df


if __name__ == '__main__':
    # All namespaces used here should be included in the ns_priority_list,
    # above
    ns_list = ('HGNC', 'FPLX', 'GO', 'MESH', 'HMDB', 'CHEBI', 'PUBCHEM')
    db_content = load_db_content('networks/db_dump_results.pkl', True, ns_list)
    df = make_dataframe('networks/stmt_df.pkl', True, db_content)
    # Convert the database query result into a set of pairwise relationships
    # Aggregate rows by genes and stmt type
    filt_df = df.filter(items=['agA_ns', 'agA_id', 'agA_name',
                               'agB_ns', 'agB_id', 'agB_name',
                               'stmt_type', 'evidence_count'])
    type_counts = filt_df.groupby(by=['agA_ns', 'agA_id', 'agA_name',
                                      'agB_ns', 'agB_id', 'agB_name',
                                      'stmt_type']).sum()
    type_counts.to_csv('networks/stmts_by_pair_type.csv')
