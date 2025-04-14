import csv
import gzip
import json
import pickle
from pathlib import Path
from collections import Counter

import networkx as nx

from indra.belief import BeliefEngine
from indra.statements import Agent, Evidence, Activation
from indra_db import get_db
from indra_db.readonly_dumping.export_assembly import calculate_belief


def test_unit_belief_calc():
    activation = Activation(
        Agent("A"),
        Agent("B"),
        evidence=[Evidence(source_api="reach") for _ in range(3)],
    )

    # Test that the belief score is calculated correctly
    assert activation.belief == 1

    # Set up default Belief Engine
    belief_engine = BeliefEngine()

    belief_engine.set_prior_probs([activation])

    assert activation.belief != 1
    assert activation.belief == 0.923


def test_calculate_belief():
    activation1 = Activation(
        Agent("A", location="nucleus"),
        Agent("B", location="cytoplasm"),
        evidence=[
            Evidence(
                source_api="reach",
                text="A activates B in vitro in a dose-dependent manner.")
        ],
    )
    hash1 = activation1.get_hash()
    activation2 = Activation(
        Agent("A", location="nucleus"),
        Agent("B"),
        evidence=[
            Evidence(source_api="reach", text="A activates B in vitro.")
        ],
    )
    hash2 = activation2.get_hash()
    activation3 = Activation(
        Agent("A"),
        Agent("B"),
        evidence=[Evidence(source_api="reach", text="A activates B.")],
    )
    hash3 = activation3.get_hash()

    # Sanity check
    assert hash1 != hash2 != hash3

    stmt_list = [(hash1, activation1), (hash2, activation2), (hash3, activation3)]

    # Dump the statements to a file
    test_statements_tsv_gz = Path(__file__).parent / "test_statements.tsv.gz"
    with gzip.open(test_statements_tsv_gz, "wt") as f:
        csv_writer = csv.writer(f, delimiter="\t")
        csv_writer.writerows(
            (sh, json.dumps(st.to_json())) for sh, st in stmt_list
        )

    source_counts = {
        hash1: {"reach": 1},
        hash2: {"reach": 1},
        hash3: {"reach": 1},
    }
    test_source_counts_pkl = Path(__file__).parent / "test_source_counts.pkl"
    with open(test_source_counts_pkl, "wb") as f:
        pickle.dump(source_counts, f)

    # Create support: activation1 -> activation2 -> activation3 in a
    # refinement graph
    refinements = {(hash1, hash2), (hash2, hash3)}
    refinement_graph = nx.DiGraph()
    refinement_graph.add_edges_from(refinements)
    assert nx.ancestors(refinement_graph, hash1) == set()
    assert nx.ancestors(refinement_graph, hash2) == {hash1}
    assert nx.ancestors(refinement_graph, hash3) == {hash1, hash2}

    # Run the belief calculation function
    db = get_db("primary")
    res = db.select_all(db.DBInfo)
    db_name_api_mapping = {r.db_name: r.source_api for r in res}
    test_belief_path = Path(__file__).parent / "test_belief_path.pkl"
    calculate_belief(
        refinements_graph=refinement_graph,
        num_batches=1,
        batch_size=len(stmt_list),
        source_mapping=db_name_api_mapping,
        unique_stmts_path=test_statements_tsv_gz,
        belief_scores_pkl_path=test_belief_path,
        source_counts_path=test_source_counts_pkl,
    )

    # Calculate the belief scores: Add evidence of supporting statements to the
    # evidence of the supported statement then calculate the prior belief
    belief_engine = BeliefEngine(refinements_graph=refinement_graph)
    to_calc_list = []
    local_beliefs = {}
    for st_hash, stmt in stmt_list:
        # Sum belief score of ancestors
        summed_src_count = Counter(source_counts[st_hash])

        if st_hash in refinement_graph.nodes:
            for anc_hash in nx.ancestors(refinement_graph, st_hash):
                summed_src_count += Counter(source_counts[anc_hash])

        ev_list_this_stmt = []
        for source, count in summed_src_count.items():
            for _ in range(count):
                ev_list_this_stmt.append(Evidence(source_api=source))

        stmt.evidence = ev_list_this_stmt
        to_calc_list.append((st_hash, stmt))

    hashes, stmts = zip(*to_calc_list)
    belief_engine.set_prior_probs(stmts)
    for st_hash2, stmt2 in zip(hashes, stmts):
        local_beliefs[st_hash2] = stmt2.belief

    # Load the belief scores
    with open(test_belief_path, "rb") as f:
        belief_dict = pickle.load(f)

    # Check that the belief scores are correct
    assert all(
        local_beliefs[st_hash] == belief_dict[st_hash]
        for st_hash in belief_dict
    )

    assert len(stmts[2].evidence) == 3
    assert all(ev.source_api == 'reach' for ev in stmts[2].evidence)
    assert belief_dict[hash3] == 0.923
