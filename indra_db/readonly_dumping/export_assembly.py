import csv
import gzip
import itertools
import json
import logging
import math
import pickle
from collections import defaultdict, Counter
from typing import Tuple, Set, Dict, List

import networkx as nx
import numpy as np
import pandas
from tqdm import tqdm

from adeft import get_available_models
from indra.belief import BeliefEngine
from indra.ontology.bio import bio_ontology
from indra.preassembler import Preassembler
from indra.statements import stmts_from_json, stmt_from_json, Statement, \
    Evidence
from indra.util import batch_iter
from indra.tools import assemble_corpus as ac

from indra_db.cli.knowledgebase import KnowledgebaseManager, local_update
from .util import clean_json_loads
from .locations import *


refinement_cycles_fpath = TEMP_DIR.joinpath("refinement_cycles.pkl")
batch_size = int(1e6)

StmtList = List[Statement]


logger = logging.getLogger("indra_db.readonly_dumping.export_assembly")


reader_versions = {
    "sparser": [
        "sept14-linux\n",
        "sept14-linux",
        "June2018-linux",
        "October2018-linux",
        "February2020-linux",
        "April2020-linux",
    ],
    "reach": [
        "61059a-biores-e9ee36",
        "1.3.3-61059a-biores-",
        "1.6.1",
        "1.6.3-e48717",
    ],
    "trips": ["STATIC", "2019Nov14", "2021Jan26"],
    "isi": ["20180503"],
    "eidos": ["0.2.3-SNAPSHOT", "1.7.1-SNAPSHOT"],
    "mti": ["1.0"],
}

version_to_reader = {}
for reader_name, reader_versions in reader_versions.items():
    for reader_version in reader_versions:
        version_to_reader[reader_version] = reader_name


def get_related(stmts: StmtList) -> Set[Tuple[int, int]]:
    global pa
    stmts_by_type = defaultdict(list)
    for stmt in stmts:
        stmts_by_type[stmt.__class__.__name__].append(stmt)
    refinements = set()
    for _, stmts_this_type in stmts_by_type.items():
        refinements |= pa._generate_relation_tuples(stmts_this_type)
    return refinements


def get_related_split(stmts1: StmtList, stmts2: StmtList) -> Set[Tuple[int, int]]:
    global pa
    stmts_by_type1 = defaultdict(list)
    stmts_by_type2 = defaultdict(list)
    for stmt in stmts1:
        stmts_by_type1[stmt.__class__.__name__].append(stmt)
    for stmt in stmts2:
        stmts_by_type2[stmt.__class__.__name__].append(stmt)
    refinements = set()
    for stmt_type, stmts_this_type1 in stmts_by_type1.items():
        stmts_this_type2 = stmts_by_type2.get(stmt_type)
        if not stmts_this_type2:
            continue
        refinements |= pa._generate_relation_tuples(
            stmts_this_type1 + stmts_this_type2, split_idx=len(stmts_this_type1) - 1
        )
    return refinements


def sample_unique_stmts(
    num: int = 100000, n_rows: int = None
) -> List[Tuple[int, Statement]]:
    """Return a random sample of Statements from unique_statements.tsv.gz

    Parameters
    ----------
    num :
        Number of Statements to return
    n_rows :
        The number of rows in the file. If not provided, the file is read in
        its entirety first to determine the number of rows.

    Returns
    -------
    :
        A list of tuples of the form (hash, Statement)
    """
    if n_rows is None:
        logger.info("Counting lines...")
        with gzip.open(unique_stmts_fpath.as_posix(), "rt") as f:
            reader = csv.reader(f, delimiter="\t")
            n_rows = sum(1 for _ in reader)

    # Generate a random sample of line indices
    logger.info(f"Sampling {num} unique statements from a total of {n_rows}")
    indices = np.random.choice(n_rows, num, replace=False)
    stmts = []
    t = tqdm(total=num, desc="Sampling statements")
    with gzip.open(unique_stmts_fpath, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for index, (sh, sjs) in enumerate(reader):
            if index in indices:
                stmts.append((int(sh), stmt_from_json(clean_json_loads(sjs))))
                t.update()
                if len(stmts) == num:
                    break

    t.close()
    return stmts


def load_text_refs_by_trid(fname: str) -> Dict:
    text_refs = {}
    for line in tqdm(
        gzip.open(fname, "rt", encoding="utf-8"),
        desc="Processing text refs into a lookup dictionary",
    ):
        ids = line.strip().split("\t")
        id_names = ["TRID", "PMID", "PMCID", "DOI", "PII", "URL", "MANUSCRIPT_ID"]
        d = {}
        for id_name, id_val in zip(id_names, ids):
            if id_val != "\\N":
                if id_name == "TRID":
                    id_val = int(id_val)
                d[id_name] = id_val
        text_refs[int(ids[0])] = d
    return text_refs


def reader_prioritize(reader_contents):
    drop = set()
    # We first organize the contents by source/text type
    versions_per_type = defaultdict(list)
    for reading_id, reader_version, source, text_type in reader_contents:
        versions_per_type[(source, text_type)].append((reader_version, reading_id))
    # Then, within each source/text_type key, we sort according to reader
    # version to be able to select the newest reader version for each key
    reading_id_per_type = {}
    for (source, text_type), versions in versions_per_type.items():
        if len(versions) > 1:
            sorted_versions = sorted(
                versions,
                key=lambda x: reader_versions[version_to_reader[x[0]]].index(x[0]),
                reverse=True,
            )
            drop |= {x[1] for x in sorted_versions[1:]}
            reading_id_per_type[(source, text_type)] = sorted_versions[0][1]
        else:
            reading_id_per_type[(source, text_type)] = versions[0][1]
    fulltexts = [
        content
        for content in reader_contents
        if content[3] == "fulltext" and content[0] not in drop
    ]
    not_fulltexts = [
        content
        for content in reader_contents
        if content[3] != "fulltext" and content[0] not in drop
    ]
    # There are 3 types of non-fulltext content: CORD-19 abstract, PubMed abstract
    # and PubMed title. If we have CORD-19, we prioritize it because it includes
    # the title so we drop any PubMed readings. Otherwise we don't drop anything
    # because we want to keep both the title and the abstract (which doesn't include
    # the title).
    if not fulltexts:
        if ("cord19_abstract", "abstract") in reading_id_per_type:
            if ("pubmed", "abstract") in reading_id_per_type:
                drop.add(reading_id_per_type[("pubmed", "abstract")])
            if ("pubmed", "title") in reading_id_per_type:
                drop.add(reading_id_per_type[("pubmed", "title")])
    # In case of fulltext, we can drop all non-fulltexts, and then drop
    # everything that is lower on the fulltext priority order
    else:
        priority = [
            "xdd-pubmed",
            "xdd",
            "xdd-biorxiv",
            "cord19_pdf",
            "elsevier",
            "cord19_pmc_xml",
            "manuscripts",
            "pmc_oa",
        ]
        drop |= {c[0] for c in not_fulltexts}
        sorted_fulltexts = sorted(
            fulltexts, key=lambda x: priority.index(x[2]), reverse=True
        )
        drop |= {c[0] for c in sorted_fulltexts[1:]}
    return drop


def distill_statements() -> Tuple[Set, Dict]:
    if not drop_readings_fpath.exists() or not reading_to_text_ref_map_fpath.exists():
        df = pandas.read_csv(
            reading_text_content_fpath,
            header=None,
            sep="\t",
            names=[
                "reading_id",
                "reader_version",
                "text_content_id",
                "text_ref_id",
                "text_content_source",
                "text_content_type",
            ],
        )
        df.sort_values("text_ref_id", inplace=True)

        drop_readings = set()
        trid = df["text_ref_id"].iloc[0]
        contents = defaultdict(list)

        # This takes around 1.5 hours
        for row in tqdm(df.itertuples(), total=len(df),
                        desc="Looping text content"):
            if row.text_ref_id != trid:
                for reader_name, reader_contents in contents.items():
                    if len(reader_contents) < 2:
                        continue
                    drop_new = reader_prioritize(reader_contents)
                    # A sanity check to make sure we don't drop all
                    # the readings
                    assert len(drop_new) < len(reader_contents)
                    drop_readings |= drop_new
                contents = defaultdict(list)
            contents[version_to_reader[row.reader_version]].append(
                (
                    row.reading_id,
                    row.reader_version,
                    row.text_content_source,
                    row.text_content_type,
                )
            )
            trid = row.text_ref_id

        with drop_readings_fpath.open("wb") as fh:
            logger.info(f"Dumping drop readings set to {drop_readings_fpath}")
            pickle.dump(drop_readings, fh)

        # Dump mapping of reading_id to text_ref_id
        reading_id_to_text_ref_id = dict(zip(df.reading_id, df.text_ref_id))
        with reading_to_text_ref_map_fpath.open("wb") as fh:
            logger.info(f"Dumping reading to text ref map to"
                        f"{reading_to_text_ref_map_fpath}")
            pickle.dump(reading_id_to_text_ref_id, fh)

    else:
        logger.info(
            f"Loading {drop_readings_fpath.as_posix()} and {reading_to_text_ref_map_fpath.as_posix()} from cache"
        )
        with drop_readings_fpath.open("rb") as fh:
            drop_readings = pickle.load(fh)
        # Get mapping of reading_id to text_ref_id
        with reading_to_text_ref_map_fpath.open("rb") as fh:
            reading_id_to_text_ref_id = pickle.load(fh)
    return drop_readings, reading_id_to_text_ref_id


def run_kb_pipeline(refresh: bool) -> Dict[int, Path]:
    """Run the knowledgebase pipeline

    Parameters
    ----------
    refresh :
        If True, generate new statements for each knowledgebase manager and
        overwrite any local files. If False (default), the local files will
        be used if they exist.

    Returns
    -------
    :
        A dictionary mapping db_info ids to the local file paths of the
        statements for that knowledgebase
    """
    from indra_db.util import get_db
    db = get_db('primary')

    res = db.select_all(db.DBInfo)
    kb_mapping = {(r.source_api, r.db_name): r.id for r in res}

    # Select all knowledgebase managers except HPRD: statements already
    # exist in db, the source data hasn't been updated since 2009 and the
    # server hosting the source data returns 500 errors when trying to
    # download it
    selected_kbs = []
    kb_file_mapping = {}
    for M in KnowledgebaseManager.__subclasses__():
        m = M()
        if m.short_name != "hprd":
            db_id = kb_mapping.get((m.source, m.short_name))
            if db_id is None:
                raise ValueError(
                    f"Could not find db_id for {m.source} {m.short_name} "
                    f"in the db_info table on the principal database. Please "
                    f"add it."
                )
            selected_kbs.append(M)
            kb_file_mapping[db_id] = m.get_local_fpath()

    local_update(kb_manager_list=selected_kbs, refresh=refresh)

    return kb_file_mapping


def preassembly(drop_readings: Set, reading_id_to_text_ref_id: Dict):
    if not processed_stmts_fpath.exists() or not source_counts_fpath.exists():
        logger.info("Preassembling statements and collecting source counts")
        text_refs = load_text_refs_by_trid(text_refs_fpath.as_posix())
        source_counts = defaultdict(lambda: defaultdict(int))
        stmt_hash_to_raw_stmt_ids = defaultdict(set)
        # Todo:
        #  - parallelize
        with gzip.open(raw_statements_fpath.as_posix(), "rt") as fh, \
                gzip.open(processed_stmts_fpath.as_posix(), "wt") as fh_out, \
                gzip.open(raw_id_info_map_fpath.as_posix(), "wt") as fh_info:
            raw_stmts_reader = csv.reader(fh, delimiter="\t")
            writer = csv.writer(fh_out, delimiter="\t")
            info_writer = csv.writer(fh_info, delimiter="\t")
            for lines in tqdm(batch_iter(raw_stmts_reader, 10000),
                              total=7536, desc="Looping raw statements"):
                paired_stmts_jsons = []
                info_rows = []
                for raw_stmt_id, db_info_id, reading_id, stmt_json_raw in lines:
                    raw_stmt_id_int = int(raw_stmt_id)
                    db_info_id = int(db_info_id) if db_info_id != "\\N" else None
                    refs = None
                    int_reading_id = None
                    if reading_id != "\\N":
                        # Skip if this is for a dropped reading
                        int_reading_id = int(reading_id)
                        if int_reading_id in drop_readings:
                            continue
                        text_ref_id = reading_id_to_text_ref_id.get(int_reading_id)
                        if text_ref_id:
                            refs = text_refs.get(text_ref_id)

                    # Append to info rows
                    info_rows.append((raw_stmt_id_int, db_info_id,
                                      int_reading_id, stmt_json_raw))
                    stmt_json = clean_json_loads(stmt_json_raw)
                    if refs:
                        stmt_json["evidence"][0]["text_refs"] = refs
                        if refs.get("PMID"):
                            stmt_json["evidence"][0]["pmid"] = refs["PMID"]
                    paired_stmts_jsons.append((raw_stmt_id_int, stmt_json))

                # Write to the info file
                info_writer.writerows(info_rows)

                raw_ids, stmts_jsons = zip(*paired_stmts_jsons)
                stmts = stmts_from_json(stmts_jsons)

                # This part ultimately calls indra_db_lite or the principal db,
                # depending on which is available
                stmts = ac.fix_invalidities(stmts, in_place=True)

                stmts = ac.map_grounding(stmts)
                stmts = ac.map_sequence(stmts)
                for raw_id, stmt in zip(raw_ids, stmts):
                    # Get the statement hash and get the source counts
                    stmt_hash = stmt.get_hash(refresh=True)
                    stmt_hash_to_raw_stmt_ids[stmt_hash].add(raw_id)
                    source_counts[stmt_hash][stmt.evidence[0].source_api] += 1
                rows = [(stmt.get_hash(), json.dumps(stmt.to_json())) for stmt in stmts]
                writer.writerows(rows)

        # Cast defaultdict to dict and pickle the source counts
        logger.info("Dumping source counts")
        source_counts = dict(source_counts)
        with source_counts_fpath.open("wb") as fh:
            pickle.dump(source_counts, fh)

        # Cast defaultdict to dict and pickle the stmt hash to raw stmt ids
        logger.info("Dumping stmt hash to raw stmt ids")
        stmt_hash_to_raw_stmt_ids = dict(stmt_hash_to_raw_stmt_ids)
        with stmt_hash_to_raw_stmt_ids_fpath.open("wb") as fh:
            pickle.dump(stmt_hash_to_raw_stmt_ids, fh)


def ground_deduplicate():
    # ~2.5-3 hours
    if not grounded_stmts_fpath.exists() or not unique_stmts_fpath.exists():
        with gzip.open(processed_stmts_fpath, "rt") as fh, gzip.open(
            grounded_stmts_fpath, "wt"
        ) as fh_out_gr, gzip.open(unique_stmts_fpath, "wt") as fh_out_uniq:
            seen_hashes = set()
            reader = csv.reader(fh, delimiter="\t")
            writer_gr = csv.writer(fh_out_gr, delimiter="\t")
            writer_uniq = csv.writer(fh_out_uniq, delimiter="\t")
            for sh, stmt_json_str in tqdm(
                reader, total=60405451, desc="Gathering grounded and unique statements"
            ):
                stmt = stmt_from_json(clean_json_loads(stmt_json_str))
                if all(
                    (set(agent.db_refs) - {"TEXT", "TEXT_NORM"})
                    for agent in stmt.real_agent_list()
                ):
                    writer_gr.writerow((sh, stmt_json_str))
                    if sh not in seen_hashes:
                        writer_uniq.writerow((sh, stmt_json_str))
                seen_hashes.add(sh)
    else:
        logger.info(
            f"Grounded and unique statements already dumped at "
            f"{grounded_stmts_fpath.as_posix()} and "
            f"{unique_stmts_fpath.as_posix()}, skipping..."
        )


def get_refinement_graph(batch_size: int, num_batches: int) -> nx.DiGraph:
    global cycles_found, pa
    """Get refinement pairs as: (more specific, less specific)

    The evidence from the more specific statement is included in the less
    specific statement

    Step 1, alternative:

    Open two CSV readers for the unique_statements.tsv.gz and then move them
    forward in batches to cover all combinations of Statement batches.

    - First batch of Stmts, internal refinement finding
    - First batch (first reader) x Second batch (second reader)
    - First batch (first reader) x Third batch (second reader)
    - ...
    - One before last batch (first reader) x Last batch (second reader)
    ---> Giant list of refinement relation pairs (hash1, hash2)

    Put the pairs in a networkx DiGraph
    """
    # Loop statements: the outer index runs all batches while the inner index
    # runs outer index < inner index <= num_batches. This way the outer
    # index runs the "diagonal" of the combinations while the inner index runs
    # the upper triangle of the combinations.

    # Open two csv readers to the same file
    if not refinements_fpath.exists():
        logger.info("Calculating refinements")
        refinements = set()
        # This takes ~9-10 hours to run
        with gzip.open(unique_stmts_fpath, "rt") as fh1:
            reader1 = csv.reader(fh1, delimiter="\t")
            for outer_batch_ix in tqdm(
                    range(num_batches), total=num_batches,
                    desc="Calculating refinements"
            ):
                # read in a batch from the first reader
                stmts1 = []
                for _ in range(batch_size):
                    try:
                        _, sjs = next(reader1)
                        stmt = stmt_from_json(
                            clean_json_loads(sjs, remove_evidence=True)
                        )
                        stmts1.append(stmt)
                    except StopIteration:
                        break

                # Get refinements for the i-th batch with itself
                refinements |= get_related(stmts1)

                # Loop batches from second reader, starting at outer_batch_ix + 1
                with gzip.open(unique_stmts_fpath, "rt") as fh2:
                    reader2 = csv.reader(fh2, delimiter="\t")
                    batch_iterator = batch_iter(reader2, batch_size=batch_size)
                    # Note: first argument is the start index, second is
                    # the stop index, but if None is used, it will iterate
                    # until possible
                    batch_iterator = itertools.islice(
                        batch_iterator, outer_batch_ix + 1, None
                    )

                    # Loop the batches
                    for inner_batch_idx, batch in tqdm(
                            enumerate(batch_iterator),
                            total=num_batches-outer_batch_ix-1,
                            leave=False
                    ):
                        stmts2 = []

                        # Loop the statements in the batch
                        for _, sjs in batch:
                            try:
                                stmt = stmt_from_json(
                                    clean_json_loads(sjs, remove_evidence=True)
                                )
                                stmts2.append(stmt)
                            except StopIteration:
                                break

                        # Get refinements for the i-th batch with the j-th batch
                        refinements |= get_related_split(stmts1, stmts2)

        # Write out the refinements as a gzipped TSV file
        with gzip.open(refinements_fpath.as_posix(), "wt") as f:
            tsv_writer = csv.writer(f, delimiter="\t")
            tsv_writer.writerows(refinements)
    else:
        logger.info(f"Loading refinements from existing file"
                    f"{refinements_fpath.as_posix()}")
        with gzip.open(refinements_fpath.as_posix(), "rt") as f:
            tsv_reader = csv.reader(f, delimiter="\t")

            # Each line is a refinement pair of two Statement hashes as ints
            refinements = {(int(h1), int(h2)) for h1, h2 in tsv_reader}

    # Perform sanity check on the refinements
    logger.info("Checking refinements")
    sample_stmts = sample_unique_stmts(n_rows=num_rows)
    sample_refinements = get_related([s for _, s in sample_stmts])
    assert sample_refinements.issubset(refinements), (
        f"Refinements are not a subset of the sample. Sample contains "
        f"{len(sample_refinements - refinements)} refinements not in "
        f"the full set."
    )

    logger.info("Checking refinements for cycles")
    ref_graph = nx.DiGraph()
    ref_graph.add_edges_from(refinements)
    try:
        cycles = nx.find_cycle(ref_graph)
        cycles_found = True
    except nx.NetworkXNoCycle:
        logger.info("No cycles found in the refinements")
        cycles = None
        cycles_found = False

    # If cycles are found, save them to a file for later inspection
    if cycles_found and cycles is not None:
        logger.warning(f"Found cycles in the refinement graph. Dumping to "
                       f"{refinement_cycles_fpath.as_posix()}")
        with refinement_cycles_fpath.open("wb") as f:
            pickle.dump(obj=cycles, file=f)
        cycles_found = True

    return ref_graph


def calculate_belief(
    refinements_graph: nx.DiGraph,
    num_batches: int,
    batch_size: int,
    unique_stmts_path: Path = unique_stmts_fpath,
):
    # The refinement set is a set of pairs of hashes, with the *first hash
    # being more specific than the second hash*, i.e. the evidence for the
    # first should be included in the evidence for the second
    #
    # The BeliefEngine expects the refinement graph to be a directed graph,
    # with edges pointing from *more specific* to *less specific* statements
    # (see docstring of indra.belief.BeliefEngine)
    #
    # => The edges represented by the refinement set are the *same* as the
    # edges expected by the BeliefEngine.

    # Initialize a belief engine
    logger.info("Initializing belief engine")
    be = BeliefEngine(refinements_graph=refinements_graph)

    # Load the source counts
    logger.info("Loading source counts")
    with source_counts_fpath.open("rb") as fh:
        source_counts = pickle.load(fh)

    # Store hash: belief score
    belief_scores = {}

    def _get_support_evidence_for_stmt(stmt_hash: int) -> List[Evidence]:
        # Find all the statements that refine the current
        # statement, i.e. all the statements that are more
        # specific than the current statement => look for ancestors
        # then add up all the source counts for the statement
        # itself and the statements that refine it
        summed_source_counts = Counter(source_counts[stmt_hash])

        # If there are refinements, add them to the source counts
        if stmt_hash in refinements_graph.nodes():
            refiner_hashes = nx.ancestors(G=refinements_graph,
                                          source=stmt_hash)
            for refiner_hash in refiner_hashes:
                summed_source_counts += Counter(source_counts[refiner_hash])

        # Mock evidence - todo: add annotations?
        # Add evidence objects for each source's count and each source
        ev_list = []
        for source, count in summed_source_counts.items():
            for _ in range(count):
                ev_list.append(Evidence(source_api=source))
        return ev_list

    def _add_belief_scores_for_batch(batch: List[Tuple[int, Statement]]):
        # Belief calculation for this batch
        hashes, stmt_list = zip(*batch)
        be.set_prior_probs(statements=stmt_list)
        for sh, st in zip(hashes, stmt_list):
            belief_scores[sh] = st.belief

    # Iterate over each unique statement
    with gzip.open(unique_stmts_path.as_posix(), "rt") as fh:
        reader = csv.reader(fh, delimiter="\t")

        for bn in tqdm(range(num_batches), desc="Calculating belief"):
            stmt_batch = []
            for _ in tqdm(range(batch_size), leave=False, desc=f"Batch {bn}"):
                try:
                    stmt_hash_string, stmt_json_string = next(reader)
                    stmt = stmt_from_json(
                        clean_json_loads(stmt_json_string, remove_evidence=True)
                    )
                    this_hash = int(stmt_hash_string)
                    stmt.evidence = _get_support_evidence_for_stmt(this_hash)
                    stmt_batch.append((this_hash, stmt))

                except StopIteration:
                    break

            _add_belief_scores_for_batch(stmt_batch)

    # Dump the belief scores
    with belief_scores_pkl_fpath.open("wb") as fo:
        pickle.dump(belief_scores, fo)


if __name__ == '__main__':
    command_line_instructions = """
    NOTE: it is essential that the file dumps are synced, i.e. run 
    immediately after each other, otherwise there is a risk that evidence 
    and publication data are missing from new statements picked up in the 
    time between dumps. Additionally, to avoid the risk of having 
    statements with missing reading data, the raw statements dump should be 
    run *first*, followed by the other two dumps:

    Raw statements

      psql -d indradb_test -h indradb-refresh.cwcetxbvbgrf.us-east-1.rds.amazonaws.com 
      -U tester -c "COPY (SELECT id, db_info_id, reading_id, convert_from(json::bytea, 'utf-8') FROM public.raw_statements) 
      TO STDOUT" | gzip > raw_statements.tsv.gz

    Time estimate: ~30-40 mins

    Text content joined with reading

      psql -d indradb_test -h indradb-refresh.cwcetxbvbgrf.us-east-1.rds.amazonaws.com
      -U tester -c "COPY (SELECT rd.id, rd.reader_version, tc.id, tc.text_ref_id, 
      tc.source, tc.text_type FROM public.text_content as tc, public.reading as rd
      WHERE tc.id = rd.text_content_id) TO STDOUT" | gzip > reading_text_content_meta.tsv.gz

    Time estimate: ~15 mins

    Text refs

      psql -d indradb_test -h indradb-refresh.cwcetxbvbgrf.us-east-1.rds.amazonaws.com
      -U tester -c "COPY (SELECT id, pmid, pmcid, doi, pii, url, manuscript_id
      FROM public.text_ref) TO STDOUT" | gzip > text_refs_principal.tsv.gz

    Time estimate: ~2.5 mins
    """
    logger.info(f"Root data path: {TEMP_DIR}")

    # 0. Dump raw data (raw statements, text content + reading, text refs)
    needed_files = [reading_text_content_fpath, text_refs_fpath,
                    raw_statements_fpath]

    if any(not f.exists() for f in needed_files):
        missing = [f.as_posix() for f in needed_files if not f.exists()]
        print(command_line_instructions)
        raise FileNotFoundError(
            f"{', '.join(missing)} missing, please run the dump commands above to get them."
        )

    import os
    if not os.environ.get("INDRA_DB_LITE_LOCATION"):
        raise ValueError("Environment variable 'INDRA_DB_LITE_LOCATION' not set")

    # The principal db is needed for the preassembly step as fallback when
    # the indra_db_lite is missing content
    from indra_db import get_db
    db = get_db("primary")
    if db is None:
        raise ValueError("Could not connect to principal db")

    if len(get_available_models()) == 0:
        raise ValueError(
            "No adeft models detected, run 'python -m adeft.download' to download models"
        )

    # Check if output from preassembly (step 2) already exists
    if not processed_stmts_fpath.exists() or not source_counts_fpath.exists():
        # 1. Distill statements
        logger.info("1. Running statement distillation")
        readings_to_drop, reading_id_textref_id_map = distill_statements()

        # 2. Preassembly (needs indra_db_lite setup)
        logger.info("2. Running preassembly")
        preassembly(drop_readings=readings_to_drop, reading_id_to_text_ref_id=reading_id_textref_id_map)
    else:
        logger.info("Output from step 2 already exists, skipping to step 3...")

    # 3. Ground and deduplicate statements (here don't discard any statements
    #    based on number of agents, as is done in cogex)
    logger.info("3. Running grounding and deduplication")
    ground_deduplicate()

    # Setup bio ontololgy for preassembler
    if not refinements_fpath.exists() or not belief_scores_pkl_fpath.exists():
        logger.info("4. Running setup for refinement calculation")
        bio_ontology.initialize()
        bio_ontology._build_transitive_closure()
        pa = Preassembler(bio_ontology)

        # Count lines in unique statements file (needed to run
        # refinement calc and belief calc)
        logger.info(f"Counting lines in {unique_stmts_fpath.as_posix()}")
        with gzip.open(unique_stmts_fpath.as_posix(), "rt") as fh:
            csv_reader = csv.reader(fh, delimiter="\t")
            num_rows = sum(1 for _ in csv_reader)

        batch_count = math.ceil(num_rows / batch_size)

        # 4. Calculate refinement graph:
        cycles_found = False
        ref_graph = get_refinement_graph(batch_size=batch_size,
                                         num_batches=batch_count)
        if cycles_found:
            logger.info(
                f"Refinement graph stored in variable 'ref_graph', "
                f"edges saved to {refinements_fpath.as_posix()}"
                f"and cycles saved to {refinement_cycles_fpath.as_posix()}"
            )

        else:
            # 5. Get belief scores, if there were no refinement cycles
            logger.info("5. Calculating belief")
            calculate_belief(
                ref_graph, num_batches=batch_count, batch_size=batch_size
            )
    else:
        logger.info("Final output already exists, stopping script")
