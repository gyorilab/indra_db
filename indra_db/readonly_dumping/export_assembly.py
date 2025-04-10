import argparse
import concurrent.futures
import csv
import ctypes
import gc
import gzip

import json
import logging

import pickle
import time
import shutil

from collections import defaultdict, Counter
from pathlib import Path
from typing import Tuple, Set, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas
import psutil
from tqdm import tqdm
import os

from adeft import get_available_models
from indra.belief import BeliefEngine
from indra.ontology.bio.sqlite_ontology import SqliteOntology
from indra.preassembler import Preassembler
from indra.statements import stmts_from_json, stmt_from_json, Statement, \
    Evidence
from indra.util import batch_iter
from indra.tools import assemble_corpus as ac

from indra_db.cli.knowledgebase import KnowledgebaseManager, local_update
from indra_db.readonly_dumping.locations import knowledgebase_source_data_fpath

from indra_db.readonly_dumping.util import clean_json_loads, \
    validate_statement_semantics, record_time, \
    download_knowledgebase_files_to_path
from indra_db.readonly_dumping.locations import *


refinement_cycles_fpath = TEMP_DIR.join(name="refinement_cycles.pkl")
batch_size = int(1e6)

StmtList = List[Statement]

logger = logging.getLogger("indra_db.readonly_dumping.export_assembly")
logger.setLevel(logging.DEBUG)
logger.propagate = False

file_handler = logging.FileHandler(pipeline_log_fpath.absolute().as_posix(), mode='a')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

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
for reader_name, versions in reader_versions.items():
    for reader_version in versions:
        version_to_reader[reader_version] = reader_name


def get_related_split(stmts1: StmtList, stmts2: StmtList, pa: Preassembler) -> Set[Tuple[int, int]]:
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
            stmts_this_type1 + stmts_this_type2,
            split_idx=len(stmts_this_type1) - 1
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
        # Columns in raw_statements.tsv.gz are:
        # raw_statement_id, db_info_id, reading_id, raw_json
        # Columns in reading_text_content_meta.tsv.gz:
        # reading_id, reader_version, text_content_id, text_ref_id, source, text_type
        # Columns in text_refs_principal.tsv.gz:
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
            f"Loading {drop_readings_fpath.as_posix()} and "
            f"{reading_to_text_ref_map_fpath.as_posix()} from cache"
        )
        with drop_readings_fpath.open("rb") as fh:
            drop_readings = pickle.load(fh)
        # Get mapping of reading_id to text_ref_id
        with reading_to_text_ref_map_fpath.open("rb") as fh:
            reading_id_to_text_ref_id = pickle.load(fh)
    return drop_readings, reading_id_to_text_ref_id


def run_kb_pipeline(
    refresh: bool,
    kb_mapping: Dict[Tuple[str, str], int]
) -> Dict[int, Path]:
    """Run the knowledgebase pipeline

    Parameters
    ----------
    refresh :
        If True, generate new statements for each knowledgebase manager and
        overwrite any local files. If False (default), the local files will
        be used if they exist.
    kb_mapping :
        A dictionary mapping source name and api name tuples to their unique
        db_info id

    Returns
    -------
    :
        A dictionary mapping db_info ids to the local file paths of the
        statements for that knowledgebase
    """
    selected_kbs = []
    kb_file_mapping = {}
    for M in KnowledgebaseManager.__subclasses__():
        m = M()
        db_id = kb_mapping.get((m.source, m.short_name))
        if db_id is None:
            raise ValueError(
                f"Could not find db_info id for {m.source} {m.short_name} "
                f"in the db_info table on the principal database. If this is a new "
                f"source, please add it to the db_info table in the principal "
                f"database."
            )
        selected_kbs.append(M)
        kb_file_mapping[db_id] = m.get_local_fpath()
    local_update(kb_manager_list=selected_kbs, refresh=refresh)

    return kb_file_mapping


def split_tsv_gz_file(input_path, output_dir, batch_size=10000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with gzip.open(input_path, "rt") as fh:
        stmts_reader = csv.reader(fh, delimiter="\t")
        for batch_index, batch in enumerate(batch_iter(stmts_reader, batch_size)):
            output_file_path = os.path.join(output_dir, f"split_{batch_index}.tsv.gz")
            with gzip.open(output_file_path, "wt") as output_file:
                writer = csv.writer(output_file, delimiter="\t")
                writer.writerows(batch)

def count_rows_in_tsv_gz(file_path):
    with gzip.open(file_path, 'rt') as file:
        reader = csv.reader(file)
        row_count = sum(1 for _ in reader)  # This counts every row in the CSV file
    return row_count

def preprocess(
        drop_readings: Set[int],
        reading_id_to_text_ref_id: Dict,
        kb_mapping: Dict[Tuple[str, str], int],
        drop_db_info_ids: Optional[Set[int]] = None,
):
    """Preassemble statements and collect source counts

    Parameters
    ----------
    drop_readings :
        A set of reading ids to drop
    reading_id_to_text_ref_id :
        A dictionary mapping reading ids to text ref ids
    kb_mapping :
        A dictionary mapping source name and api name tuples to their unique db_info id
    drop_db_info_ids :
        A set of db_info ids to drop
    """
    if (
            not processed_stmts_reading_fpath.exists() or
            not source_counts_reading_fpath.exists()
    ):
        logger.info("Preassembling statements, collecting source counts, "
                    "mapping from stmt hash to raw statement ids and mapping "
                    "from raw statement ids to db info and reading ids")
        db_info_id_name_map = {
            db_id: name for (src_api, name), db_id in kb_mapping.items()
        }
        text_refs = load_text_refs_by_trid(text_refs_fpath.as_posix())
        source_counts = defaultdict(Counter)
        stmt_hash_to_raw_stmt_ids = defaultdict(set)
        with gzip.open(raw_statements_fpath.as_posix(), "rt") as fh, \
             gzip.open(processed_stmts_reading_fpath.as_posix(), "wt") as fh_out, \
             gzip.open(raw_id_info_map_reading_fpath.as_posix(), "wt") as fh_info:
            raw_stmts_reader = csv.reader(fh, delimiter="\t")
            writer = csv.writer(fh_out, delimiter="\t")
            info_writer = csv.writer(fh_info, delimiter="\t")
            for batch_ix, lines in enumerate(
                    tqdm(batch_iter(raw_stmts_reader, 10000),
                    total=7727, desc="Looping raw statements")):

                paired_stmts_jsons = []
                info_rows = []
                for raw_stmt_id, db_info_id, reading_id, stmt_json_raw in lines:
                    raw_stmt_id_int = int(raw_stmt_id)
                    db_info_id_int = int(db_info_id) if db_info_id != "\\N" else None
                    refs = None

                    # Skip if this is for a dropped knowledgebase or reading
                    if drop_db_info_ids and db_info_id_int and \
                            db_info_id_int in drop_db_info_ids:
                        continue
                    if reading_id != "\\N":
                        int_reading_id = int(reading_id)
                        if int_reading_id in drop_readings:
                            continue
                        text_ref_id = reading_id_to_text_ref_id.get(int_reading_id)
                        if text_ref_id:
                            refs = text_refs.get(text_ref_id)

                    # Append to info rows
                    info_rows.append((raw_stmt_id_int, db_info_id_int or "\\N",
                                      int_reading_id, stmt_json_raw))
                    stmt_json = clean_json_loads(stmt_json_raw)
                    if refs:
                        stmt_json["evidence"][0]["text_refs"] = refs
                        if refs.get("PMID"):
                            stmt_json["evidence"][0]["pmid"] = refs["PMID"]
                    paired_stmts_jsons.append((raw_stmt_id_int, stmt_json, db_info_id_int))
                # Write to the info file
                # "raw_stmt_id_to_info_map_reading.tsv.gz"
                info_writer.writerows(info_rows)
                if paired_stmts_jsons:
                    raw_ids, stmts_jsons, db_info_ids = zip(*paired_stmts_jsons)
                stmts = stmts_from_json(stmts_jsons)

                # Use UUID mapping to keep track of the statements after
                # assemble corpus is called, since the statements can be
                # skipped or modified.
                stmt_uuid_map = {
                    st.uuid: (rid, dbiid)
                    for rid, st, dbiid in zip(raw_ids, stmts, db_info_ids)
                }

                # This part ultimately calls indra_db_lite or the principal db,
                # depending on which is available
                stmts = ac.fix_invalidities(stmts, in_place=True)

                stmts = ac.map_grounding(stmts)
                stmts = ac.map_sequence(stmts)
                for stmt in stmts:
                    # Get the statement hash and get the source counts
                    raw_id, dbi_id = stmt_uuid_map[stmt.uuid]
                    stmt_hash = stmt.get_hash(refresh=True)
                    stmt_hash_to_raw_stmt_ids[stmt_hash].add(raw_id)

                    if dbi_id:
                        # If this is a knowledgebase statement, source_name is
                        # given by the db_name field in the db_info table,
                        # here provided by db_info_id_name_map
                        source_name = db_info_id_name_map[dbi_id]
                    else:
                        # For readers, source_api == source name
                        source_name = stmt.evidence[0].source_api
                    source_counts[stmt_hash][source_name] += 1
                rows = [(stmt.get_hash(), json.dumps(stmt.to_json()))
                        for stmt in stmts]
                writer.writerows(rows)

        # Cast Counter to dict and pickle the source counts
        logger.info("Dumping source counts")
        source_counts = dict(source_counts)
        with source_counts_reading_fpath.open("wb") as fh:
            pickle.dump(source_counts, fh)
            print("Source count saved")

        # Cast defaultdict to dict and pickle the stmt hash to raw stmt ids
        logger.info("Dumping stmt hash to raw stmt ids")
        stmt_hash_to_raw_stmt_ids = dict(stmt_hash_to_raw_stmt_ids)
        with stmt_hash_to_raw_stmt_ids_reading_fpath.open("wb") as fh:
            pickle.dump(stmt_hash_to_raw_stmt_ids, fh)


def merge_processed_statements():
    """Merge processed statements from reading and knowledgebases"""

    # List the processed statement file to be merged
    proc_stmts_reading = processed_stmts_reading_fpath.absolute().as_posix()
    kb_folder_path = TEMP_DIR.join('knowledgebases').absolute().as_posix()
    all_files = [os.path.join(kb_folder_path, file)
                 for file in os.listdir(kb_folder_path)
                 if os.path.isfile(os.path.join(kb_folder_path, file))
                 and file.endswith('.tsv.gz')]
    all_files.append(proc_stmts_reading)

    if not processed_stmts_fpath.exists():
        logger.info("Merging processed statements")
        with gzip.open(processed_stmts_fpath.absolute().as_posix(), 'wb') as f_out:
                for file in all_files:
                    with gzip.open(file, 'rb') as f_in:
                        logger.info(f"Merging {file} into processed statements")
                        shutil.copyfileobj(f_in, f_out)
    else:
        logger.info(f"Processed statements already merged at "
                    f"{processed_stmts_fpath.absolute().as_posix()}, skipping...")

    # Merge source counts
    if not source_counts_fpath.exists():
        # Open the reading source counts
        with source_counts_reading_fpath.open("rb") as f:
            reading_source_counts = pickle.load(f)

        # Open the knowledgebase source counts
        with source_counts_knowledgebases_fpath.open("rb") as f:
            kb_source_counts = pickle.load(f)

        # Merge the KB source counts into the reading source counts
        for stmt_hash, kb_counts in kb_source_counts.items():
            counts = reading_source_counts.get(stmt_hash, Counter())
            # NOTE: This assumes `counts` is an instance of Counter
            counts.update(kb_counts)
            reading_source_counts[stmt_hash] = counts

        # Dump the merged source counts
        with source_counts_fpath.open("wb") as f:
            pickle.dump(reading_source_counts, f)


    # Merge the raw statement id to info map
    if not raw_id_info_map_fpath.exists():
        logger.info("Merging raw statement id to info mappings")
        with gzip.open(raw_id_info_map_fpath.absolute().as_posix(), 'wb') as f_out:
                for file in [
                    raw_id_info_map_knowledgebases_fpath.absolute().as_posix(),
                    raw_id_info_map_reading_fpath.absolute().as_posix()
                                ]:
                    with gzip.open(file, 'rb') as f_in:
                        logger.info(f"Merging {file} into processed statements")
                        shutil.copyfileobj(f_in, f_out)
        assert raw_id_info_map_fpath.exists()


    # Merge the stmt hash to raw stmt ids
    if not stmt_hash_to_raw_stmt_ids_fpath.exists():
        logger.info("Merging stmt hash to raw stmt ids")

        # Open the reading stmt hash to raw stmt ids
        with stmt_hash_to_raw_stmt_ids_reading_fpath.open("rb") as f:
            reading_stmt_hash_to_raw_stmt_ids = pickle.load(f)

        # Open the knowledgebase stmt hash to raw stmt ids
        with stmt_hash_to_raw_stmt_ids_knowledgebases_fpath.open("rb") as f:
            kb_stmt_hash_to_raw_stmt_ids = pickle.load(f)

        # Merge the KB stmt hash to raw stmt ids into the reading stmt hash
        # to raw stmt ids
        for stmt_hash, kb_raw_stmt_ids in kb_stmt_hash_to_raw_stmt_ids.items():
            raw_stmt_ids = reading_stmt_hash_to_raw_stmt_ids.get(stmt_hash, set())
            raw_stmt_ids |= kb_raw_stmt_ids
            reading_stmt_hash_to_raw_stmt_ids[stmt_hash] = raw_stmt_ids

        # Dump the merged stmt hash to raw stmt ids
        with stmt_hash_to_raw_stmt_ids_fpath.open("wb") as f:
            pickle.dump(reading_stmt_hash_to_raw_stmt_ids, f)


def deduplicate():
    # NOTE: As opposed to INDRA CoGEx we don't filter out statements
    # without db_refs for the readonly DB as we want to allow statements that
    # are valid but ungrounded
    # ~2.5-3 hours
    if not unique_stmts_fpath.exists():
        with gzip.open(processed_stmts_fpath, "rt") as fh, \
                gzip.open(unique_stmts_fpath, "wt") as fh_out_uniq:
            seen_hashes = set()
            reader = csv.reader(fh, delimiter="\t")
            writer_uniq = csv.writer(fh_out_uniq, delimiter="\t")
            for sh, stmt_json_str in tqdm(
                    reader, total=60405451, desc="Gathering unique statements"
            ):
                stmt = stmt_from_json(clean_json_loads(stmt_json_str))
                if not validate_statement_semantics(stmt):
                    continue

                if sh not in seen_hashes:
                    writer_uniq.writerow((sh, stmt_json_str))
                    seen_hashes.add(sh)
    else:
        logger.info(
            f"Unique statements already dumped at {unique_stmts_fpath.as_posix()}, "
            f"skipping..."
        )


def load_statements_from_file(file_path):
    stmts = []
    with gzip.open(file_path, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for _, sjs in reader:
            stmt = stmt_from_json(clean_json_loads(sjs, remove_evidence=True))
            stmts.append(stmt)
    return stmts

def calculate_belief(
    refinements_graph: nx.DiGraph,
    num_batches: int,
    batch_size: int,
    source_mapping: Dict[str, str],
    unique_stmts_path: Path = unique_stmts_fpath,
    belief_scores_pkl_path: Path = belief_scores_pkl_fpath,
    source_counts_path: Path = source_counts_fpath,

):
    """Calculate belief scores for unique statements from the refinement graph

    Parameters
    ----------
    refinements_graph :
        A directed graph where the edges point from more specific to less
        specific statements
    num_batches :
        The number of batches to process from the unique statements file
    batch_size :
        The number of statements to process in each batch
    source_mapping :
        A dictionary mapping source names to source api names
    unique_stmts_path :
        The path to the unique statements file
    source_counts_path :
        Pickle mapping ``stmt_hash → {source_name: count}``.
    belief_scores_pkl_path :
         ``{stmt_hash: belief}`` dict.
    """
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
    with source_counts_path.open("rb") as fh:
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
        for source_name, count in summed_source_counts.items():
            for _ in range(count):
                source_api = source_mapping.get(source_name, source_name)
                ev_list.append(Evidence(source_api=source_api))
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
    with belief_scores_pkl_path.open("wb") as fo:
        pickle.dump(belief_scores, fo)


def process_batch_pair(file):
    sqlite_ontology = SqliteOntology(
        db_path=sql_ontology_db_fpath.absolute().as_posix())
    sqlite_ontology.initialize()
    pa = Preassembler(sqlite_ontology)
    file1, file2 = file
    stmts1 = load_statements_from_file(file1)
    stmts2 = load_statements_from_file(file2)
    refinements = get_related_split(stmts1, stmts2, pa)
    logging.info("Processing batch pair: %s, %s", file1, file2)
    return refinements

def get_related(stmts: StmtList, pa: Preassembler) -> Set[Tuple[int, int]]:
    stmts_by_type = defaultdict(list)
    for stmt in stmts:
        stmts_by_type[stmt.__class__.__name__].append(stmt)
    refinements = set()
    for _, stmts_this_type in stmts_by_type.items():
        refinements |= pa._generate_relation_tuples(stmts_this_type)
    return refinements

def parallel_process_files(split_files, num_processes = 1):
    sqlite_ontology = SqliteOntology(
        db_path=sql_ontology_db_fpath.absolute().as_posix())
    sqlite_ontology.initialize()
    pa = Preassembler(sqlite_ontology)
    split_files = split_files[::-1]
    tasks = []
    refinements = set()
    num_files = len(split_files)
    for i in range(num_files):
        for j in range(i + 1, num_files):
            tasks.append((split_files[i], split_files[j]))
    logging.info("Completed all tasks")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_processes) as executor:
        results = list(tqdm(executor.map(process_batch_pair, tasks),
                            total=len(tasks)))

    for result in results:
        refinements |= result

    for i in range(num_files):
        refinements |= get_related(load_statements_from_file(split_files[i]),
                                   pa=pa)

    return refinements

def get_n_process():
    available_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    # about 12 Gb per each process
    max_processes_by_memory = int(available_memory_gb // 5)
    max_processes_by_cores = os.cpu_count()
    num_processes = min(max_processes_by_memory, max_processes_by_cores)
    # leave space for memory
    return num_processes

def get_refinement_graph(n_rows: int, split_files: list) -> nx.DiGraph:
    global cycles_found
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
        logger.info("6. Calculating refinements")

        n_process = get_n_process()
        logger.info(f"{n_process} processes starting")
        refinements = parallel_process_files(split_files,
                                             num_processes=n_process)

        # Write out the refinements as a gzipped TSV file
        with gzip.open(refinements_fpath.as_posix(), "wt") as f:
            tsv_writer = csv.writer(f, delimiter="\t")
            tsv_writer.writerows(refinements)
    else:
        logger.info(f"6. Loading refinements from existing file"
                    f"{refinements_fpath.as_posix()}")
        with gzip.open(refinements_fpath.as_posix(), "rt") as f:
            tsv_reader = csv.reader(f, delimiter="\t")

            # Each line is a refinement pair of two Statement hashes as ints
            refinements = {(int(h1), int(h2)) for h1, h2 in tsv_reader}

    #This can only check for full data
    # Perform sanity check on the refinements
    logger.info("Checking refinements")
    sqlite_ontology = SqliteOntology(
        db_path=sql_ontology_db_fpath.absolute().as_posix())
    sqlite_ontology.initialize()
    pa = Preassembler(sqlite_ontology)
    sample_stmts = sample_unique_stmts(n_rows=n_rows)
    sample_refinements = get_related([s for _, s in sample_stmts], pa=pa)
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

def release_memory():
    # Force garbage collection on Linux system
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass


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
    parser = argparse.ArgumentParser("Run the export and assembly pipeline")
    parser.add_argument("--refresh-kb", action="store_true",
                        help="If set, overwrite any existing local files "
                             "with new ones for the knowledgebase statements")
    args = parser.parse_args()
    logger.info(f"Root data path: {TEMP_DIR.base}")

    # 0. Dump raw data (raw statements, text content + reading, text refs)

    needed_files = [reading_text_content_fpath, text_refs_fpath,
                    raw_statements_fpath]

    if any(not f.exists() for f in needed_files):
        missing = [f.as_posix() for f in needed_files if not f.exists()]
        print(command_line_instructions)
        raise FileNotFoundError(
            f"{', '.join(missing)} missing, please run the dump commands above to get them."
        )

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

    # Get the mapping of knowledgebase sources to db_info ids, this is required to get
    # unique source counts
    res = db.select_all(db.DBInfo)
    db_info_mapping = {(r.source_api, r.db_name): r.id for r in res}

    # 1. Run knowledge base pipeline if the output files don't exist
    start_time = time.time()
    if not knowledgebase_source_data_fpath.exists():
        os.makedirs(knowledgebase_source_data_fpath.absolute().as_posix())

    logger.info("Downloading knowlegebase file sources")
    download_knowledgebase_files_to_path(
        knowledgebase_source_data_fpath.absolute().as_posix())

    logger.info("1. Running knowledgebase pipeline")
    kb_updates = run_kb_pipeline(refresh=args.refresh_kb, kb_mapping=db_info_mapping)

    end_time = time.time()
    record_time(export_benchmark.absolute().as_posix(),
                (end_time - start_time) / 3600,
                'Processing knowledgebase step', 'w')

    # Check if output from preassembly (step 2) already exists
    if (
            not processed_stmts_reading_fpath.exists() or
            not source_counts_reading_fpath.exists()
    ):
        # 2. Distill statements
        start_time = time.time()
        logger.info("2. Running statement distillation")
        readings_to_drop, reading_id_textref_id_map = distill_statements()
        # 3. Preassembly (needs indra_db_lite setup)
        logger.info("3. Running preprocess")
        preprocess(
            drop_readings=readings_to_drop,
            reading_id_to_text_ref_id=reading_id_textref_id_map,
            kb_mapping=db_info_mapping,
            drop_db_info_ids=set(kb_updates.keys()),
        )
        end_time = time.time()
        record_time(export_benchmark.absolute().as_posix(),
                    (end_time - start_time) / 3600,
                    'Distill/Preassembly step', 'a')
    else:
        logger.info(
            "Output from step 2 & 3 already exists, skipping to step 4..."
        )

    # 4. Merge the processed raw statements with the knowledgebase statements
    start_time = time.time()
    logger.info(
        "4. Merging processed knowledgebase statements with processed raw statements"
    )
    merge_processed_statements()
    end_time = time.time()
    record_time(export_benchmark.absolute().as_posix(),
                (end_time - start_time) / 3600,
                'Merge step', 'a')

    # 5. Ground and deduplicate statements (here don't discard any statements
    #    based on number of agents, as is done in cogex)
    logger.info("5. Running grounding and deduplication")
    start_time = time.time()
    deduplicate()
    end_time = time.time()
    record_time(export_benchmark.absolute().as_posix(),
                (end_time - start_time) / 3600,
                'Deduplicate step', 'a')
    release_memory()
