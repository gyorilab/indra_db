import argparse
import csv
import gzip
import json
import logging
import pickle
import re
import subprocess
import uuid
from collections import defaultdict, Counter
from hashlib import md5
from textwrap import dedent
from typing import Tuple, Iterable

import requests
from bs4 import BeautifulSoup
from lxml import etree
from sqlalchemy import create_engine
from tqdm import tqdm

from indra.literature.pubmed_client import _get_annotations
from indra.statements import stmt_from_json, ActiveForm
from indra_db.config import get_databases
from indra_db.databases import ReadonlyDatabaseManager
from indra_db.schemas.mixins import ReadonlyTable
from schemas.readonly_schema import ro_type_map, ro_role_map, SOURCE_GROUPS

from .locations import *
from .util import load_statement_json

logger = logging.getLogger(__name__)

LOCAL_RO_PASSWORD = os.environ["LOCAL_RO_PASSWORD"]
LOCAL_RO_USER = os.environ["LOCAL_RO_USER"]
LOCAL_RO_PORT = int(os.environ.get("LOCAL_RO_PORT", "5432"))
LOCAL_RO_DB_NAME = os.environ.get("LOCAL_RO_DB_NAME", "indradb_readonly_local")


def table_has_content(
    ro_mngr_local: ReadonlyDatabaseManager,
    table_name: str,
    count: int = 0,
) -> bool:
    """Check that the table is not empty

    Parameters
    ----------
    ro_mngr_local :
        The local readonly database manager
    table_name :
        The name of the table to check
    count :
        The number of rows above which the table is considered to have content.
        Defaults to 0.

    Returns
    -------
    :
        True if the table has content, False otherwise
    """
    res = ro_mngr_local.engine.execute(
        f"SELECT COUNT(*) FROM readonly.{table_name}"
    )
    return res.fetchone()[0] > count


def get_stmt_hash_mesh_pmid_counts():
    # Check the existence of the output files
    if not all(fp.exists() for fp in [pmid_mesh_term_counts_fpath,
                                      mk_hash_pmid_sets_fpath,
                                      mesh_term_ref_counts_fpath]):

        # Check the existence of the input files
        if not pmid_mesh_map_fpath.exists() and pmid_stmt_hash_fpath.exists():
            ensure_pubmed_mesh_data()

        # dump pmid-mesh from mti_ref_annotations_test for concept and term
        # separately
        mti_mesh_query = dedent("""
        SELECT pmid_num, mesh_num, is_concept 
        FROM mti_ref_annotations_test""")
        principal_query_to_csv(
            query=mti_mesh_query, output_location=pmid_mesh_mti_fpath
        )

        # Load the pmid-mesh map
        pmid_mesh_mapping = pickle.load(pmid_mesh_map_fpath.open("rb"))

        # Load the pmid-mesh map from the mti_ref_annotations_test table and
        # merge it with the existing map
        with pmid_mesh_mti_fpath.open("r") as fh_in:
            reader = csv.reader(fh_in, delimiter="\t")
            for pmid, mesh_num, is_concept in reader:
                _type = "concept" if is_concept == "t" else "term"
                if pmid in pmid_mesh_mapping:
                    if _type in pmid_mesh_mapping[pmid]:
                        pmid_mesh_mapping[pmid][_type].add(mesh_num)
                    else:
                        pmid_mesh_mapping[pmid][_type] = {mesh_num}
                else:
                    pmid_mesh_mapping[pmid] = {_type: {mesh_num}}

        # Load the pmid-stmt_hash map
        pmid_stmt_hashes = pickle.load(pmid_stmt_hash_fpath.open("rb"))

        # Dicts for ref_count (count of PMIDs per mk_hash, mesh_num pair)
        pmid_mesh_concept_counts = Counter()
        pmid_mesh_term_counts = Counter()

        # Dict for hash -> pmid set (distinct pmids associated with a hash,
        # i.e. from multiple sources)
        mk_hash_pmid_sets = defaultdict(set)

        # Get pmid_count and ref_count (for concept and term separately)
        for pmid_num, mk_hash_set in tqdm(pmid_stmt_hashes.items(),
                                          desc="Getting counts"):
            for mk_hash in mk_hash_set:
                mk_hash_pmid_sets[mk_hash].add(pmid_num)
                if pmid_num in pmid_mesh_mapping:
                    # For concepts
                    mesh_concepts = pmid_mesh_mapping[pmid_num].get("concept", set())
                    for mesh_concept in mesh_concepts:
                        pmid_mesh_concept_counts[(mk_hash, mesh_concept)] += 1
                    # For terms
                    mesh_terms = pmid_mesh_mapping[pmid_num].get("term", set())
                    for mesh_term in mesh_terms:
                        pmid_mesh_term_counts[(mk_hash, mesh_term)] += 1

        # Cache the results
        pickle.dump(pmid_mesh_concept_counts,
                    pmid_mesh_concept_counts_fpath.open("wb"))
        pickle.dump(pmid_mesh_term_counts,
                    pmid_mesh_term_counts_fpath.open("wb"))
        pickle.dump(mk_hash_pmid_sets, mk_hash_pmid_sets_fpath.open("wb"))
    else:
        # Load the cached results
        pmid_mesh_concept_counts = pickle.load(
            pmid_mesh_concept_counts_fpath.open("rb")
        )
        pmid_mesh_term_counts = pickle.load(
            pmid_mesh_term_counts_fpath.open("rb")
        )
        mk_hash_pmid_sets = pickle.load(mk_hash_pmid_sets_fpath.open("rb"))

    return pmid_mesh_concept_counts, pmid_mesh_term_counts, mk_hash_pmid_sets


# MeshTerm/ConceptRefCounts
# These two tables enumerate stmt_hash, mesh_num, ref_count, pmid_count
# for each stmt_hash, mesh_num pair.
# - The **ref_count** is the number of distinct PMIDs that support the
#   stmt_hash, mesh_num pair.
# - The **pmid_count** is the number of distinct PMIDs that support the
#   stmt_hash.
# The PaRefLink table is not generated, but the equivalent contents is
# generated here to produce the input to the MeshTerm/ConceptRefCounts tables.
def ensure_pa_ref_link():
    # Check the existence of the output files
    if mesh_concept_ref_counts_fpath.exists() and \
            mesh_term_ref_counts_fpath.exists():
        return
    pmid_mesh_concept_counts, pmid_mesh_term_counts, mk_hash_pmid_sets = \
        get_stmt_hash_mesh_pmid_counts()

    # Dump mk_hash, mesh_num, ref_count, pmid_count for concepts and terms
    # to separate tsv files
    with mesh_term_ref_counts_fpath.open("w") as terms_fh:
        terms_writer = csv.writer(terms_fh, delimiter="\t")

        for (mk_hash, mesh_num), ref_count in tqdm(
                pmid_mesh_term_counts.items(), desc="MeshTermRefCounts"
        ):
            assert ref_count > 0  # Sanity check
            pmid_set = mk_hash_pmid_sets.get(mk_hash, set())
            if len(pmid_set) > 0:
                terms_writer.writerow(
                    (mk_hash, mesh_num, ref_count, len(pmid_set))
                )

    with mesh_concept_ref_counts_fpath.open("w") as concepts_fh:
        concepts_writer = csv.writer(concepts_fh, delimiter="\t")
        for (mk_hash, mesh_num), ref_count in tqdm(
                pmid_mesh_concept_counts.items(),
                desc="MeshConceptRefCounts",
        ):
            assert ref_count > 0  # Sanity check
            pmid_set = mk_hash_pmid_sets.get(mk_hash, set())
            if len(pmid_set) > 0:
                concepts_writer.writerow(
                    (mk_hash, mesh_num, ref_count, len(pmid_set))
                )


# MeshTermRefCounts
def mesh_term_ref_counts(local_ro_mngr: ReadonlyDatabaseManager):
    # Create the source tsv file
    ensure_pa_ref_link()

    # Load the tsv file into the local readonly db
    load_data_file_into_local_ro(
        table_name="readonly.mesh_term_ref_counts",
        column_order="mk_hash, mesh_num, ref_count, pmid_count",
        tsv_file=mesh_term_ref_counts_fpath.absolute().as_posix(),
    )

    # Build index
    mesh_term_ref_counts_table: ReadonlyTable = local_ro_mngr.tables[
        "mesh_term_ref_counts"]
    logger.info(
        f"Building index on {mesh_term_ref_counts_table.full_name()}"
    )
    mesh_term_ref_counts_table.build_indices(local_ro_mngr)


# MeshConceptRefCounts
def mesh_concept_ref_counts(local_ro_mngr: ReadonlyDatabaseManager):
    # Create the source tsv file
    ensure_pa_ref_link()

    # Load the tsv file into the local readonly db
    load_data_file_into_local_ro(
        table_name="readonly.mesh_concept_ref_counts",
        column_order="mk_hash, mesh_num, ref_count, pmid_count",
        tsv_file=mesh_concept_ref_counts_fpath.absolute().as_posix(),
    )

    # Build index
    mesh_concept_ref_counts_table: ReadonlyTable = local_ro_mngr.tables[
        "mesh_concept_ref_counts"]
    logger.info(
        f"Building index on {mesh_concept_ref_counts_table.full_name()}"
    )
    mesh_concept_ref_counts_table.build_indices(local_ro_mngr)


# Belief
def belief(local_ro_mngr: ReadonlyDatabaseManager):
    """Dump belief scores into the belief table on the local readonly db

    depends on: raw_statements, text_content, reading
    requires assembly: True
    assembly process: (see indra_cogex.sources.indra_db.raw_export)
    """
    logger.info("Reading belief score pickle file")
    with belief_scores_pkl_fpath.open("rb") as pkl_in:
        belief_dict = pickle.load(pkl_in)

    logger.info("Dumping belief scores to tsv file")
    with belief_scores_tsv_fpath.open("w") as fh_out:
        writer = csv.writer(fh_out, delimiter="\t")
        writer.writerows(((sh, bs) for sh, bs in belief_dict.items()))

    load_data_file_into_local_ro(table_name="readonly.belief",
                                 column_order="mk_hash, belief",
                                 tsv_file=belief_scores_tsv_fpath.absolute().as_posix())

    logger.info(f"Deleting {belief_scores_tsv_fpath.absolute().as_posix()}")
    os.remove(belief_scores_tsv_fpath.absolute().as_posix())

    # Build index
    belief_table: ReadonlyTable = local_ro_mngr.tables["belief"]
    logger.info(f"Building index on {belief_table.full_name()}")
    belief_table.build_indices(local_ro_mngr)


# RawStmtSrc
def raw_stmt_src(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the raw statement source table with data

    Depends on: raw_statements, text_content, reading
    Requires assembly: False

    Original SQL query to get data:

        SELECT raw_statements.id AS sid, lower(reading.reader) AS src
        FROM raw_statements, reading
        WHERE reading.id = raw_statements.reading_id
        UNION
        SELECT raw_statements.id AS sid,
        lower(db_info.db_name) AS src
        FROM raw_statements, db_info
        WHERE db_info.id = raw_statements.db_info_id
    """
    dump_file = raw_stmt_source_tsv_fpath
    principal_dump_sql = dedent(
        """SELECT raw_statements.id AS sid, lower(reading.reader) AS src 
           FROM raw_statements, reading 
           WHERE reading.id = raw_statements.reading_id 
           UNION 
           SELECT raw_statements.id AS sid, lower(db_info.db_name) AS src 
           FROM raw_statements, db_info 
           WHERE db_info.id = raw_statements.db_info_id"""
    )
    columns = "sid, src"

    # Dump the query output
    principal_query_to_csv(query=principal_dump_sql,
                           output_location=dump_file)

    load_data_file_into_local_ro(table_name="readonly.raw_stmt_src",
                                 column_order=columns,
                                 tsv_file=dump_file.absolute().as_posix())

    # Delete the dump file
    logger.info(f"Deleting {dump_file.absolute().as_posix()}")
    os.remove(dump_file.absolute().as_posix())

    # Build the index
    raw_stmt_src_table: ReadonlyTable = local_ro_mngr.tables["raw_stmt_src"]
    logger.info(f"Building index on {raw_stmt_src_table.full_name()}")
    raw_stmt_src_table.build_indices(local_ro_mngr)


# PaAgentCounts
def pa_agent_counts(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the pa_agent_counts table with data"""
    # Load cache
    pa_hash_act_type_ag_count = get_activity_type_ag_count()

    # Loop entries in pa_hash count cache and write to tsv
    with pa_agents_counts_tsv.open("w") as fh_out:
        writer = csv.writer(fh_out, delimiter="\t")
        writer.writerows(
            (
                (mk_hash, agent_count)
                for mk_hash, (_, _, _, agent_count)
                in (pa_hash_act_type_ag_count.items())
            )
        )

    # Load tsv into local readonly db
    load_data_file_into_local_ro(
        table_name="readonly.pa_agent_counts",
        column_order="pa_hash, agent_count",
        tsv_file=pa_agents_counts_tsv.absolute().as_posix()
    )

    # Build index
    pa_agent_counts_table: ReadonlyTable = local_ro_mngr.tables["pa_agent_counts"]
    logger.info(f"Building index on {pa_agent_counts_table.full_name()}")
    pa_agent_counts_table.build_indices(local_ro_mngr)


# ReadingRefLink
def reading_ref_link(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the reading ref link table with data

    depends on: text_ref, text_content, reading
    requires assembly: False

    Original SQL query to get data
        SELECT pmid, pmid_num, pmcid, pmcid_num,
               pmcid_version, doi, doi_ns, doi_id,
               tr.id AS trid, pii, url, manuscript_id,
               tc.id AS tcid, source, r.id AS rid, reader
        FROM text_ref AS tr
        JOIN text_content AS tc
            ON tr.id = tc.text_ref_id
        JOIN reading AS r
            ON tc.id = r.text_content_id
    """
    # Create a temp file to put the query in
    dump_file = reading_ref_link_tsv_fpath

    principal_dump_sql = dedent(
        """SELECT pmid, pmid_num, pmcid, pmcid_num,
                  pmcid_version, doi, doi_ns, doi_id,
                  tr.id AS trid, pii, url, manuscript_id,
                  tc.id AS tcid, source, r.id AS rid, reader
        FROM text_ref AS tr
        JOIN text_content AS tc
            ON tr.id = tc.text_ref_id
        JOIN reading AS r
            ON tc.id = r.text_content_id""")
    column_order = (
        "pmid, pmid_num, pmcid, pmcid_num, pmcid_version, doi, doi_ns, "
        "doi_id, trid, pii, url, manuscript_id, tcid, source, rid, reader"
    )

    # Dump the query from principal
    principal_query_to_csv(query=principal_dump_sql, output_location=dump_file)

    # todo: if you want to switch to gzipped files you can do
    #  "copy table_name from program 'zcat /tmp/tp.csv.gz';"
    load_data_file_into_local_ro(table_name="readonly.reading_ref_link",
                                 column_order=column_order,
                                 tsv_file=dump_file.absolute().as_posix())

    # Delete the dump file
    logger.info(f"Deleting {dump_file.absolute().as_posix()}")
    os.remove(dump_file)

    # Build the index
    reading_ref_link_table: ReadonlyTable = local_ro_mngr.tables["reading_ref_link"]
    logger.info(f"Building index for table {reading_ref_link_table.full_name()}")
    reading_ref_link_table.build_indices(local_ro_mngr)


# EvidenceCounts
def evidence_counts(local_ro_mngr: ReadonlyDatabaseManager):
    # Basically just upload the source counts file as a table

    if not source_counts_fpath.exists():
        raise ValueError(f"Surce counts {source_counts_fpath} does not exist")

    source_counts = pickle.load(source_counts_fpath.open("rb"))

    with evidence_counts_tsv.open("w") as ev_counts_f:
        writer = csv.writer(ev_counts_f, delimiter="\t")

        for mk_hash, src_counts in tqdm(source_counts.items(),
                                        desc="EvidenceCounts"):
            ev_count = sum(src_counts.values())
            writer.writerow([mk_hash, ev_count])

    load_data_file_into_local_ro(
        table_name="readonly.evidence_counts",
        column_order="mk_hash, ev_count",
        tsv_file=evidence_counts_tsv.absolute().as_posix()
    )

    # Build the index
    evidence_counts_table: ReadonlyTable = local_ro_mngr.tables["evidence_counts"]
    logger.info(f"Building index for table {evidence_counts_table.full_name()}")
    evidence_counts_table.build_indices(local_ro_mngr)


# AgentInteractions
def agent_interactions(local_ro_mngr: ReadonlyDatabaseManager):
    # This table depends completely on source_meta and name_meta in the
    # readonly database, so we can just run the table's create method
    if not table_has_content(local_ro_mngr, "source_meta") or \
            not table_has_content(local_ro_mngr, "name_meta"):
        raise ValueError("source_meta and name_meta must be filled before "
                         "agent_interactions can be filled")
    agent_interactions_table: ReadonlyTable = local_ro_mngr.tables[
        "agent_interactions"]

    # Create the table
    logger.info(f"Creating table {agent_interactions_table.full_name()}")
    agent_interactions_table.create(local_ro_mngr)

    # Build the index
    logger.info(f"Building index for table {agent_interactions_table.full_name()}")
    agent_interactions_table.build_indices(local_ro_mngr)


def get_local_ro_uri() -> str:
    # postgresql://<username>:<password>@localhost[:port]/[name]
    return f"postgresql://{LOCAL_RO_USER}:{LOCAL_RO_PASSWORD}@localhost:" \
           f"{LOCAL_RO_PORT}/{LOCAL_RO_DB_NAME}"


def load_data_file_into_local_ro(
    table_name: str, column_order: str, tsv_file: str, null_value: str = None
):
    """Load data from a file to the local ro database

    Mental note: COPY FROM copies data from a file to a table

    Parameters
    ----------
    table_name :
        The name of the table to transfer to, e.g. readonly.reading_ref_link
    column_order :
        A string of comma separated column names as they appear in the file,
        where the names correspond to the naming in the table in the readonly
        database, e.g. "reading_id, db_info_id, ref_id, ref_type, ref_text"
    tsv_file :
        The path to the tab separated file to be (up)loaded into the database.
    null_value :
        The value to be interpreted as null, e.g. "NULL" or "\N". If None,
        the default value of the database is used and the upload command
        will be without the 'NULL AS ...' part.
    """
    # todo: if you want to switch to gzipped files you can do
    #  "copy table_name from program 'zcat /tmp/tp.tsv.gz';"
    null_clause = f", NULL AS '{null_value}'" if null_value else ""
    command = [
        "psql",
        get_local_ro_uri(),
        "-c",
        (
            f"\\copy {table_name} ({column_order}) from '{tsv_file}' with "
            # fixme: test if the null clause works
            f"(format csv, delimiter E'\t', header{null_clause})"
        ),
    ]
    logger.info(f"Loading data into table {table_name} from {tsv_file}")
    subprocess.run(command)


# FastRawPaLink
def fast_raw_pa_link(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the fast_raw_pa_link table in the local readonly database

    Depends on:
    (principal)
    raw_statements, (pa_statements), raw_unique_links

    (readonly)
    raw_stmt_src

    (requires statement type map a.k.a ro_type_map.get_with_clause() to be
    inserted, here only showing the first 5. The full type map has almost 50
    items)

    Steps:
    Iterate through the grounded statements and:
    1. Dump (sid, src) from raw_stmt_src and load it into a dictionary
    2. Load stmt hash - raw statement id mapping into a dictionary
    3. Get raw statement id mapping to a) db info id and b) raw statement json
    4. Iterate over the grounded statements and get the following values:
        raw statement id
        raw statement json
        raw db info id      <-- (get from raw statements)
        assembled statement hash
        assembled statement json
        type num (from ro_type_map; (0, 'Acetylation'), (1, 'Activation'), ...)
        raw statement source (mapped from the raw_stmt_src dictionary)
    """
    table_name = "fast_raw_pa_link"
    assert table_name in local_ro_mngr.tables

    # Load the raw_stmt_src table into a dictionary
    logger.info("Loading raw_stmt_src table into a dictionary")
    local_ro_mngr.grab_session()
    query = local_ro_mngr.session.query(local_ro_mngr.RawStmtSrc.sid,
                                        local_ro_mngr.RawStmtSrc.src)
    reading_id_source_map = {int(read_id): src for read_id, src in query.all()}
    if len(reading_id_source_map) == 0:
        raise ValueError("No data in readonly.raw_stmt_src")

    # Load statement hash - raw statement id mapping into a dictionary
    hash_to_raw_id_map = pickle.load(stmt_hash_to_raw_stmt_ids_fpath.open("rb"))

    # Iterate over the raw statements to get mapping from
    # raw statement id to reading id, db info id and raw json
    logger.info("Loading mapping from raw statement id to info id")
    with gzip.open(raw_id_info_map_fpath.as_posix(), "rt") as fh:
        reader = csv.reader(fh, delimiter="\t")
        raw_id_to_info = {}
        for raw_stmt_id, db_info_id, reading_id, stmt_json_raw in reader:
            info = {"raw_json": stmt_json_raw}
            if db_info_id:
                info["db_info_id"] = int(db_info_id)
            if reading_id:
                info["reading_id"] = int(reading_id)
            raw_id_to_info[int(raw_stmt_id)] = info

    # For each grounded statement, get all associated raw statement ids and
    # get the following values:
    #   - raw statement id,
    #   - raw statement json,
    #   - db info id,
    #   - assembled statement hash,
    #   - assembled statement json,
    #   - type num (from ro_type_map)
    #   - raw statement source (mapped from the raw_stmt_src dictionary)
    temp_tsv = TEMP_DIR.joinpath(f"{uuid.uuid4()}.tsv")
    logger.info("Iterating over grounded statements")
    with gzip.open(unique_stmts_fpath.as_posix(), "rt") as fh,\
            temp_tsv.open("w") as out_fh:
        unique_stmts_reader = csv.reader(fh, delimiter="\t")
        writer = csv.writer(out_fh, delimiter="\t")

        for statement_hash_string, stmt_json_string in unique_stmts_reader:
            this_hash = int(statement_hash_string)
            stmt_json = load_statement_json(stmt_json_string)
            for raw_stmt_id in hash_to_raw_id_map[this_hash]:
                info_dict = raw_id_to_info.get(raw_stmt_id, {})
                raw_stmt_src_name = reading_id_source_map[raw_stmt_id]
                type_num = ro_type_map._str_to_int(stmt_json["type"])
                writer.writerow([
                    raw_stmt_id,
                    info_dict.get("raw_json"),
                    info_dict.get("reading_id"),
                    info_dict.get("db_info_id"),
                    statement_hash_string,
                    stmt_json_string,
                    type_num,
                    raw_stmt_src_name,
                ])

    # Load the data into the fast_raw_pa_link table
    column_order = ("id, raw_json, reading_id, db_info_id, mk_hash, pa_json, "
                    "type_num, src")
    load_data_file_into_local_ro(table_name, column_order, temp_tsv)

    # Remove the temporary file
    os.remove(temp_tsv)

    # Build the index
    table: ReadonlyTable = local_ro_mngr.tables[table_name]
    logger.info(f"Building index on {table.full_name()}")
    table.build_indices(local_ro_mngr)


def ensure_source_meta_source_files(local_ro_mngr: ReadonlyDatabaseManager):
    """Generate the source files for the SourceMeta table"""
    # NOTE: this function depends on the readonly NameMeta table
    # Load source_counts
    ro = local_ro_mngr
    source_counts = pickle.load(source_counts_fpath.open("rb"))

    # Dump out from NameMeta
    # mk_hash, type_num, activity, is_active, ev_count, belief, agent_count
    # where is_complex_dup == False
    # todo If it's too large, we can do it in chunks of 100k
    # Typical count for namemeta 90_385_263
    res = local_ro_mngr.select_all([ro.NameMeta.ev_count,
                                    ro.NameMeta.belief,
                                    ro.NameMeta.type_num,
                                    ro.NameMeta.activity,
                                    ro.NameMeta.is_active,
                                    ro.NameMeta.agent_count],
                                   ro.NameMeta.is_complex_dup == False)

    # Loop the dump and create the following columns as well:
    # source count, src_json (==source counts dict per mk_hash), only_src (if
    # only one source), has_rd (if the source is in the reading sources),
    # has_db (if the source is in the db sources)
    # fixme: What is the proper null value? None, "null", "\\N", 0?
    null = ""
    all_sources = list(
        {*SOURCE_GROUPS["reader"], *SOURCE_GROUPS["database"]}
    )
    with source_meta_tsv.open("w") as out_fh:
        writer = csv.writer(out_fh, delimiter="\t")
        for (mk_hash, ev_count, belief, type_num,
             activity, is_active, agent_count) in res:

            # Get the source count
            src_count_dict = source_counts.get(mk_hash)
            if src_count_dict is None:
                continue

            num_srcs = len(src_count_dict)
            has_rd = SOURCE_GROUPS["reader"] in src_count_dict
            has_db = SOURCE_GROUPS["database"] in src_count_dict
            only_src = list(src_count_dict.keys())[0] if num_srcs == 1 else None
            sources_tuple = tuple(
                src_count_dict.get(src, null) for src in all_sources
            )

            # Write the following columns:
            #  - mk_hash
            #  - *sources (a splat of all sources,
            #              null if not present in the source count dict)
            #  - ev_count
            #  - belief
            #  - num_srcs
            #  - src_json  # fixme: Should it be dumped to a string? binary?
            #  - only_src - if only one source, the name of that source
            #  - has_rd  boolean - true if any source is from reading
            #  - has_db  boolean - true if any source is from a database
            #  - type_num
            #  - activity
            #  - is_active
            #  - agent_count

            # SourceMeta
            row = [
                mk_hash,
                *sources_tuple,
                ev_count,
                belief,
                num_srcs,
                json.dumps(src_count_dict).replace('"', '""'),
                only_src,
                has_rd,
                has_db,
                type_num,
                activity,
                is_active,
                agent_count
            ]
            writer.writerow(row)

    # Return the column order in the file
    all_sources_str = ", ".join(all_sources)
    col_names = (
        "ev_count, belief, num_srcs, src_json, only_src, has_rd, has_db, "
        "type_num, activity, is_active, agent_count"
    )
    return "mk_hash, " + all_sources_str + ", " + col_names


# SourceMeta
def source_meta(local_ro_mngr: ReadonlyDatabaseManager):
    col_order = ensure_source_meta_source_files(local_ro_mngr)
    table_name = "readonly.source_meta"

    logger.info(f"Loading data into {table_name}")
    load_data_file_into_local_ro(
        table_name, col_order, source_meta_tsv, null_value=""
    )

    # Build the index
    table: ReadonlyTable = local_ro_mngr.tables[table_name]
    logger.info(f"Building index on {table.full_name()}")
    table.build_indices(local_ro_mngr)


# PaMeta - the table itself is not generated on the readonly db, but the
# tables derived from it are (NameMeta, TextMeta and OtherMeta)
def ensure_pa_meta_table():
    """Generate the source files for the Name/Text/OtherMeta tables"""
    if all([name_meta_tsv.exists(),
            text_meta_tsv.exists(),
            other_meta_tsv.exists()]):
        return

    #  Depends on:
    #   - PaActivity (from principal schema)
    #   - PaAgents (from principal schema)
    #   - PAStatements (from principal schema)
    # - Belief (from readonly schema)
    # - EvidenceCounts (from readonly schema)
    # - PaAgentCounts (from readonly schema)
    # - TypeMap (from ???)
    # - RoleMap (from ???)
    # To get the needed info from the principal schema, select the
    # appropriate columns from the tables on the principal schema, join them
    # by mk_hash and then read in belief, agent count and evidence count
    # from the belief dump, unique statement file and source counts file.
    #
    # **Role num** is the integer mapping subject, other, object to
    # (-1, 0, 1), see schemas.readonly_schema.RoleMapping
    #
    # **Type num** is the integer mapping of the statement type to
    # the integer in the type map, see
    # schemas.readonly_schema.StatementTypeMapping
    #
    # *pa_agents* has the columns:
    # stmt_mk_hash, db_name, db_id, role, ag_num, agent_ref_hash
    #
    # *pa_statements* has the columns:
    # mk_hash, matches_key, uuid, type, indra_version, json, create_date
    #
    # *pa_activity* has the columns:
    # id, stmt_mk_hash, statements, activity, is_active
    pa_meta_query = ("""
    SELECT pa_agents.db_name,
           pa_agents.db_id,
           pa_agents.id AS ag_id, 
           pa_agents.ag_num,
           pa_agent.role,
           pa_statements.mk_hash
    FROM pa_agents INNER JOIN
         pa_statements ON
             pa_agents.stmt_mk_hash = pa_statements.mk_hash
    WHERE LENGTH(pa_agents.db_id) < 2000
    """)
    # fixme:
    #  - Add 'LENGTH(pa_agents.db_id) < 2000' as condition to the query??
    #  - Skip the INNER JOIN and just use the mk_hash from pa_agents?
    principal_query_to_csv(pa_meta_query, pa_meta_fpath)

    # Load the belief dump into a dictionary
    logger.info("Loading belief scores")
    belief_dict = pickle.load(belief_scores_pkl_fpath.open("rb"))

    # Load source counts (can generate evidence counts from this)
    logger.info("Loading source counts")
    source_counts = pickle.load(source_counts_fpath.open("rb"))

    # Load agent count, activity and type mapping
    stmt_hash_to_activity_type_count = get_activity_type_ag_count()

    # Loop pa_meta dump and write load files for NameMeta, TextMeta, OtherMeta
    logger.info("Iterating over pa_meta dump")
    nones = (None, None, None, None)
    with gzip.open(pa_meta_fpath.as_posix(), "rt") as fh, \
            name_meta_tsv.open("wt") as name_fh, \
            text_meta_tsv.open("wt") as text_fh, \
            other_meta_tsv.open("wt") as other_fh:
        reader = csv.reader(fh, delimiter="\t")
        name_writer = csv.writer(name_fh, delimiter="\t")
        text_writer = csv.writer(text_fh, delimiter="\t")
        other_writer = csv.writer(other_fh, delimiter="\t")

        for db_name, db_id, ag_id, ag_num, role, stmt_hash_str in reader:
            stmt_hash = int(stmt_hash_str)

            # Get the belief score
            belief_score = belief_dict.get(stmt_hash)
            if belief_score is None:
                # todo: debug log, remove later
                logger.warning(f"Missing belief score for {stmt_hash}")
                continue

            # Get the agent count, activity and type count
            activity, is_active, type_num, agent_count = \
                stmt_hash_to_activity_type_count.get(stmt_hash, nones)
            if type_num is None and agent_count is None:
                continue

            # Get the evidence count
            ev_count = sum(
                source_counts.get(stmt_hash, {}).values()
            )
            if ev_count == 0:
                # todo: debug log, remove later
                logger.warning(f"Missing evidence count for {stmt_hash}")
                continue

            # Get role num
            role_num = ro_role_map.get_int(role)
            is_complex_dup = True if type_num == ro_type_map.get_int(
                "Complex") else False

            # NameMeta - db_name == "NAME"
            # TextMeta - db_name == "TEXT"
            # OtherMeta - other db_names not part of ("NAME", "TEXT")
            # Columns are:
            # ag_id, ag_num, [db_name,] db_id, role_num, type_num, mk_hash,
            # ev_count, belief, activity, is_active, agent_count,
            # is_complex_dup
            row_start = [
                ag_id,
                ag_num,
            ]
            row_end = [
                db_id,
                role_num,
                type_num,
                stmt_hash,
                ev_count,
                belief_score,
                activity,
                is_active,
                agent_count,
                is_complex_dup,
            ]
            if db_name == "NAME":
                name_writer.writerow(row_start + row_end)
            elif db_name == "TEXT":
                text_writer.writerow(row_start + row_end)
            else:
                other_writer.writerow(row_start + [db_name] + row_end)


# NameMeta, TextMeta, OtherMeta
def name_meta(local_ro_mngr: ReadonlyDatabaseManager):
    # Ensure the pa_meta file exists
    ensure_pa_meta_table()
    # ag_id, ag_num, db_id, role_num, type_num, mk_hash,
    # ev_count, belief, activity, is_active, agent_count, is_complex_dup
    colum_order = (
        "ag_id, ag_num, db_id, role_num, type_num, mk_hash, ev_count, "
        "belief, activity, is_active, agent_count, is_complex_dup"
    )

    # Load into local ro
    logger.info("Loading name_meta into local ro")
    load_data_file_into_local_ro("readonly.name_meta",
                                 colum_order,
                                 name_meta_tsv.absolute().as_posix())

    # Build indices
    name_meta_table: ReadonlyTable = local_ro_mngr.tables["name_meta"]
    logger.info("Building indices for name_meta")
    name_meta_table.build_indices(local_ro_mngr)


def text_meta(local_ro_mngr: ReadonlyDatabaseManager):
    # Ensure the pa_meta file exists
    ensure_pa_meta_table()
    # ag_id, ag_num, db_id, role_num, type_num, mk_hash,
    # ev_count, belief, activity, is_active, agent_count, is_complex_dup
    colum_order = (
        "ag_id, ag_num, db_id, role_num, type_num, mk_hash, ev_count, "
        "belief, activity, is_active, agent_count, is_complex_dup"
    )

    # Load into local ro
    logger.info("Loading text_meta into local ro")
    load_data_file_into_local_ro("readonly.text_meta",
                                 colum_order,
                                 text_meta_tsv.absolute().as_posix())

    # Build indices
    text_meta_table: ReadonlyTable = local_ro_mngr.tables["text_meta"]
    logger.info("Building indices for text_meta")
    text_meta_table.build_indices(local_ro_mngr)


def other_meta(local_ro_mngr: ReadonlyDatabaseManager):
    # Ensure the pa_meta file exists
    ensure_pa_meta_table()
    # ag_id, ag_num, db_name, db_id, role_num, type_num, mk_hash,
    # ev_count, belief, activity, is_active, agent_count, is_complex_dup
    colum_order = (
        "ag_id, ag_num, db_name, db_id, role_num, type_num, mk_hash, "
        "ev_count, belief, activity, is_active, agent_count, is_complex_dup"
    )

    # Load into local ro
    logger.info("Loading other_meta into local ro")
    load_data_file_into_local_ro("readonly.other_meta",
                                 colum_order,
                                 other_meta_tsv.absolute().as_posix())

    # Build indices
    other_meta_table: ReadonlyTable = local_ro_mngr.tables["other_meta"]
    logger.info("Building indices for other_meta")
    other_meta_table.build_indices(local_ro_mngr)


def ensure_pubmed_xml_files(xml_dir: Path = pubmed_xml_gz_dir,
                            retries: int = 3) -> int:
    """Downloads the PubMed XML files if they are not already present"""

    def _get_urls(url: str) -> Iterable[str]:
        """Get the paths to all XML files on the PubMed FTP server."""
        logger.info("Getting URL paths from %s" % url)

        # Get page
        response = requests.get(url)
        response.raise_for_status()

        # Make soup
        soup = BeautifulSoup(response.text, "html.parser")

        # Append trailing slash if not present
        url = url if url.endswith("/") else url + "/"

        # Loop over all links
        for link in soup.find_all("a"):
            href = link.get("href")
            # yield if href matches
            # 'pubmed<2 digit year>n<4 digit file index>.xml.gz'
            # but skip the md5 files
            if href and href.startswith("pubmed") and href.endswith(".xml.gz"):
                yield url + href

    def _download_xml_gz(xml_url: str, xml_file: Path, md5_check: bool = True):
        try:
            resp = requests.get(xml_url)
            resp.raise_for_status()
        except requests.exceptions.RequestException:
            return False

        if md5_check:
            md5_resp = requests.get(xml_url + ".md5")
            checksum = md5(resp.content).hexdigest()
            expected_checksum = re.search(
                r"[0-9a-z]+(?=\n)", md5_resp.content.decode("utf-8")
            ).group()
            if checksum != expected_checksum:
                logger.warning(
                    f"Checksum mismatch for {xml_url}, skipping download"
                )
                raise ValueError("Checksum mismatch")

        # Write the file xml.gz file
        with xml_file.open("wb") as fh:
            fh.write(resp.content)

        return True

    if retries < 0:
        raise ValueError("retries must be >= 0")

    # Define some constants
    pubmed_base_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
    pubmed_update_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"

    # Create the directory if it doesn't exist
    xml_dir.mkdir(exist_ok=True, parents=True)

    # Download the files if they don't exist
    num_files = 0
    basefiles = [u for u in _get_urls(pubmed_base_url)]
    updatefiles = [u for u in _get_urls(pubmed_update_url)]
    for xml_url in tqdm(
            basefiles + updatefiles, desc="Downloading PubMed XML files"
    ):
        xml_file_path = xml_dir.joinpath(xml_url.split("/")[-1])

        # Download the file if it doesn't exist
        if not xml_file_path.exists():
            success = False
            for _ in range(retries + 1):
                try:
                    success = _download_xml_gz(xml_url, xml_file_path)
                except ValueError:
                    logger.warning(f"Checksum mismatch, skipping {xml_url}")
                    break
                # If success, break retry loop and continue to next file
                if success:
                    num_files += 1
                    break
            if not success:
                logger.error(
                    f"Failed to download {xml_url} after {retries} retries"
                )
        else:
            num_files += 1

    return num_files


def _load_pmid_to_raw_stmt_id():
    # Load necessary mappings #
    # Mapping sequence from mesh to raw statement id:
    # mesh -> pmid -> text_ref_id -> reading_id -> raw_statement_id
    # mesh -> pmid comes from pubmed (pubmed xml - outer loop in this function)
    # pmid -> text_ref_id comes from text_refs_fpath
    # text_ref_id -> reading_id comes from reading_to_text_ref_map_fpath
    # reading_id -> raw_statement_id comes from raw_id_info_map_fpath
    text_ref_to_pmid_cache = PUBMED_MESH_DIR.joinpath("text_ref_to_pmid.pkl")
    reading_id_to_pmid_cache = \
        PUBMED_MESH_DIR.joinpath("reading_id_to_pmid.pkl")
    pmid_to_raw_stmt_id_cache = \
        PUBMED_MESH_DIR.joinpath("pmid_to_raw_stmt_id.pkl")

    def _load_create_text_ref_to_pmid():
        if text_ref_to_pmid_cache.exists():
            logger.info("Loading text_ref -> pmid map from cache")
            with text_ref_to_pmid_cache.open("rb") as fh:
                local_dict = pickle.load(fh)
        else:
            local_dict = {}
            with gzip.open(text_refs_fpath.as_posix(), "rt") as fh:
                reader = csv.reader(fh, delimiter="\t")
                # Rows are:
                # TRID, PMID, PMCID, DOI, PII, URL, MANUSCRIPT_ID
                for meta_row in tqdm(reader):
                    if meta_row[1] != "\\N":
                        # trid values are unique, PMIDs are not
                        trid = int(meta_row[0])
                        pmid = int(meta_row[1])
                        local_dict[trid] = pmid

            # Save to cache
            with text_ref_to_pmid_cache.open("wb") as fh:
                pickle.dump(local_dict, fh)

        return local_dict

    def _load_create_reading_id_to_pmid():
        if reading_id_to_pmid_cache.exists():
            logger.info("Loading reading_id -> pmid map from cache")
            with reading_id_to_pmid_cache.open("rb") as fh:
                local_dict = pickle.load(fh)
        else:
            text_ref_to_pmid = _load_create_text_ref_to_pmid()
            local_dict = {}
            with gzip.open(reading_to_text_ref_map_fpath.as_posix(),
                           "rt") as fh:
                reader = csv.reader(fh, delimiter="\t")
                # Rows are:
                # reading_id,  reader_version, text_content_id,
                # text_ref_id, source,         text_type
                for meta_row in tqdm(reader):
                    if meta_row[3] != "\\N":
                        trid = int(meta_row[3])
                        rid = int(meta_row[0])
                        # One text ref id can have multiple reading ids
                        # (because one text source can have multiple readings
                        # from different readers and/or reader versions) so
                        # text_ref_id -> reading_id is a one-to-many mapping
                        # => reading_id -> pmid is a many-to-one mapping
                        pmid = text_ref_to_pmid.get(trid)
                        if pmid:
                            local_dict[rid] = pmid

            # Save the cache
            with reading_id_to_pmid_cache.open("wb") as fh:
                pickle.dump(local_dict, fh)

        return local_dict

    def _load_create_pmid_to_raw_stmt_id():
        if pmid_to_raw_stmt_id_cache.exists():
            logger.info("Loading pmid -> raw_stmt_id map from cache")
            with pmid_to_raw_stmt_id_cache.open("rb") as fh:
                local_dict = pickle.load(fh)
        else:
            reading_id_to_pmid = _load_create_reading_id_to_pmid()
            local_dict = defaultdict(set)
            with gzip.open(raw_id_info_map_fpath.as_posix(), "rt") as fh:
                reader = csv.reader(fh, delimiter="\t")
                # Rows are:
                # raw_stmt_id_int, db_info_id, int_reading_id, stmt_json_raw
                for meta_row in tqdm(reader):
                    if meta_row[2] != "\\N":
                        rid = int(meta_row[2])
                        raw_stmt_id = int(meta_row[0])
                        pmid = reading_id_to_pmid.get(rid)
                        if pmid:
                            local_dict[pmid].add(raw_stmt_id)

            # Cast as regular dict
            local_dict = dict(local_dict)

            # Save to cache
            with pmid_to_raw_stmt_id_cache.open("wb") as fh:
                pickle.dump(local_dict, fh)

        return local_dict

    return _load_create_pmid_to_raw_stmt_id()


def get_activity_type_ag_count():
    if pa_hash_act_type_ag_count_cache.exists():
        cache = pickle.load(pa_hash_act_type_ag_count_cache.open("rb"))
        return cache

    # Get stmt_hash -> type_num, activity, is_active, agent_count
    # Loop unique_stmts_path
    logger.info("Mapping pre-assembled statement hashes to activity data and "
                "agent counts")
    stmt_hash_to_activity_type_count = {}
    with gzip.open(unique_stmts_fpath.as_posix(), "rt") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for stmt_hash_string, stmt_json_string in tqdm(reader):
            stmt_json = load_statement_json(stmt_json_string)
            stmt = stmt_from_json(stmt_json)
            stmt_hash = int(stmt_hash_string)
            agent_count = len(stmt.agent_list())
            type_num = ro_type_map.get_int(stmt_json["type"])
            if isinstance(stmt, ActiveForm):
                activity = stmt.activity
                is_active = stmt.is_active
            else:
                activity = None
                is_active = False  # is_active is a boolean column
            stmt_hash_to_activity_type_count[stmt_hash] = (
                activity, is_active, type_num, agent_count
            )

    # Save to cache
    with pa_hash_act_type_ag_count_cache.open("wb") as fh:
        pickle.dump(stmt_hash_to_activity_type_count, fh)


def ensure_pubmed_mesh_data():
    """Get the of PubMed XML gzip files for pmid-mesh processing"""
    # Check if the output files already exist

    if all(f.exists() for f in [mesh_concepts_meta,
                                mesh_terms_meta,
                                raw_stmt_mesh_concepts,
                                raw_stmt_mesh_terms,
                                pmid_mesh_map_fpath,
                                pmid_stmt_hash_fpath]):
        return

    def _pmid_mesh_extractor(xml_gz_path: Path) -> Iterable[Tuple[str, str]]:
        tree = etree.parse(xml_gz_path.as_posix())

        for article in tree.findall("PubmedArticle"):
            medline_citation = article.find("MedlineCitation")
            pubmed_id = medline_citation.find("PMID").text

            mesh_ann = _get_annotations(medline_citation)
            yield pubmed_id, mesh_ann["mesh_annotations"]

    num_files = ensure_pubmed_xml_files(xml_dir=pubmed_xml_gz_dir)
    if num_files == 0:
        raise FileNotFoundError("No PubMed XML files found")

    # Get the raw statement id -> stmt hash mapping
    hash_to_raw_stmt_id = pickle.load(stmt_hash_to_raw_stmt_ids_fpath.open("rb"))
    raw_stmt_id_to_hash = {raw_id: stmt_hash for stmt_hash, raw_ids in
                           hash_to_raw_stmt_id.items() for raw_id in raw_ids}
    del hash_to_raw_stmt_id

    # Load the pmid -> raw statement id mapping
    pmid_to_raw_stmt_id = _load_pmid_to_raw_stmt_id()

    stmt_hash_to_activity_type_count = get_activity_type_ag_count()

    # Get source counts ({hash: {source: count}})
    logger.info("Loading source counts")
    source_counts = pickle.load(source_counts_fpath.open("rb"))

    # Get belief scores (hash -> belief)
    logger.info("Loading belief scores")
    belief_scores = pickle.load(belief_scores_pkl_fpath.open("rb"))

    stmt_hashes = set()
    pmid_mesh_mapping = {}
    pmid_stmt_hash = defaultdict(set)
    logger.info("Generating tsv ingestion files")
    with (
        gzip.open(
            mesh_concepts_meta.as_posix(), "wt") as concepts_meta_fh,
        gzip.open(
            mesh_terms_meta.as_posix(), "wt") as terms_meta_fh,
        gzip.open(
            raw_stmt_mesh_concepts.as_posix(), "wt") as raw_concepts_fh,
        gzip.open(
            raw_stmt_mesh_terms.as_posix(), "wt") as raw_terms_fh
    ):
        concepts_meta_writer = csv.writer(concepts_meta_fh, delimiter="\t")
        terms_meta_writer = csv.writer(terms_meta_fh, delimiter="\t")
        raw_concepts_writer = csv.writer(raw_concepts_fh, delimiter="\t")
        raw_terms_writer = csv.writer(raw_terms_fh, delimiter="\t")

        xml_files = list(pubmed_xml_gz_dir.glob("*.xml.gz"))
        for xml_file_path in tqdm(xml_files, desc="Pubmed processing"):
            # Extract the data from the XML file
            for pmid, mesh_annotations in _pmid_mesh_extractor(xml_file_path):
                pmid_mesh_map = {"concepts": set(), "terms": set()}
                for annot in mesh_annotations:
                    mesh_id = annot["mesh_id"]
                    mesh_num = int(mesh_id[1:])
                    is_concept = mesh_id.startswith("C")
                    # major_topic = annot["major_topic"]  # Unused

                    # Save each pmid-mesh mapping
                    if is_concept:
                        pmid_mesh_map["concepts"].add(mesh_num)
                    else:
                        pmid_mesh_map["terms"].add(mesh_num)

                    # For each pmid, get the raw statement id
                    for raw_stmt_id in pmid_to_raw_stmt_id.get(pmid, set()):
                        raw_row = [raw_stmt_id, mesh_num]
                        # RawStmtMeshConcepts
                        if is_concept:
                            raw_concepts_writer.writerow(raw_row)
                        # RawStmtMeshTerms
                        else:
                            raw_terms_writer.writerow(raw_row)

                        # Now write to the meta tables, one row per
                        # (stmt hash, mesh id) pair
                        stmt_hash = raw_stmt_id_to_hash.get(raw_stmt_id)

                        # Save the pmid -> stmt hash mapping
                        if stmt_hash:
                            pmid_stmt_hash[pmid].add(stmt_hash)

                        if stmt_hash and stmt_hash not in stmt_hashes:
                            tup = \
                                stmt_hash_to_activity_type_count.get(stmt_hash)

                            if tup is not None:
                                act, is_act, type_num, agent_count = tup
                            else:
                                stmt_hashes.add(stmt_hash)
                                continue

                            # Get evidence count for this statement
                            ec = sum(
                                source_counts.get(stmt_hash, {}).values()
                            )
                            bs = belief_scores.get(stmt_hash)

                            meta_row = [
                                stmt_hash,
                                ec,
                                bs,
                                mesh_num,
                                type_num,
                                act,
                                is_act,
                                agent_count
                            ]
                            if is_concept:
                                # MeshConceptMeta
                                concepts_meta_writer.writerow(meta_row)
                            else:
                                # MeshTermMeta
                                terms_meta_writer.writerow(meta_row)

                            stmt_hashes.add(stmt_hash)

                # If the pmid is already in pmid_mesh_mapping, update the
                # mesh concepts and terms
                if pmid in pmid_mesh_mapping:
                    pmid_mesh_mapping[pmid]["concepts"].update(
                        pmid_mesh_map["concepts"]
                    )
                    pmid_mesh_mapping[pmid]["terms"].update(
                        pmid_mesh_map["terms"]
                    )
                else:
                    pmid_mesh_mapping[pmid] = pmid_mesh_map

    # Save the pmid-mesh and pmid-stmt hash mappings to cache
    logger.info("Saving pmid-stmt hash mappings to cache")
    pmid_stmt_hash = dict(pmid_stmt_hash)
    with pmid_stmt_hash_fpath.open("wb") as pmid_stmt_hash_fh:
        pickle.dump(pmid_stmt_hash, pmid_stmt_hash_fh)
    logger.info("Saving pmid-mesh mappings to cache")
    with pmid_mesh_map_fpath.open("wb") as pmid_mesh_mapping_fh:
        pickle.dump(pmid_mesh_mapping, pmid_mesh_mapping_fh)


# RawStmtMeshConcepts
def create_raw_stmt_mesh_concepts(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the raw_stmt_mesh_concepts table."""
    full_table_name = "readonly.raw_stmt_mesh_concepts"
    column_order = "sid, mesh_num"
    _mesh_create(local_ro_mngr, full_table_name, column_order)


# RawStmtMeshTerms
def create_raw_stmt_mesh_terms(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the raw_stmt_mesh_terms table"""
    full_table_name = "readonly.raw_stmt_mesh_terms"
    column_order = "sid, mesh_num"
    _mesh_create(local_ro_mngr, full_table_name, column_order)


# MeshConceptMeta
def create_mesh_concept_meta(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the mesh_concept_meta table."""
    full_table_name = "readonly.mesh_concept_meta"
    column_order = (
        "mk_hash, ev_count, belief, mesh_num, "
        "type_num, activity, is_active, agent_count"
    )
    _mesh_create(local_ro_mngr, full_table_name, column_order)


# MeshTermMeta
def create_mesh_term_meta(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the mesh_term_meta table."""
    full_table_name = "readonly.mesh_term_meta"
    column_order = (
        "mk_hash, ev_count, belief, mesh_num, "
        "type_num, activity, is_active, agent_count"
    )
    _mesh_create(local_ro_mngr, full_table_name, column_order)


def _mesh_create(
    local_ro_mngr: ReadonlyDatabaseManager,
    full_table_name: str,
    column_order: str,
):
    # Ensure tsv files exist
    ensure_pubmed_mesh_data()

    # Load data into table
    load_data_file_into_local_ro(
        full_table_name, column_order, raw_stmt_mesh_terms.as_posix()
    )

    # Build index
    table = local_ro_mngr.tables[full_table_name.split(".")[1]]
    logger.info(f"Building index for {table.full_name}")
    table.build_indices(local_ro_mngr)


def principal_query_to_csv(
        query: str, output_location: str, db: str = "primary"
) -> None:
    """Dump results of query into a csv file.

    Parameters
    ----------
    query : str
        Postgresql query that returns a list of rows. Queries that perform
        writes of any kind should not be used. A simple keyword search is
        used to check for queries that may perform writes, but be careful,
        it shouldn't be trusted completely.

    output_location : str
        Path to file where output is to be stored.

    db : Optional[str]
        Database from list of defaults in indra_db config. Default: "primary".

    Raises
    ------
    AssertionError
        If input query contains a disallowed keyword that could lead to
        modifications of the database.
    """

    disallowed_keywords = _find_disallowed_keywords(query)

    if disallowed_keywords:
        raise ValueError(
            f"Query '{query}' uses disallowed keywords: {disallowed_keywords}"
        )

    defaults = get_databases()
    try:
        principal_db_uri = defaults[db]
    except KeyError as err:
        raise KeyError(f"db {db} not available. Check db_config.ini") from err

    command = [
        "psql",
        principal_db_uri,
        "-c",
        f"\\copy ({query}) to {output_location} with (format csv, delimiter "
        f"E'\t', header)",
    ]
    logger.info(f"Running SQL query {query} with psql")
    subprocess.run(command)


def _find_disallowed_keywords(query: str) -> list:
    """Returns list of disallowed keywords in query if any exist.

    Keywords are disallowed if they can lead to modifications of the
    database or its settings. The disallowed keywords are:

    alter, call, commit, create, delete, drop, explain, grant, insert,
    lock, merge, rename,  revoke, savepoint, set, rollback, transaction,
    truncate, update.

    Matching is case insensitive.

    Parameters
    ----------
    query : str
        A string containing a Postgresql query.

    Returns
    -------
    list
        List of keywords within the query that are within the disallowed
        list. Entries are returned in all lower case.
    """
    disallowed = [
        "alter",
        "call",
        "commit",
        "create",
        "delete",
        "drop",
        "explain",
        "grant",
        "insert",
        "lock",
        "merge",
        "rename",
        "revoke",
        "savepoint",
        "set",
        "rollback",
        "transaction",
        "truncate",
        "update",
    ]

    query_token_set = set(token.lower() for token in query.split())
    return list(query_token_set & set(disallowed))


def get_temp_file(suffix: str) -> Path:
    """Return a temporary file path.

    Parameters
    ----------
    suffix :
        The suffix to use for the temporary file, e.g. '.csv'.

    Returns
    -------
    :
        A Path object pointing to the temporary file.
    """
    return TEMP_DIR.joinpath(f"{uuid.uuid4()}{suffix}")


def drop_schema(ro_mngr_local: ReadonlyDatabaseManager):
    logger.info("Dropping schema 'readonly'")
    ro_mngr_local.drop_schema('readonly')


def create_ro_tables(ro_mngr: ReadonlyDatabaseManager, postgres_url):
    engine = create_engine(postgres_url)
    logger.info("Connected to local db")

    # Create schema
    ro_mngr.create_schema('readonly')

    # Create empty tables
    tables_metadata = ro_mngr.tables['belief'].metadata
    logger.info("Creating tables")
    tables_metadata.create_all(bind=engine)
    logger.info("Done creating tables")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__name__)
    parser.add_argument("--db-name", default="indradb_readonly_local")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--hostname", default="localhost")
    parser.add_argument("--port", default=5432,
                        help="The port the local db server listens to.")

    args = parser.parse_args()

    # Get a ro manager for the local readonly db
    # postgresql://<username>:<password>@localhost[:port]/[name]
    postgres_url = get_local_ro_uri()
    ro_manager = ReadonlyDatabaseManager(postgres_url)

    # Create the tables
    create_ro_tables(ro_manager, postgres_url)

    # For each table, run the function that will fill out the table with data
    # todo: add all table filling functions
