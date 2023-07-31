import argparse
import csv
import gzip
import logging
import os
import pickle
import re
import subprocess
import uuid
from collections import defaultdict
from hashlib import md5
from pathlib import Path
from textwrap import dedent
from typing import Tuple, Iterable

from lxml import etree
from sqlalchemy import create_engine
from tqdm import tqdm

from indra.literature.pubmed_client import _get_annotations
from indra.statements import stmt_from_json, ActiveForm
from indra_db.config import get_databases
from indra_db.databases import ReadonlyDatabaseManager
from indra_db.schemas.mixins import ReadonlyTable
from schemas.readonly_schema import ro_type_map

from .locations import *
from .util import load_statement_json

logger = logging.getLogger(__name__)

LOCAL_RO_PASSWORD = os.environ["LOCAL_RO_PASSWORD"]
LOCAL_RO_USER = os.environ["LOCAL_RO_USER"]
LOCAL_RO_PORT = int(os.environ.get("LOCAL_RO_PORT", "5432"))
LOCAL_RO_DB_NAME = os.environ.get("LOCAL_RO_DB_NAME", "indradb_readonly_local")


# Tables to create (currently in no particular order):

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
           SELECT raw_statements.id AS sid, 
           lower(db_info.db_name) AS src 
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


def get_local_ro_uri() -> str:
    # postgresql://<username>:<password>@localhost[:port]/[name]
    return f"postgresql://{LOCAL_RO_USER}:{LOCAL_RO_PASSWORD}@localhost:" \
           f"{LOCAL_RO_PORT}/{LOCAL_RO_DB_NAME}"


def load_data_file_into_local_ro(table_name: str, column_order: str,
                                 tsv_file: str):
    """Load data from a file to the local ro database

    Parameters
    ----------
    table_name :
        The name of the table to transfer to, e.g. readonly.reading_ref_link
    column_order :
        A string of comma separated column names as they appear in the file.
    tsv_file :
        The path to the file to be uploaded.

    COPY FROM copies data from a file to a table
    """
    command = [
        "psql",
        get_local_ro_uri(),
        "-c",
        f"\\copy {table_name} ({column_order}) from '{tsv_file}' with (format csv, delimiter E'\t', header)",
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

    # For each grounded statements, get all associated raw statement ids and
    # get the following values:
    #   - raw statement id,
    #   - raw statement json,
    #   - db info id,
    #   - assembled statement hash,
    #   - assembled statement json,
    #   - type num (from ro_type_map)
    #   - raw statement source (mapped from the raw_stmt_src dictionary)
    temp_tsv = f"{uuid.uuid4()}.tsv"
    logger.info("Iterating over grounded statements")
    with gzip.open(unique_stmts_fpath.as_posix(), "rt") as fh, open(
            temp_tsv, "w") as out_fh:
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


def ensure_pubmed_xml_files(xml_dir: Path = pubmed_xml_gz_dir,
                            retries: int = 3) -> int:
    """Downloads the PubMed XML files if they are not already present"""
    if retries < 0:
        raise ValueError("retries must be >= 0")

    import requests

    # Define some constants
    year_index = 23  # Check https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
    max_file_index = 1166  # Check https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
    max_update_index = 1218  # Check https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/
    xml_file_template = "pubmed%sn{index}.xml.gz" % year_index
    pubmed_base_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
    pubmed_update_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"

    # Create the directory if it doesn't exist
    xml_dir.mkdir(exist_ok=True, parents=True)

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
        with open(xml_file, "wb") as fh:
            fh.write(resp.content)

        return True

    # Download the files if they don't exist
    num_files = 0
    for index in tqdm(range(1, max_update_index + 10)):
        xml_file_name = xml_file_template.format(index=index)
        xml_file_path = xml_dir.joinpath(xml_file_name)

        # Download the file if it doesn't exist
        if not xml_file_path.exists():
            base_url = pubmed_base_url if index <= max_file_index else \
                pubmed_update_url
            xml_url = base_url + xml_file_name
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
                logger.error(f"Failed to download {xml_url} after {retries} retries")
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


def ensure_pubmed_mesh_data():
    """Get the of PubMed XML gzip files for pmid-mesh processing"""
    # Check if the output files already exist

    if all(f.exists() for f in [mesh_concepts_meta, mesh_terms_meta,
                                raw_stmt_mesh_concepts, raw_stmt_mesh_terms]):
        return

    def _pmid_mesh_extractor(xml_gz_path: Path) -> Iterable[Tuple[str, str]]:
        tree = etree.parse(xml_gz_path.as_posix())

        for article in tree.findall("PubmedArticle"):
            medline_citation = article.find("MedlineCitation")
            pmid = medline_citation.find("PMID").text

            mesh_annotations = _get_annotations(medline_citation)
            yield pmid, mesh_annotations["mesh_annotations"]

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

    # Get source counts ({hash: {source: count}})
    logger.info("Loading source counts")
    source_counts = pickle.load(source_counts_fpath.open("rb"))

    # Get belief scores (hash -> belief)
    logger.info("Loading belief scores")
    belief_scores = pickle.load(belief_scores_pkl_fpath.open("rb"))

    stmt_hashes = set()
    logger.info("Generating tsv ingestion files")
    with gzip.open(mesh_concepts_meta.as_posix(), "wt") as concepts_meta_fh,\
            gzip.open(mesh_terms_meta.as_posix(), "wt") as terms_meta_fh,\
            gzip.open(raw_stmt_mesh_concepts.as_posix(), "wt") as raw_concepts_fh,\
            gzip.open(raw_stmt_mesh_terms.as_posix(), "wt") as raw_terms_fh:
        concepts_meta_writer = csv.writer(concepts_meta_fh, delimiter="\t")
        terms_meta_writer = csv.writer(terms_meta_fh, delimiter="\t")
        raw_concepts_writer = csv.writer(raw_concepts_fh, delimiter="\t")
        raw_terms_writer = csv.writer(raw_terms_fh, delimiter="\t")

        xml_files = list(pubmed_xml_gz_dir.glob("*.xml.gz"))
        for xml_file_path in tqdm(xml_files, desc="Pubmed XML files"):
            # Extract the data from the XML file
            for pmid, mesh_annotations in _pmid_mesh_extractor(xml_file_path):
                for annot in mesh_annotations:
                    mesh_id = annot["mesh_id"]
                    mesh_num = int(mesh_id[1:])
                    is_concept = mesh_id.startswith("C")
                    # major_topic = annot["major_topic"]  # Unused

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

    try:
        assert not disallowed_keywords
    except AssertionError:
        logger.exception(
            f'Query "{query}" uses disallowed keywords: {disallowed_keywords}'
        )

    defaults = get_databases()
    try:
        principal_db_uri = defaults[db]
    except KeyError:
        logger.exception(f"db {db} not available. Check db_config.ini")

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

    # Get a ro manager
    # postgresql://<username>:<password>@localhost[:port]/[name]
    postgres_url = (
        f"postgresql://{args.username}:{args.password}"
        f"@{args.hostname}:{args.port}/indradb_readonly_local"
    )
    ro_manager = ReadonlyDatabaseManager(postgres_url)

    # Create the tables
    create_ro_tables(ro_manager, postgres_url)

    # For each table, run the function that will fill out the table with data
    # todo: add all tables filling functions
