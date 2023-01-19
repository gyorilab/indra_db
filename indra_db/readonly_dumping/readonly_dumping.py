import argparse
import csv
import gzip
import logging
import os
import pickle
import subprocess
import uuid
from textwrap import dedent

from sqlalchemy import create_engine

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
