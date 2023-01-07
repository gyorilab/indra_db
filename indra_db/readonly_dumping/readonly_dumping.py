import argparse
import csv
import logging
import os
import pickle
import subprocess
from textwrap import dedent

from sqlalchemy import create_engine

from indra_db.config import get_databases
from indra_db.databases import ReadonlyDatabaseManager
from indra_db.schemas.mixins import ReadonlyTable

from .locations import *

logger = logging.getLogger(__name__)

LOCAL_RO_PASSWORD = os.environ["LOCAL_RO_PASSWORD"]
LOCAL_RO_USER = os.environ["LOCAL_RO_USER"]
LOCAL_RO_PORT = int(os.environ.get("LOCAL_RO_PORT", "5432"))
LOCAL_RO_DB_NAME = os.environ.get("LOCAL_RO_DB_NAME", "indradb_readonly_local")


# Tables to create (currently in no particular order):

# Belief
def belief():
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

    os.remove(belief_scores_tsv_fpath.absolute().as_posix())


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


# ReadingRefLink
def reading_ref_link(ro_mngr: ReadonlyDatabaseManager,
                     reading_ref_link_table: ReadonlyTable):
    """Fill the reading ref link table with data
    depends on: text_ref, text_content, reading
    requires assembly: False
    Table definition:

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
    dump_file = TEMPDIR.joinpath("reading_ref_link.tsv").absolute().as_posix()

    sql = dedent(
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
    principal_query_to_csv(query=sql, output_location=dump_file)

    # todo: if you want to switch to gzipped files you can do
    #  "copy table_name from program 'zcat /tmp/tp.csv.gz';"
    load_data_file_into_local_ro(table_name="readonly.reading_ref_link",
                                 column_order=column_order,
                                 tsv_file=dump_file)

    # Build the index
    reading_ref_link_table.build_indices(ro_mngr)

    # Delete the dump file after upload
    os.remove(dump_file)


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
