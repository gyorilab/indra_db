import codecs
import difflib
import json
import os
import re
import shutil
import subprocess
from urllib.parse import urlparse

from indra.statements import Statement
from indra.statements.validate import assert_valid_statement_semantics
from indra_db.readonly_dumping.locations import *


class StatementJSONDecodeError(Exception):
    pass


def load_statement_json(
        json_str: str,
        attempt: int = 1,
        max_attempts: int = 5,
        remove_evidence: bool = False
):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        if attempt < max_attempts:
            json_str = codecs.escape_decode(json_str)[0].decode()
            sj = load_statement_json(
                json_str, attempt=attempt + 1, max_attempts=max_attempts
            )
            if remove_evidence:
                sj["evidence"] = []
            return sj
    raise StatementJSONDecodeError(
        f"Could not decode statement JSON after " f"{attempt} attempts: {json_str}"
    )


class UnicodeEscapeError(Exception):
    """Exception raised when unicode escapes cannot be removed from a string"""


def clean_escaped_stmt_json_string(stmt_json_str: str) -> str:
    """Clean up escaped characters in a statement JSON string.

    Parameters
    ----------
    stmt_json_str :
        The JSON string to clean up.

    Returns
    -------
    :
        The cleaned up JSON string.

    """
    # Replace escaped backslashes with unescaped backslashes
    stmt_json_str = stmt_json_str.replace('\\\\', '\\')
    return stmt_json_str


def clean_json_loads(stmt_json_str: str, remove_evidence: bool = False):
    """Clean up escaped characters in a statement JSON string and load it.

    Parameters
    ----------
    stmt_json_str :
        The JSON string to clean up and load.
    remove_evidence :
        If True, remove the evidence from the returned statement JSON.

    Returns
    -------
    :
        The loaded JSON object.
    """
    # The logic in this function comes from looking at two aspects of
    # de-serializing the raw statement json string dumped from the principal
    # database:
    # 1. Can the loaded statement reproduce the original matches hash of the
    #    raw statement json with stmt.get_hash(refresh=True) after being
    #    initialized via `indra.statements.io.stmt_from_json`?
    # 2. Does json.loads error?
    # Denoting a matching hash as T or F for matching or not, and an error
    # as 'error' the following table is observed:
    #
    # | # | json.loads       | cleanup + json.loads | pick                 |
    # |   | > stmt_from_json | > stmt_from_json     |                      |
    # |---|------------------|----------------------|----------------------|
    # | 1 | T                | T                    | cleanup + json.loads |
    # | 2 | F                | T                    | cleanup + json.loads |
    # | 3 | error            | T                    | cleanup + json.loads |
    # | 4 | T                | error                | json.loads           |
    #
    # This means the json string has to be loaded twice, once without
    # cleanup and once with cleanup, to check both conditions before
    # returning the correct json object.
    #
    # NOTE: F | F is also possible, and has happened in a few cases (<100 out
    # of >75 M raw statements). On inspection, none of these had any escaped
    # characters in the json string, so the reason for the mismatch with the
    # matches hash is unknown, but is at least not related to the issue of
    # doubly escaped characters which this function is meant to address.
    # All other combinations of T, F and error have not been observed.
    if not stmt_json_str:
        raise ValueError("Empty json string")

    # Try clean+load first. If there is no error (this is the vast majority
    # of cases), return the cleaned json (case 1, 2 and 3 above). Otherwise,
    # return the uncleaned json (case 4 above).

    # Cleaned load
    try:
        cleaned_str = clean_escaped_stmt_json_string(stmt_json_str)
        stmt_json = json.loads(cleaned_str)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Uncleaned load
        try:
            stmt_json = json.loads(stmt_json_str)
        except Exception as err:
            raise UnicodeEscapeError(
                f"Could not load statement json string:{err}"
            ) from err

    if remove_evidence:
        stmt_json["evidence"] = []
    return stmt_json


def validate_statement_semantics(stmt: Statement) -> bool:
    """Validate the semantics of a statement.

    Parameters
    ----------
    stmt :
        The statement to validate.

    Returns
    -------
    :
        True if the statement is semantically valid, False otherwise.
    """
    try:
        assert_valid_statement_semantics(stmt)
        return True
    except ValueError:
        return False



def generate_db_snapshot(postgres_url, output_file):
    try:
        parsed_url = urlparse(postgres_url)
        username = parsed_url.username
        password = parsed_url.password
        host = parsed_url.hostname
        port = parsed_url.port or 5432
        database_name = parsed_url.path.lstrip('/')

        if password:
            os.environ['PGPASSWORD'] = password

        command = [
            'pg_dump',
            '-U', username,
            '-h', host,
            '-p', str(port),
            '-d', database_name,
            '-F', 'p',  # 'p' for plain text format
            '-s',
            '-v',
            '-f', output_file
        ]

        subprocess.run(command, check=True)
        print(f"Snapshot saved to {output_file}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

def extract_tables_and_columns(snapshot_path):
    with open(snapshot_path, 'r') as file:
        lines = file.readlines()

    tables = {}
    current_table = None

    for line in lines:
        # Check for the start of a table definition
        table_match = re.match(r'CREATE TABLE ([\w\.]+) \(', line)
        if table_match:
            current_table = table_match.group(1)
            tables[current_table] = []
            continue

        # Check for the end of a table definition
        if line.startswith(");"):
            current_table = None
            continue

        # If we are inside a table definition, capture column names and data types
        if current_table and re.match(r'\s+\w+', line):
            column_definition = line.strip().rstrip(',')
            tables[current_table].append(column_definition)

    return tables

def compare_snapshots(snapshot1, snapshot2):
    """Check if new database snapshot and current database snapshot are identical

    Parameters
    ----------
    snapshot1 :
        The current database snapshot file from the AWS
    snapshot2:
        The generated new database snapshot file


    Returns
    -------
    :
        True if the two database are conformed
    """
    print("Comparing snapshots")
    tables1 = extract_tables_and_columns(snapshot1)
    tables2 = extract_tables_and_columns(snapshot2)
    print(len(tables1), tables1.keys())
    print(len(tables2), tables2.keys())

    # Compare the columns in each table
    for table in tables1:
        if tables1[table] != tables2[table]:
            diff = difflib.unified_diff(
                tables1[table], tables2[table],
                fromfile='snapshot1',
                tofile='snapshot2',
                lineterm=''
            )
            for line in diff:
                if " json" in line and line.startswith("- "):
                    return False
    # FIXME: Need to determine in which extend the snapshots need match(e.g integer type);
    print("All tables and columns match.")
    return True


def record_time(file, time, text='', mode='w'):
    with open(file, mode) as f:
        f.write(f"{text}, {time:.4f}\n")

def pipeline_files_clean_up():
    file_variable_names = [
        refinements_fpath,
        belief_scores_pkl_fpath,
        stmt_hash_to_raw_stmt_ids_knowledgebases_fpath,
        source_counts_knowledgebases_fpath,
        raw_id_info_map_knowledgebases_fpath,
        unique_stmts_fpath,
        stmt_hash_to_raw_stmt_ids_fpath,
        raw_id_info_map_fpath,
        source_counts_fpath,
        processed_stmts_fpath,
        stmt_hash_to_raw_stmt_ids_reading_fpath,
        source_counts_reading_fpath,
        processed_stmts_reading_fpath,
        raw_id_info_map_reading_fpath,
        reading_to_text_ref_map_fpath,
        drop_readings_fpath,
        text_refs_fpath,
        reading_text_content_fpath,
        raw_statements_fpath
    ]
    for f in file_variable_names:
        if os.path.exists(f.absolute().as_posix()):
            os.remove(f.absolute().as_posix())
        else:
            print(f"{f.absolute().as_posix()} does not exist.")

    if os.path.exists(split_unique_statements_folder_fpath.absolute().as_posix()):
        shutil.rmtree(split_unique_statements_folder_fpath.absolute().as_posix())
    else:
        print(f"{split_unique_statements_folder_fpath.absolute().as_posix()} "
              f"does not exist.")

