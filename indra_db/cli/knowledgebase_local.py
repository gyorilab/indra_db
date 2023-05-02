import codecs
import csv
import gzip
import json
import logging

from tqdm import tqdm

from indra.sources.signor import process_from_web


logger = logging.getLogger(__name__)


class StatementJSONDecodeError(Exception):
    pass


def load_statement_json(
        json_str: str, attempt: int = 1, max_attempts: int = 5
):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        if attempt < max_attempts:
            json_str = codecs.escape_decode(json_str)[0].decode()
            return load_statement_json(
                json_str, attempt=attempt + 1, max_attempts=max_attempts
            )
    raise StatementJSONDecodeError(
        f"Could not decode statement JSON after " f"{attempt} attempts: {json_str}"
    )


def meta_filter(*filter_funs):
    def _meta_filter(stmt_json):
        for filter_fun in filter_funs:
            if not filter_fun(stmt_json):
                return False
        return True
    return _meta_filter


def filter_old_signor(raw_stmt_json):
    """Return False if stmt is from signor"""
    return raw_stmt_json["evidence"][0]["source_api"] != "signor"


def update_signor(csv_writer):
    """Update the signor knowledgebase with new statements"""
    # Todo: what is the correct way to provide missing values?
    null = "\\N"

    # Load/yield the new statements from the signor processor
    logger.info("Loading new statements from signor processor")
    sp = process_from_web()

    # Append the new statements to the old statements
    # Start at start_index if given, otherwise use None to append?
    # Columns: index (should be set when inserted), db_info_id == 3,
    # reading_id (missing), raw_stmt_json_str
    rows = (
        (null, 3, null, json.dumps(stmt_json)) for stmt_json in sp.statements
    )
    logger.info(f"Writing {len(sp.statements)} new statements from signor")
    csv_writer.writerows(rows=rows)


def update(raw_stmts_path: str, new_stmts_path: str):
    """Update the knowledgebases with new statements

    Parameters
    ----------
    raw_stmts_path :
        Path to the raw statements file
    new_stmts_path :
        Path to the new raw statements file
    """

    # Add more filters as they are implemented
    filters = meta_filter(filter_old_signor)

    with gzip.open(new_stmts_path, "wt") as f_out, \
            gzip.open(raw_stmts_path, "rt") as f_in:
        reader = csv.reader(f_in, delimiter="\t")
        writer = csv.writer(f_out, delimiter="\t")
        for raw_stmt_id, db_info_id, reading_id, rsjs in tqdm(
                reader, total=75816146, desc="Updating knowledgebases"
        ):
            raw_stmt_json = load_statement_json(rsjs)
            if filters(raw_stmt_json):
                writer.writerow([raw_stmt_id, db_info_id, reading_id, rsjs])

        update_signor(writer)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "raw_stmts_path",
        help="Path to the raw statements file",
        default="raw_statements.tsv.gz",
    )
    parser.add_argument(
        "new_stmts_path",
        help="Path to the new raw statements file",
        default="new_raw_statements.tsv.gz",
    )
    args = parser.parse_args()

    update(args.raw_stmts_path, args.new_stmts_path)
