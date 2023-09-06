import codecs
import json


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
