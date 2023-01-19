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
