__all__ = [
    "TITLE",
    "DEPLOYMENT",
    "BASE_URL",
    "VUE_ROOT",
    "MAX_STMTS",
    "MAX_LIST_LEN",
    "REDACT_MESSAGE",
    "TESTING",
    "jwt_nontest_optional",
    "CURATOR_SALT",
]

from os import environ
from pathlib import Path
from flask_jwt_extended import jwt_required

TITLE = "The INDRA Database"
DEPLOYMENT = environ.get("INDRA_DB_API_DEPLOYMENT")
BASE_URL = environ.get("INDRA_DB_API_BASE_URL")
CURATOR_SALT = environ.get("INDRA_DB_API_CURATOR_SALT")
VUE_ROOT = environ.get("INDRA_DB_API_VUE_ROOT")
if VUE_ROOT is not None and not VUE_ROOT.startswith("http"):
    VUE_ROOT = Path(VUE_ROOT).expanduser()
    if not VUE_ROOT.is_absolute():
        VUE_ROOT = Path(__file__).parent.absolute() / VUE_ROOT
MAX_STMTS = 500
MAX_LIST_LEN = 2000
REDACT_MESSAGE = "[MISSING/INVALID CREDENTIALS: limited to 200 char for Elsevier]"

TESTING = {}
if environ.get("TESTING_DB_APP") == "1":
    TESTING["status"] = True
else:
    TESTING["status"] = False
TESTING["deployment"] = ""
TESTING["vue-root"] = ""


def jwt_nontest_optional(func):
    if TESTING["status"]:
        return func
    else:
        return jwt_required(optional=True)(func)
