from os import environ
from flask_jwt_extended import jwt_optional

TITLE = "The INDRA Database"
DEPLOYMENT = environ.get('INDRA_DB_API_DEPLOYMENT')
VUE_ROOT = environ.get('INDRA_DB_API_VUE_ROOT')
if VUE_ROOT is not None and VUE_ROOT.endswith('/'):
    # Peal off the trailing slash.
    VUE_ROOT = VUE_ROOT[:-1]
MAX_STMTS = 500
MAX_LIST_LEN = 2000
REDACT_MESSAGE = '[MISSING/INVALID CREDENTIALS: limited to 200 char for Elsevier]'

TESTING = {}
if environ.get('TESTING_DB_APP') == '1':
    TESTING['status'] = True
else:
    TESTING['status'] = False


def jwt_nontest_optional(func):
    if TESTING['status']:
        return func
    else:
        return jwt_optional(func)