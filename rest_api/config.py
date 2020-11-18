from os import environ
from flask_jwt_extended import jwt_optional

TITLE = "The INDRA Database"
DEPLOYMENT = environ.get('INDRA_DB_API_DEPLOYMENT')
VUE_ROOT = environ.get('INDRA_DB_API_VUE')
MAX_STMTS = int(0.5e3)
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