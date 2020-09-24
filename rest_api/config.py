from os import environ

TITLE = "The INDRA Database"
DEPLOYMENT = environ.get('INDRA_DB_API_DEPLOYMENT')
MAX_STMTS = int(0.5e3)
REDACT_MESSAGE = '[MISSING/INVALID CREDENTIALS: limited to 200 char for Elsevier]'
