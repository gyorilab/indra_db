"""This module contains tools designed to access content in the db.

Specifically, this is for direct access to the database, not through the web
api.

All the functions defined require direct access to the database, which is in
general restricted. For broad access, see the indra_db_rest api client in
INDRA.

There are two key ways of accessing statements from the INDRA Database:
directly and using the materialize views. Only the `get_statement_jsons`
functions are limited to using the views. Most other functions access the
primary tables of the database and are generally slower. The
`get_statement_jsons` functions are the most heavily optimized for fast
recall, as they are the back-end to the REST API.
"""

from .datasets import *
from .readonly import *
from .principal import *
