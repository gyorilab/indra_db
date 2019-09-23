Miscellaneous Client APIs (Mostly Deprecated)
=============================================

There are some, generally archaic, client functions which use both readonly
and principal resources. I make no guarantee that these will work.

Get Datasets (:py:mod:`indra_db.client.datasets`)
-------------------------------------------------

An early attempt at something very like the :py:mod:`indra_db.client.readonly.interactions`
approach to getting superficial data out of the database.

.. automodule:: indra_db.client.datasets
   :members:


Get Statements (:py:mod:`indra_db.client.statements`)
-----------------------------------------------------

The first round of tools written to get Statements out of the database,
utilizing far too many queries and taking absurdly long to complete. Most of
their functions have been outmoded, with the exception of getting PA Statements
from the principal database, which (as of this writing) has yet to be
implemented.

.. automodule:: indra_db.client.statements
   :members:
