The Principal Database Client
=============================

This is the set of client tools to access the most-nearly ground truth
knowledge stored on the principal database.


Access Readings and Text Content (:py:mod:`indra_db.client.principal.content`)
------------------------------------------------------------------------------

This defines a simple API to access the content that we store on the database
for external purposes.

.. automodule:: indra_db.client.principal.content
   :members:


Submit and Retrieve Curations (:py:mod:`indra_db.client.principal.curation`)
----------------------------------------------------------------------------

On our services, users have the ability to curate the results we present,
indicating whether they are correct or not, and how they may be incorrect. The
API for adding and retrieving that input is defined here.

.. automodule:: indra_db.client.principal.curation
   :members:


Get Raw Statements (:py:mod:`indra_db.client.principal.raw_statements`)
-----------------------------------------------------------------------

Get the raw, uncleaned and un-merged Statements based on agent and type or by
paper(s) of origin.

.. automodule:: indra_db.client.principal.raw_statements
   :members:
