The Readonly Client
===================

Here are our primary tools intended for retrieving Statements, in particular
Pre-Assembled (PA) Statements, from the readonly database. This is some of the
most heavily optimized access code in the repo, and is the backbone of most
external or outward facing applications.

The readonly database, as the name suggests, is designed to take only read
requests, and is updated via dump only once a week. This allows users of
our database to access it even as we perform daily updates on the principal
database, without worrying about queries interfering.


Get Pre-Assembled Statements (:py:mod:`indra_db.client.readonly.pa_statements`)
-------------------------------------------------------------------------------

Here are the tools used to get PA Statements from the readonly database, with
the goal of retrieving at least 1,000 Statements with 10 evidence each in under
30 seconds.

.. automodule:: indra_db.client.readonly.pa_statements
   :members:


Get Simple Interactions from Metadata (:py:mod:`indra_db.client.readonly.interactions`)
---------------------------------------------------------------------------------------

This provides an API to get somewhat less detailed data than above, using just
the metadata of the database (not looking into the Statement JSONs), but is
much faster. These tools can be sufficient if, for example, all that is needed
is an interactome.

.. automodule::indra_db.client.readonly.interactions
   :memebrs:
