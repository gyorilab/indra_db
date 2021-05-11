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


Construct composable queries (:py:mod:`indra_db.client.readonly.query`)
-------------------------------------------------------------------------------

This is a sophisticated system of classes that can be used to form queires
for preassembled statements from the readonly database.

.. automodule:: indra_db.client.readonly.query
   :members:
   :member-order: bysource

