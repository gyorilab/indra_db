Database Schemas
================

Here are defined the schemas for the principal and readonly databases, as well
as some useful mixin classes.

Principal Database Schema (:py:mod:`indra_db.schemas.principal_schema`)
-----------------------------------------------------------------------

Defines the `get_schema` function for the principal database, which represents
the "ground truth" of the knowledge we aggregate.

.. automodule:: indra_db.schemas.principal_schema
   :members:

Readonly Database Schema (:py:mod:`indra_db.schemas.readonly_schema`)
---------------------------------------------------------------------

Defines the `get_schema` function for the readonly database, which is used by
external services to access the Statement knowledge we acquire.

.. automodule:: indra_db.schemas.readonly_schema
   :members:

Class Mix-ins (:py:mod:`indra_db.schemas.mixins`)
-------------------------------------------------

This defines class mixins that are used to add general features to SQLAlchemy
table objects via multiple inheritance.

.. automodule:: indra_db.schemas.mixins
   :members:

Indexes (:py:mod:`indra_db.schemas.indexes`)
--------------------------------------------

This defines the classes needed to create and maintain indices in the database,
the other part of the infrastructure of which is included in the `IndraDBTable`
class mixin definition.

.. automodule:: indra_db.schemas.indexes
   :members:
