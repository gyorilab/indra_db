Some Miscellaneous Modules
==========================

Here are some modules and files that live on their own, and don't fit neatly
into other categories.


Low Level Database Interface (:py:mod:`indra_db.databases`)
-----------------------------------------------------------

The Database Manager classes are the lowest level interface with the database,
implemented with SQLAlchemy, providing useful short-cuts but also allowing full
access to SQLAlchemy's API.

.. automodule:: indra_db.databases
   :members:
   :member-order: bysource


Belief Calculator (:py:mod:`indra_db.belief`)
---------------------------------------------

The belief in the knowledge of a Statement is a measure of our confidence that
the Statement is an accurate representation of the text, _NOT_ our confidence
in the validity of what was in that text. Given the size of the content in the
database, some special care is needed when calculating this value, which
depends heavily on the support relations between pre-assembled Statements.

.. automodule:: indra_db.belief
   :members:
