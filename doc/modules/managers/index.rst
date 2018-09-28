These are the core tools that are used to manage the content in the database
and perform updates and operations on/with that content.

.. automodule:: indra_db.managers
    :members:

Database Manager (:py:mod:`indra_db.managers.database_manager`)
---------------------------------------------------------------

The Database Manager object is the core interface with the database,
implemented with SQLAlchemy, providing useful short-cuts but also
allowing full access to SQLAlchemy's API.

.. automodule:: indra_db.managers.database_manager
    :members:

Content Manager (:py:mod:`indra_db.managers.content_manager`)
-------------------------------------------------------------

The Content Managers, as the name suggests, manage the text content that is
stored in the database. A parent class is defined, and managers for different
sources (e.g. PubMed) can be defined by inheriting from this parent. This file
is also used as the shell command to run updates of the content.

.. automodule:: indra_db.managers.content_manager
    :members:

Reading Manager (:py:mod:`indra_db.managers.reading_manager`)
-------------------------------------------------------------

The Reading Managers handle the reading of the text contend and the processing
of those readings into statements. As with Content Managers, different reading
pipelines can be handled by defining children of a parent class.

.. automodule:: indra_db.managers.reading_manager
    :members:


PreAssembly Manager (:py:mod:`indra_db.managers.preassembly_manager`)
---------------------------------------------------------------------

The Preassembly Manager performs the complex process of preassembly, where
after cleaning the raw extractions and fixing groundings and sites, the
numerous duplicate Statements extracted from databases and literature are
distilled into a corpus of unique statements, with links back to the raw
Statements, and their history and provenance.

.. automodule:: indra_db.managers.preassembly_manager
    :members:
