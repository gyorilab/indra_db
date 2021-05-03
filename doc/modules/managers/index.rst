Pipeline Managers
=================

This module contains the pipelines used to update content and knowledge in the
database, and move or transform that knowledge on a regular basis.

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


Knowledge Base Manager (:py:mod:`indra_db.managers.knowledgebase_manager`)
--------------------------------------------------------------------------

The INDRA Databases also derives much of its knowledge from external databases
and other resources not extracted from plain text, referred to in this repo as
"knowledge bases", so as to avoid the ambiguity of "database". This manager
handles the updates of those knowledge bases, each of which requires different
handling.

.. automodule:: indra_db.managers.knowledgebase_manager
   :members:


Static Dump Manager (:py:mod:`indra_db.managers.dump_manager`)
--------------------------------------------------------------

This handles the generation of static dumps, including the readonly database 
from the principal database.

.. automodule:: indra_db.managers.dump_manager
   :members:
   :member-order: bysource
