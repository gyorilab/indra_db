Pipeline Management CLI
=======================

This module creates a CLI for managing the pipelines used to update
content and knowledge in the database, and move or transform that
knowledge on a regular basis.

.. click:: indra_db.cli:main
   :prog: indra-db
   :nested: full


Pipeline CLI Implementations
============================

Content (:py:mod:`indra_db.cli.content`)
----------------------------------------

The Content CLI manages the text content that is
stored in the database. A parent class is defined, and managers for different
sources (e.g. PubMed) can be defined by inheriting from this parent. This file
is also used as the shell command to run updates of the content.

.. automodule:: indra_db.cli.content
   :members:
   :member-order: bysource


Reading (:py:mod:`indra_db.cli.reading`)
----------------------------------------

The Reading CLI handles the reading of the text contend and the processing
of those readings into statements. As with Content CLI, different reading
pipelines can be handled by defining children of a parent class.

.. automodule:: indra_db.cli.reading
   :members:
   :member-order: bysource


PreAssembly (:py:mod:`indra_db.cli.preassembly`)
------------------------------------------------

The Preassembly CLI manages the preassembly pipeline, running deploying
preassembly jobs to Batch.

.. automodule:: indra_db.cli.preassembly
   :members:
   :member-order: bysource


Knowledge Bases (:py:mod:`indra_db.cli.knowledgebase`)
------------------------------------------------------

The INDRA Databases also derives much of its knowledge from external databases
and other resources not extracted from plain text, referred to in this repo as
"knowledge bases", so as to avoid the ambiguity of "database". This CLI
handles the updates of those knowledge bases, each of which requires different
handling.

.. automodule:: indra_db.cli.knowledgebase
   :members:
   :member-order: bysource


Static Dumps (:py:mod:`indra_db.cli.dump`)
------------------------------------------

This handles the generation of static dumps, including the readonly database 
from the principal database.

.. automodule:: indra_db.cli.dump
   :members:
   :member-order: bysource
