Utilities
=========

Here live the more mundane and backend utilities used throughout other modules
of the codebase, and potentially elsewhere, although they are not intended for
external use in general. Several more-or-less bespoke scripts are also stored
here.


Database Session Constructors (:py:mod:`indra_db.util.constructors`)
--------------------------------------------------------------------

Constructors to get interfaces to the different databases, selecting among
the various physical instances defined in the config file.

.. automodule:: indra_db.util.constructors
   :members:


Scripts to Get Content (:py:mod:`indra_db.util.content_scripts`)
----------------------------------------------------------------

General scripts for getting content by various IDs.

.. automodule:: indra_db.util.content_scripts
   :members:


Distilling Raw Statements (:py:mod:`indra_db.util.distill_statements`)
----------------------------------------------------------------------

Do some pre-pre-assembly cleansing of the raw Statements to account for various
kinds of duplicity that are artifacts of our content collection and reading
pipelines rather than representing actually duplicated knowledge in the
literature.

.. automodule:: indra_db.util.distill_statements
   :members:


Script to Create a SIF Dump (:py:mod:`indra_db.util.dump_sif`)
--------------------------------------------------------------

Create an interactome from metadata in the database and dump the results as a
sif file.

.. automodule:: indra_db.util.dump_sif
   :members:


General Helper Functions (:py:mod:`indra_db.util.helpers`)
----------------------------------------------------------

Functions with broad utility throughout the repository, but otherwise
miscellaneous.

.. automodule:: indra_db.util.helpers
   :members:


Routines for Inserting Statements and Content (:py:mod:`indra_db.util.insert`)
------------------------------------------------------------------------------

Inserting content into the database can be a rather involved process, but here
are defined high-level utilities to uniformly accomplish the task.

.. automodule:: indra_db.util.insert
   :members:
