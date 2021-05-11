Database Integrated Preassembly Tools
=====================================

The database runs incremental preassembly on the raw statements to generate
the preassembled (PA) Statements. The code to accomplish this task is defined
here, principally in :class:`DbPreassembler
<indra_db.preassembly.preassemble_db.DbPreassembler>`. This module also
defines proceedures for running these jobs on AWS.

Database Preassembly (:py:mod:`indra_db.preassembly.preassemble_db`)
--------------------------------------------------------------------

This module defines a class that manages preassembly for a given list of
statement types on the local machine.

.. automodule:: indra_db.preassembly.preassemble_db
   :members:
   :member-order: bysource


A Class to Manage and Monitor AWS Batch Jobs (:py:mod:`indra_db.preassembly.submitter`)
---------------------------------------------------------------------------------------

Allow a manager to monitor the Batch jobs to prevent runaway jobs, and smooth
out job runs and submissions.

.. automodule:: indra_db.preassembly.submitter
   :members:
   :member-order: bysource

