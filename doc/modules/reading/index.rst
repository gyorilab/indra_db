Database Integrated Reading Tools
=================================

Here are defined the procedures for reading content on the database, stashing
the reading outputs, and producing statements from the readings, and inserting
those raw statements into the database.

The Database Readers (:py:mod:`indra_db.reading.read_db`)
---------------------------------------------------------

A reader is defined as a python class which implements the machinery needed to
process the text content we store, read it, and extract Statements from the
reading results, storing the readings along the way. The reader must conform
to a standard interface, which then allows readers to be run in a plug-and-play
manner.

.. automodule:: indra_db.reading.read_db
    :members:


The Database Script for Running on AWS (:py:mod:`indra_db.reading.read_db_aws`)
-------------------------------------------------------------------------------

This is the script used to run reading on AWS Batch, generally run from an
AWS Lambda function.

.. automodule:: indra_db.reading.read_db_aws
    :members:

The Database Reporter (:py:mod:`indra_db.reading.report_db_aws`)
----------------------------------------------------------------

Create an object that is used to aggregate and report on the reading process,
allowing for effective monitoring.

.. automodule:: indra_db.reading.report_db_aws
    :members:

A Class to Manage and Monitor AWS Batch Jobs (:py:mod:`indra_db.reading.submit_reading_pipeline`)
-------------------------------------------------------------------------------------------------

Allow a manager to monitor the Batch jobs to prevent runaway jobs, and smooth
out job runs and submissions.

.. automodule:: indra_db.reading.submit_reading_pipeline
    :members:

