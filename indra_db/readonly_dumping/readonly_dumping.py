import argparse
import csv
import gzip
import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import defaultdict, Counter
from hashlib import md5
from pathlib import Path
from textwrap import dedent
from typing import Tuple, Iterable

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from lxml import etree
from pyspark.sql.types import StructType, IntegerType, StructField, LongType, FloatType, StringType, ShortType, \
    BooleanType, BinaryType
from tqdm import tqdm
from pyspark.sql import SparkSession
from indra.literature.pubmed_client import _get_annotations
from indra.statements import stmt_from_json, ActiveForm
from indra.util.statement_presentation import db_sources, reader_sources
from indra_db import get_db
from indra_db.config import get_databases
from indra_db.databases import ReadonlyDatabaseManager
from indra_db.schemas.mixins import ReadonlyTable
from indra_db.schemas.readonly_schema import (
    ro_type_map,
    ro_role_map,
    SOURCE_GROUPS
)
from sqlalchemy import create_engine
from pyspark.sql.functions import to_json, col
from .locations import *
from .util import clean_json_loads, generate_db_snapshot, compare_snapshots, \
    pipeline_files_clean_up

logger = logging.getLogger("indra_db.readonly_dumping.export_assembly")
logger.setLevel(logging.DEBUG)
logger.propagate = False

file_handler = logging.FileHandler(pipeline_log_fpath.absolute().as_posix(), mode='a')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

SQL_NULL = "\\N"
LOCAL_RO_DB_NAME = os.environ["LOCAL_RO_DB_NAME"]
LOCAL_RO_PASSWORD = os.environ["LOCAL_RO_PASSWORD"]
LOCAL_RO_USER = os.environ["LOCAL_RO_USER"]

RUN_ORDER = [
    "belief",
    "raw_stmt_src",
    "reading_ref_link",
    "evidence_counts",
    "pa_agent_counts",
    "mesh_concept_ref_counts",
    "mesh_term_ref_counts",
    "name_meta",
    "text_meta",
    "other_meta",
    "source_meta",
    "agent_interactions",
    "fast_raw_pa_link",
    "raw_stmt_mesh_concepts",
    "raw_stmt_mesh_terms",
    "mesh_concept_meta",
    "mesh_term_meta",
]


def create_primary_key(ro_mngr_local: ReadonlyDatabaseManager,
                       table_name: str,
                       keys):
    if isinstance(keys, str):
        keys = [keys]
    keys_str = ", ".join(keys)
    sql = f"ALTER TABLE readonly.{table_name} \
    ADD CONSTRAINT {table_name}_pkey PRIMARY KEY ({keys_str});"
    ro_mngr_local.execute(sql)


def table_has_content(
        ro_mngr_local: ReadonlyDatabaseManager,
        table_name: str,
        count: int = 0,
) -> bool:
    """Check that the table is not empty

    Parameters
    ----------
    ro_mngr_local :
        The local readonly database manager
    table_name :
        The name of the table to check
    count :
        The number of rows above which the table is considered to have content.
        Defaults to 0.

    Returns
    -------
    :
        True if the table has content, False otherwise
    """
    res = ro_mngr_local.execute(
        f"SELECT COUNT(*) FROM readonly.{table_name}"
    )
    return res.fetchone()[0] > count


def get_stmt_hash_mesh_pmid_counts():
    # Check the existence of the output files
    if not all(fp.exists() for fp in [pmid_mesh_term_counts_fpath,
                                      mk_hash_pmid_sets_fpath,
                                      mesh_term_ref_counts_fpath]):

        # Check the existence of the input files
        if not (pmid_mesh_map_fpath.exists() and pmid_stmt_hash_fpath.exists()):
            ensure_pubmed_mesh_data()

        # dump pmid-mesh from mti_ref_annotations_test for concept and term
        # separately
        mti_mesh_query = dedent("""
        SELECT pmid_num, mesh_num, is_concept 
        FROM mti_ref_annotations_test""")
        principal_query_to_csv(
            query=mti_mesh_query, output_location=pmid_mesh_mti_fpath
        )

        # Load the pmid-mesh map
        pmid_mesh_mapping = pickle.load(pmid_mesh_map_fpath.open("rb"))

        # Load the pmid-mesh map from the mti_ref_annotations_test table and
        # merge it with the existing map
        with pmid_mesh_mti_fpath.open("r") as fh_in:
            reader = csv.reader(fh_in, delimiter="\t")
            for pmid, mesh_num, is_concept in reader:
                _type = "concept" if is_concept == "t" else "term"
                if pmid in pmid_mesh_mapping:
                    if _type in pmid_mesh_mapping[pmid]:
                        pmid_mesh_mapping[pmid][_type].add(mesh_num)
                    else:
                        pmid_mesh_mapping[pmid][_type] = {mesh_num}
                else:
                    pmid_mesh_mapping[pmid] = {_type: {mesh_num}}

        # Load the pmid-stmt_hash map
        pmid_stmt_hashes = pickle.load(pmid_stmt_hash_fpath.open("rb"))

        # Dicts for ref_count (count of PMIDs per mk_hash, mesh_num pair)
        pmid_mesh_concept_counts = Counter()
        pmid_mesh_term_counts = Counter()

        # Dict for hash -> pmid set (distinct pmids associated with a hash,
        # i.e. from multiple sources)
        mk_hash_pmid_sets = defaultdict(set)

        # Get pmid_count and ref_count (for concept and term separately)
        for pmid_num, mk_hash_set in tqdm(pmid_stmt_hashes.items(),
                                          desc="Getting counts"):
            for mk_hash in mk_hash_set:
                mk_hash_pmid_sets[mk_hash].add(pmid_num)
                if pmid_num in pmid_mesh_mapping:
                    # For concepts
                    mesh_concepts = pmid_mesh_mapping[pmid_num].get("concept", set())
                    for mesh_concept in mesh_concepts:
                        pmid_mesh_concept_counts[(mk_hash, mesh_concept)] += 1
                    # For terms
                    mesh_terms = pmid_mesh_mapping[pmid_num].get("term", set())
                    for mesh_term in mesh_terms:
                        pmid_mesh_term_counts[(mk_hash, mesh_term)] += 1

        # Cache the results
        pickle.dump(pmid_mesh_concept_counts,
                    pmid_mesh_concept_counts_fpath.open("wb"))
        pickle.dump(pmid_mesh_term_counts,
                    pmid_mesh_term_counts_fpath.open("wb"))
        pickle.dump(mk_hash_pmid_sets, mk_hash_pmid_sets_fpath.open("wb"))
    else:
        # Load the cached results
        pmid_mesh_concept_counts = pickle.load(
            pmid_mesh_concept_counts_fpath.open("rb")
        )
        pmid_mesh_term_counts = pickle.load(
            pmid_mesh_term_counts_fpath.open("rb")
        )
        mk_hash_pmid_sets = pickle.load(mk_hash_pmid_sets_fpath.open("rb"))

    return pmid_mesh_concept_counts, pmid_mesh_term_counts, mk_hash_pmid_sets


# MeshTerm/ConceptRefCounts
# These two tables enumerate stmt_hash, mesh_num, ref_count, pmid_count
# for each stmt_hash, mesh_num pair.
# - The **ref_count** is the number of distinct PMIDs that support the
#   stmt_hash, mesh_num pair.
# - The **pmid_count** is the number of distinct PMIDs that support the
#   stmt_hash.
# The PaRefLink table is not generated, but the equivalent contents is
# generated here to produce the input to the MeshTerm/ConceptRefCounts tables.
def ensure_pa_ref_link():
    # Check the existence of the output files
    if mesh_concept_ref_counts_fpath.exists() and \
            mesh_term_ref_counts_fpath.exists():
        return
    pmid_mesh_concept_counts, pmid_mesh_term_counts, mk_hash_pmid_sets = \
        get_stmt_hash_mesh_pmid_counts()

    # Dump mk_hash, mesh_num, ref_count, pmid_count for concepts and terms
    # to separate tsv files
    with mesh_term_ref_counts_fpath.open("w") as terms_fh:
        terms_writer = csv.writer(terms_fh, delimiter="\t")

        for (mk_hash, mesh_num), ref_count in tqdm(
                pmid_mesh_term_counts.items(), desc="MeshTermRefCounts"
        ):
            assert ref_count > 0  # Sanity check
            pmid_set = mk_hash_pmid_sets.get(mk_hash, set())
            if len(pmid_set) > 0:
                terms_writer.writerow(
                    (mk_hash, mesh_num, ref_count, len(pmid_set))
                )

    with mesh_concept_ref_counts_fpath.open("w") as concepts_fh:
        concepts_writer = csv.writer(concepts_fh, delimiter="\t")
        for (mk_hash, mesh_num), ref_count in tqdm(
                pmid_mesh_concept_counts.items(),
                desc="MeshConceptRefCounts",
        ):
            assert ref_count > 0  # Sanity check
            pmid_set = mk_hash_pmid_sets.get(mk_hash, set())
            if len(pmid_set) > 0:
                concepts_writer.writerow(
                    (mk_hash, mesh_num, ref_count, len(pmid_set))
                )

# MeshConceptRefCounts
def mesh_concept_ref_counts(local_ro_mngr: ReadonlyDatabaseManager):
    # Create the source tsv file
    ensure_pa_ref_link()

    schema = StructType([
        StructField("mk_hash", LongType(), True),
        StructField("mesh_num", IntegerType(), True),
        StructField("ref_count", IntegerType(), True),
        StructField("pmid_count", IntegerType(), True)
    ])
    # Load the tsv file into the local readonly db
    load_file_to_table_spark(
        table_name="readonly.mesh_concept_ref_counts",
        schema=schema,
        column_order="mk_hash, mesh_num, ref_count, pmid_count",
        tsv_file=mesh_concept_ref_counts_fpath.absolute().as_posix(),
    )
    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name='mesh_concept_ref_counts',
                       keys=['mk_hash', 'mesh_num'])
    # Build index
    mesh_concept_ref_counts_table: ReadonlyTable = local_ro_mngr.tables[
        "mesh_concept_ref_counts"]
    logger.info(
        f"Building index on {mesh_concept_ref_counts_table.full_name()}"
    )
    mesh_concept_ref_counts_table.build_indices(local_ro_mngr)


# MeshTermRefCounts
def mesh_term_ref_counts(local_ro_mngr: ReadonlyDatabaseManager):
    # Create the source tsv file
    ensure_pa_ref_link()
    schema = StructType([
        StructField("mk_hash", LongType(), True),
        StructField("mesh_num", IntegerType(), True),
        StructField("ref_count", IntegerType(), True),
        StructField("pmid_count", IntegerType(), True)
    ])
    # Load the tsv file into the local readonly db
    load_file_to_table_spark(
        table_name="readonly.mesh_term_ref_counts",
        schema=schema,
        column_order="mk_hash, mesh_num, ref_count, pmid_count",
        tsv_file=mesh_term_ref_counts_fpath.absolute().as_posix(),
    )

    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name='mesh_term_ref_counts',
                       keys=['mk_hash', 'mesh_num'])
    # Build index
    mesh_term_ref_counts_table: ReadonlyTable = local_ro_mngr.tables[
        "mesh_term_ref_counts"]
    logger.info(
        f"Building index on {mesh_term_ref_counts_table.full_name()}"
    )
    mesh_term_ref_counts_table.build_indices(local_ro_mngr)

    logger.info(f"Deleting {mesh_term_ref_counts_fpath.absolute().as_posix()}")
    os.remove(mesh_term_ref_counts_fpath.absolute().as_posix())
    logger.info(f"Deleting {mesh_concept_ref_counts_fpath.absolute().as_posix()}")
    os.remove(mesh_concept_ref_counts_fpath.absolute().as_posix())
    logger.info(f"Deleting {mk_hash_pmid_sets_fpath.absolute().as_posix()}")
    os.remove(mk_hash_pmid_sets_fpath.absolute().as_posix())
    logger.info(f"Deleting {pmid_mesh_term_counts_fpath.absolute().as_posix()}")
    os.remove(pmid_mesh_term_counts_fpath.absolute().as_posix())
    logger.info(f"Deleting {pmid_mesh_concept_counts_fpath.absolute().as_posix()}")
    os.remove(pmid_mesh_concept_counts_fpath.absolute().as_posix())


#Belief
def belief(local_ro_mngr: ReadonlyDatabaseManager):
    """Dump belief scores into the belief table on the local readonly db

    depends on: raw_statements, text_content, reading
    requires assembly: True
    assembly process: (see indra_db.readonly_dumping.export_assembly_refinement)
    """
    logger.info("Reading belief score pickle file")
    with belief_scores_pkl_fpath.open("rb") as pkl_in:
        belief_dict = pickle.load(pkl_in)
    logger.info(ro_manager.url)
    logger.info("Dumping belief scores to tsv file")
    with belief_scores_tsv_fpath.open("w") as fh_out:
        writer = csv.writer(fh_out, delimiter="\t")
        writer.writerows(((sh, bs) for sh, bs in belief_dict.items()))
    schema = StructType([
        StructField("mk_hash", LongType(), True),
        StructField("belief", FloatType(), True)
    ])

    load_file_to_table_spark(table_name="readonly.belief",
                             schema=schema,
                             column_order="mk_hash, belief",
                             tsv_file=belief_scores_tsv_fpath.absolute().as_posix())
    logger.info("Belief loaded")
    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name="belief",
                       keys="mk_hash")

    logger.info(f"Deleting {belief_scores_tsv_fpath.absolute().as_posix()}")
    os.remove(belief_scores_tsv_fpath.absolute().as_posix())

    # Build index
    belief_table: ReadonlyTable = local_ro_mngr.tables["belief"]
    logger.info(f"Building index on {belief_table.full_name()}")
    belief_table.build_indices(local_ro_mngr)


# RawStmtSrc
def raw_stmt_src(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the raw statement source table with data

    Depends on: raw_statements, reading, db_info
    Requires assembly: False

    Original SQL query to get data:

        SELECT raw_statements.id AS sid, lower(reading.reader) AS src
        FROM raw_statements, reading
        WHERE reading.id = raw_statements.reading_id
        UNION
        SELECT raw_statements.id AS sid,
        lower(db_info.db_name) AS src
        FROM raw_statements, db_info
        WHERE db_info.id = raw_statements.db_info_id
    """
    dump_file = raw_stmt_source_tsv_fpath
    columns = "sid, src"

    with gzip.open(raw_statements_fpath.as_posix(), "rt") as fh, \
            dump_file.open("w") as out_fh:
        reader = csv.reader(fh, delimiter="\t")
        writer = csv.writer(out_fh, delimiter="\t")
        for line in reader:
            stmt = clean_json_loads(line[3])
            source = stmt['evidence'][0]['source_api']
            writer.writerow([line[0], source])

    schema = StructType([
        StructField("sid", IntegerType(), True),
        StructField("src", StringType(), True)
    ])

    load_file_to_table_spark(table_name="readonly.raw_stmt_src",
                             schema=schema,
                             column_order=columns,
                             tsv_file=dump_file.absolute().as_posix())

    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name="raw_stmt_src",
                       keys="sid")
    # Delete the dump file
    logger.info(f"Deleting {dump_file.absolute().as_posix()}")
    os.remove(dump_file.absolute().as_posix())

    # Build the index
    raw_stmt_src_table: ReadonlyTable = local_ro_mngr.tables["raw_stmt_src"]
    logger.info(f"Building index on {raw_stmt_src_table.full_name()}")
    raw_stmt_src_table.build_indices(local_ro_mngr)


# PaAgentCounts
def pa_agent_counts(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the pa_agent_counts table with data"""
    # Load cache
    pa_hash_act_type_ag_count = get_activity_type_ag_count()

    # Loop entries in pa_hash count cache and write to tsv
    with pa_agents_counts_tsv.open("w") as fh_out:
        writer = csv.writer(fh_out, delimiter="\t")
        writer.writerows(
            (
                (mk_hash, agent_count)
                for mk_hash, (_, _, _, agent_count)
                in (pa_hash_act_type_ag_count.items())
            )
        )

    schema = StructType([
        StructField("mk_hash", LongType(), True),
        StructField("agent_count", IntegerType(), True)
    ])

    # Load tsv into local readonly db
    load_file_to_table_spark(
        table_name="readonly.pa_agent_counts",
        schema=schema,
        column_order="mk_hash, agent_count",
        tsv_file=pa_agents_counts_tsv.absolute().as_posix()
    )
    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name='pa_agent_counts',
                       keys='mk_hash')

    # Build index
    pa_agent_counts_table: ReadonlyTable = local_ro_mngr.tables["pa_agent_counts"]
    logger.info(f"Building index on {pa_agent_counts_table.full_name()}")
    pa_agent_counts_table.build_indices(local_ro_mngr)

    logger.info(f"Deleting {pa_agents_counts_tsv.absolute().as_posix()}")
    os.remove(pa_agents_counts_tsv.absolute().as_posix())


# ReadingRefLink
def reading_ref_link(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the reading ref link table with data

    depends on: text_ref, text_content, reading
    requires assembly: False

    Original SQL query to get data
        SELECT pmid, pmid_num, pmcid, pmcid_num,
               pmcid_version, doi, doi_ns, doi_id,
               tr.id AS trid, pii, url, manuscript_id,
               tc.id AS tcid, source, r.id AS rid, reader
        FROM text_ref AS tr
        JOIN text_content AS tc
            ON tr.id = tc.text_ref_id
        JOIN reading AS r
            ON tc.id = r.text_content_id
    """
    # Create a temp file to put the query in
    dump_file = reading_ref_link_tsv_fpath

    principal_dump_sql = dedent(
        """SELECT pmid, pmid_num, pmcid, pmcid_num,
                  pmcid_version, doi, doi_ns, doi_id,
                  tr.id AS trid, pii, url, manuscript_id,
                  tc.id AS tcid, source, r.id AS rid, reader
        FROM text_ref AS tr
        JOIN text_content AS tc
            ON tr.id = tc.text_ref_id
        JOIN reading AS r
            ON tc.id = r.text_content_id""")
    column_order = (
        "pmid, pmid_num, pmcid, pmcid_num, pmcid_version, doi, doi_ns, "
        "doi_id, trid, pii, url, manuscript_id, tcid, source, rid, reader"
    )

    # Dump the query from principal
    principal_query_to_csv(query=principal_dump_sql, output_location=dump_file)

    # todo: if you want to switch to gzipped files you can do
    #  "copy table_name from program 'zcat /tmp/tp.csv.gz';"
    schema = StructType([
        StructField("pmid", StringType(), True),
        StructField("pmid_num", IntegerType(), True),
        StructField("pmcid", StringType(), True),
        StructField("pmcid_num", IntegerType(), True),
        StructField("pmcid_version", IntegerType(), True),
        StructField("doi", StringType(), True),
        StructField("doi_ns", IntegerType(), True),
        StructField("doi_id", StringType(), True),
        StructField("trid", IntegerType(), True),
        StructField("pii", StringType(), True),
        StructField("url", StringType(), True),
        StructField("manuscript_id", StringType(), True),
        StructField("tcid", IntegerType(), True),
        StructField("source", StringType(), True),
        StructField("rid", LongType(), True),
        StructField("reader", StringType(), True)
    ])
    load_file_to_table_spark(table_name="readonly.reading_ref_link",
                             schema=schema,
                             column_order=column_order,
                             tsv_file=dump_file.absolute().as_posix(),
                             header=True)

    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name='reading_ref_link',
                       keys='rid')

    # Delete the dump file
    logger.info(f"Deleting {dump_file.absolute().as_posix()}")
    os.remove(dump_file)

    # Build the index
    reading_ref_link_table: ReadonlyTable = local_ro_mngr.tables["reading_ref_link"]
    logger.info(f"Building index for table {reading_ref_link_table.full_name()}")
    reading_ref_link_table.build_indices(local_ro_mngr)


def load_file_to_table_spark(table_name, schema,
                             column_order, tsv_file,
                             null_value: str = None,
                             header: bool = False):
    spark = SparkSession.builder \
        .appName("TSV to PostgreSQL") \
        .config("spark.jars", postgresql_jar.absolute().as_posix()) \
        .config("spark.driver.memory", "80g") \
        .config("spark.local.dir", "/data/spark_tmp") \
        .getOrCreate()
    if table_name == 'readonly.fast_raw_pa_link':
        df = spark.read.parquet(*tsv_file)
    elif table_name == 'readonly.source_meta':
        df = spark.read.parquet(tsv_file)
        df = df.withColumn("src_json", to_json(col("src_json")))
        df = df.withColumn("belief", col("belief").cast(FloatType()))
    else:
        if null_value:
            df = spark.read.csv(tsv_file, schema=schema, sep='\t', nullValue=null_value, header=header)
        else:
            df = spark.read.csv(tsv_file, schema=schema, sep='\t', header=header)

    url = "jdbc:postgresql://localhost:5432/indradb_readonly_local_test"

    custom_column_names = column_order.split(", ")
    df = df.toDF(*custom_column_names)
    df = df.select(custom_column_names)
    if table_name in ['readonly.raw_stmt_mesh_concepts',
                      'readonly.raw_stmt_mesh_terms']:
        df = df.dropDuplicates(['sid', 'mesh_num'])
    elif table_name in ['readonly.name_meta', 'readonly.text_meta',
                        'readonly.other_meta']:
        df = df.filter(col("ag_id").isNotNull() & col("ag_num").isNotNull())
    elif table_name in ['readonly.mesh_term_meta',
                        'readonly.mesh_concept_meta']:
        df = df.dropDuplicates(['mk_hash', 'mesh_num'])

    properties = {
        "user": LOCAL_RO_USER,
        "password": LOCAL_RO_PASSWORD,
        "driver": "org.postgresql.Driver"
    }
    if table_name == 'readonly.fast_raw_pa_link':
        df.write.jdbc(url=url, table=table_name, mode="append", properties=properties)
    else:
        df.write.jdbc(url=url, table=table_name, mode="overwrite", properties=properties)


# EvidenceCounts
def evidence_counts(local_ro_mngr: ReadonlyDatabaseManager):
    # Basically just upload the source counts, with the counts summed up per
    # mk_hash, as a table
    logger.info("Start Loading evidence_count table")
    if not source_counts_fpath.exists():
        raise ValueError(f"Source counts {source_counts_fpath} does not exist")

    source_counts = pickle.load(source_counts_fpath.open("rb"))

    with evidence_counts_tsv.open("w") as ev_counts_f:
        writer = csv.writer(ev_counts_f, delimiter="\t")

        for mk_hash, src_counts in tqdm(source_counts.items(),
                                        desc="EvidenceCounts"):
            ev_count = sum(src_counts.values())
            writer.writerow([mk_hash, ev_count])
    schema = StructType([
        StructField("mk_hash", LongType(), True),
        StructField("ev_count", IntegerType(), True)
    ])
    load_file_to_table_spark(table_name="readonly.evidence_counts",
                             schema=schema,
                             column_order="mk_hash, ev_count",
                             tsv_file=evidence_counts_tsv.absolute().as_posix())

    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name='evidence_counts',
                       keys='mk_hash')

    # Build the index
    evidence_counts_table: ReadonlyTable = local_ro_mngr.tables["evidence_counts"]
    logger.info(f"Building index for table {evidence_counts_table.full_name()}")
    evidence_counts_table.build_indices(local_ro_mngr)

    logger.info(f"Deleting {evidence_counts_tsv.absolute().as_posix()}")
    os.remove(evidence_counts_tsv.absolute().as_posix())


# AgentInteractions
def agent_interactions(local_ro_mngr: ReadonlyDatabaseManager):
    # This table depends completely on source_meta and name_meta in the
    # readonly database, so we can just run the table's create method
    if not table_has_content(local_ro_mngr, "source_meta") or \
            not table_has_content(local_ro_mngr, "name_meta"):
        raise ValueError("source_meta and name_meta must be filled before "
                         "agent_interactions can be filled")
    agent_interactions_table: ReadonlyTable = local_ro_mngr.tables[
        "agent_interactions"]
    define = ("SELECT\n"
              "  low_level_names.mk_hash AS mk_hash, \n"
              "  jsonb_object(\n"
              "    array_agg(\n"
              "      CAST(\n"
              "        low_level_names.ag_num AS VARCHAR)),\n"
              "    array_agg(low_level_names.db_id)\n"
              "  ) AS agent_json, \n"
              "  low_level_names.type_num AS type_num, \n"
              "  low_level_names.agent_count AS agent_count, \n"
              "  low_level_names.ev_count AS ev_count, \n"
              "  low_level_names.belief AS belief, \n"
              "  low_level_names.activity AS activity, \n"
              "  low_level_names.is_active AS is_active, \n"
              "  low_level_names.src_json::jsonb AS src_json, \n"
              "  false AS is_complex_dup\n"
              "FROM \n"
              "  (\n"
              "    SELECT \n"
              "      readonly.name_meta.mk_hash AS mk_hash,\n"
              "      readonly.name_meta.db_id AS db_id,\n"
              "      readonly.name_meta.ag_num AS ag_num,\n"
              "      readonly.name_meta.type_num AS type_num,\n"
              "      readonly.name_meta.agent_count\n"
              "        AS agent_count,\n"
              "      readonly.name_meta.ev_count AS ev_count,\n"
              "      readonly.name_meta.belief AS belief,\n"
              "      readonly.name_meta.activity AS activity,\n"
              "      readonly.name_meta.is_active\n"
              "        AS is_active,\n"
              "      readonly.source_meta.src_json\n"
              "        AS src_json\n"
              "    FROM \n"
              "      readonly.name_meta, \n"
              "      readonly.source_meta\n"
              "    WHERE \n"
              "      readonly.name_meta.mk_hash \n"
              "        = readonly.source_meta.mk_hash\n"
              "      AND NOT readonly.name_meta.is_complex_dup"
              "  ) AS low_level_names \n"
              "GROUP BY \n"
              "  low_level_names.mk_hash, \n"
              "  low_level_names.type_num, \n"
              "  low_level_names.agent_count, \n"
              "  low_level_names.ev_count, \n"
              "  low_level_names.belief, \n"
              "  low_level_names.activity, \n"
              "  low_level_names.is_active, \n"
              "  low_level_names.src_json::jsonb")

    # For each row, create a new row for each pair of agents, if
    # the interaction is not a self-interaction (i.e., if there
    # are more than one agent in the interaction)

    drop_sql = "DROP TABLE IF EXISTS readonly.agent_interactions CASCADE"
    local_ro_mngr.execute(drop_sql)
    sql = "CREATE TABLE readonly.agent_interactions AS \n" + define
    logger.info(f"sql executing")
    local_ro_mngr.execute(sql)

    # Create the table
    logger.info(f"Creating table {agent_interactions_table.full_name()}")
    agent_interactions_table.create(local_ro_mngr)

    # Build the index
    logger.info(f"Building index for table {agent_interactions_table.full_name()}")
    agent_interactions_table.build_indices(local_ro_mngr)


def get_postgres_uri(
        username, password,
        port, db_name,
) -> str:
    # postgresql://<username>:<password>@localhost[:port]/[name]
    # username = ro.url.username
    # password = ro.url.password
    # port = ro.url.port
    # db_name = ro.url.database
    return f"postgresql://{username}:{password}@localhost:{port}/{db_name}"


def load_data_file_into_local_ro(
        table_name: str, column_order: str, tsv_file: str, null_value: str = None
):
    """Load data from a file to the local ro database

    Mental note: COPY FROM copies data from a file to a table

    Parameters
    ----------
    table_name :
        The name of the table to transfer to, e.g. readonly.reading_ref_link
    column_order :
        A string of comma separated column names as they appear in the file,
        where the names correspond to the naming in the table in the readonly
        database, e.g. "reading_id, db_info_id, ref_id, ref_type, ref_text"
    tsv_file :
        The path to the tab separated file to be (up)loaded into the database.
    null_value :
        The value to be interpreted as null, e.g. "NULL" or "\\N". If None,
        the default value of the database is used and the upload command
        will be without the 'NULL AS ...' part.
    """
    # todo: if you want to switch to gzipped files you can do
    #  "copy table_name from program 'zcat /tmp/tp.tsv.gz';"
    null_clause = f", NULL AS '{null_value}'" if null_value else ""
    command = [
        "psql",
        get_postgres_uri(),
        "-c",
        (
            f"COPY {table_name} ({column_order}) from '{tsv_file}' with "
            # fixme: test if the null clause works
            f"(format csv, delimiter E'\t', header{null_clause})"
        ),
    ]
    logger.info(f"Loading data into table {table_name} from {tsv_file}")
    subprocess.run(command)


# FastRawPaLink
def fast_raw_pa_link_helper(local_ro_mngr):
    local_ro_mngr.grab_session()
    query = local_ro_mngr.session.query(local_ro_mngr.RawStmtSrc.sid,
                                        local_ro_mngr.RawStmtSrc.src)
    db = get_db("primary")

    # Execute query to get db_indo_id tp source_api mapping
    rows = db.select_all(db.DBInfo)
    id_to_source_api = {r.id: r.db_name for r in rows}

    raw_stmt_id_source_map = {
        int(raw_stmt_id): src for raw_stmt_id, src in query.all()
    }
    if len(raw_stmt_id_source_map) == 0:
        raise ValueError("No data in readonly.raw_stmt_src")

    # Load statement hash - raw statement id mapping into a dictionary
    hash_to_raw_id_map = pickle.load(stmt_hash_to_raw_stmt_ids_fpath.open("rb"))

    # Iterate over the raw statements to get mapping from
    # raw statement id to reading id, db info id and raw json
    logger.info("Loading mapping from raw statement id to info id")
    with gzip.open(raw_id_info_map_fpath.as_posix(), "rt") as fh:
        reader = csv.reader(fh, delimiter="\t")
        raw_id_to_info = {}
        for raw_stmt_id, db_info_id, reading_id, stmt_json_raw in reader:
            info = {"raw_json": stmt_json_raw}
            if db_info_id and db_info_id != "\\N":
                info["db_info_id"] = int(db_info_id)
                info["src"] = id_to_source_api[int(db_info_id)]
            if reading_id and reading_id != "\\N":
                info["reading_id"] = int(reading_id)
            raw_id_to_info[int(raw_stmt_id)] = info

    # For each grounded statement, get all associated raw statement ids and
    # get the following values:
    #   - raw statement id,
    #   - raw statement json,
    #   - db info id,
    #   - assembled statement hash,
    #   - assembled statement json,
    #   - type num (from ro_type_map)
    #   - raw statement source (mapped from the raw_stmt_src dictionary)

    split_pa_link_folder_fpath = TEMP_DIR.join(name='split_parquet')
    os.makedirs(split_pa_link_folder_fpath, exist_ok=True)
    logger.info("Iterating over grounded statements")
    split_unique_files = [os.path.join(split_unique_statements_folder_fpath, f)
                          for f in os.listdir(split_unique_statements_folder_fpath)
                          if f.endswith(".gz")]
    split_unique_files = sorted(split_unique_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    for num, f in enumerate(split_unique_files):
        rows = []
        with gzip.open(f, "rt") as fh:
            logger.info(f"generating parque{num} from {f}")
            chunk_num = 0

            unique_stmts_reader = csv.reader(fh, delimiter="\t")

            for statement_hash_string, stmt_json_string in unique_stmts_reader:
                this_hash = int(statement_hash_string)
                stmt_json = clean_json_loads(stmt_json_string)

                for raw_stmt_id in hash_to_raw_id_map[this_hash]:
                    info_dict = raw_id_to_info.get(raw_stmt_id, {})
                    raw_json = info_dict.get("raw_json", "")
                    reading_id = info_dict.get("reading_id")
                    db_info_id = info_dict.get("db_info_id")
                    raw_stmt_src_name = info_dict.get("src")
                    type_num = ro_type_map._str_to_int[stmt_json["type"]]
                    rows.append([
                        raw_stmt_id,
                        raw_json.encode(encoding="utf-8"),
                        reading_id,
                        db_info_id,
                        this_hash,
                        stmt_json_string.encode(encoding="utf-8"),
                        type_num,
                        raw_stmt_src_name,
                    ])
                if len(rows) >= 100000:
                    temp_parquet_file = split_pa_link_folder_fpath.joinpath(
                        f"split_file_{num}_chunk_{chunk_num}.parquet")
                    df = pd.DataFrame(rows, columns=[
                        'raw_stmt_id', 'raw_json', 'reading_id', 'db_info_id',
                        'statement_hash', 'stmt_json', 'type_num', 'raw_stmt_src_name'
                    ])
                    df['reading_id'] = pd.to_numeric(df['reading_id'], errors='coerce').astype('Int64')
                    df['db_info_id'] = pd.to_numeric(df['db_info_id'], errors='coerce').astype('Int32')
                    df['type_num'] = pd.to_numeric(df['type_num'], errors='coerce').astype('Int16')
                    df['raw_stmt_src_name'] = df['raw_stmt_src_name'].astype('string')
                    df.to_parquet(temp_parquet_file, index=False)
                    rows.clear()  # Clear rows after saving
                    chunk_num += 1  # Increment chunk number

                # Save any remaining rows that didn't reach the chunk size
            if rows:
                temp_parquet_file = split_pa_link_folder_fpath.joinpath(
                    f"split_file_{num}_chunk_{chunk_num}.parquet")
                df = pd.DataFrame(rows, columns=[
                    'raw_stmt_id', 'raw_json', 'reading_id', 'db_info_id',
                    'statement_hash', 'stmt_json', 'type_num', 'raw_stmt_src_name'
                ])
                df['reading_id'] = pd.to_numeric(df['reading_id'], errors='coerce').astype('Int64')
                df['db_info_id'] = pd.to_numeric(df['db_info_id'], errors='coerce').astype('Int32')
                df['type_num'] = pd.to_numeric(df['type_num'], errors='coerce').astype('Int16')
                df.to_parquet(temp_parquet_file, index=False)
                rows.clear()  # Clear remaining rows


def fast_raw_pa_link(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the fast_raw_pa_link table in the local readonly database

    Depends on:
    (principal)
    raw_statements, (pa_statements), raw_unique_links

    (readonly)
    raw_stmt_src

    (requires statement type map a.k.a ro_type_map.get_with_clause() to be
    inserted)

    Steps:
    Iterate through the grounded statements and:
    1. Dump (sid, src) from raw_stmt_src and load it into a dictionary
    2. Load stmt hash - raw statement id mapping into a dictionary
    3. Get raw statement id mapping to a) db info id and b) raw statement json
    4. Iterate over the grounded statements and get the following values:
        raw statement id
        raw statement json
        raw db info id      <-- (get from raw statements)
        assembled statement hash
        assembled statement json
        type num (from ro_type_map; (0, 'Acetylation'), (1, 'Activation'), ...)
        raw statement source (mapped from the raw_stmt_src dictionary)
    """
    table_name = "fast_raw_pa_link"

    assert table_name in local_ro_mngr.tables

    # Load the raw_stmt_src table into a dictionary
    logger.info("Loading raw_stmt_src table into a dictionary")
    if not table_has_content(local_ro_mngr, "raw_stmt_src"):
        raise ValueError(
            "raw_stmt_src must be filled before fast_raw_pa_link can be filled"
        )

    fast_raw_pa_link_helper(local_ro_mngr)

    # Load the data into the fast_raw_pa_link table
    column_order = "id, raw_json, reading_id, db_info_id, mk_hash, pa_json, " \
                   "type_num, src"
    schema = StructType([
        StructField("id", LongType(), True),
        StructField("raw_json", BinaryType(), True),
        StructField("reading_id", LongType(), True),
        StructField("db_info_id", IntegerType(), True),
        StructField("mk_hash", LongType(), True),
        StructField("pa_json", BinaryType(), True),
        StructField("type_num", ShortType(), True),
        StructField("src", StringType(), True)
    ])
    split_unique_files = [os.path.join(split_pa_link_folder_fpath, f)
                          for f in os.listdir(split_pa_link_folder_fpath)
                          if f.endswith(".parquet")]
    split_unique_files = sorted(split_unique_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    load_file_to_table_spark("readonly.fast_raw_pa_link",
                             schema, column_order,
                             split_unique_files)
    # create_primary_key(ro_mngr_local=local_ro_mngr,
    #                    table_name='fast_raw_pa_link',
    #                    keys='id')

    # Build the index
    table: ReadonlyTable = local_ro_mngr.tables[table_name]
    logger.info(f"Building index on {table.full_name()}")
    table.build_indices(local_ro_mngr)

    logger.info(f"Deleting {split_pa_link_folder_fpath.absolute().as_posix()}")
    shutil.rmtree(split_pa_link_folder_fpath.absolute().as_posix())
    logger.info(f"Deleting {pa_hash_act_type_ag_count_cache.absolute().as_posix()}")
    os.remove(pa_hash_act_type_ag_count_cache.absolute().as_posix())


def ensure_source_meta_source_files(local_ro_mngr: ReadonlyDatabaseManager):
    """Generate the source files for the SourceMeta table"""
    if not table_has_content(local_ro_mngr, "name_meta"):
        raise ValueError(
            "name_meta must be filled before source_meta can be filled"
        )
    col_names = "ev_count, belief, num_srcs, " \
                "src_json, only_src, has_rd, has_db, " \
                "type_num, activity, is_active, agent_count"
    all_sources = list(
        {*SOURCE_GROUPS["reader"], *SOURCE_GROUPS["database"]}
    )
    all_sources_str = ", ".join(all_sources)

    if source_meta_parquet.exists():
        return "mk_hash, " + all_sources_str + ", " + col_names, all_sources
    logger.info("Loading source counts")
    # Load source_counts
    ro = local_ro_mngr
    source_counts = pickle.load(source_counts_fpath.open("rb"))

    # Dump out from NameMeta
    # mk_hash, type_num, activity, is_active, ev_count, belief, agent_count
    # where is_complex_dup == False
    # todo If it's too large, we can do it in chunks of 100k
    # Typical count for namemeta 90_385_263
    res = local_ro_mngr.select_all([ro.NameMeta.mk_hash,
                                    ro.NameMeta.ev_count,
                                    ro.NameMeta.belief,
                                    ro.NameMeta.type_num,
                                    ro.NameMeta.activity,
                                    ro.NameMeta.is_active,
                                    ro.NameMeta.agent_count],
                                   ro.NameMeta.is_complex_dup == False)

    # Loop the dump and create the following columns as well:
    # source count, src_json (==source counts dict per mk_hash), only_src (if
    # only one source), has_rd (if the source is in the reading sources),
    # has_db (if the source is in the db sources)
    mk_hash_set = set()
    rows = []
    logger.info("Generating source_meta parquet")
    for (mk_hash, ev_count, belief, type_num,
         activity, is_active, agent_count) in res:
        if mk_hash in mk_hash_set:
            continue
        mk_hash_set.add(mk_hash)
        # Get the source count
        src_count_dict = source_counts.get(mk_hash)
        if src_count_dict is None:
            continue

        num_srcs = len(src_count_dict)
        has_rd = any(source in src_count_dict for source in reader_sources)
        has_db = any(source in src_count_dict for source in db_sources)
        only_src = list(src_count_dict.keys())[0] if num_srcs == 1 else None
        sources_tuple = tuple(
            src_count_dict.get(src) for src in all_sources
        )
        #print(len(sources_tuple),sources_tuple)
        # Write the following columns:
        #  - mk_hash
        #  - *sources (a splat of all sources,
        #              null if not present in the source count dict)
        #  - ev_count
        #  - belief
        #  - num_srcs
        #  - src_json
        #  - only_src - if only one source, the name of that source
        #  - has_rd  boolean - true if any source is from reading
        #  - has_db  boolean - true if any source is from a database
        #  - type_num
        #  - activity
        #  - is_active
        #  - agent_count

        # SourceMeta
        rows.append([
            mk_hash,
            *(np.int32(value) if value is not None else None for value in sources_tuple),
            np.int32(ev_count),
            belief,
            np.int32(num_srcs),
            src_count_dict,
            only_src,
            has_rd,
            has_db,
            type_num,
            activity,
            is_active,
            np.int32(agent_count)
        ])
    columns = ["mk_hash"] + all_sources + [
        "ev_count", "belief", "num_srcs",
        "src_json", "only_src", "has_rd", "has_db",
        "type_num", "activity", "is_active", "agent_count"
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_parquet(source_meta_parquet.absolute().as_posix(), index=False)

    return "mk_hash, " + all_sources_str + ", " + col_names, all_sources


# SourceMeta
def source_meta(local_ro_mngr: ReadonlyDatabaseManager):
    col_order, source_names = ensure_source_meta_source_files(local_ro_mngr)
    table_name = "readonly.source_meta"
    source_fields = [StructField(source_name, IntegerType(), True)
                     for source_name in source_names]

    schema = StructType([
                            StructField("mk_hash", LongType(), True)
                        ] + source_fields + [
                            StructField("ev_count", IntegerType(), True),
                            StructField("belief", FloatType(), True),
                            StructField("num_srcs", IntegerType(), True),
                            StructField("src_json", StructType(), True),
                            StructField("only_src", StringType(), True),
                            StructField("has_rd", BooleanType(), True),
                            StructField("has_db", BooleanType(), True),
                            StructField("type_num", ShortType(), True),
                            StructField("activity", StringType(), True),
                            StructField("is_active", BooleanType(), True),
                            StructField("agent_count", IntegerType(), True)
                        ])
    logger.info(f"Loading data into {table_name}")
    load_file_to_table_spark(
        table_name, schema, col_order, source_meta_parquet.absolute().as_posix())
    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name='source_meta',
                       keys='mk_hash')
    # Build the index
    table: ReadonlyTable = local_ro_mngr.tables['source_meta']
    logger.info(f"Building index on {table.full_name()}")
    table.build_indices(local_ro_mngr)
    local_ro_mngr.execute(f"ALTER TABLE {table_name}\
                          ALTER COLUMN src_json TYPE JSON\
                          USING src_json::JSON;")

    logger.info(f"Deleting {source_meta_parquet.absolute().as_posix()}")
    os.remove(source_meta_parquet.absolute().as_posix())


# PaMeta - the table itself is not generated on the readonly db, but the
# tables derived from it are (NameMeta, TextMeta and OtherMeta)
def ensure_pa_meta():
    """Generate the source files for the Name/Text/OtherMeta tables"""
    if all([name_meta_tsv.exists(),
            text_meta_tsv.exists(),
            other_meta_tsv.exists()]):
        return

    #  Depends on:
    #   - PaActivity (from principal schema)
    #   - PaAgents (from principal schema)
    #   - PAStatements (from principal schema)
    # - Belief (from readonly schema)
    # - EvidenceCounts (from readonly schema)
    # - PaAgentCounts (from readonly schema)
    # - TypeMap (from ???)
    # - RoleMap (from ???)
    # To get the needed info from the principal schema, select the
    # appropriate columns from the tables on the principal schema, join them
    # by mk_hash and then read in belief, agent count and evidence count
    # from the belief dump, unique statement file and source counts file.
    #
    # **Role num** is the integer mapping subject, other, object to
    # (-1, 0, 1), see schemas.readonly_schema.RoleMapping
    #
    # **Type num** is the integer mapping of the statement type to
    # the integer in the type map, see
    # schemas.readonly_schema.StatementTypeMapping
    #
    # *pa_agents* has the columns:
    # stmt_mk_hash, db_name, db_id, role, ag_num, agent_ref_hash
    #
    # *pa_statements* has the columns:
    # mk_hash, matches_key, uuid, type, indra_version, json, create_date
    #
    # *pa_activity* has the columns:
    # id, stmt_mk_hash, statements, activity, is_active
    pa_meta_query = ("""
    SELECT pa_agents.db_name,
           pa_agents.db_id,
           pa_agents.id AS ag_id,
           pa_agents.ag_num,
           pa_agents.role,
           pa_statements.mk_hash
    FROM pa_agents INNER JOIN
         pa_statements ON
             pa_agents.stmt_mk_hash = pa_statements.mk_hash
    WHERE LENGTH(pa_agents.db_id) < 2000
    """)
    principal_query_to_csv(pa_meta_query, pa_meta_fpath)

    # Load the belief dump into a dictionary
    logger.info("Loading belief scores")
    belief_dict = pickle.load(belief_scores_pkl_fpath.open("rb"))

    # Load source counts (can generate evidence counts from this)
    logger.info("Loading source counts")
    source_counts = pickle.load(source_counts_fpath.open("rb"))

    # Load agent count, activity and type mapping
    stmt_hash_to_activity_type_count = get_activity_type_ag_count()

    # Loop pa_meta dump and write load files for NameMeta, TextMeta, OtherMeta
    logger.info("Iterating over pa_meta dump")
    nones = (None, None, None, None)

    def db_id_clean(s):
        return s.replace('\n', ' ').replace('\r', ' ') if s else s

    with pa_meta_fpath.open('rt') as fh, \
            name_meta_tsv.open("wt") as name_fh, \
            text_meta_tsv.open("wt") as text_fh, \
            other_meta_tsv.open("wt") as other_fh:
        reader = csv.reader(fh, delimiter="\t")
        name_writer = csv.writer(name_fh, delimiter="\t")
        text_writer = csv.writer(text_fh, delimiter="\t")
        other_writer = csv.writer(other_fh, delimiter="\t")
        next(reader)  #skip column name
        for db_name, db_id, ag_id, ag_num, role, stmt_hash_str in reader:
            stmt_hash = int(stmt_hash_str)

            # Get the belief score
            belief_score = belief_dict.get(stmt_hash)
            if belief_score is None:
                # todo: debug log, remove later
                #logger.warning(f"Missing belief score for {stmt_hash}")
                continue

            # Get the agent count, activity and type count
            activity, is_active, type_num, agent_count = \
                stmt_hash_to_activity_type_count.get(stmt_hash, nones)
            if type_num is None and agent_count is None:
                continue

            # Get the evidence count
            ev_count = sum(
                source_counts.get(stmt_hash, {}).values()
            )
            if ev_count == 0:
                # todo: debug log, remove later
                #logger.warning(f"Missing evidence count for {stmt_hash}")
                continue

            # Get role num
            role_num = ro_role_map.get_int(role)

            # NameMeta - db_name == "NAME"
            # TextMeta - db_name == "TEXT"
            # OtherMeta - other db_names not part of ("NAME", "TEXT")
            # Columns are:
            # ag_id, ag_num, [db_name,] db_id, role_num, type_num, mk_hash,
            # ev_count, belief, activity, is_active, agent_count,
            # is_complex_dup
            row_start = [
                ag_id,
                ag_num,
            ]
            # db_name here if "other"
            row_end = [
                db_id_clean(db_id),
                role_num,
                type_num,
                stmt_hash,
                ev_count,
                belief_score,
                activity,
                is_active,
                agent_count,
                False,
            ]
            if db_name == "NAME":
                name_writer.writerow(row_start + row_end)
            elif db_name == "TEXT":
                text_writer.writerow(row_start + row_end)
            else:
                other_writer.writerow(row_start + [db_name] + row_end)

            if type_num == ro_type_map.get_int("Complex"):
                dup1 = [
                    db_id_clean(db_id),
                    -1,  # role_num
                    type_num, stmt_hash,
                    ev_count, belief_score,
                    activity, is_active,
                    agent_count,
                    True  # is_complex_dup = True
                ]

                dup2 = [
                    db_id_clean(db_id),
                    1,  # role_num
                    type_num, stmt_hash,
                    ev_count, belief_score,
                    activity, is_active,
                    agent_count,
                    True
                ]
                if db_name == "NAME":
                    name_writer.writerow(
                        row_start[:1] + [0] + dup1)  # ag_num=0
                    name_writer.writerow(
                        row_start[:1] + [1] + dup2)  # ag_num=1
                elif db_name == "TEXT":
                    text_writer.writerow(row_start[:1] + [0] + dup1)
                    text_writer.writerow(row_start[:1] + [1] + dup2)
                else:
                    other_writer.writerow(
                        row_start[:1] + [0] + [db_name] + dup1)
                    other_writer.writerow(
                        row_start[:1] + [1] + [db_name] + dup2)


# NameMeta, TextMeta, OtherMeta
def name_meta(local_ro_mngr: ReadonlyDatabaseManager):
    # Ensure the pa_meta file exists
    ensure_pa_meta()
    # ag_id, ag_num, db_id, role_num, type_num, mk_hash,
    # ev_count, belief, activity, is_active, agent_count, is_complex_dup
    colum_order = (
        "ag_id, ag_num, db_id, role_num, type_num, mk_hash, ev_count, "
        "belief, activity, is_active, agent_count, is_complex_dup"
    )
    schema = StructType([
        StructField("ag_id", IntegerType(), True),
        StructField("ag_num", IntegerType(), True),
        StructField("db_id", StringType(), True),
        StructField("role_num", ShortType(), True),
        StructField("type_num", ShortType(), True),
        StructField("mk_hash", LongType(), True),
        StructField("ev_count", IntegerType(), True),
        StructField("belief", FloatType(), True),
        StructField("activity", StringType(), True),
        StructField("is_active", BooleanType(), True),
        StructField("agent_count", IntegerType(), True),
        StructField("is_complex_dup", BooleanType(), True)
    ])
    # Load into local ro
    logger.info("Loading name_meta into local ro")
    load_file_to_table_spark("readonly.name_meta",
                             schema,
                             colum_order,
                             name_meta_tsv.absolute().as_posix())
    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name='name_meta',
                       keys=['ag_id', 'mk_hash', 'role_num', 'ag_num'])
    # Build indices
    name_meta_table: ReadonlyTable = local_ro_mngr.tables["name_meta"]
    logger.info("Building indices for name_meta")
    name_meta_table.build_indices(local_ro_mngr)


def text_meta(local_ro_mngr: ReadonlyDatabaseManager):
    # Ensure the pa_meta file exists
    ensure_pa_meta()
    # ag_id, ag_num, db_id, role_num, type_num, mk_hash,
    # ev_count, belief, activity, is_active, agent_count, is_complex_dup
    colum_order = (
        "ag_id, ag_num, db_id, role_num, type_num, mk_hash, ev_count, "
        "belief, activity, is_active, agent_count, is_complex_dup"
    )
    schema = StructType([
        StructField("ag_id", IntegerType(), True),
        StructField("ag_num", IntegerType(), True),
        StructField("db_id", StringType(), True),
        StructField("role_num", ShortType(), True),
        StructField("type_num", ShortType(), True),
        StructField("mk_hash", LongType(), True),
        StructField("ev_count", IntegerType(), True),
        StructField("belief", FloatType(), True),
        StructField("activity", StringType(), True),
        StructField("is_active", BooleanType(), True),
        StructField("agent_count", IntegerType(), True),
        StructField("is_complex_dup", BooleanType(), True)
    ])
    # Load into local ro
    logger.info("Loading text_meta into local ro")
    load_file_to_table_spark("readonly.text_meta",
                             schema,
                             colum_order,
                             text_meta_tsv.absolute().as_posix())

    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name='text_meta',
                       keys=['ag_id', 'mk_hash', 'role_num', 'ag_num'])

    # Build indices
    text_meta_table: ReadonlyTable = local_ro_mngr.tables["text_meta"]
    logger.info("Building indices for text_meta")
    text_meta_table.build_indices(local_ro_mngr)


def other_meta(local_ro_mngr: ReadonlyDatabaseManager):
    # Ensure the pa_meta file exists
    ensure_pa_meta()
    # ag_id, ag_num, db_name, db_id, role_num, type_num, mk_hash,
    # ev_count, belief, activity, is_active, agent_count, is_complex_dup
    colum_order = (
        "ag_id, ag_num, db_name, db_id, role_num, type_num, mk_hash, "
        "ev_count, belief, activity, is_active, agent_count, is_complex_dup"
    )
    schema = StructType([
        StructField("ag_id", IntegerType(), True),
        StructField("ag_num", IntegerType(), True),
        StructField("db_name", StringType(), True),
        StructField("db_id", StringType(), True),
        StructField("role_num", ShortType(), True),
        StructField("type_num", ShortType(), True),
        StructField("mk_hash", LongType(), True),
        StructField("ev_count", IntegerType(), True),
        StructField("belief", FloatType(), True),
        StructField("activity", StringType(), True),
        StructField("is_active", BooleanType(), True),
        StructField("agent_count", IntegerType(), True),
        StructField("is_complex_dup", BooleanType(), True)
    ])
    # Load into local ro
    logger.info("Loading other_meta into local ro")
    load_file_to_table_spark("readonly.other_meta",
                             schema,
                             colum_order,
                             other_meta_tsv.absolute().as_posix())

    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name='other_meta',
                       keys=['ag_id', 'mk_hash', 'role_num', 'ag_num'])
    # Build indices
    other_meta_table: ReadonlyTable = local_ro_mngr.tables["other_meta"]
    logger.info("Building indices for other_meta")
    other_meta_table.build_indices(local_ro_mngr)

    logger.info(f"Deleting {name_meta_tsv.absolute().as_posix()}")
    os.remove(name_meta_tsv.absolute().as_posix())
    logger.info(f"Deleting {text_meta_tsv.absolute().as_posix()}")
    os.remove(text_meta_tsv.absolute().as_posix())
    logger.info(f"Deleting {other_meta_tsv.absolute().as_posix()}")
    os.remove(other_meta_tsv.absolute().as_posix())
    logger.info(f"Deleting {pa_meta_fpath.absolute().as_posix()}")
    os.remove(pa_meta_fpath.absolute().as_posix())


def ensure_pubmed_xml_files(xml_dir: Path = pubmed_xml_gz_dir,
                            retries: int = 3) -> int:
    """Downloads the PubMed XML files if they are not already present"""

    def _get_urls(url: str) -> Iterable[str]:
        """Get the paths to all XML files on the PubMed FTP server."""
        logger.info("Getting URL paths from %s" % url)

        # Get page
        response = requests.get(url)
        response.raise_for_status()

        # Make soup
        soup = BeautifulSoup(response.text, "html.parser")

        # Append trailing slash if not present
        url = url if url.endswith("/") else url + "/"

        # Loop over all links
        for link in soup.find_all("a"):
            href = link.get("href")
            # yield if href matches
            # 'pubmed<2 digit year>n<4 digit file index>.xml.gz'
            # but skip the md5 files
            if href and href.startswith("pubmed") and href.endswith(".xml.gz"):
                yield url + href

    def _download_xml_gz(xml_url: str, xml_file: Path, md5_check: bool = True):
        try:
            resp = requests.get(xml_url)
            resp.raise_for_status()
        except requests.exceptions.RequestException:
            return False

        if md5_check:
            md5_resp = requests.get(xml_url + ".md5")
            checksum = md5(resp.content).hexdigest()
            expected_checksum = re.search(
                r"[0-9a-z]+(?=\n)", md5_resp.content.decode("utf-8")
            ).group()
            if checksum != expected_checksum:
                logger.warning(
                    f"Checksum mismatch for {xml_url}, skipping download"
                )
                raise ValueError("Checksum mismatch")

        # Write the file xml.gz file
        with xml_file.open("wb") as fh:
            fh.write(resp.content)

        return True

    if retries < 0:
        raise ValueError("retries must be >= 0")

    # Define some constants
    pubmed_base_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
    pubmed_update_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"

    # Create the directory if it doesn't exist
    xml_dir.mkdir(exist_ok=True, parents=True)

    # Download the files if they don't exist
    num_files = 0
    basefiles = [u for u in _get_urls(pubmed_base_url)]
    updatefiles = [u for u in _get_urls(pubmed_update_url)]
    for xml_url in tqdm(
            basefiles + updatefiles, desc="Downloading PubMed XML files"
    ):
        xml_file_path = xml_dir.joinpath(xml_url.split("/")[-1])

        # Download the file if it doesn't exist
        if not xml_file_path.exists():
            success = False
            for _ in range(retries + 1):
                try:
                    success = _download_xml_gz(xml_url, xml_file_path)
                except ValueError:
                    logger.warning(f"Checksum mismatch, skipping {xml_url}")
                    break
                # If success, break retry loop and continue to next file
                if success:
                    num_files += 1
                    break
            if not success:
                logger.error(
                    f"Failed to download {xml_url} after {retries} retries"
                )
        else:
            num_files += 1

    return num_files


def _load_pmid_to_raw_stmt_id():
    # Load necessary mappings #
    # Mapping sequence from mesh to raw statement id:
    # mesh -> pmid -> text_ref_id -> reading_id -> raw_statement_id
    # mesh -> pmid comes from pubmed (pubmed xml - outer loop in this function)
    # pmid -> text_ref_id comes from text_refs_fpath
    # text_ref_id -> reading_id comes from reading_to_text_ref_map_fpath
    # reading_id -> raw_statement_id comes from raw_id_info_map_fpath
    text_ref_to_pmid_cache = PUBMED_MESH_DIR.join(name="text_ref_to_pmid.pkl")
    reading_id_to_pmid_cache = \
        PUBMED_MESH_DIR.join(name="reading_id_to_pmid.pkl")
    pmid_to_raw_stmt_id_cache = \
        PUBMED_MESH_DIR.join(name="pmid_to_raw_stmt_id.pkl")

    def _load_create_text_ref_to_pmid():
        if text_ref_to_pmid_cache.exists():
            logger.info("Loading text_ref -> pmid map from cache")
            with text_ref_to_pmid_cache.open("rb") as fh:
                local_dict = pickle.load(fh)
        else:
            local_dict = {}
            with gzip.open(text_refs_fpath.as_posix(), "rt") as fh:
                reader = csv.reader(fh, delimiter="\t")
                # Rows are:
                # TRID, PMID, PMCID, DOI, PII, URL, MANUSCRIPT_ID
                for meta_row in tqdm(reader):
                    if meta_row[1] != SQL_NULL:
                        # In the text_refs table, trid values are unique while
                        # PMIDs are not
                        trid = int(meta_row[0])
                        pmid = int(meta_row[1])
                        local_dict[trid] = pmid

            # Save to cache
            with text_ref_to_pmid_cache.open("wb") as fh:
                pickle.dump(local_dict, fh)

        return local_dict

    def _load_create_reading_id_to_pmid():
        if reading_id_to_pmid_cache.exists():
            logger.info("Loading reading_id -> pmid map from cache")
            with reading_id_to_pmid_cache.open("rb") as fh:
                local_dict = pickle.load(fh)
        else:
            text_ref_to_pmid = _load_create_text_ref_to_pmid()
            local_dict = {}
            with gzip.open(reading_text_content_fpath.as_posix(),
                           "rt") as fh:
                reader = csv.reader(fh, delimiter="\t")
                # Rows are:
                # reading_id,  reader_version, text_content_id,
                # text_ref_id, source,         text_type
                for meta_row in tqdm(reader):
                    if meta_row[3] != SQL_NULL:
                        trid = int(meta_row[3])  # text_ref_id
                        rid = int(meta_row[0])  # reading_id
                        # One text ref id can have multiple reading ids
                        # (because one text source can have multiple readings
                        # from different readers and/or reader versions) so
                        # text_ref_id -> reading_id is a one-to-many mapping
                        # => reading_id -> pmid is a many-to-one mapping
                        pmid = text_ref_to_pmid.get(trid)
                        if pmid:
                            local_dict[rid] = pmid

            # Save the cache
            with reading_id_to_pmid_cache.open("wb") as fh:
                pickle.dump(local_dict, fh)

        return local_dict

    def _load_create_pmid_to_raw_stmt_id():
        if pmid_to_raw_stmt_id_cache.exists():
            logger.info("Loading pmid -> raw_stmt_id map from cache")
            with pmid_to_raw_stmt_id_cache.open("rb") as fh:
                local_dict = pickle.load(fh)
        else:
            reading_id_to_pmid = _load_create_reading_id_to_pmid()
            local_dict = defaultdict(set)
            with gzip.open(raw_id_info_map_fpath.as_posix(), "rt") as fh:
                reader = csv.reader(fh, delimiter="\t")
                # Rows are:
                # raw_stmt_id_int, db_info_id, int_reading_id, stmt_json_raw
                for meta_row in tqdm(reader):
                    if meta_row[2] != SQL_NULL:
                        rid = int(meta_row[2])
                        raw_stmt_id = int(meta_row[0])
                        pmid = reading_id_to_pmid.get(rid)
                        if pmid:
                            local_dict[pmid].add(raw_stmt_id)

            # Cast as regular dict
            local_dict = dict(local_dict)

            # Save to cache
            with pmid_to_raw_stmt_id_cache.open("wb") as fh:
                pickle.dump(local_dict, fh)

        return local_dict

    return _load_create_pmid_to_raw_stmt_id()


def get_activity_type_ag_count():
    if pa_hash_act_type_ag_count_cache.exists():
        logger.info("Loading pre-assembled statement hash -> activity type "
                    "and agent count map from cache")
        cache = pickle.load(pa_hash_act_type_ag_count_cache.open("rb"))
        return cache

    # Get stmt_hash -> type_num, activity, is_active, agent_count
    # Loop unique_stmts_path
    logger.info("Mapping pre-assembled statement hashes to activity data and "
                "agent counts")
    stmt_hash_to_activity_type_count = {}
    with gzip.open(unique_stmts_fpath.as_posix(), "rt") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for stmt_hash_string, stmt_json_string in tqdm(reader):
            stmt_json = clean_json_loads(stmt_json_string)
            stmt = stmt_from_json(stmt_json)
            stmt_hash = int(stmt_hash_string)
            agent_count = len(stmt.agent_list())
            type_num = ro_type_map.get_int(stmt_json["type"])
            if isinstance(stmt, ActiveForm):
                activity = stmt.activity
                is_active = stmt.is_active
            else:
                activity = None
                is_active = False  # is_active is a boolean column
            stmt_hash_to_activity_type_count[stmt_hash] = (
                activity, is_active, type_num, agent_count
            )

    # Save to cache
    with pa_hash_act_type_ag_count_cache.open("wb") as fh:
        pickle.dump(stmt_hash_to_activity_type_count, fh)
    return stmt_hash_to_activity_type_count


def ensure_pubmed_mesh_data():
    """Get the of PubMed XML gzip files for pmid-mesh processing"""
    # Check if the output files already exist

    if all(f.exists() for f in [mesh_concepts_meta_fpath,
                                mesh_terms_meta_fpath,
                                raw_stmt_mesh_concepts_fpath,
                                raw_stmt_mesh_terms_fpath,
                                pmid_mesh_map_fpath,
                                pmid_stmt_hash_fpath]):
        return

    def _pmid_mesh_extractor(xml_gz_path: Path) -> Iterable[Tuple[str, str]]:
        tree = etree.parse(xml_gz_path.as_posix())

        for article in tree.findall("PubmedArticle"):
            medline_citation = article.find("MedlineCitation")
            pubmed_id = medline_citation.find("PMID").text

            mesh_ann = _get_annotations(medline_citation)
            yield pubmed_id, mesh_ann["mesh_annotations"]

    num_files = ensure_pubmed_xml_files(xml_dir=pubmed_xml_gz_dir)
    if num_files == 0:
        raise FileNotFoundError("No PubMed XML files found")

    # Get the raw statement id -> stmt hash mapping
    hash_to_raw_stmt_id = pickle.load(stmt_hash_to_raw_stmt_ids_fpath.open("rb"))
    raw_stmt_id_to_hash = {raw_id: stmt_hash for stmt_hash, raw_ids in
                           hash_to_raw_stmt_id.items() for raw_id in raw_ids}
    del hash_to_raw_stmt_id

    # Load the pmid -> raw statement id mapping
    pmid_to_raw_stmt_id = _load_pmid_to_raw_stmt_id()

    stmt_hash_to_activity_type_count = get_activity_type_ag_count()

    # Get source counts ({hash: {source: count}})
    logger.info("Loading source counts")
    source_counts = pickle.load(source_counts_fpath.open("rb"))

    # Get belief scores (hash -> belief)
    logger.info("Loading belief scores")
    belief_scores = pickle.load(belief_scores_pkl_fpath.open("rb"))

    stmt_hashes = set()
    pmid_mesh_mapping = {}
    pmid_stmt_hash = defaultdict(set)
    logger.info("Generating tsv ingestion files")
    with open(
            mesh_terms_meta_fpath.as_posix(), "wt") as terms_meta_fh, \
            open(
                raw_stmt_mesh_concepts_fpath.as_posix(), "wt") as raw_concepts_fh, \
            open(
                raw_stmt_mesh_terms_fpath.as_posix(), "wt") as raw_terms_fh:
        terms_meta_writer = csv.writer(terms_meta_fh, delimiter="\t")
        raw_concepts_writer = csv.writer(raw_concepts_fh, delimiter="\t")
        raw_terms_writer = csv.writer(raw_terms_fh, delimiter="\t")

        xml_files = list(pubmed_xml_gz_dir.glob("*.xml.gz"))
        for xml_file_path in tqdm(xml_files, desc="Pubmed processing"):
            # Extract the data from the XML file
            for pmid, mesh_annotations in _pmid_mesh_extractor(xml_file_path):
                pmid_mesh_map = {"concepts": set(), "terms": set()}
                for annot in mesh_annotations:
                    mesh_id = annot["mesh"]
                    mesh_num = int(mesh_id[1:])
                    is_concept = mesh_id.startswith("C")
                    # major_topic = annot["major_topic"]  # Unused

                    # Save each pmid-mesh mapping
                    if is_concept:
                        pmid_mesh_map["concepts"].add(mesh_num)
                    else:
                        pmid_mesh_map["terms"].add(mesh_num)

                    # For each pmid, get the raw statement id
                    for raw_stmt_id in pmid_to_raw_stmt_id.get(int(pmid), set()):
                        raw_row = [raw_stmt_id, mesh_num]
                        # RawStmtMeshConcepts
                        if is_concept:
                            raw_concepts_writer.writerow(raw_row)
                        # RawStmtMeshTerms
                        else:
                            raw_terms_writer.writerow(raw_row)

                        # Now write to the meta tables, one row per
                        # (stmt hash, mesh id) pair
                        stmt_hash = raw_stmt_id_to_hash.get(int(raw_stmt_id))

                        # Save the pmid -> stmt hash mapping
                        if stmt_hash:
                            pmid_stmt_hash[pmid].add(stmt_hash)

                        if stmt_hash and stmt_hash not in stmt_hashes:
                            tup = \
                                stmt_hash_to_activity_type_count.get(stmt_hash)

                            if tup is not None:
                                act, is_act, type_num, agent_count = tup
                            else:
                                stmt_hashes.add(stmt_hash)
                                continue

                            # Get evidence count for this statement
                            ec = sum(
                                source_counts.get(stmt_hash, {}).values()
                            )
                            bs = belief_scores.get(stmt_hash)

                            meta_row = [
                                stmt_hash,
                                ec,
                                bs,
                                mesh_num,
                                type_num,
                                act,
                                is_act,
                                agent_count
                            ]
                            if not is_concept:
                                terms_meta_writer.writerow(meta_row)

                            stmt_hashes.add(stmt_hash)

                # If the pmid is already in pmid_mesh_mapping, update the
                # mesh concepts and terms
                if pmid in pmid_mesh_mapping:
                    pmid_mesh_mapping[pmid]["concepts"].update(
                        pmid_mesh_map["concepts"]
                    )
                    pmid_mesh_mapping[pmid]["terms"].update(
                        pmid_mesh_map["terms"]
                    )
                else:
                    pmid_mesh_mapping[pmid] = pmid_mesh_map
    with open(raw_stmt_mesh_concepts_fpath, 'rt') as fh, \
            open(
                mesh_concepts_meta_fpath.as_posix(), "wt") as concepts_meta_fh:
        reader = csv.reader(fh, delimiter="\t")
        concepts_meta_writer = csv.writer(concepts_meta_fh, delimiter="\t")
        for i, line in enumerate(reader):
            raw_stmt_id = line[0]
            mesh_num = line[1]
            stmt_hash = raw_stmt_id_to_hash.get(int(raw_stmt_id))
            if stmt_hash:
                tup = stmt_hash_to_activity_type_count.get(stmt_hash)
                if tup is not None:
                    act, is_act, type_num, agent_count = tup
                    ec = sum(
                        source_counts.get(stmt_hash, {}).values()
                    )
                    bs = belief_scores.get(stmt_hash)

                    meta_row = [
                        stmt_hash,
                        ec,
                        bs,
                        mesh_num,
                        type_num,
                        act,
                        is_act,
                        agent_count
                    ]
                    concepts_meta_writer.writerow(meta_row)

    # Save the pmid-mesh and pmid-stmt hash mappings to cache
    logger.info("Saving pmid-stmt hash mappings to cache")
    pmid_stmt_hash = dict(pmid_stmt_hash)
    with pmid_stmt_hash_fpath.open("wb") as pmid_stmt_hash_fh:
        pickle.dump(pmid_stmt_hash, pmid_stmt_hash_fh)
    logger.info("Saving pmid-mesh mappings to cache")
    with pmid_mesh_map_fpath.open("wb") as pmid_mesh_mapping_fh:
        pickle.dump(pmid_mesh_mapping, pmid_mesh_mapping_fh)


# RawStmtMeshConcepts
def raw_stmt_mesh_concepts(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the raw_stmt_mesh_concepts table."""
    full_table_name = "readonly.raw_stmt_mesh_concepts"
    column_order = "sid, mesh_num"
    ensure_pubmed_mesh_data()

    schema = StructType([
        StructField("sid", IntegerType(), True),
        StructField("mesh_num", IntegerType(), True)
    ])
    # Load data into table
    load_file_to_table_spark(
        full_table_name, schema, column_order, raw_stmt_mesh_concepts_fpath.as_posix()
    )

    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name='raw_stmt_mesh_concepts',
                       keys=['sid', 'mesh_num'])
    # Build index
    table: ReadonlyTable = local_ro_mngr.tables["raw_stmt_mesh_concepts"]
    logger.info(f"Building index for {table.full_name}")
    table.build_indices(local_ro_mngr)


# RawStmtMeshTerms
def raw_stmt_mesh_terms(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the raw_stmt_mesh_terms table"""
    full_table_name = "readonly.raw_stmt_mesh_terms"
    column_order = "sid, mesh_num"
    ensure_pubmed_mesh_data()

    schema = StructType([
        StructField("sid", IntegerType(), True),
        StructField("mesh_num", IntegerType(), True)
    ])
    # Load data into table
    load_file_to_table_spark(
        full_table_name, schema, column_order, raw_stmt_mesh_terms_fpath.as_posix()
    )
    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name='raw_stmt_mesh_terms',
                       keys=['sid', 'mesh_num'])
    # Build index
    table: ReadonlyTable = local_ro_mngr.tables["raw_stmt_mesh_terms"]
    logger.info(f"Building index for {table.full_name}")
    table.build_indices(local_ro_mngr)


# MeshConceptMeta
def mesh_concept_meta(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the mesh_concept_meta table."""
    full_table_name = "readonly.mesh_concept_meta"
    column_order = (
        "mk_hash, ev_count, belief, mesh_num, "
        "type_num, activity, is_active, agent_count"
    )
    ensure_pubmed_mesh_data()

    schema = StructType([
        StructField("mk_hash", LongType(), True),
        StructField("ev_count", IntegerType(), True),
        StructField("belief", FloatType(), True),
        StructField("mesh_num", IntegerType(), True),
        StructField("type_num", ShortType(), True),
        StructField("activity", StringType(), True),
        StructField("is_active", BooleanType(), True),
        StructField("agent_count", IntegerType(), True)
    ])
    # Load data into table
    load_file_to_table_spark(
        full_table_name, schema, column_order, mesh_concepts_meta_fpath.as_posix()
    )
    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name='mesh_concept_meta',
                       keys=['mk_hash', 'mesh_num'])
    # Build index
    table: ReadonlyTable = local_ro_mngr.tables["mesh_concept_meta"]
    logger.info(f"Building index for {table.full_name}")
    table.build_indices(local_ro_mngr)


# MeshTermMeta
def mesh_term_meta(local_ro_mngr: ReadonlyDatabaseManager):
    """Fill the mesh_term_meta table."""
    full_table_name = "readonly.mesh_term_meta"
    column_order = (
        "mk_hash, ev_count, belief, mesh_num, "
        "type_num, activity, is_active, agent_count"
    )
    ensure_pubmed_mesh_data()

    schema = StructType([
        StructField("mk_hash", LongType(), True),
        StructField("ev_count", IntegerType(), True),
        StructField("belief", FloatType(), True),
        StructField("mesh_num", IntegerType(), True),
        StructField("type_num", ShortType(), True),
        StructField("activity", StringType(), True),
        StructField("is_active", BooleanType(), True),
        StructField("agent_count", IntegerType(), True)
    ])
    # Load data into table
    load_file_to_table_spark(
        full_table_name, schema, column_order, mesh_terms_meta_fpath.as_posix()
    )
    create_primary_key(ro_mngr_local=local_ro_mngr,
                       table_name='mesh_term_meta',
                       keys=['mk_hash', 'mesh_num'])
    # Build index
    table: ReadonlyTable = local_ro_mngr.tables["mesh_term_meta"]
    logger.info(f"Building index for {table.full_name}")
    table.build_indices(local_ro_mngr)
    pubmed_mesh_path = PUBMED_MESH_DIR.base
    logger.info(f"Deleting {pubmed_mesh_path.absolute().as_posix()}")
    shutil.rmtree(pubmed_mesh_path.absolute().as_posix())

def principal_query_to_csv(
        query: str, output_location: str, db: str = "primary"
) -> None:
    """Dump results of a query to the principal database to a tsv file

    Parameters
    ----------
    query : str
        Postgresql query that returns a list of rows. Queries that perform
        writes of any kind should not be used. A simple keyword search is
        used to check for queries that may perform writes, but be careful,
        it shouldn't be trusted completely.

    output_location : str
        Path to file where output is to be stored.

    db : Optional[str]
        Database from list of defaults in indra_db config. Default: "primary".

    Raises
    ------
    AssertionError
        If input query contains a disallowed keyword that could lead to
        modifications of the database.
    """

    disallowed_keywords = _find_disallowed_keywords(query)

    if disallowed_keywords:
        raise ValueError(
            f"Query '{query}' uses disallowed keywords: {disallowed_keywords}"
        )
    print("connecting to db")
    defaults = get_databases()
    print("got connection")
    try:
        principal_db_uri = defaults[db]
    except KeyError as err:
        raise KeyError(f"db {db} not available. Check db_config.ini") from err
    print(principal_db_uri)
    # fixme: allow tsv.gz output
    # Note: 'copy ... to' copies data to a file on the server, while
    # 'copy ... from' copies data to the client. The former is faster, but
    # requires from a file on the server. We use the former here.
    command = [
        "psql",
        principal_db_uri,
        "-c",
        f"\\copy ({query}) to {output_location} with (format csv, delimiter "
        f"E'\t', header)",
    ]
    logger.info(f"Running SQL query {query} with psql")
    subprocess.run(command)


def _find_disallowed_keywords(query: str) -> list:
    """Returns list of disallowed keywords in query if any exist.

    Keywords are disallowed if they can lead to modifications of the
    database or its settings. The disallowed keywords are:

    alter, call, commit, create, delete, drop, explain, grant, insert,
    lock, merge, rename, revoke, savepoint, set, rollback, transaction,
    truncate, update.

    Matching is case-insensitive.

    Parameters
    ----------
    query : str
        A string containing a Postgresql query.

    Returns
    -------
    list
        List of keywords within the query that are within the disallowed
        list. Entries are returned in all lower case.
    """
    disallowed = [
        "alter",
        "call",
        "commit",
        "create",
        "delete",
        "drop",
        "explain",
        "grant",
        "insert",
        "lock",
        "merge",
        "rename",
        "revoke",
        "savepoint",
        "set",
        "rollback",
        "transaction",
        "truncate",
        "update",
    ]

    query_token_set = set(token.lower() for token in query.split())
    return list(query_token_set & set(disallowed))


def get_temp_file(suffix: str) -> Path:
    """Return a temporary file path.

    Parameters
    ----------
    suffix :
        The suffix to use for the temporary file, e.g. '.csv'.

    Returns
    -------
    :
        A Path object pointing to the temporary file.
    """
    return TEMP_DIR.join(name=f"{uuid.uuid4()}{suffix}")


def drop_schema(ro_mngr_local: ReadonlyDatabaseManager):
    """Drop the readonly schema from the local db.

    This will drop all tables in the readonly schema, so use with caution.

    Parameters
    ----------
    ro_mngr_local : ReadonlyDatabaseManager
        The local readonly database manager.
    """
    logger.info("Dropping schema 'readonly'")
    ro_mngr_local.drop_schema('readonly')


def create_ro_tables(
        ro_mngr_local: ReadonlyDatabaseManager,
        force: bool = False
):
    """Create the readonly tables on the local db.

    Parameters
    ----------
    ro_mngr_local : ReadonlyDatabaseManager
        The local readonly database manager.
    force : bool
        If True, drop the readonly schema if it already exists before creating
        the tables.
    """
    postgres_uri = str(ro_mngr_local.url)
    engine = create_engine(postgres_uri)
    logger.info("Connected to db")

    # Drop schema if force is set
    if force:
        drop_schema(ro_mngr_local)

    # Create schema, this step uses '... IF NOT EXISTS ...', so it's idempotent
    ro_mngr_local.create_schema('readonly')

    # Create empty tables if they don't exist - MetaData.create_all() skips
    # existing tables by default
    tables_metadata = ro_mngr_local.tables['belief'].metadata
    logger.info("Creating tables")
    tables_metadata.create_all(bind=engine)
    logger.info("Done creating tables")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__name__)
    parser.add_argument("--db-name",
                        default=LOCAL_RO_DB_NAME)
    parser.add_argument("--user", required=True,
                        help="Username for the local readonly db.")
    parser.add_argument("--password", required=True,
                        help="Password for the local readonly db.")
    parser.add_argument("--hostname", default="localhost")
    parser.add_argument("--port", default=5432,
                        help="The port the local db server listens to.")
    parser.add_argument("--force", action="store_true",
                        help="If set, the script will delete data from "
                             "tables in the readonly schema if it already "
                             "exists.")

    args = parser.parse_args()

    # Get a ro manager for the local readonly db
    # postgresql://<username>:<password>@localhost[:port]/[name]
    postgres_url = get_postgres_uri(
        username=args.user,
        password=args.password,
        port=args.port,
        db_name=args.db_name,
    )
    logger.info(f"postgres_url: {postgres_url}")
    ro_manager = ReadonlyDatabaseManager(postgres_url, protected=False)
    # Create the tables
    print("DATABASE NAME IS ",args.db_name)
    create_ro_tables(ro_manager, force=args.force)

    ro_manager.create_schema('readonly')
    # For each table, run the function that will fill out the table with data
    time_benchmark = {}
    for table_name in tqdm(RUN_ORDER, desc="Creating tables"):
        start_time = time.time()
        if hasattr(sys.modules[__name__], table_name):
            build_script = getattr(sys.modules[__name__], table_name)
            build_script(ro_manager)
        else:
            raise ValueError(f"Table function for {table_name} not found")

        end_time = time.time()
        elapsed_time_seconds = end_time - start_time
        elapsed_time_hours = elapsed_time_seconds / 3600
        time_benchmark[table_name] = elapsed_time_hours

    with open(table_benchmark.absolute().as_posix(), 'w') as f:
        f.write("Table Name,Processing Time (hours)\n")
        for table_name, processing_time in time_benchmark.items():
            f.write(f"{table_name},{processing_time:.4f}\n")

    if not standard_readonly_snapshot.exists():
        logger.error("No standard snapshot to compare with")
    else:
        logger.info("Extracting new snapshot")
        generate_db_snapshot(postgres_url, new_readonly_snapshot.absolute().as_posix())
        if not compare_snapshots(standard_readonly_snapshot.absolute().as_posix(),
                                 new_readonly_snapshot.absolute().as_posix()):
            raise TypeError(f"Snapshots are not identical")
    pipeline_files_clean_up()
