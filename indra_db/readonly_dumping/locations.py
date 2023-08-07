import os
from pathlib import Path

TEMP_DIR = os.environ.get("PRINCIPAL_DUMPING_ROOT")
if not TEMP_DIR:
    TEMP_DIR = Path.home().joinpath(".data/indra/db")
else:
    TEMP_DIR = Path(TEMP_DIR)

TEMP_DIR.mkdir(exist_ok=True, parents=True)

# Dump files and their derivatives
raw_statements_fpath = TEMP_DIR.joinpath("raw_statements.tsv.gz")
reading_text_content_fpath = TEMP_DIR.joinpath("reading_text_content_meta.tsv.gz")
text_refs_fpath = TEMP_DIR.joinpath("text_refs_principal.tsv.gz")
drop_readings_fpath = TEMP_DIR.joinpath("drop_readings.pkl")
reading_to_text_ref_map_fpath = TEMP_DIR.joinpath("reading_to_text_ref_map.pkl")
processed_stmts_fpath = TEMP_DIR.joinpath("processed_statements.tsv.gz")
source_counts_fpath = TEMP_DIR.joinpath("source_counts.pkl")
stmt_hash_to_raw_stmt_ids_fpath = TEMP_DIR.joinpath(
    "stmt_hash_to_raw_stmt_ids.pkl")
raw_id_info_map_fpath = TEMP_DIR.joinpath("raw_stmt_id_to_info_map.tsv.gz")
grounded_stmts_fpath = TEMP_DIR.joinpath("grounded_statements.tsv.gz")
unique_stmts_fpath = TEMP_DIR.joinpath("unique_statements.tsv.gz")
refinements_fpath = TEMP_DIR.joinpath("refinements.tsv.gz")
belief_scores_pkl_fpath = TEMP_DIR.joinpath("belief_scores.pkl")
pa_hash_act_type_ag_count_cache = TEMP_DIR.joinpath(
    "pa_hash_act_type_ag_count_cache.pkl")

# Temporary tsv files used for load into readonly db
belief_scores_tsv_fpath = TEMP_DIR.joinpath("belief_scores.tsv")
reading_ref_link_tsv_fpath = TEMP_DIR.joinpath("reading_ref_link.tsv")
raw_stmt_source_tsv_fpath = TEMP_DIR.joinpath("raw_stmt_source.tsv")

# Pubmed XML files
PUBMED_MESH_DIR = TEMP_DIR.joinpath("pubmed_mesh")
pubmed_xml_gz_dir = PUBMED_MESH_DIR.joinpath("pubmed_xml_gz")

# MeshConceptMeta and MeshTermMeta
mesh_concepts_meta = PUBMED_MESH_DIR.joinpath("mesh_concepts_meta.tsv")
mesh_terms_meta = PUBMED_MESH_DIR.joinpath("mesh_terms_meta.tsv")

# RawStmtMeshConcepts and RawStmtMeshTerms
raw_stmt_mesh_concepts = PUBMED_MESH_DIR.joinpath("raw_stmt_mesh_concepts.tsv")
raw_stmt_mesh_terms = PUBMED_MESH_DIR.joinpath("raw_stmt_mesh_terms.tsv")

# PaMeta and derived files
pa_meta_fpath = TEMP_DIR.joinpath("pa_meta.tsv.gz")
name_meta_tsv = TEMP_DIR.joinpath("name_meta.tsv")
text_meta_tsv = TEMP_DIR.joinpath("text_meta.tsv")
other_meta_tsv = TEMP_DIR.joinpath("other_meta.tsv")
