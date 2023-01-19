import os
from pathlib import Path

__all__ = ["raw_statements_fpath", "source_counts_fpath", "text_refs_fpath",
           "reading_text_content_fpath", "drop_readings_fpath",
           "reading_to_text_ref_map_fpath", "processed_stmts_fpath",
           "grounded_stmts_fpath", "unique_stmts_fpath", "refinements_fpath",
           "belief_scores_pkl_fpath", "stmt_hash_to_raw_stmt_ids_fpath",
           "raw_id_info_map_fpath", "TEMP_DIR",
           "belief_scores_tsv_fpath", "reading_ref_link_tsv_fpath",
           "raw_stmt_source_tsv_fpath"]

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
stmt_hash_to_raw_stmt_ids_fpath = TEMP_DIR.joinpath("stmt_hash_to_raw_stmt_ids.pkl")
raw_id_info_map_fpath = TEMP_DIR.joinpath("raw_stmt_id_to_info_map.tsv.gz")
grounded_stmts_fpath = TEMP_DIR.joinpath("grounded_statements.tsv.gz")
unique_stmts_fpath = TEMP_DIR.joinpath("unique_statements.tsv.gz")
refinements_fpath = TEMP_DIR.joinpath("refinements.tsv.gz")
belief_scores_pkl_fpath = TEMP_DIR.joinpath("belief_scores.pkl")

# Temporary tsv files used for load into readonly db
belief_scores_tsv_fpath = TEMP_DIR.joinpath("belief_scores.tsv")
reading_ref_link_tsv_fpath = TEMP_DIR.joinpath("reading_ref_link.tsv")
raw_stmt_source_tsv_fpath = TEMP_DIR.joinpath("raw_stmt_source.tsv")
