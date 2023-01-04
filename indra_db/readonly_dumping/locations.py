import os
from pathlib import Path

__all__ = ["raw_statements_fpath", "source_counts_fpath", "text_refs_fpath",
           "reading_text_content_fpath", "drop_readings_fpath",
           "reading_to_text_ref_map_fpath", "processed_stmts_fpath",
           "TEMP_DIR"]

TEMP_DIR = os.environ.get("PRINCIPAL_DUMPING_ROOT")
if not TEMP_DIR:
    TEMP_DIR = Path.home().joinpath(".data/indra/db")
else:
    TEMP_DIR = Path(TEMP_DIR)

TEMP_DIR.mkdir(exist_ok=True, parents=True)

raw_statements_fpath = TEMP_DIR.joinpath("raw_statements.tsv.gz")
reading_text_content_fpath = TEMP_DIR.joinpath("reading_text_content_meta.tsv.gz")
text_refs_fpath = TEMP_DIR.joinpath("text_refs_principal.tsv.gz")
drop_readings_fpath = TEMP_DIR.joinpath("drop_readings.pkl")
reading_to_text_ref_map_fpath = TEMP_DIR.joinpath("reading_to_text_ref_map.pkl")
processed_stmts_fpath = TEMP_DIR.joinpath("processed_statements.tsv.gz")
source_counts_fpath = TEMP_DIR.joinpath("source_counts.pkl")
