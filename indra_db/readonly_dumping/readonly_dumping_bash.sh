# INITIAL DUMPING

# Get password to the principal database from the user
echo "Enter password for the principal database:"
read -s PGPASSWORD # -s flag hides the password
export PGPASSWORD

# If the password is empty, exit
if [ -z "$PGPASSWORD" ]
then
    echo "Password is empty. Exiting."
    exit 1
fi

# Get file paths for initial dump files
RAW_STMTS_FPATH=`python3 -m indra_db.readonly_dumping.locations raw_statements`
export RAW_STMTS_FPATH
READING_TEXT_CONTENT_META_FPATH=`python3 -m indra_db.readonly_dumping.locations reading_text_content`
export READING_TEXT_CONTENT_META_FPATH
TEXT_REFS_PRINCIPAL_FPATH=`python3 -m indra_db.readonly_dumping.locations text_refs_principal`
export TEXT_REFS_PRINCIPAL_FPATH

# Exit if any of the file names are empty
if [ -z "$RAW_STMTS_FPATH" ] || [ -z "$READING_TEXT_CONTENT_META_FPATH" ] || [ -z "$TEXT_REFS_PRINCIPAL_FPATH" ]
then
    echo "One or more of the file paths are empty. Exiting."
    exit 1
fi

# Run dumps
# Only run the dumps if the files don't exist
if [ ! -f "$RAW_STMTS_FPATH" ]
then
    echo "Dumping raw statements"

    psql -d indradb_test \
         -h indradb-refresh.cwcetxbvbgrf.us-east-1.rds.amazonaws.com \
         -U tester \
         -c "COPY (SELECT id, db_info_id, reading_id,
                   convert_from (json::bytea, 'utf-8')
                   FROM public.raw_statements)
             TO STDOUT" \
          | gzip > "$RAW_STMTS_FPATH"
fi

if [ ! -f "$READING_TEXT_CONTENT_META_FPATH" ]
then
    echo "Dumping reading text content meta"

    psql -d indradb_test \
         -h indradb-refresh.cwcetxbvbgrf.us-east-1.rds.amazonaws.com \
         -U tester \
         -c "COPY (SELECT rd.id, rd.reader_version, tc.id, tc.text_ref_id,
                          tc.source, tc.text_type
                   FROM public.text_content as tc, public.reading as rd
                   WHERE tc.id = rd.text_content_id)
             TO STDOUT" \
         | gzip > "$READING_TEXT_CONTENT_META_FPATH"
fi

if [ ! -f "$TEXT_REFS_PRINCIPAL_FPATH" ]
then
    echo "Dumping text refs principal"

    psql -d indradb_test \
         -h indradb-refresh.cwcetxbvbgrf.us-east-1.rds.amazonaws.com \
         -U tester \
         -c "COPY (SELECT id, pmid, pmcid, doi, pii, url, manuscript_id
                   FROM public.text_ref)
             TO STDOUT" \
         | gzip > text_refs_principal.tsv.gz
fi

# LOCAL DB CREATION AND DUMPING

# Run export assembly script
python3 -m indra_db.readonly_dumping.export_assembly # --refresh-kb

# Create db; todo: how to pass password?
psql -h localhost -c 'create database indradb_readonly_local;' -U postgres

# Run import script
python3 -m indra_db.readonly_dumping.readonly_dumping

# Dump the db

# copy to s3

# clean up