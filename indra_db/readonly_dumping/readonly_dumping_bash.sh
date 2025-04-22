# shellcheck disable=SC1090
# SETUP
set -e
# Get password to the principal database from the user
echo "Enter password for the principal database:"
read -s PGPASSWORD # -s flag hides the password
# Use the PGPASSWORD environment variable to set the password, see:
# https://www.postgresql.org/docs/13/libpq-envars.html
export PGPASSWORD

# If the password is empty, exit
if [ -z "$PGPASSWORD" ]
then
    echo "Password is empty. Exiting."
    exit 1
fi



#Set the user for the local db
LOCAL_RO_USER="postgres"
export LOCAL_RO_USER

# Set the password for the local db
echo "Provide password for the local database:"
read -s LOCAL_RO_PASSWORD
export LOCAL_RO_PASSWORD

# Set the name of the local db
LOCAL_RO_DB_NAME="indradb_readonly_local_test"
export LOCAL_RO_DB_NAME
echo "Local db name: $LOCAL_RO_DB_NAME"

# Get the current date and time
START_DATE_TIME=`date '+%Y-%m-%d %H:%M:%S'`
START_DATE=`date '+%Y-%m-%d'`
echo "{\"datetime\": \"$START_DATE_TIME\", \"date_stamp\": \"$START_DATE\"}" > start.json
S3_PATH="s3://bigmech/indra-db/dumps/$START_DATE"
aws s3 cp start.json "$S3_PATH/start.json"
echo "Start date marked as: $START_DATE"
# INITIAL DUMPING

# Get file paths for initial dump files
RAW_STMTS_FPATH=`python3 -m indra_db.readonly_dumping.locations raw_statements`
export RAW_STMTS_FPATH
READING_TEXT_CONTENT_META_FPATH=`python3 -m indra_db.readonly_dumping.locations reading_text_content`
export READING_TEXT_CONTENT_META_FPATH
TEXT_REFS_PRINCIPAL_FPATH=`python3 -m indra_db.readonly_dumping.locations text_refs`
export TEXT_REFS_PRINCIPAL_FPATH


if [ -z "$RAW_STMTS_FPATH" ] || [ -z "$READING_TEXT_CONTENT_META_FPATH" ] || [ -z "$TEXT_REFS_PRINCIPAL_FPATH" ]
then
    if [ -z "$RAW_STMTS_FPATH" ]
    then
        echo "Raw statements file path is empty"
    fi
    if [ -z "$READING_TEXT_CONTENT_META_FPATH" ]
    then
        echo "Reading text content meta file path is empty"
    fi
    if [ -z "$TEXT_REFS_PRINCIPAL_FPATH" ]
    then
        echo "Text refs principal file path is empty"
    fi
    exit 1
else
    echo "Raw statements file path: $RAW_STMTS_FPATH"
    echo "Reading text content meta file path: $READING_TEXT_CONTENT_META_FPATH"
    echo "Text refs principal file path: $TEXT_REFS_PRINCIPAL_FPATH"
fi

# Exit if any of the file names are empty
if [ ! -f "$RAW_STMTS_FPATH" ]
then
    echo "Dumping raw statements"
    start=$(date +%s)
    psql -d indradb_test \
         -h indradb-refresh.cvyak4iikv71.us-east-1.rds.amazonaws.com \
         -U tester \
         -w \
         -c "COPY (SELECT id, db_info_id, reading_id,
                   convert_from (json::bytea, 'utf-8')
                   FROM public.raw_statements)
             TO STDOUT" \
          | gzip > "$RAW_STMTS_FPATH"
    end=$(date +%s)
    runtime=$((end-start))
    echo "Dumped raw statements in $runtime seconds"
else
    echo "Raw statements file already exists, skipping dump"
fi

if [ ! -f "$READING_TEXT_CONTENT_META_FPATH" ]
then
    echo "Dumping reading text content meta"
    start=$(date +%s)
    psql -d indradb_test \
         -h indradb-refresh.cvyak4iikv71.us-east-1.rds.amazonaws.com \
         -U tester \
         -w \
         -c "COPY (SELECT rd.id, rd.reader_version, tc.id, tc.text_ref_id,
                          tc.source, tc.text_type
                   FROM public.text_content as tc, public.reading as rd
                   WHERE tc.id = rd.text_content_id)
             TO STDOUT" \
         | gzip > "$READING_TEXT_CONTENT_META_FPATH"
    end=$(date +%s)
    runtime=$((end-start))
    echo "Dumped reading text content meta in $runtime seconds"
else
    echo "Reading text content meta file already exists, skipping dump"
fi

if [ ! -f "$TEXT_REFS_PRINCIPAL_FPATH" ]
then
    echo "Dumping text refs principal"
    start=$(date +%s)
    psql -d indradb_test \
         -h indradb-refresh.cvyak4iikv71.us-east-1.rds.amazonaws.com \
         -U tester \
         -w \
         -c "COPY (SELECT id, pmid, pmcid, doi, pii, url, manuscript_id
                   FROM public.text_ref)
             TO STDOUT" \
         | gzip > "$TEXT_REFS_PRINCIPAL_FPATH"
    end=$(date +%s)
    runtime=$((end-start))
    echo "Dumped text refs in $runtime seconds"
else
    echo "Text refs principal file already exists, skipping dump"
fi

# LOCAL DB CREATION AND DUMPING
week_number=$(date +%V)
#Get a full update using pubmed for corner cases every 10 weeks
if (( 10#$week_number % 10 == 0 )); then
  echo "Skipping export"
  unset MAPPING_SKIP_PUBMED
else
  echo "Setting MAPPING_SKIP_PUBMED=false"
  export MAPPING_SKIP_PUBMED=false
fi

python -m indra_db.readonly_dumping.export_assembly
python -m indra_db.readonly_dumping.export_assembly_refinement

# Create db;
PGPASSWORD=$LOCAL_RO_PASSWORD
export PGPASSWORD

psql -h localhost -U postgres -c "DROP DATABASE IF EXISTS $LOCAL_RO_DB_NAME"
psql -h localhost -U postgres -c "CREATE DATABASE $LOCAL_RO_DB_NAME"
## Run import script
python3 -m indra_db.readonly_dumping.readonly_dumping \
        --db-name $LOCAL_RO_DB_NAME \
        --user $LOCAL_RO_USER \
        --password "$LOCAL_RO_PASSWORD"
        # --force  # Use if you want to overwrite an existing db, if it exists

# Dump the db, once done importing
pg_dump -h localhost \
        -U postgres \
        -w \
        -f "${LOCAL_RO_DB_NAME}.dump" $LOCAL_RO_DB_NAME

## copy to s3
aws s3 cp "${LOCAL_RO_DB_NAME}.dump" "s3://bigmech/indra-db/dumps/"

# Remove dump file only after it has been copied to s3 successfully
#rm "${LOCAL_RO_DB_NAME}.dump"

# Upload an end date file to S3
# This is used to keep track of the end date of the dump
# The file is uploaded to the indra-db/dumps/ directory
# The file name is the current date and time

# Get the current date and time
END_DATE_TIME=`date '+%Y-%m-%d %H:%M:%S'`
END_DATE=`date '+%Y-%m-%d'`
echo "{\"datetime\": \"$END_DATE_TIME\", \"date_stamp\": \"$END_DATE\"}" > end.json
aws s3 cp end.json "$S3_PATH/end.json"

# At this point, if a new readonly instance is already created, we could run
# the following command to update the instance (assuming the password is set
# in PGPASSWORD, which will be read if -w is set):
# pg_restore -h <readonly-instance>.us-east-1.rds.amazonaws.com \
#            -U <user-name> \
#            -f <dump-file> \
#            -w \
#            -d indradb_readonly \
#            --no-owner