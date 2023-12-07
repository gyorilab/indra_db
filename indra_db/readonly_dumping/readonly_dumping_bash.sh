# SETUP

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

# Set the user for the local db
LOCAL_RO_USER="postgres"
export LOCAL_RO_USER

# Set the password for the local db
echo "Set password for the local database:"
read -s LOCAL_RO_PASSWORD
export LOCAL_RO_PASSWORD

# Set the name of the local db
LOCAL_RO_DB_NAME="indradb_readonly_local"
export LOCAL_RO_DB_NAME
echo "Local db name: $LOCAL_RO_DB_NAME"

# Upload a JSON file with the start date to S3
# This is used to keep track of the start date of the dump
# The file is uploaded to the indra-db/dumps/ directory
# The file name is the current date and time

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
TEXT_REFS_PRINCIPAL_FPATH=`python3 -m indra_db.readonly_dumping.locations text_refs_principal`
export TEXT_REFS_PRINCIPAL_FPATH

# Exit if any of the file names are empty
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

# Run dumps
# Only run the dumps if the files don't exist
if [ ! -f "$RAW_STMTS_FPATH" ]
then
    echo "Dumping raw statements"

    psql -d indradb_test \
         -h indradb-refresh.cwcetxbvbgrf.us-east-1.rds.amazonaws.com \
         -U tester \
         -w \
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
         -w \
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
         -w \
         -c "COPY (SELECT id, pmid, pmcid, doi, pii, url, manuscript_id
                   FROM public.text_ref)
             TO STDOUT" \
         | gzip > text_refs_principal.tsv.gz
fi

# LOCAL DB CREATION AND DUMPING

# Run export assembly script
python3 -m indra_db.readonly_dumping.export_assembly # --refresh-kb

# Create db;
# todo: how to pass or set password? Will it interfere with PGPASSWORD set above?
psql -h localhost -c "create database $LOCAL_RO_DB_NAME" -U postgres

# Run import script
python3 -m indra_db.readonly_dumping.readonly_dumping \
        --db-name $LOCAL_RO_DB_NAME \
        --user $LOCAL_RO_USER \
        --password "$LOCAL_RO_PASSWORD"
        # --force  # Use if you want to overwrite an existing db, if it exists

# Dump the db, once done importing
PGPASSWORD=$LOCAL_RO_PASSWORD
export PGPASSWORD
pg_dump -h localhost \
        -U postgres \
        -w \
        -f "${LOCAL_RO_DB_NAME}.dump" $LOCAL_RO_DB_NAME

# copy to s3
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