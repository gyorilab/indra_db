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
LOCAL_RO_DB_NAME="indradb_readonly_local"
export LOCAL_RO_DB_NAME
echo "Local db name: $LOCAL_RO_DB_NAME"



# INITIAL DUMPING

# Get file paths for initial dump files
RAW_STMTS_FPATH=`python3 -m indra_db.readonly_dumping.locations raw_statements`
export RAW_STMTS_FPATH
READING_TEXT_CONTENT_META_FPATH=`python3 -m indra_db.readonly_dumping.locations reading_text_content`
export READING_TEXT_CONTENT_META_FPATH
TEXT_REFS_PRINCIPAL_FPATH=`python3 -m indra_db.readonly_dumping.locations text_refs`
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

#get three initial file

# LOCAL DB CREATION AND DUMPING


# Create db;
# todo: how to pass or set password? Will it interfere with PGPASSWORD set above?
#psql -h localhost -c "create database $LOCAL_RO_DB_NAME" -U postgres

# Run import script
python3 -m indra_db.readonly_dumping.readonly_dumping \
        --db-name $LOCAL_RO_DB_NAME \
        --user $LOCAL_RO_USER \
        --password "$LOCAL_RO_PASSWORD"
        # --force  # Use if you want to overwrite an existing db, if it exists

# Dump the db, once done importing
#PGPASSWORD=$LOCAL_RO_PASSWORD
#export PGPASSWORD
#pg_dump -h localhost \
#        -U postgres \
#        -w \
#        -f "${LOCAL_RO_DB_NAME}.dump" $LOCAL_RO_DB_NAME



# Remove dump file only after it has been copied to s3 successfully
#rm "${LOCAL_RO_DB_NAME}.dump"

# Upload an end date file to S3
# This is used to keep track of the end date of the dump
# The file is uploaded to the indra-db/dumps/ directory
# The file name is the current date and time

## At this point, if a new readonly instance is already created, we could run
## the following command to update the instance (assuming the password is set
## in PGPASSWORD, which will be read if -w is set):
## pg_restore -h <readonly-instance>.us-east-1.rds.amazonaws.com \
##            -U <user-name> \
##            -f <dump-file> \
##            -w \
##            -d indradb_readonly \
##            --no-owner