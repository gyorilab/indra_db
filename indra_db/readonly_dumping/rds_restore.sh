#!/bin/bash

# Log start time
echo "[$(date)] Starting RDS instance creation..."

# Step 1: Create the RDS instance
aws rds create-db-instance \
    --db-instance-identifier readonly-test \
    --db-instance-class db.m5.xlarge \
    --engine postgres \
    --allocated-storage 500 \
    --master-username masteruser \
    --master-user-password testpassword \
    --vpc-security-group-ids sg-0c49d0d42c8ae49c1 \
    --availability-zone us-east-1a \
    --backup-retention-period 7 \
    --db-name postgres \
    --publicly-accessible

# Log progress
echo "[$(date)] RDS instance creation initiated. Waiting for it to be available..."

# Step 2: Wait for the RDS instance to become available
aws rds wait db-instance-available --db-instance-identifier readonly-test
echo "[$(date)] RDS instance is now available."

# Step 3: Get the RDS endpoint
RDS_ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier readonly-test \
    --query "DBInstances[0].Endpoint.Address" \
    --output text)

if [[ -z "$RDS_ENDPOINT" ]]; then
    echo "[$(date)] Failed to retrieve RDS endpoint."
    exit 1
fi

echo "[$(date)] RDS Endpoint: $RDS_ENDPOINT"

# Step 4: Connect to the RDS instance and create a database

echo "[$(date)] Connecting to RDS to create the database..."
PGPASSWORD=testpassword psql -h $RDS_ENDPOINT -U masteruser -d postgres -p 5432 -c "DROP DATABASE IF EXISTS indradb_readonly_test;"
PGPASSWORD=testpassword psql -h $RDS_ENDPOINT -U masteruser -d postgres -p 5432 -c "CREATE DATABASE indradb_readonly_test;"

if [[ $? -ne 0 ]]; then
    echo "[$(date)] Failed to create the database."
    exit 1
fi

echo "[$(date)] Database 'indradb_readonly_test' created successfully."

# Step 5: Restore the dump file from S3 into the new database
echo "[$(date)] Restoring dump file into the database..."

aws s3 cp s3://bigmech/indra-db/dumps/indradb_readonly_local_test.dump - | \
PGPASSWORD=testpassword psql -h $RDS_ENDPOINT -U masteruser -d indradb_readonly_test

if [[ $? -ne 0 ]]; then
    echo "[$(date)] Failed to restore the dump file."
    exit 1
fi

echo "[$(date)] Dump file restored successfully into 'indradb_readonly_test'."