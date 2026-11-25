#!/bin/bash

# Dynamically set GILDA_TERMS environment variable based on installed gilda version
GILDA_VERSION=$(python -c "import gilda; print(gilda.__version__)")
export GILDA_TERMS="/root/.data/gilda/${GILDA_VERSION}/grounding_terms.db"

exec gunicorn -w 4 --threads 2 -t 330 -b 0.0.0.0:8090 \
  -c /sw/indra_db/indra_db_service/gunicorn.conf.py \
  --access-logfile - \
  --forwarded-allow-ips '*' \
  indra_db_service.api:app
