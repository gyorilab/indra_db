#!/bin/bash

# Dynamically set GILDA_TERMS environment variable based on installed gilda version
GILDA_VERSION=$(python -c "import gilda; print(gilda.__version__)")
export GILDA_TERMS="/root/.data/gilda/${GILDA_VERSION}/grounding_terms.db"

exec gunicorn -w 2 -c /sw/indra_db/indra_db_service/gunicorn.conf.py -t 330 -b 0.0.0.0:8090 indra_db_service.api:app
