FROM 292075781285.dkr.ecr.us-east-1.amazonaws.com/indra:latest

ARG BUILD_BRANCH
ARG INDRA_BRANCH

ENV DIRPATH /sw
ENV PYTHONPATH "$PYTHONPATH:${DIRPATH}/covid-19"
WORKDIR $DIRPATH

RUN cd indra && \
    git fetch --all && \
    git checkout $INDRA_BRANCH && \
    echo "INDRA_BRANCH=" $INDRA_BRANCH && \
    pip install -e . -U

# Install libpq5 and some other necessities.
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - && \
    apt-get update && \
    apt-get install -y lsb-core && \
    echo "deb http://apt.postgresql.org/pub/repos/apt/ `lsb_release -cs`-pgdg main" | tee  /etc/apt/sources.list.d/pgdg.list && \
    apt-get update && \
    apt-get install -y libpq5 libpq-dev postgresql-client-13 postgresql-client-common && \
    pip install awscli

# Install psycopg2
RUN git clone https://github.com/psycopg/psycopg2.git && \
    cd psycopg2 && \
    python setup.py build && \
    python setup.py install

# Install pgcopy
RUN git clone https://github.com/pagreene/pgcopy.git && \
    cd pgcopy && \
    python setup.py install

# Install covid-19
RUN git clone https://github.com/indralab/covid-19.git

# Install sqlalchemy < 1.4 (due to indirect dependencies, it may be a later
# version in the indra:db image)
RUN pip install "sqlalchemy<1.4"

# Install indra_db
RUN git clone https://github.com/indralab/indra_db.git && \
    cd indra_db && \
    pip install -e .[all] && \
    pip list && \
    echo "PYTHONPATH =" $PYTHONPATH && \
    git checkout $BUILD_BRANCH && \
    echo "BUILD_BRANCH =" $BUILD_BRANCH && \
    git branch && \
    echo "[indra]" > /root/.config/indra/config.ini

