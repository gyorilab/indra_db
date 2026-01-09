FROM ubuntu:22.04

# Set working folder
ENV DIRPATH /sw
WORKDIR $DIRPATH

RUN apt-get update && \
    # Install Java
    apt-get install -y openjdk-8-jdk && \
    # jnius-indra requires cython which requires gcc
    apt-get install -y git wget zip unzip bzip2 gcc graphviz graphviz-dev \
        pkg-config python3-pip cmake libxml2-dev swig && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get install -y locales && \
    locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

RUN mkdir -p /root/.config/indra && \
    echo "[indra]" > /root/.config/indra/config.ini && \
    # Install Python dependencies
    pip install --upgrade pip && \
    # Install cython first for pyjnius
    pip install "cython<3" && \
    pip install "pyjnius==1.1.4" --no-build-isolation && \
    pip install --prefer-binary python-libsbml || true && \
    pip install git+https://github.com/gyorilab/indra.git#egg=indra[all] && \
    pip uninstall -y enum34 && \
    # Install gilda
    pip install git+https://github.com/gyorilab/gilda.git && \
    # Install things related to API deployment
    pip install gunicorn psycopg2-binary && \
    git clone https://github.com/gyorilab/ui_util.git && \
    cd ui_util/indralab_auth_tools && \
    pip install . && \
    cd ../..

# Download resources
RUN \
    # Download protmapper resources
    python -m protmapper.resources && \
    # Pre-build the bio ontology
    python -m indra.ontology.bio build && \
    # Build the sqlite ontology cache
    python -c "from indra.ontology.bio.sqlite_ontology import build_sqlite_ontology; build_sqlite_ontology()" && \
    # Download Adeft models
    python -m adeft.download && \
    # Download gilda resources
    python -m gilda.resources && \
    # Set up SQLite adapter for Gilda
    python -m gilda.resources.sqlite_adapter

# INDRA DB Service setup
RUN git clone https://github.com/gyorilab/indra_db.git && \
    cd indra_db && \
    pip install -e .[service] && \
    cd ..

RUN mkdir /app
WORKDIR /app

ENV GILDA_TERMS /root/.data/gilda/1.5.0/grounding_terms.db

ENTRYPOINT ["gunicorn", "-w", "2", "-c", "/sw/indra_db/indra_db_service/gunicorn.conf.py", "-t", "330", "-b", "0.0.0.0:8090", "indra_db_service.api:app"]
