FROM ubuntu:24.04

# Set working folder
ENV DIRPATH /sw
WORKDIR $DIRPATH

RUN apt-get update && \
    # Install Java
    apt-get install -y openjdk-8-jdk && \
    # jnius-indra requires cython which requires gcc
    apt-get install -y git wget zip unzip bzip2 gcc graphviz graphviz-dev \
        pkg-config python3 python3-pip python3.12-venv && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get install -y locales && \
    locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

# Create virtual environment (required for ubuntu 24.04)
ENV VENV_PATH=$DIRPATH/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

RUN mkdir -p /root/.config/indra && \
    echo "[indra]" > /root/.config/indra/config.ini && \
    # Install Python dependencies
    pip install --upgrade pip && \
    # Install cython first for pyjnius
    pip install "cython<3.0" setuptools setuptools_scm wheel && \
    pip install "pyjnius==1.1.4" --no-build-isolation && \
    pip install --prefer-binary python-libsbml || true && \
    pip install "pysb>=1.17.0" && \
    pip install "indra[all] @ git+https://github.com/gyorilab/indra.git" && \
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

COPY . /sw/indra_db
RUN cd /sw/indra_db && pip install -e .[service]

RUN mkdir /app
WORKDIR /app

# Copy startup script that dynamically sets GILDA_TERMS based on installed version
COPY indra_db_service/docker/docker_startup.sh /app/docker_startup.sh
RUN chmod +x /app/docker_startup.sh

ENTRYPOINT ["/app/docker_startup.sh"]
