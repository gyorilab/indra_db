__all__ = ['TasManager', 'CBNManager', 'HPRDManager', 'SignorManager',
           'BiogridManager', 'BelLcManager', 'PathwayCommonsManager',
           'RlimspManager', 'TrrustManager', 'PhosphositeManager',
           'CTDManager', 'VirHostNetManager', 'PhosphoElmManager',
           'DrugBankManager']

import csv
import gzip
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Type

import click
import pickle
import logging
import tempfile
from itertools import count as iter_count
from collections import Counter, defaultdict

import requests
import yaml
from bs4 import BeautifulSoup

from indra.util import batch_iter
from tqdm import tqdm

from indra.statements.validate import assert_valid_statement
from indra.tools import assemble_corpus as ac
from indra_db.readonly_dumping.locations import knowledgebase_source_data_fpath, knowledgebase_version_record
from indra_db.util import insert_db_stmts
from indra_db.util.distill_statements import extract_duplicates, KeyFunc
from indra_db.readonly_dumping.locations import *

from indra_db.cli.util import format_date

logger = logging.getLogger(__name__)

KB_DIR = TEMP_DIR.module("knowledgebases")


class KnowledgebaseManager(object):
    """This is a class to lay out the methods for updating a dataset."""
    name: str = NotImplemented
    short_name: str = NotImplemented
    source: str = NotImplemented

    def upload(self, db):
        """Upload the content for this dataset into the database."""
        dbid = self._check_reference(db)
        stmts = self.get_statements()
        # Raise any validity issues with statements as exceptions here
        # to avoid uploading invalid content.
        for stmt in stmts:
            assert_valid_statement(stmt)
        insert_db_stmts(db, stmts, dbid)
        return

    def update(self, db):
        """Add any new statements that may have come into the dataset."""
        dbid = self._check_reference(db, can_create=False)
        if dbid is None:
            raise ValueError("This knowledge base has not yet been "
                             "registered.")
        existing_keys = set(db.select_all([db.RawStatements.mk_hash,
                                           db.RawStatements.source_hash],
                                          db.RawStatements.db_info_id == dbid))
        stmts = self.get_statements()
        filtered_stmts = [s for s in stmts
                          if (s.get_hash(), s.evidence[0].get_source_hash())
                          not in existing_keys]
        insert_db_stmts(db, filtered_stmts, dbid)
        return

    @classmethod
    def get_last_update(cls, db):
        """Get the last time the row was updated or created."""
        dbinfo = db.select_one(db.DBInfo, db.DBInfo.db_name == cls.short_name)
        if dbinfo.last_updated:
            return dbinfo.last_updated
        else:
            return dbinfo.create_date

    def _check_reference(self, db, can_create=True):
        """Ensure that this database has an entry in the database."""
        dbinfo = db.select_one(db.DBInfo, db.DBInfo.db_name == self.short_name)
        if dbinfo is None:
            if can_create:
                dbid = db.insert(db.DBInfo, db_name=self.short_name,
                                 source_api=self.source, db_full_name=self.name)
            else:
                return None
        else:
            dbid = dbinfo.id
            if dbinfo.source_api != self.source:
                dbinfo.source_api = self.source
                db.commit("Could not update source_api for %s."
                          % dbinfo.db_name)
        return dbid

    def get_statements(self, **kwargs):
        raise NotImplementedError("Statement retrieval must be defined in "
                                  "each child.")

    def get_source_version(self):
        raise NotImplementedError("Version retrieval must be defined in "
                                  "each child.")

    def get_local_fpath(self) -> Path:
        """Return the local path to the knowledge base file."""
        if self.short_name == self.source:
            local_name = self.short_name
        elif self.short_name in self.source or self.source in self.short_name:
            # Pick the longer name
            local_name = self.short_name \
                if len(self.short_name) > len(self.source) else self.source
        else:
            local_name = f"{self.short_name}_{self.source}"
        return KB_DIR.join(name=f"processed_stmts_{local_name}.tsv.gz")


class TasManager(KnowledgebaseManager):
    """This manager handles retrieval and processing of the TAS dataset."""
    # TODO: Data is simply a CSV from S3
    name = 'TAS'
    short_name = 'tas'
    source = 'tas'

    def get_statements(self):
        from indra.sources import tas
        # The settings we use here are justified as follows:
        # - only affinities that indicate binding are included
        # - only agents that have some kind of a name available are
        #   included, with ones that get just an ID as a name are
        #   not included.
        # - we do not require full standardization, thereby allowing
        #   set of drugs to be extracted for which we have a name from CHEBML,
        #   HMS-LINCS, or DrugBank
        logger.info('Processing TAS from web')
        tp = tas.process_csv(knowledgebase_source_data_fpath.joinpath("tas.csv"))
        logger.info('Expanding evidences and deduplicating')
        filtered_stmts = [s for s in _expanded(tp.statements)]
        unique_stmts, _ = extract_duplicates(filtered_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts

    def get_source_version(self):
        # In get_statement, it checks the checksum of the source
        # So the version can be directly used
        from indra.sources.tas.api import tas_resource_md5
        return tas_resource_md5


class SignorManager(KnowledgebaseManager):
    name = 'Signor'
    short_name = 'signor'
    source = 'signor'

    def get_statements(self, **kwargs):
        from indra.sources.signor import process_from_web, process_from_file
        if kwargs.get("signor_data_file"):
            data_file = kwargs.pop("signor_data_file")
            complexes_file = kwargs.pop("signor_complexes_file", None)
            proc = process_from_file(signor_data_file=data_file,
                                     signor_complexes_file=complexes_file,
                                     **kwargs)
        else:
            proc = process_from_web()
        return proc.statements

    def get_source_version(self):
        data_url = 'https://signor.uniroma2.it/download_entity.php'
        complex_url = 'https://signor.uniroma2.it/download_complexes.php'
        res = requests.post(data_url, data={'organism': 'human', 'format': 'csv', 'submit': 'Download'})
        complex_res = requests.post(complex_url, data={'submit': 'Download complex data'})
        if res.status_code == 200 and complex_res.status_code == 200:
            content1 = res.content
            content2 = complex_res.content
            checksum1 = hashlib.md5(content1).hexdigest()
            checksum2 = hashlib.md5(content2).hexdigest()
            return checksum1+checksum2
        else:
            logger.error("Could not get checksum(version)")



class CBNManager(KnowledgebaseManager):
    """This manager handles retrieval and processing of CBN network files"""
    name = 'Causal Bionet'
    short_name = 'cbn'
    source = 'bel'

    def __init__(
            self,
            archive_url="https://github.com/pybel/cbn-bel/raw/master/Human-2.0.zip"
    ):
        self.archive_url = archive_url

    def get_statements(self):
        import requests
        from zipfile import ZipFile
        from indra.sources.bel.api import process_cbn_jgif_file
        import tempfile

        cbn_dir = tempfile.mkdtemp('cbn_manager')

        logger.info('Retrieving CBN network zip archive')
        tmp_zip = os.path.join(cbn_dir, 'cbn_human.zip')
        resp = requests.get(self.archive_url)
        resp.raise_for_status()
        with open(tmp_zip, 'wb') as f:
            f.write(resp.content)

        stmts = []
        tmp_dir = os.path.join(cbn_dir, 'cbn')
        os.mkdir(tmp_dir)
        with ZipFile(tmp_zip) as zipf:
            logger.info('Extracting archive to %s' % tmp_dir)
            zipf.extractall(path=tmp_dir)
            logger.info('Processing jgif files')
            for jgif in tqdm(zipf.namelist()):
                if jgif.endswith('.jgf') or jgif.endswith('.jgif'):
                    pbp = process_cbn_jgif_file(os.path.join(tmp_dir, jgif))
                    stmts += pbp.statements

        uniques, dups = extract_duplicates(stmts,
                                           key_func=KeyFunc.mk_and_one_ev_src)

        logger.info("Deduplicating...")
        print('\n'.join(str(dup) for dup in dups))
        print(len(dups))

        return uniques

    def get_source_version(self):
        start = self.archive_url.find("/raw/")
        if start != -1:
            version = self.archive_url[start:]
        else:
            version = None
        return version


class BiogridManager(KnowledgebaseManager):
    name = 'BioGRID'
    short_name = 'biogrid'
    source = 'biogrid'

    def get_statements(self):
        from indra.sources import biogrid
        bp = biogrid.BiogridProcessor()
        return list(_expanded(bp.statements))

    def get_source_version(self):
        url = "https://downloads.thebiogrid.org/BioGRID/Release-Archive"
        soup = BeautifulSoup(requests.get(url).content, "html.parser")
        matches = re.findall(r'BIOGRID-(\d+\.\d+\.\d+)', soup.text)
        if matches:
            latest_version = max(matches,
                                 key=lambda s: tuple(map(int, s.split('.'))))
            return latest_version
        return None


class PathwayCommonsManager(KnowledgebaseManager):
    name = 'Pathway Commons'
    short_name = 'pc'
    source = 'biopax'
    skips = {'psp', 'hprd', 'biogrid', 'phosphosite', 'phosphositeplus',
             'ctd', 'drugbank'}

    def __init__(self, *args, **kwargs):
        self.url = "https://download.baderlab.org/PathwayCommons/PC2/v14/pc-biopax.owl.gz"
        self.counts = Counter()
        super(PathwayCommonsManager, self).__init__(*args, **kwargs)

    def _can_include(self, stmt):
        num_ev = len(stmt.evidence)
        assert num_ev == 1, "Found statement with %d evidence." % num_ev

        ev = stmt.evidence[0]
        ssid = ev.annotations['source_sub_id']
        self.counts[ssid] += 1

        return ssid not in self.skips

    def get_statements(self):
        v = self.get_source_version()
        from indra.sources import biopax
        response = requests.get(self.url, stream=True)
        if response.status_code == 200:
            # download the source file
            with open(knowledgebase_source_data_fpath.joinpath(
                    f"{v}_pc-biopax.owl.gz"), 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        else:
            logger.error(f"Failed to download the file. Status code: {response.status_code}")

        prc = biopax.process_owl_gz(knowledgebase_source_data_fpath.joinpath(
            f"{v}_pc-biopax.owl.gz"))
        stmts = prc.statements

        logger.info('Loading PC statements from pickle')

        logger.info('Expanding evidences and deduplicating')
        filtered_stmts = [s for s in _expanded(stmts) if self._can_include(s)]
        unique_stmts, _ = extract_duplicates(filtered_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts

    def get_source_version(self):
        match = re.search(r'/v\d+/', self.url)
        if match:
            version = match.group(0).strip('/')
            return version
        else:
            logger.error("Version not found in the URL")



class CTDManager(KnowledgebaseManager):
    name = 'CTD'
    source = 'ctd'
    short_name = 'ctd'
    subsets = ['gene_disease', 'chemical_disease',
               'chemical_gene']

    def get_statements(self):
        from indra.sources.ctd import process_from_web
        all_stmts = []
        for subset in tqdm(self.subsets, desc="CTD subsets"):
            ctd_processor = process_from_web(subset)
            all_stmts += [s for s in _expanded(ctd_processor.statements)]
        # Return exactly one of multiple statements that are exactly the same
        # in terms of content and evidence.
        unique_stmts, _ = extract_duplicates(all_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts
    def get_source_version(self):
        from indra.sources.ctd.api import urls
        # Check if 'Last-Modified' is in the headers
        version = ''
        for subset, url in urls.items():
            response = requests.head(url)
            if 'Last-Modified' in response.headers:
                last_modified = response.headers['Last-Modified']
                version += + subset + last_modified
            else:
                logger.error("Last-Modified header not found")
        return version


class DrugBankManager(KnowledgebaseManager):
    name = 'DrugBank'
    short_name = 'drugbank'
    source = 'drugbank'

    def get_statements(self):
        from indra.sources import drugbank
        # For now. Load from local since aws is not setted up
        prc = drugbank.process_xml(knowledgebase_source_data_fpath.joinpath("drugbank.xml"))
        expanded_stmts = [s for s in _expanded(prc.statements)]
        # Return exactly one of multiple statements that are exactly the same
        # in terms of content and evidence.
        unique_stmts, _ = extract_duplicates(expanded_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts
    def get_source_version(self):
        with open(knowledgebase_source_data_fpath.joinpath("drugbank.xml"), 'r', encoding='utf-8') as file:
            version = None
            for _ in range(10):
                line = file.readline()
                if '<drugbank' in line and 'version=' in line:
                    version = line.split('version="')[1].split('"')[0]
                    break
        return version


class VirHostNetManager(KnowledgebaseManager):
    name = 'VirHostNet'
    short_name = 'vhn'
    source = 'virhostnet'

    def get_statements(self):
        from indra.sources import virhostnet
        vp = virhostnet.process_from_web()
        return [s for s in _expanded(vp.statements)]

    def get_source_version(self):
        url = "http://virhostnet.prabi.fr:9090/psicquic/webservices/current/search/query/*"
        response = requests.get(url)
        response.raise_for_status()
        content = response.content
        md5_checksum = hashlib.md5(content).hexdigest()
        return md5_checksum


class PhosphoElmManager(KnowledgebaseManager):
    name = 'Phospho.ELM'
    short_name = 'pe'
    source = 'phosphoelm'

    def get_statements(self):
        from indra.sources import phosphoelm

        pp = phosphoelm.process_from_dump(knowledgebase_source_data_fpath.joinpath("phosphoELM_all_2015-04.dump"))

        stmts = [s for s in _expanded(pp.statements)]
        # Return exactly one of multiple statements that are exactly the same
        # in terms of content and evidence.
        # Now make sure we don't include exact duplicates
        unique_stmts, _ = extract_duplicates(stmts, KeyFunc.mk_and_one_ev_src)
        return unique_stmts

    def get_source_version(self):
        latest_file = max(
            (f for f in knowledgebase_source_data_fpath.glob("phosphoELM*.dump")),
            key=lambda f: tuple(map(int, f.stem.split('_')[-1].split('-'))),
            default=None
        )
        return latest_file.stem.split('_')[-1]



class HPRDManager(KnowledgebaseManager):
    name = 'HPRD'
    short_name = 'hprd'
    source = 'hprd'

    def get_statements(self):
        import tarfile
        import requests
        from indra.sources import hprd

        # Download the files.
        url = "https://rescued.omnipathdb.org/HPRD_FLAT_FILES_041310.tar.gz"
        resp = requests.get(url)
        resp.raise_for_status()
        tmp_dir = tempfile.mkdtemp('hprd_files')
        tmp_tarfile = os.path.join(tmp_dir, 'hprd_files.tar.gz')
        with open(tmp_tarfile, 'wb') as f:
            f.write(resp.content)

        # Extract the files.
        with tarfile.open(tmp_tarfile, 'r:gz') as tf:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tf, tmp_dir)

        # Find the relevant files.
        dirs = os.listdir(tmp_dir)
        for files_dir in dirs:
            if files_dir.startswith('FLAT_FILES'):
                break
        else:
            # Loop doesn't break: FLAT_FILES directory not found
            raise NotADirectoryError('Could not find FLAT_FILES directory.')
        files_path = os.path.join(tmp_dir, files_dir)
        file_names = {'id_mappings_file': 'HPRD_ID_MAPPINGS',
                      'complexes_file': 'PROTEIN_COMPLEXES',
                      'ptm_file': 'POST_TRANSLATIONAL_MODIFICATIONS',
                      'ppi_file': 'BINARY_PROTEIN_PROTEIN_INTERACTIONS',
                      'seq_file': 'PROTEIN_SEQUENCES'}
        kwargs = {kw: os.path.join(files_path, fname + '.txt')
                  for kw, fname in file_names.items()}

        # Run the processor
        hp = hprd.process_flat_files(**kwargs)

        # Filter out exact duplicates
        unique_stmts, dups = \
            extract_duplicates(_expanded(hp.statements),
                               key_func=KeyFunc.mk_and_one_ev_src)
        print('\n'.join(str(dup) for dup in dups))

        return unique_stmts


class BelLcManager(KnowledgebaseManager):
    name = 'BEL Large Corpus'
    short_name = 'bel_lc'
    source = 'bel'

    def get_statements(self):
        from indra.sources import bel

        pbp = bel.process_large_corpus()
        stmts = pbp.statements
        pbp = bel.process_small_corpus()
        stmts += pbp.statements
        stmts, dups = extract_duplicates(stmts,
                                         key_func=KeyFunc.mk_and_one_ev_src)
        print('\n'.join(str(dup) for dup in dups))
        print(len(stmts), len(dups))
        return stmts

    def get_source_version(self):
        from indra.sources.bel.api import version
        return version


class PhosphositeManager(KnowledgebaseManager):
    name = 'Phosphosite Plus'
    short_name = 'psp'
    source = 'biopax'

    def get_statements(self):
        from indra.sources import biopax
        bp = biopax.process_owl_gz(knowledgebase_source_data_fpath.joinpath('Kinase_substrates.owl.gz'))
        stmts, dups = extract_duplicates(bp.statements,
                                         key_func=KeyFunc.mk_and_one_ev_src)
        print('\n'.join(str(dup) for dup in dups))
        print(len(stmts), len(dups))
        return stmts

    def get_source_version(self):
        md5_hash = hashlib.md5()
        with open(knowledgebase_source_data_fpath.joinpath('Kinase_substrates.owl.gz'), 'rb') as file:
            for chunk in iter(lambda: file.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()


class RlimspManager(KnowledgebaseManager):
    name = 'RLIMS-P'
    short_name = 'rlimsp'
    source = 'rlimsp'
    _rlimsp_root = 'https://hershey.dbi.udel.edu/textmining/export/'
    _rlimsp_files = [('rlims.medline.json', 'pmid'),
                     ('rlims.pmc.json', 'pmcid')]

    def get_statements(self):
        from indra.sources import rlimsp
        import requests

        stmts = []
        for fname, id_type in self._rlimsp_files:
            print("Processing %s..." % fname)
            res = requests.get(self._rlimsp_root + fname)
            res.raise_for_status()
            jsonl_str = res.content.decode('utf-8')
            rp = rlimsp.process_jsonl_str(jsonl_str, id_type)
            stmts += rp.statements
            print("Added %d more statements from %s..."
                  % (len(rp.statements), fname))

        stmts, dups = extract_duplicates(_expanded(stmts),
                                         key_func=KeyFunc.mk_and_one_ev_src)
        print('\n'.join(str(dup) for dup in dups))
        print(len(stmts), len(dups))

        return stmts

    def get_source_version(self):
        response = requests.get(self._rlimsp_root)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        version=''
        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 4:
                file_name = cells[1].text.strip()
                if file_name in [f[0] for f in self._rlimsp_files]:
                    last_modified = cells[2].text.strip()
                    version=version+file_name+last_modified
        return version

class TrrustManager(KnowledgebaseManager):
    name = 'TRRUST'
    short_name = 'trrust'
    source = 'trrust'

    def get_statements(self):
        from indra.sources import trrust
        tp = trrust.process_from_web()
        unique_stmts, dups = \
            extract_duplicates(_expanded(tp.statements),
                               key_func=KeyFunc.mk_and_one_ev_src)
        return unique_stmts

    def get_source_version(self):
        from indra.sources.trrust.api import trrust_human_url
        response = requests.get(trrust_human_url, stream=True)
        response.raise_for_status()
        md5_hash = hashlib.md5()
        for chunk in response.iter_content(chunk_size=8192):
            md5_hash.update(chunk)
        return md5_hash.hexdigest()


def _expanded(stmts):
    for stmt in stmts:
        # Only one evidence is allowed for each statement.
        if len(stmt.evidence) > 1:
            for ev in stmt.evidence:
                new_stmt = stmt.make_generic_copy()
                new_stmt.evidence.append(ev)
                yield new_stmt
        else:
            yield stmt


class DgiManager(KnowledgebaseManager):
    """This manager handles retrieval and processing of the DGI dataset."""
    name = 'DGI'
    short_name = 'dgi'
    source = 'dgi'

    def get_statements(self):
        from indra.sources import dgi
        logger.info('Processing DGI from web')
        dp = dgi.process_version()
        logger.info('Expanding evidences and deduplicating')
        filtered_stmts = [s for s in _expanded(dp.statements)]
        unique_stmts, _ = extract_duplicates(filtered_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts

    def get_source_version(self):
        url = "https://raw.githubusercontent.com/dgidb/dgidb-v5/main/server/data_version.yml"
        response = requests.get(url)
        response.raise_for_status()
        version_data = yaml.safe_load(response.text)
        return version_data['version']


class CrogManager(KnowledgebaseManager):
    """This manager handles retrieval and processing of the CRoG dataset."""
    name = 'CRoG'
    short_name = 'crog'
    source = 'crog'

    def get_statements(self):
        from indra.sources import crog
        logger.info('Processing CRoG from web')
        cp = crog.process_from_web()
        logger.info('Expanding evidences and deduplicating')
        filtered_stmts = [s for s in _expanded(cp.statements)]
        unique_stmts, _ = extract_duplicates(filtered_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts

    def get_source_version(self):
        url = "https://api.github.com/repos/chemical-roles/chemical-roles/commits?path=docs/_data/crog.indra.json"
        response = requests.get(url)
        response.raise_for_status()
        commits = response.json()
        return commits[0]['commit']['committer']['date']


class ConibManager(KnowledgebaseManager):
    """This manager handles retrieval and processing of the CONIB dataset."""
    name = 'CONIB'
    short_name = 'conib'
    source = 'bel'

    def get_statements(self):
        import pybel
        import requests
        from indra.sources.bel import process_pybel_graph
        logger.info('Processing CONIB from web')
        url = 'https://github.com/pharmacome/conib/raw/master/conib' \
              '/_cache.bel.nodelink.json'
        res_json = requests.get(url).json()
        graph = pybel.from_nodelink(res_json)
        # Get INDRA statements
        pbp = process_pybel_graph(graph)

        # Fix and issue with PMID spaces
        for stmt in pbp.statements:
            for ev in stmt.evidence:
                if ev.pmid:
                    ev.pmid = ev.pmid.strip()
                if ev.text_refs.get('PMID'):
                    ev.text_refs['PMID'] = ev.text_refs['PMID'].strip()

        logger.info('Expanding evidences and deduplicating')
        filtered_stmts = [s for s in _expanded(pbp.statements)]
        unique_stmts, _ = extract_duplicates(filtered_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts

    def get_source_version(self):
        url = "https://api.github.com/repos/pharmacome/conib/commits?path=conib/_cache.bel.nodelink.json"
        response = requests.get(url)
        response.raise_for_status()
        commits = response.json()
        return commits[0]['commit']['committer']['date']


class UbiBrowserManager(KnowledgebaseManager):
    """This manager handles retrieval and processing of UbiBrowser data."""
    name = 'UbiBrowser'
    short_name = 'ubibrowser'
    source = 'ubibrowser'

    def get_statements(self):
        from indra.sources import ubibrowser
        logger.info('Processing UbiBrowser from web')
        up = ubibrowser.process_from_web()
        logger.info('Expanding evidences and deduplicating')
        filtered_stmts = [s for s in _expanded(up.statements)]
        unique_stmts, _ = extract_duplicates(filtered_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts

    def get_source_version(self):
        from indra.sources.ubibrowser.api import DOWNLOAD_URL
        E3_URL = DOWNLOAD_URL + 'literature.E3.txt'
        DUB_URL = DOWNLOAD_URL + 'literature.DUB.txt'
        e3_response = requests.get(E3_URL)
        e3_md5 = hashlib.md5(e3_response.content).hexdigest()
        dub_response = requests.get(DUB_URL)
        dub_md5 = hashlib.md5(dub_response.content).hexdigest()
        return e3_md5+dub_md5



def local_update(
        kb_manager_list: List[Type[KnowledgebaseManager]],
        local_files: Dict[str, Dict] = None,
        refresh: bool = False,
):
    """Update the knowledgebases of a local raw statements file dump

    Parameters
    ----------
    kb_manager_list :
        List of the (un-instantiated) classes of the knowledgebase managers
        to use in update.
    local_files :
        Dictionary of local files to use in the update. Keys are the
        knowledgebase short names, values are kwargs to pass to the
        knowledgebase manager get_statements method.
    refresh :
        If True, the local files will be recreated even if they already exist.
    """

    def _get_kb_info_map():
        # Get db info id mapping
        from indra_db.util import get_db
        db = get_db('primary')

        res = db.select_all(db.DBInfo)
        kb_mapping = {(r.source_api, r.db_name): r.id for r in res}
        return kb_mapping

    if not kb_manager_list:
        raise ValueError(
            "No knowledgebase managers provided, nothing to update"
        )

    # Get the ids of the knowledgebases to update
    kbs_to_run = []
    existing_kbs = []
    versions = {}
    now = datetime.now()
    modified_date = now.strftime("%b-%d-%y")

    #load kb versions
    with open(knowledgebase_version_record, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            key, version, date = row
            versions[key] = (version, date)

    for kb_manager in kb_manager_list:
        kbm = kb_manager()
        # Add the knowledgebase to the list if it is not already there
        # or if we are refreshing

        #check if we need updates according to versions
        old_version = versions.get(kbm.source+kbm.short_name, (None,))[0]


        try:
            #try/except deal with cases when source website is down
            new_version = kbm.get_source_version()
        except:
            new_version = old_version

        if new_version != old_version:
            versions[kbm.source+kbm.short_name] = (new_version, modified_date)
            need_update = True
        else:
            need_update = False

        if refresh or not kbm.get_local_fpath().exists() or need_update:
            kbs_to_run.append(kb_manager)
        else:
            assert kbm.get_local_fpath().exists()
            existing_kbs.append(kb_manager)

    #save the updated kb versions
    with open(knowledgebase_version_record.absolute().as_posix(), 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for key, value in versions.items():
            writer.writerow([key, value[0], value[1]])

    if kbs_to_run:
        logger.info("Generating new statements from these knowledgebases:")
        for kb_manager in kbs_to_run:
            logger.info(f"  {kb_manager.name} ({kb_manager.short_name})")
    else:
        # If no new KBs need to be run, check if all outputs already exist,
        # and if so, do nothing
        if all(
                of.exists() for of in [
                    raw_id_info_map_knowledgebases_fpath,
                    stmt_hash_to_raw_stmt_ids_knowledgebases_fpath,
                    source_counts_knowledgebases_fpath
                ]
        ):
            logger.info("All knowledgebases already exist, nothing to do.")

    if existing_kbs:
        logger.info("Using existing statements from the following knowledgebases:")
        for kb_manager in existing_kbs:
            logger.info(f"  {kb_manager.name} ({kb_manager.short_name})")

    total = len(kbs_to_run) + len(existing_kbs)
    source_counts = {}
    counts = Counter()
    stmt_hash_to_raw_id = defaultdict(set)
    db_info_map = _get_kb_info_map()

    # Generate fake raw statement id for each statement, from -1 and down
    fake_raw_id_generator = iter_count(-1, -1)
    with gzip.open(
            raw_id_info_map_knowledgebases_fpath.as_posix(), "wt"
    ) as info_fh:
        kb_info_writer = csv.writer(info_fh, delimiter="\t")

        outer_tqdm = tqdm(
            desc="Knowledgebases", total=total, unit="knowledgebase"
        )
        # First run through the existing knowledgebases
        logger.info(f"Loading existing statements")
        for ix, Mngr in enumerate(existing_kbs):
            kbm = Mngr()
            tqdm.write(f"[{ix + 1}/{total}] {kbm.name} ({kbm.short_name})")
            with gzip.open(kbm.get_local_fpath(), "rt") as stmts_fh:
                stmts_reader = csv.reader(stmts_fh, delimiter="\t")
                for row in tqdm(stmts_reader, leave=False):
                    stmt_hash_str, stmt_json_str = row
                    raw_id = next(fake_raw_id_generator)
                    stmt_hash = int(stmt_hash_str)
                    db_info_id = db_info_map[(kbm.source, kbm.short_name)]

                    # Add to raw id mappings
                    stmt_hash_to_raw_id[stmt_hash].add(raw_id)
                    # raw_id, db_info_id, (reading_id), stmt_json
                    kb_info_writer.writerow(
                        (raw_id, db_info_id, "\\N", stmt_json_str)
                    )
                    # Update count
                    counts[(kbm.source, kbm.short_name)] += 1

                    # Get source count for this statement, or create new
                    # Counter if it doesn't exist, and increment the count
                    source_count_dict = source_counts.get(stmt_hash, Counter())
                    source_count_dict[kbm.source] += 1
                    source_counts[stmt_hash] = source_count_dict
            outer_tqdm.update(1)

        # Then run through the knowledgebases that are generating new output
        for ix, Mngr in enumerate(kbs_to_run, start=len(existing_kbs)):
            kbm = Mngr()
            db_info_id = db_info_map[(kbm.source, kbm.short_name)]
            tqdm.write(
                f"[{ix + 1}/{total}] {kbm.name} ({kbm.short_name})"
            )

            # Write statements for this knowledgebase to a file
            fname = kbm.get_local_fpath().as_posix()
            with gzip.open(fname, "wt") as proc_stmts_fh:
                proc_stmts_writer = csv.writer(proc_stmts_fh, delimiter="\t")

                if local_files is not None:
                    kb_kwargs = local_files.get(kbm.short_name, {})
                else:
                    kb_kwargs = {}
                stmts = kbm.get_statements(**kb_kwargs)

                # Do preassembly
                if len(stmts) > 100000:
                    stmts_iter = batch_iter(stmts, 100000)
                    batches = True
                    t = tqdm(desc=f"Preassembling {kbm.short_name}",
                             total=len(stmts) // 100000 + 1)
                else:
                    stmts_iter = [stmts]
                    batches = False
                    t = None
                    logger.info(f"Preassembling {kbm.short_name}")

                for stmts in stmts_iter:
                    # Pre-process statements
                    stmts = ac.fix_invalidities(list(stmts), in_place=True)
                    stmts = ac.map_grounding(stmts)
                    stmts = ac.map_sequence(stmts)
                    rows = []
                    kb_info_rows = []
                    for stmt in tqdm(stmts, leave=not batches):
                        raw_id = next(fake_raw_id_generator)
                        # Get the statement hash and update the source count
                        stmt_hash = stmt.get_hash(refresh=True)

                        # Get source count for this statement, or create a new
                        # Counter if it doesn't exist, and increment the count
                        source_count_dict = source_counts.get(stmt_hash, Counter())
                        source_count_dict[kbm.short_name] += 1
                        source_counts[stmt_hash] = source_count_dict

                        # Append to various raw id mappings
                        stmt_hash_to_raw_id[stmt_hash].add(raw_id)
                        kb_info_rows.append(
                            # raw_id, db_info_id, (reading_id), stmt_json
                            (raw_id, db_info_id, "\\N", stmt.to_json())
                        )

                        rows.append((stmt_hash, json.dumps(stmt.to_json())))
                    proc_stmts_writer.writerows(rows)
                    kb_info_writer.writerows(kb_info_rows)
                    if batches:
                        t.update(1)
                    counts[(kbm.source, kbm.short_name)] += len(stmts)
            if batches:
                t.close()

    logger.info("Statements produced per knowledgebase:")
    for (source, short_name), count in counts.most_common():
        logger.info(f"  - {source} {short_name}: {count}")
    logger.info(f"Total rows added: {sum(counts.values())}")

    # Dump source counts
    with source_counts_knowledgebases_fpath.open("wb") as src_count_fh:
        pickle.dump(source_counts, src_count_fh)

    # Dump stmt hash to raw stmt id mapping
    stmt_hash_to_raw_id = dict(stmt_hash_to_raw_id)
    with stmt_hash_to_raw_stmt_ids_knowledgebases_fpath.open("wb") as hr_fh:
        pickle.dump(stmt_hash_to_raw_id, hr_fh)


@click.group()
def kb():
    """Manage the Knowledge Bases used by the database."""


@kb.command()
@click.argument("task", type=click.Choice(["upload", "update", "local-update"]))
@click.argument("sources", nargs=-1, type=click.STRING, required=False)
def run(
        task: str,
        sources: List[str],
):
    """Upload/update the knowledge bases used by the database.

    Parameters
    ----------
    task :
        The task to perform. One of: upload, update, local-update
    sources :
        The knowledge bases to update. If not specified, all knowledge bases
        will be updated.

    \b
    Usage tasks are:
     - upload: use if the knowledge bases have not yet been added.
     - update: if they have been added, but need to be updated.
     - local-update: if you have a local raw statements file dump, and want
       to update the knowledge bases from that and create a new raw
       statements file dump.

    Specify which knowledge base sources to update by their name, e.g. "Pathway
    Commons" or "pc". If not specified, all sources will be updated.
    """
    from indra_db.util import get_db
    db = get_db('primary')

    # Determine which sources we are working with
    if sources:
        source_set = {s.lower() for s in sources}
        selected_kbs = [
            M for M in KnowledgebaseManager.__subclasses__()
            if M.name.lower() in source_set or
               M.short_name.lower() in source_set
        ]
    else:
        selected_kbs = [KnowledgebaseManager.__subclasses__()]

    logger.info(f"Selected knowledgebases: "
                f"{', '.join([M.name for M in selected_kbs])}")

    # Handle the list option.
    if task == 'list':
        return

    # Handle the other tasks.
    logger.info(f"Running {task}...")
    if task == "local-update":
        local_update(kb_manager_list=selected_kbs)
    else:
        for Manager in selected_kbs:
            kbm = Manager()

            if task == 'upload':
                print(f'Uploading {kbm.name}...')
                kbm.upload(db)
            elif task == 'update':
                print(f'Updating {kbm.name}...')
                kbm.update(db)


@kb.command('list')
def show_list():
    """List the knowledge sources and their status."""
    import tabulate
    from indra_db.util import get_db
    db = get_db('primary')
    rows = [(M.name, M.short_name, format_date(M.get_last_update(db)))
            for M in KnowledgebaseManager.__subclasses__()]
    print(tabulate.tabulate(rows, ('Name', 'Short Name', 'Last Updated'),
                            tablefmt='simple'))
