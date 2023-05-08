__all__ = ['TasManager', 'CBNManager', 'HPRDManager', 'SignorManager',
           'BiogridManager', 'BelLcManager', 'PathwayCommonsManager',
           'RlimspManager', 'TrrustManager', 'PhosphositeManager',
           'CTDManager', 'VirHostNetManager', 'PhosphoElmManager',
           'DrugBankManager']

import codecs
import csv
import gzip
import json
import os
import zlib
import boto3
import click
import pickle
import logging
import tempfile
from collections import defaultdict

from tqdm import tqdm

from indra.statements.validate import assert_valid_statement
from indra_db.util import insert_db_stmts
from indra_db.util.distill_statements import extract_duplicates, KeyFunc

from .util import format_date

logger = logging.getLogger(__name__)


class StatementJSONDecodeError(Exception):
    pass


def load_statement_json(
        json_str: str, attempt: int = 1, max_attempts: int = 5
):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        if attempt < max_attempts:
            json_str = codecs.escape_decode(json_str)[0].decode()
            return load_statement_json(
                json_str, attempt=attempt + 1, max_attempts=max_attempts
            )
    raise StatementJSONDecodeError(
        f"Could not decode statement JSON after " f"{attempt} attempts: {json_str}"
    )


class KnowledgebaseManager(object):
    """This is a class to lay out the methods for updating a dataset."""
    name = NotImplemented
    short_name = NotImplemented
    source = NotImplemented

    def upload(self, db):
        """Upload the content for this dataset into the database."""
        dbid = self._check_reference(db)
        stmts = self._get_statements()
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
        stmts = self._get_statements()
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

    def _get_statements(self):
        raise NotImplementedError("Statement retrieval must be defined in "
                                  "each child.")


class TasManager(KnowledgebaseManager):
    """This manager handles retrieval and processing of the TAS dataset."""
    # TODO: Data is simply a CSV from S3
    name = 'TAS'
    short_name = 'tas'
    source = 'tas'

    def _get_statements(self):
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
        tp = tas.process_from_web(affinity_class_limit=2,
                                  named_only=True,
                                  standardized_only=False)
        logger.info('Expanding evidences and deduplicating')
        filtered_stmts = [s for s in _expanded(tp.statements)]
        unique_stmts, _ = extract_duplicates(filtered_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts


class SignorManager(KnowledgebaseManager):
    name = 'Signor'
    short_name = 'signor'
    source = 'signor'

    def _get_statements(self):
        from indra.sources.signor import process_from_web
        proc = process_from_web()
        return proc.statements


class CBNManager(KnowledgebaseManager):
    """This manager handles retrieval and processing of CBN network files"""
    name = 'Causal Bionet'
    short_name = 'cbn'
    source = 'bel'

    def __init__(self, archive_url=None):
        if not archive_url:
            self.archive_url = ('http://www.causalbionet.com/Content'
                                '/jgf_bulk_files/Human-2.0.zip')
        else:
            self.archive_url = archive_url
        return

    def _get_statements(self):
        import requests
        from zipfile import ZipFile
        from indra.sources.bel.api import process_cbn_jgif_file
        import tempfile

        cbn_dir = tempfile.mkdtemp('cbn_manager')

        logger.info('Retrieving CBN network zip archive')
        tmp_zip = os.path.join(cbn_dir, 'cbn_human.zip')
        resp = requests.get(self.archive_url)
        with open(tmp_zip, 'wb') as f:
            f.write(resp.content)

        stmts = []
        tmp_dir = os.path.join(cbn_dir, 'cbn')
        os.mkdir(tmp_dir)
        with ZipFile(tmp_zip) as zipf:
            logger.info('Extracting archive to %s' % tmp_dir)
            zipf.extractall(path=tmp_dir)
            logger.info('Processing jgif files')
            for jgif in zipf.namelist():
                if jgif.endswith('.jgf') or jgif.endswith('.jgif'):
                    logger.info('Processing %s' % jgif)
                    pbp = process_cbn_jgif_file(os.path.join(tmp_dir, jgif))
                    stmts += pbp.statements

        uniques, dups = extract_duplicates(stmts,
                                           key_func=KeyFunc.mk_and_one_ev_src)

        logger.info("Deduplicating...")
        print('\n'.join(str(dup) for dup in dups))
        print(len(dups))

        return uniques


class BiogridManager(KnowledgebaseManager):
    name = 'BioGRID'
    short_name = 'biogrid'
    source = 'biogrid'

    def _get_statements(self):
        from indra.sources import biogrid
        bp = biogrid.BiogridProcessor()
        return list(_expanded(bp.statements))


class PathwayCommonsManager(KnowledgebaseManager):
    name = 'Pathway Commons'
    short_name = 'pc'
    source = 'biopax'
    skips = {'psp', 'hprd', 'biogrid', 'phosphosite', 'phosphositeplus',
             'ctd', 'drugbank'}

    def __init__(self, *args, **kwargs):
        self.counts = defaultdict(lambda: 0)
        super(PathwayCommonsManager, self).__init__(*args, **kwargs)

    def _can_include(self, stmt):
        num_ev = len(stmt.evidence)
        assert num_ev == 1, "Found statement with %d evidence." % num_ev

        ev = stmt.evidence[0]
        ssid = ev.annotations['source_sub_id']
        self.counts[ssid] += 1

        return ssid not in self.skips

    def _get_statements(self):
        s3 = boto3.client('s3')

        logger.info('Loading PC content pickle from S3')
        resp = s3.get_object(Bucket='bigmech',
                             Key='indra-db/biopax_pc12_pybiopax.pkl')
        logger.info('Loading PC statements from pickle')
        stmts = pickle.loads(resp['Body'].read())

        logger.info('Expanding evidences and deduplicating')
        filtered_stmts = [s for s in _expanded(stmts) if self._can_include(s)]
        unique_stmts, _ = extract_duplicates(filtered_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts


class CTDManager(KnowledgebaseManager):
    name = 'CTD'
    source = 'ctd'
    short_name = 'ctd'
    subsets = ['gene_disease', 'chemical_disease',
               'chemical_gene']

    def _get_statements(self):
        s3 = boto3.client('s3')
        all_stmts = []
        for subset in self.subsets:
            logger.info('Fetching CTD subset %s from S3...' % subset)
            key = 'indra-db/ctd_%s.pkl' % subset
            resp = s3.get_object(Bucket='bigmech', Key=key)
            stmts = pickle.loads(resp['Body'].read())
            all_stmts += [s for s in _expanded(stmts)]
        # Return exactly one of multiple statements that are exactly the same
        # in terms of content and evidence.
        unique_stmts, _ = extract_duplicates(all_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts


class DrugBankManager(KnowledgebaseManager):
    name = 'DrugBank'
    short_name = 'drugbank'
    source = 'drugbank'

    def _get_statements(self):
        s3 = boto3.client('s3')
        logger.info('Fetching DrugBank statements from S3...')
        key = 'indra-db/drugbank_5.1.pkl'
        resp = s3.get_object(Bucket='bigmech', Key=key)
        stmts = pickle.loads(resp['Body'].read())
        expanded_stmts = [s for s in _expanded(stmts)]
        # Return exactly one of multiple statements that are exactly the same
        # in terms of content and evidence.
        unique_stmts, _ = extract_duplicates(expanded_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts


class VirHostNetManager(KnowledgebaseManager):
    name = 'VirHostNet'
    short_name = 'vhn'
    source = 'virhostnet'

    def _get_statements(self):
        from indra.sources import virhostnet
        vp = virhostnet.process_from_web()
        return [s for s in _expanded(vp.statements)]


class PhosphoElmManager(KnowledgebaseManager):
    name = 'Phospho.ELM'
    short_name = 'pe'
    source = 'phosphoelm'

    def _get_statements(self):
        from indra.sources import phosphoelm
        logger.info('Fetching PhosphoElm dump from S3...')
        s3 = boto3.resource('s3')
        tmp_dir = tempfile.mkdtemp('phosphoelm_files')
        dump_file = os.path.join(tmp_dir, 'phosphoelm.dump')
        s3.meta.client.download_file('bigmech',
                                     'indra-db/phosphoELM_all_2015-04.dump',
                                     dump_file)
        logger.info('Processing PhosphoElm dump...')
        pp = phosphoelm.process_from_dump(dump_file)
        logger.info('Expanding evidences on PhosphoElm statements...')
        # Expand evidences just in case, though this processor always
        # produces a single evidence per statement.
        stmts = [s for s in _expanded(pp.statements)]
        # Return exactly one of multiple statements that are exactly the same
        # in terms of content and evidence.
        # Now make sure we don't include exact duplicates
        unique_stmts, _ = extract_duplicates(stmts, KeyFunc.mk_and_one_ev_src)
        return unique_stmts


class HPRDManager(KnowledgebaseManager):
    name = 'HPRD'
    short_name = 'hprd'
    source = 'hprd'

    def _get_statements(self):
        import tarfile
        import requests
        from indra.sources import hprd

        # Download the files.
        hprd_base = 'http://www.hprd.org/RELEASE9/'
        resp = requests.get(hprd_base + 'HPRD_FLAT_FILES_041310.tar.gz')
        tmp_dir = tempfile.mkdtemp('hprd_files')
        tmp_tarfile = os.path.join(tmp_dir, 'hprd_files.tar.gz')
        with open(tmp_tarfile, 'wb') as f:
            f.write(resp.content)

        # Extract the files.
        with tarfile.open(tmp_tarfile, 'r:gz') as tf:
            tf.extractall(tmp_dir)

        # Find the relevant files.
        dirs = os.listdir(tmp_dir)
        for files_dir in dirs:
            if files_dir.startswith('FLAT_FILES'):
                break
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
    # Todo: New data is available
    name = 'BEL Large Corpus'
    short_name = 'bel_lc'
    source = 'bel'

    def _get_statements(self):
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


class PhosphositeManager(KnowledgebaseManager):
    # TODO: New data is available - check
    #  s3://bigmech/indra-db/Kinase_substrates.owl.gz
    name = 'Phosphosite Plus'
    short_name = 'psp'
    source = 'biopax'

    def _get_statements(self):
        from indra.sources import biopax

        s3 = boto3.client('s3')
        resp = s3.get_object(Bucket='bigmech',
                             Key='indra-db/Kinase_substrates.owl.gz')
        owl_gz = resp['Body'].read()
        owl_str = \
            zlib.decompress(owl_gz, zlib.MAX_WBITS + 32).decode('utf-8')
        bp = biopax.process_owl_str(owl_str)
        stmts, dups = extract_duplicates(bp.statements,
                                         key_func=KeyFunc.mk_and_one_ev_src)
        print('\n'.join(str(dup) for dup in dups))
        print(len(stmts), len(dups))
        return stmts


class RlimspManager(KnowledgebaseManager):
    name = 'RLIMS-P'
    short_name = 'rlimsp'
    source = 'rlimsp'
    _rlimsp_root = 'https://hershey.dbi.udel.edu/textmining/export/'
    _rlimsp_files = [('rlims.medline.json', 'pmid'),
                     ('rlims.pmc.json', 'pmcid')]

    def _get_statements(self):
        from indra.sources import rlimsp
        import requests

        stmts = []
        for fname, id_type in self._rlimsp_files:
            print("Processing %s..." % fname)
            res = requests.get(self._rlimsp_root + fname)
            jsonish_str = res.content.decode('utf-8')
            rp = rlimsp.process_from_jsonish_str(jsonish_str, id_type)
            stmts += rp.statements
            print("Added %d more statements from %s..."
                  % (len(rp.statements), fname))

        stmts, dups = extract_duplicates(_expanded(stmts),
                                         key_func=KeyFunc.mk_and_one_ev_src)
        print('\n'.join(str(dup) for dup in dups))
        print(len(stmts), len(dups))

        return stmts


class TrrustManager(KnowledgebaseManager):
    name = 'TRRUST'
    short_name = 'trrust'
    source = 'trrust'

    def _get_statements(self):
        from indra.sources import trrust
        tp = trrust.process_from_web()
        unique_stmts, dups = \
            extract_duplicates(_expanded(tp.statements),
                               key_func=KeyFunc.mk_and_one_ev_src)
        print(len(dups))
        return unique_stmts


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

    def _get_statements(self):
        from indra.sources import dgi
        logger.info('Processing DGI from web')
        dp = dgi.process_version('2020-Nov')
        logger.info('Expanding evidences and deduplicating')
        filtered_stmts = [s for s in _expanded(dp.statements)]
        unique_stmts, _ = extract_duplicates(filtered_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts


class CrogManager(KnowledgebaseManager):
    """This manager handles retrieval and processing of the CRoG dataset."""
    name = 'CRoG'
    short_name = 'crog'
    source = 'crog'

    def _get_statements(self):
        from indra.sources import crog
        logger.info('Processing CRoG from web')
        cp = crog.process_from_web()
        logger.info('Expanding evidences and deduplicating')
        filtered_stmts = [s for s in _expanded(cp.statements)]
        unique_stmts, _ = extract_duplicates(filtered_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts


class ConibManager(KnowledgebaseManager):
    """This manager handles retrieval and processing of the CONIB dataset."""
    name = 'CONIB'
    short_name = 'conib'
    source = 'bel'

    def _get_statements(self):
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


class UbiBrowserManager(KnowledgebaseManager):
    """This manager handles retrieval and processing of UbiBrowser data."""
    name = 'UbiBrowser'
    short_name = 'ubibrowser'
    source = 'ubibrowser'

    def _get_statements(self):
        from indra.sources import ubibrowser
        logger.info('Processing UbiBrowser from web')
        up = ubibrowser.process_from_web()
        logger.info('Expanding evidences and deduplicating')
        filtered_stmts = [s for s in _expanded(up.statements)]
        unique_stmts, _ = extract_duplicates(filtered_stmts,
                                             KeyFunc.mk_and_one_ev_src)
        return unique_stmts


def local_update(
    raw_stmts_tsv_gz_path: str,
    out_tsv_gz_path: str,
    kb_manager_list,
    kb_mapping,
):
    """Update the knowledgebases of a local raw statements file dump

    Parameters
    ----------
    raw_stmts_tsv_gz_path :
        Path to the raw statements file dump
    out_tsv_gz_path :
        Path to the output file
    kb_manager_list :
        List of the classes of the knowledgebase managers to use in update
    kb_mapping :
        Mapping of knowledgebase source api and name, to db info id. Keyed
        by tuple of (source api, db name) from db info table.
    """
    assert raw_stmts_tsv_gz_path != out_tsv_gz_path, \
        "Input and output paths cannot be the same"
    null = "\\N"

    def _keep(stmt_json):
        # Return true if the statement's source is not from any of the
        # knowledgebases
        for mngr in kb_manager_list:
            if mngr.source == stmt_json["evidence"][0]["source_api"]:
                return False
        return True

    with gzip.open(raw_stmts_tsv_gz_path, 'rt') as in_fh, \
            gzip.open(out_tsv_gz_path, 'wt') as out_fh:
        reader = csv.reader(in_fh, delimiter='\t')
        writer = csv.writer(out_fh, delimiter='\t')

        # Filter out old knowledgebase statements
        for raw_stmt_id, db_info_id, reading_id, rsjs in tqdm(
                reader, total=75816146, desc="Filtering raw statements"
        ):
            raw_stmt_json = load_statement_json(rsjs)
            if _keep(raw_stmt_json):
                writer.writerow([raw_stmt_id, db_info_id, reading_id, rsjs])

        # Update the knowledgebases
        for Mngr in tqdm(kb_manager_list, desc="Updating knowledgebases"):
            kbm = Mngr()
            db_id = kb_mapping[(kbm.source, kbm.short_name)]
            stmts = kbm._get_statements()
            logger.info(f"Updating {kbm.name} with {len(stmts)} statements")
            rows = (
                (null, db_id, null, json.dumps(stmt.to_json()))
                for stmt in stmts
            )
            writer.writerows(rows)


@click.group()
def kb():
    """Manage the Knowledge Bases used by the database."""


@kb.command()
@click.argument("task", type=click.Choice(["upload", "update", "local-update"]))
@click.argument("sources", nargs=-1, type=click.STRING, required=False)
@click.option("--raw-stmts-tsvgz", type=click.STRING, required=False,
              help="Path to the raw statements tsv.gz file when using the "
                   "local option.")
@click.option("--raw-tsvgz-out", type=click.STRING, required=False,
              help="Path to the output raw statements tsv.gz file when "
                   "using the local option.")
def run(task, sources, raw_stmts_tsvgz, raw_tsvgz_out):
    """Upload/update the knowledge bases used by the database.

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

    res = db.select_all(db.DBInfo)
    kb_mapping = {(r.source_api, r.db_name): r.id for r in res}

    if task == "local-update":
        if not raw_stmts_tsvgz:
            raise ValueError("Must specify raw statements tsv.gz file when "
                             "using the local option.")
        elif not os.path.exists(raw_stmts_tsvgz):
            raise FileNotFoundError("Raw statements tsv.gz file does not "
                                    "exist: %s" % raw_stmts_tsvgz)
        elif not raw_stmts_tsvgz.endswith(".tsv.gz"):
            raise ValueError("Raw statements file must be tsv gzipped file. "
                             "Expected extension: .tsv.gz - got: "
                             f"{raw_stmts_tsvgz.split('.')[-1]}")

        if not raw_tsvgz_out:
            # Just add a suffix to the input file name
            raw_tsvgz_out = \
                raw_stmts_tsvgz.split("/")[-1].split(".")[0] + "_updated.tsv.gz"

        logger.info(f"Using output file name: {raw_tsvgz_out}")

    # Determine which sources we are working with
    source_set = None
    if sources:
        source_set = {s.lower() for s in sources}
    selected_kbs = (M for M in KnowledgebaseManager.__subclasses__()
                    if not source_set or M.name.lower() in source_set
                    or M.short_name in source_set)

    logger.info(f"Selected knowledgebases: "
                f"{', '.join([M.name for M in selected_kbs])}")

    # Handle the list option.
    if task == 'list':
        return

    # Handle the other tasks.
    logger.info(f"Running {task}...")
    if task == "local-update":
        local_update(raw_stmts_tsvgz, raw_tsvgz_out, selected_kbs, kb_mapping)
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
