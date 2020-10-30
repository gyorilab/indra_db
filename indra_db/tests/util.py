import json
import random
import pickle
import logging
from datetime import datetime
from functools import wraps
from os import path

import indra_db.util as dbu
from indra.util.get_version import get_version
from indra_db.config import get_s3_dump
from indra_db.databases import PrincipalDatabaseManager, \
    ReadonlyDatabaseManager
from indra_db.util import insert_raw_agents

logger = logging.getLogger(__name__)


THIS_DIR = path.dirname(path.abspath(__file__))


def assert_contents_equal(list1, list2, msg=None):
    "Check that the contents of two lists are the same, regardless of order."
    res = set(list1) == set(list2)
    err_msg = "Contents of lists do not match:\n%s\n%s\n" % (list1, list2)
    if msg is not None:
        err_msg += msg
    assert res, err_msg


def capitalize_list_of_tpls(l):
    return [tuple([i.upper() if isinstance(i, str) else i for i in e])
            for e in l]


def get_temp_db(clear=False):
    """Get a DatabaseManager for the test database."""
    db = PrincipalDatabaseManager('postgresql://postgres:@localhost/indradb_test')
    if clear:
        db._clear(force=True)
    db.grab_session()
    db.session.rollback()
    return db


def get_temp_ro(clear=False):
    """Get a manager for a Readonly Database."""
    ro = ReadonlyDatabaseManager('postgresql://postgres:@localhost/indradb_ro_test')
    if clear:
        ro._clear(force=True)
    ro.grab_session()
    ro.session.rollback()
    return ro


TEST_FTP_PATH = path.abspath(path.join(path.dirname(__file__), path.pardir,
                                       'resources', 'test_ftp'))


def get_test_ftp_url():
    if not path.exists(TEST_FTP_PATH):
        print("Creating test directory. This could take a while...")
        from indra_db.resources.build_sample_set import build_set
        build_set(2, TEST_FTP_PATH)
    return TEST_FTP_PATH


def get_db_with_pubmed_content():
    "Populate the database with sample content from pubmed."
    from indra_db.managers.content_manager import Pubmed
    db = get_temp_db(clear=True)
    Pubmed(ftp_url=get_test_ftp_url(), local=True).populate(db)
    return db


def get_db_with_ftp_content():
    "Populate database with content from all the ftp services"
    from indra_db.managers.content_manager import PmcOA, Manuscripts
    db = get_db_with_pubmed_content()
    PmcOA(ftp_url=get_test_ftp_url(), local=True).populate(db)
    Manuscripts(ftp_url=get_test_ftp_url(), local=True).populate(db)
    return db


def _with_quiet_db_logs(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Make the database manager logger quieter.
        from indra_db.databases import logger as dbm_logger
        original_level = dbm_logger.level
        dbm_logger.setLevel(logging.WARNING)

        try:
            ret = func(*args, **kwargs)
        finally:
            # Make sure we go back to ordinary logging
            dbm_logger.setLevel(original_level)
        return ret

    return wrapper


TEST_DATA = {}


class DatabaseEnv(object):
    """This object is used to setup the test database into various configs."""
    def __init__(self, max_total_stmts):
        self.test_db = get_temp_db(clear=True)
        if not TEST_DATA:
            with open(path.join(THIS_DIR,
                                'db_pa_test_input_1M.pkl'), 'rb') as f:
                TEST_DATA.update(pickle.load(f))
        self.test_data = TEST_DATA

        if max_total_stmts < len(self.test_data['raw_statements']['tuples']):
            self.stmt_tuples = random.sample(
                self.test_data['raw_statements']['tuples'],
                max_total_stmts
                )
        else:
            self.stmt_tuples = self.test_data['raw_statements']['tuples']

        self.used_stmt_tuples = set()
        return

    def get_available_stmt_tuples(self):
        return list(set(self.stmt_tuples) - self.used_stmt_tuples)

    @_with_quiet_db_logs
    def load_background(self):
        """Load in all the background provenance metadata (e.g. text_ref).

        Note: This must be done before you try to load any statements.
        """
        # Abbreviate this variable to avoid excessive line breaks.
        td = self.test_data
        tables = ['text_ref', 'text_content', 'reading', 'db_info']

        # Handle the case where we aren't using all the statements.
        if len(self.stmt_tuples) < len(td['raw_statements']['tuples']):
            # Look up the indices for easy access.
            rdg_idx = td['raw_statements']['cols'].index('reading_id')
            tc_idx = td['reading']['cols'].index('text_content_id')
            tr_idx = td['text_content']['cols'].index('text_ref_id')

            # Select only the necessary refs
            inputs = {tbl: set() for tbl in tables}

            # Take all the db_info (there aren't many).
            inputs['db_info'] = set(td['db_info']['tuples'])

            # Filter out un-needed reading provenance.
            for stmt_tpl in self.stmt_tuples:
                rid = stmt_tpl[rdg_idx]
                if not rid:
                    continue
                # Select the reading.
                rdg_tpl = td['reading']['dict'][stmt_tpl[rdg_idx]]
                inputs['reading'].add(rdg_tpl)

                # Select the text content.
                tc_tpl = td['text_content']['dict'][rdg_tpl[tc_idx]]
                inputs['text_content'].add(tc_tpl)

                # Select the text ref.
                inputs['text_ref'].add(td['text_ref']['dict'][tc_tpl[tr_idx]])
        else:
            inputs = {tbl: set(td[tbl]['tuples']) for tbl in tables}

        # Insert the necessary content.
        for tbl in tables:
            print("Loading %s..." % tbl)
            self.test_db.copy(tbl, inputs[tbl], td[tbl]['cols'])
        return

    @_with_quiet_db_logs
    def insert_the_statements(self, input_tuples):
        print("Loading %d statements..." % len(input_tuples))
        cols = self.test_data['raw_statements']['cols']
        new_input_dict = {}

        batch_id_set = set()

        for t in input_tuples:
            batch_id_set.add(t[cols.index('batch_id')])

            rid = t[cols.index('reading_id')]
            dbid = t[cols.index('db_info_id')]
            mk_hash = t[cols.index('mk_hash')]
            if rid is not None:
                key = (mk_hash, rid, t[cols.index('text_hash')])
            elif dbid is not None:
                key = (mk_hash, dbid, t[cols.index('source_hash')])
            else:
                raise ValueError("Either rid or dbid must be non-none.")
            new_input_dict[key] = t

        logger.debug("Loading %d/%d statements."
                     % (len(new_input_dict), len(input_tuples)))

        self.test_db.copy_push('raw_statements', new_input_dict.values(), cols,
                               constraint='reading_raw_statement_uniqueness')

        print("Inserting agents...")
        for batch_id in batch_id_set:
            dbu.insert_raw_agents(self.test_db, batch_id)

        return set(new_input_dict.values())

    def add_statements(self):
        """Add statements and agents to the database."""
        input_tuples = self.get_available_stmt_tuples()
        self.insert_the_statements(input_tuples)
        self.used_stmt_tuples |= set(input_tuples)
        return

    @_with_quiet_db_logs
    def insert_pa_statements(self, with_agents=False):
        """Insert pickled preassembled statements."""
        existing_sids = {t[0] for t in self.used_stmt_tuples}
        link_tuples = []
        pa_tuples = []
        hash_set = set()
        pa_stmt_dict = self.test_data['pa_statements']['dict']
        for mk_hash, sid in self.test_data['raw_unique_links']['tuples']:
            if sid in existing_sids:
                link_tuples.append((mk_hash, sid))
                if mk_hash not in hash_set:
                    pa_tuples.append(pa_stmt_dict[mk_hash])
                    hash_set.add(mk_hash)
        self.test_db.copy('pa_statements', pa_tuples,
                          self.test_data['pa_statements']['cols'])
        self.test_db.copy('raw_unique_links', link_tuples,
                          self.test_data['raw_unique_links']['cols'])
        supps_tuples = {t for t in self.test_data['pa_support_links']['tuples']
                        if set(t).issubset(hash_set)}
        self.test_db.copy('pa_support_links', supps_tuples,
                          self.test_data['pa_support_links']['cols'])
        if with_agents:
            ag_dict = self.test_data['pa_agents']['dict']
            self.test_db.copy('pa_agents',
                              sum([ag_dict[h] for h in hash_set], []),
                              self.test_data['pa_agents']['cols'])
        return


def get_prepped_db(num_stmts, with_pa=False, with_agents=False):
    dts = DatabaseEnv(num_stmts)
    dts.load_background()
    dts.add_statements()
    if with_pa:
        dts.insert_pa_statements(with_agents)
    return dts.test_db


class PaDatabaseEnv(DatabaseEnv):
    """This object is used to setup the test database into various configs."""
    def add_statements(self, fraction=1, pam=None):
        """Add statements and agents to the database.

        Parameters
        ----------
        fraction : float between 0 and 1
            Default is 1. The fraction of remaining statements to be added.
        with_pa : bool
            Default False. Choose to run pre-assembly/incremental-preassembly
            on the added statements.
        """
        available_tuples = self.get_available_stmt_tuples()
        if fraction is not 1:
            num_stmts = int(fraction*len(available_tuples))
            input_tuples = random.sample(available_tuples, num_stmts)
        else:
            input_tuples = available_tuples

        self.insert_the_statements(input_tuples)

        if pam:
            print("Preassembling new statements...")
            if self.used_stmt_tuples:
                pam.supplement_corpus(self.test_db)
            else:
                pam.create_corpus(self.test_db)

        self.used_stmt_tuples |= set(input_tuples)
        return


def get_pa_loaded_db(num_stmts, split=None, pam=None):
    print("Creating and filling a test database:")
    dts = PaDatabaseEnv(num_stmts)
    dts.load_background()

    if split is None:
        dts.add_statements(pam=pam)
    else:
        dts.add_statements(split, pam=pam)
        dts.add_statements()
    dts.test_db.session.close()
    dts.test_db.grab_session()
    return dts.test_db


def get_filled_ro(num_stmts):
    db = get_prepped_db(num_stmts, with_pa=True, with_agents=True)
    db.generate_readonly()
    s3_base = get_s3_dump()
    assert s3_base, "No s3 config available for db dumps."
    s3_path = dbu.S3Path.from_string(s3_base.to_string() + '-test')
    now_str = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
    s3_path.get_element_path('readonly-%s.dump' % now_str)
    db.dump_readonly(s3_path)
    ro = get_temp_ro()
    ro.load_dump(s3_path)
    return ro


def insert_test_stmts(db, stmts_dict):
    """Insert raw statements from readings into the database.

    `stmts_dict` must be of the form {<source_type>: {<source_id>: [stmts]}}
    where `source_type` is "reading" or "databases", and source_id would be a
    reading ID or a db_info_id, respectively.
    """
    batch_id = db.make_copy_batch_id()

    stmt_data = []
    cols = ('uuid', 'mk_hash', 'db_info_id', 'reading_id',
            'type', 'json', 'batch_id', 'source_hash', 'indra_version')

    all_stmts = []
    for category, stmts in stmts_dict.items():
        for src_id, stmt_list in stmts.items():
            for stmt in stmt_list:
                stmt_info = {
                    'uuid': stmt.uuid,
                    'mk_hash': stmt.get_hash(refresh=True),
                    'type': stmt.__class__.__name__,
                    'json': json.dumps(stmt.to_json()).encode('utf-8'),
                    'batch_id': batch_id,
                    'source_hash': -1,
                    'indra_version': get_version()
                }
                if category == 'reading':
                    stmt_info['reading_id'] = src_id
                else:
                    stmt_info['db_info_id'] = src_id
                stmt_data.append(tuple(stmt_info.get(col) for col in cols))
                all_stmts.append(stmt)

    db.copy('raw_statements', stmt_data, cols)
    insert_raw_agents(db, batch_id, all_stmts)
