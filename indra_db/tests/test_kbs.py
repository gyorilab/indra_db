from nose.plugins.attrib import attr

from indra.statements.statements import Agent, Phosphorylation, Complex, \
    Evidence

from indra_db.managers.knowledgebase_manager import *
from indra_db.util import insert_db_stmts
from indra_db.tests.util import get_temp_db


def _check_kbm(Kb, *args, **kwargs):
    db = get_temp_db(clear=True)
    dbid = db.select_one(db.DBInfo.id, db.DBInfo.db_name == Kb.name)
    assert dbid is None
    kbm = Kb(*args, **kwargs)
    kbm.upload(db)
    dbid = db.select_one(db.DBInfo.id, db.DBInfo.db_name == Kb.name)[0]
    assert dbid is not None
    db_stmts = db.select_all(db.RawStatements)
    print(len(db_stmts))
    assert len(db_stmts)
    assert all(s.db_info_id == dbid for s in db_stmts)
    db.session.close()


@attr("nonpublic")
def test_tas():
    _check_kbm(TasManager)


@attr('nonpublic')
def test_cbn():
    s3_url = 'https://s3.amazonaws.com/bigmech/travis/Hox-2.0-Hs.jgf.zip'
    _check_kbm(CBNManager, archive_url=s3_url)


@attr('nonpublic', 'slow')
def test_hprd():
    _check_kbm(HPRDManager)


@attr('nonpublic')
def test_signor():
    _check_kbm(SignorManager)


@attr('nonpublic', 'slow')
def test_biogrid():
    _check_kbm(BiogridManager)


@attr('nonpublic', 'slow')
def test_bel_lc():
    _check_kbm(BelLcManager)


@attr('nonpublic', 'slow')
def test_pathway_commons():
    _check_kbm(PathwayCommonsManager)


@attr('nonpublic', 'slow')
def test_rlimsp():
    _check_kbm(RlimspManager)


@attr('nonpublic')
def test_trrust():
    _check_kbm(TrrustManager)


@attr('nonpublic')
def test_phosphosite():
    _check_kbm(PhosphositeManager)


@attr('nonpublic')
def test_simple_db_insert():
    db = get_temp_db()
    db._clear(force=True)
    stmts = [Phosphorylation(Agent('MEK', db_refs={'FPLX': 'MEK'}),
                             Agent('ERK', db_refs={'FPLX': 'ERK'}),
                             evidence=Evidence(source_api='test')),
             Complex([Agent(n, db_refs={'FPLX': n}) for n in ('MEK', 'ERK')],
                     evidence=Evidence(source_api='test'))]
    dbid = db.insert(db.DBInfo, db_name='test', source_api='tester')
    insert_db_stmts(db, stmts, dbid)
    db_stmts = db.select_all(db.RawStatements)
    db_agents = db.select_all(db.RawAgents)
    assert len(db_stmts) == 2, len(db_stmts)
    assert len(db_agents) == 8, len(db_agents)
    db.session.close()
