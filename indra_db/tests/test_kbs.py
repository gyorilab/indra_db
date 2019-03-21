from nose.plugins.attrib import attr

from indra.statements.statements import Statement, Agent, Phosphorylation, \
    Complex, Evidence

from indra_db.managers.knowledgebase_manager import TasManager, CBNManager, \
    HPRDManager, SignorManager, BiogridManager
from indra_db.util import get_test_db, insert_db_stmts


def _check_kb(Kb):
    db = get_test_db()
    db._clear(force=True)
    dbid = db.select_one(db.DBInfo.id, db.DBInfo.db_name == Kb.name)
    assert dbid is None
    kbm = Kb()
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
    _check_kb(TasManager)


@attr('nonpublic')
def test_cbn():
    s3_url = 'https://s3.amazonaws.com/bigmech/travis/Hox-2.0-Hs.jgf.zip'
    tmp_archive = './temp_Hox-2.0-Hs.jgf.zip'
    temp_extract = './temp/'

    cbn_mgr = CBNManager(tmp_archive=tmp_archive,
                         temp_extract=temp_extract,
                         archive_url=s3_url)

    db = get_test_db()
    db._clear(force=True)
    cbn_mgr.upload(db)
    db_stmts = db.select_all(db.RawStatements)
    assert len(db_stmts)


@attr('nonpublic')
def test_hprd():
    _check_kb(HPRDManager)


@attr('nonpublic')
def test_signor():
    _check_kb(SignorManager)


@attr('nonpublic')
def test_biogrid():
    _check_kb(BiogridManager)


@attr('nonpublic')
def test_simple_db_insert():
    db = get_test_db()
    db._clear(force=True)
    stmts = [Phosphorylation(Agent('MEK', db_refs={'FPLX': 'MEK'}),
                             Agent('ERK', db_refs={'FPLX': 'ERK'}),
                             evidence=Evidence(source_api='test')),
             Complex([Agent(n, db_refs={'FPLX': n}) for n in ('MEK', 'ERK')],
                     evidence=Evidence(source_api='test'))]
    dbid = db.insert(db.DBInfo, db_name='test')
    insert_db_stmts(db, stmts, dbid)
    db_stmts = db.select_all(db.RawStatements)
    db_agents = db.select_all(db.RawAgents)
    assert len(db_stmts) == 2, len(db_stmts)
    assert len(db_agents) == 8, len(db_agents)
    db.session.close()
