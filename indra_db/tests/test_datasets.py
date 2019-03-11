from nose.plugins.attrib import attr

from indra_db.managers.dataset_manager import TasManager
from indra_db.util import get_test_db


@attr("nonpublic")
def test_tas():
    db = get_test_db()
    db._clear(force=True)
    dbid = db.select_one(db.DBInfo.id, db.DBInfo.db_name == TasManager.name)
    assert dbid is None
    tm = TasManager()
    tm.upload(db)
    dbid = db.select_on(db.DBInfo, db.DBInfo.db_name == TasManager.name)
    assert dbid is not None
    db_stmts = db.select_all(db.RawStatements)
    print(len(db_stmts))
    assert len(db_stmts)
    assert all(s.db_info_id == dbid for s in db_stmts)
