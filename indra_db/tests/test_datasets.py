from indra_db.managers.dataset_manager import TasManager
from indra_db.util import get_test_db


def test_tas():
    db = get_test_db()
    db._clear(force=True)
    tm = TasManager()
    tm.upload(db)
    db_stmts = db.select_all(db.RawStatements)
    assert len(db_stmts)
