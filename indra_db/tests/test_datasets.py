from indra_db.managers.dataset_manager import TasManager
from indra_db.util import get_test_db


def test_tas():
    db = get_test_db()
    tm = TasManager()
    tm.upload(db)
