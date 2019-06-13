from indra_db.tests.util import get_temp_db


def test_db_presence():
    db = get_temp_db()
    db.insert(db.TextRef, pmid='12345')
