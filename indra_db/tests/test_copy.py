from indra_db.tests.util import get_temp_db


COLS = ('pmid', 'pmcid')


def _ref_set(db):
    return set(db.select_all([db.TextRef.pmid, db.TextRef.pmcid]))


def _assert_set_equal(s1, s2):
    assert s1 == s2, '%s != %s' % (s1, s2)


def test_vanilla_copy():
    db = get_temp_db(True)
    inps = {('a', '1'), ('b', '1')}
    db.copy('text_ref', inps, COLS)
    assert inps == _ref_set(db)

    try:
        db.copy('text_ref', inps, COLS)
    except:
        return
    assert False, "Copy of duplicate data succeeded."


def _do_init_copy(db):
    inps_1 = {('a', '1'), ('b', '2')}
    db.copy('text_ref', inps_1, COLS)
    _assert_set_equal(inps_1, _ref_set(db))
    return inps_1


def test_lazy_copy():
    db = get_temp_db(True)
    inps_1 = _do_init_copy(db)
    inps_2 = {('b', '2'), ('c', '1'), ('d', '3')}
    db.copy_lazy('text_ref', inps_2, COLS)
    _assert_set_equal(inps_1 | inps_2, _ref_set(db))


def test_lazy_report_copy():
    db = get_temp_db(True)
    inps_1 = _do_init_copy(db)
    inps_2 = {('b', '2'), ('c', '1'), ('d', '3')}

    left_out = db.copy_report_lazy('text_ref', inps_2, COLS)
    _assert_set_equal(inps_1 | inps_2, _ref_set(db))
    _assert_set_equal(inps_1 & inps_2, {t[:2] for t in left_out})


def test_push_copy():
    db = get_temp_db(True)
    inps_1 = _do_init_copy(db)
    inps_2 = {('b', '2'), ('c', '1'), ('d', '3')}

    original_date = db.select_one(db.TextRef.create_date,
                                  db.TextRef.pmid == 'b')

    db.copy_push('text_ref', inps_2, COLS)
    _assert_set_equal(inps_1 | inps_2, _ref_set(db))
    new_date = db.select_one(db.TextRef.create_date,
                             db.TextRef.pmid == 'b')
    assert new_date != original_date, "PMID b was not updated."


def test_push_report_copy():
    db = get_temp_db(True)
    inps_1 = _do_init_copy(db)
    inps_2 = {('b', '2'), ('c', '1'), ('d', '3')}

    original_date = db.select_one(db.TextRef.create_date,
                                  db.TextRef.pmid == 'b')

    updated = db.copy_report_push('text_ref', inps_2, COLS)
    _assert_set_equal(inps_1 | inps_2, _ref_set(db))
    _assert_set_equal(inps_1 & inps_2, {t[:2] for t in updated})
    new_date = db.select_one(db.TextRef.create_date,
                             db.TextRef.pmid == 'b')
    assert new_date != original_date, 'PMID b was not updated.'


def test_returning_copy():
    db = get_temp_db(True)
    inps_1 = _do_init_copy(db)
    inps_2 = {('b', '2'), ('c', '1'), ('d', '3')}

    exiting_ids = {trid for trid, in db.select_all(db.TextRef.id)}

    ids, skipped = db.copy_report_and_return_lazy('text_ref', inps_2, COLS)
    _assert_set_equal(inps_1 | inps_2, _ref_set(db))
    _assert_set_equal(inps_1 & inps_2, {t[:2] for t in skipped})
    assert {trid for trid, in ids} != exiting_ids


def test_returning_copy_pmid_and_id():
    db = get_temp_db(True)
    inps_1 = _do_init_copy(db)
    inps_2 = {('b', '2'), ('c', '1'), ('d', '3')}

    existing_id_dict = {pmid: trid for trid, pmid
                        in db.select_all([db.TextRef.id, db.TextRef.pmid])}

    ids, skipped = db.copy_report_and_return_lazy('text_ref', inps_2, COLS,
                                                  ('pmid', 'id'))
    new_id_dict = {pmid: trid for pmid, trid in ids}
    _assert_set_equal(inps_1 | inps_2, _ref_set(db))
    _assert_set_equal(inps_1 & inps_2, {t[:2] for t in skipped})
    assert set(existing_id_dict.keys()) != set(new_id_dict.keys())
