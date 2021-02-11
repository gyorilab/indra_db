import json

from indra.statements import Agent, Phosphorylation, Complex, Activation
from indra_db.client.principal import *
from indra_db.util import insert_pa_agents

from indra_db.tests.util import get_temp_db
from indra_db.tests.db_building_util import DbBuilder


def _construct_database():
    db = get_temp_db(clear=True)
    db_builder = DbBuilder(db)
    db_builder.add_text_refs([
        ('12345', 'PMC54321'),
        ('24680', 'PMC08642')
    ])
    db_builder.add_text_content([
        ['pubmed-abs', 'pmc_oa'],
        ['pubmed-abs']
    ])
    db_builder.add_readings([
        ['REACH'],
        ['REACH'],
        ['REACH', 'SPARSER']
    ])

    mek = Agent('MEK', db_refs={'FPLX': 'MEK'})
    erk = Agent('ERK', db_refs={'FPLX': 'ERK'})
    raf = Agent('RAF', db_refs={'FPLX': 'RAF'})

    db_builder.add_raw_reading_statements([
        [Phosphorylation(mek, erk), Complex([mek, erk])],
        [Phosphorylation(mek, erk)],
        [Activation(mek, erk)],
        [Complex([mek, erk]), Complex([raf, erk])]
    ])

    raw_stmts = db_builder.raw_statements

    pa_statements_pre = [
        (Phosphorylation(mek, erk), [raw_stmts[0], raw_stmts[2]]),
        (Complex([mek, erk]), [raw_stmts[1], raw_stmts[4]]),
        (Activation(mek, erk), [raw_stmts[3]]),
        (Complex([raf, erk]), [raw_stmts[5]])
    ]
    pa_stmts = []
    raw_unique_links = []
    for pa_stmt, raw_stmt_list in pa_statements_pre:
        pa_json = pa_stmt.to_json()
        h = pa_stmt.get_hash()
        for raw_stmt in raw_stmt_list:
            raw_unique_links.append(
                db.RawUniqueLinks(raw_stmt_id=raw_stmt.id, pa_stmt_mk_hash=h)
            )
        pa_stmts.append(
            db.PAStatements(mk_hash=h, json=json.dumps(pa_json).encode('utf-8'),
                            type=pa_json['type'], uuid=pa_stmt.uuid,
                            matches_key=pa_stmt.matches_key(),
                            indra_version='test')
        )
    db.session.add_all(pa_stmts)
    db.session.commit()
    db.session.add_all(raw_unique_links)
    db.session.commit()

    insert_pa_agents(db, [s for s, _ in pa_statements_pre])

    return db


def test_get_raw_statements_all():
    db = _construct_database()
    res = get_raw_stmt_jsons(db=db)
    assert len(res) == 6


def test_raw_statement_retrieval_from_agents_type_only():
    db = _construct_database()
    res = get_raw_stmt_jsons_from_agents(stmt_type='Complex', db=db)
    assert len(res) > 0
    assert len(res) < 6
    assert all(sj['type'] == 'Complex' for sj in res.values())


def test_raw_statement_retrieval_from_agents_mek():
    db = _construct_database()
    res = get_raw_stmt_jsons_from_agents(agents=[(None, 'MEK', 'FPLX')], db=db)
    assert len(res) > 0
    assert len(res) < 6
    assert all('MEK' in json.dumps(sj) for sj in res.values())


def test_raw_statement_retrieval_generic():
    db = _construct_database()
    res = get_raw_stmt_jsons([db.Reading.reader == 'REACH',
                              db.Reading.id == db.RawStatements.reading_id],
                             db=db)
    assert len(res) > 0
    assert len(res) < 6
    assert all(sj['evidence'][0]['source_api'] == 'reach'
               for sj in res.values())


def test_pa_statement_retrieval_generic():
    db = _construct_database()
    res = get_pa_stmt_jsons(db=db)
    assert len(res) == 4


def test_pa_statement_retrieval_by_type():
    db = _construct_database()
    res = get_pa_stmt_jsons([db.PAStatements.type == 'Complex'], db=db)
    assert len(res) == 2
    assert all(j['stmt']['type'] == 'Complex' for j in res.values())
