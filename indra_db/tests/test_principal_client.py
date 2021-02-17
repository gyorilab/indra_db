import json

from indra_db.client.principal import *
from indra.statements import Agent, Phosphorylation, Complex, Activation

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

    db_builder.add_databases(['signor'])
    db_builder.add_raw_database_statements([
        [Complex([raf, erk])]
    ])
    db_builder.add_pa_statements([
        (Phosphorylation(mek, erk), [0, 2]),
        (Complex([mek, erk]), [1, 4]),
        (Activation(mek, erk), [3]),
        (Complex([raf, erk]), [5, 6])
    ])
    return db


def test_get_raw_statements_all():
    db = _construct_database()
    res = get_raw_stmt_jsons(db=db)
    assert len(res) == 7, len(res)


def test_raw_statement_retrieval_from_agents_type_only():
    db = _construct_database()
    res = get_raw_stmt_jsons_from_agents(stmt_type='Complex', db=db)
    assert len(res) > 0
    assert len(res) < 7
    assert all(sj['type'] == 'Complex' for sj in res.values())


def test_raw_statement_retrieval_from_agents_mek():
    db = _construct_database()
    res = get_raw_stmt_jsons_from_agents(agents=[(None, 'MEK', 'FPLX')], db=db)
    assert len(res) > 0
    assert len(res) < 7
    assert all('MEK' in json.dumps(sj) for sj in res.values())


def test_raw_statement_retrieval_generic():
    db = _construct_database()
    res = get_raw_stmt_jsons([db.Reading.reader == 'REACH',
                              db.Reading.id == db.RawStatements.reading_id],
                             db=db)
    assert len(res) > 0
    assert len(res) < 7
    assert all(sj['evidence'][0]['source_api'] == 'reach'
               for sj in res.values())


def test_raw_statements_get_database_only():
    db = _construct_database()
    res = get_raw_stmt_jsons([db.RawStatements.reading_id.is_(None)], db=db)
    assert len(res) == 1, len(res)
    assert all(sj['evidence'][0]['source_api'] == 'signor'
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
