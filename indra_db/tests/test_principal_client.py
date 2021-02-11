import json

from indra.statements import Agent, Phosphorylation, Evidence, Complex, \
    Activation
from indra_db.client.principal import *

from indra_db.tests.util import get_temp_db
from indra_db.util import insert_raw_agents, insert_pa_agents


def _construct_database():
    db = get_temp_db(clear=True)

    text_refs = [
        db.TextRef.new('12345', 'PMC54321'),
        db.TextRef.new('24680', 'PMC08642')
    ]
    db.session.add_all(text_refs)
    db.session.commit()

    text_content = [
        db.TextContent(text_ref_id=text_refs[0].id, source='pubmed',
                       format='txt', text_type='abstract'),
        db.TextContent(text_ref_id=text_refs[0].id, source='pmc_oa',
                       format='xml', text_type='fulltext'),
        db.TextContent(text_ref_id=text_refs[1].id, source='pubmed',
                       format='txt', text_type='abstract')
    ]
    db.session.add_all(text_content)
    db.session.commit()

    readings = [
        db.Reading(text_content_id=text_content[0].id, reader='REACH',
                   reader_version='v0', format='xml', batch_id=1),
        db.Reading(text_content_id=text_content[1].id, reader='REACH',
                   reader_version='v0', format='xml', batch_id=1),
        db.Reading(text_content_id=text_content[2].id, reader='REACH',
                   reader_version='v0', format='xml', batch_id=1),
        db.Reading(text_content_id=text_content[2].id, reader='SPARSER',
                   reader_version='v0', format='json', batch_id=1)
    ]
    db.session.add_all(readings)
    db.session.commit()

    mek = Agent('MEK', db_refs={'FPLX': 'MEK'})
    erk = Agent('ERK', db_refs={'FPLX': 'ERK'})

    def ev(ridx, pmid=None):
        if pmid is None:
            pmid = readings[ridx].text_content.text_ref.pmid
        return Evidence(readings[ridx].reader.lower(), pmid=pmid,
                        text="This is evidence text.")

    raw_statements_pre = {
        0: [
            Phosphorylation(mek, erk, evidence=[ev(0)]),
            Complex([mek, erk], [ev(0)])
        ],
        1: [
            Phosphorylation(mek, erk, evidence=[ev(1)])
        ],
        2: [
            Activation(mek, erk, evidence=[ev(2)])
        ],
        3: [
            Complex([mek, erk], evidence=[ev(3, None)])
        ]
    }

    raw_stmts = []
    for ridx, stmt_list in raw_statements_pre.items():
        reading = readings[ridx]
        for stmt in stmt_list:
            raw_json = stmt.to_json()
            src_hash = stmt.evidence[0].get_source_hash()
            raw_stmts.append(
                db.RawStatements(reading_id=reading.id,
                                 json=json.dumps(raw_json).encode('utf-8'),
                                 type=raw_json['type'], uuid=stmt.uuid,
                                 batch_id=0, source_hash=src_hash,
                                 mk_hash=stmt.get_hash(), indra_version="test")
            )
    db.session.add_all(raw_stmts)
    db.session.commit()

    insert_raw_agents(db, 0, [s for sl in raw_statements_pre.values()
                              for s in sl])

    pa_statements_pre = [
        (Phosphorylation(mek, erk), [raw_stmts[0], raw_stmts[2]]),
        (Complex([mek, erk]), [raw_stmts[1], raw_stmts[4]]),
        (Activation(mek, erk), [raw_stmts[3]])
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


def test_raw_statement_retrieval_from_agents_type_only():
    db = _construct_database()
    res = get_raw_stmt_jsons_from_agents(stmt_type='Complex', db=db)
    assert len(res) == 2
    assert set(res.keys()) == {2, 5}


def test_raw_statement_retrieval_generic():
    db = _construct_database()
    res = get_raw_stmt_jsons([db.Reading.reader == 'REACH',
                              db.Reading.id == db.RawStatements.reading_id],
                             db=db)
    assert len(res) > 0
    assert len(res) < 5
    assert all(sj['evidence'][0]['source_api'] == 'reach'
               for sj in res.values())


def test_pa_statement_retrieval_generic():
    db = _construct_database()
    res = get_pa_stmt_jsons(db=db)
    assert len(res) == 3


def test_pa_statement_retrieval_by_type():
    db = _construct_database()
    res = get_pa_stmt_jsons([db.PAStatements.type == 'Complex'], db=db)
    assert len(res) == 1
    assert all(j['stmt']['type'] == 'Complex' for j in res.values())
