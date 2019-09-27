import os
import pickle
import random
from collections import defaultdict
from unittest import SkipTest

from nose.plugins.attrib import attr

from indra.literature import pubmed_client as pubc
from indra.statements import stmts_from_json

from indra_db import util as dbu
from indra_db import client as dbc
from indra_db.tests.util import get_prepped_db, get_filled_ro, get_temp_db

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@attr('nonpublic', 'slow')
def test_get_statements():
    num_stmts = 10000
    db = get_prepped_db(num_stmts)

    # Test getting all statements
    stmts = dbc.get_statements([], preassembled=False, db=db)
    assert len(stmts) == num_stmts, len(stmts)

    stmts = dbc.get_statements([db.RawStatements.reading_id.isnot(None)],
                               preassembled=False, db=db)
    pmids = {s.evidence[0].pmid for s in random.sample(stmts, 100)}
    assert pmids
    assert pmids != {None}
    md_list = pubc.get_metadata_for_ids([pmid for pmid in pmids
                                         if pmid is not None])
    assert len(md_list) == len(pmids - {None}),\
        (len(md_list), len(pmids - {None}))

    # Test getting some statements
    stmt_uuid = stmts[0].uuid
    stmts = dbc.get_statements([db.RawStatements.uuid != stmt_uuid],
                               preassembled=False, db=db)
    assert len(stmts) == num_stmts-1, len(stmts)

    # Test getting statements without fix refs.
    stmts = dbc.get_statements([db.RawStatements.reading_id.isnot(None),
                                db.RawStatements.reading_id == db.Reading.id,
                                db.Reading.reader == 'SPARSER'],
                               preassembled=False, fix_refs=False, db=db)
    assert 0 < len(stmts) < num_stmts, len(stmts)
    pmids = {s.evidence[0].pmid for s in random.sample(stmts, 200)}
    assert None in pmids, pmids


@attr('nonpublic', 'slow')
def test_get_statements_by_grot():
    """Test get statements by gene-role-type."""
    num_stmts = 10000
    db = get_prepped_db(num_stmts, with_agents=True)

    stmts = dbc.get_statements_by_gene_role_type('MAP2K1', preassembled=False,
                                                 db=db)
    assert stmts

    stmts = dbc.get_statements_by_gene_role_type('MEK', agent_ns='FPLX',
                                                 preassembled=False, db=db)
    assert stmts

    stmts = dbc.get_statements_by_gene_role_type('MAP2K1', preassembled=False,
                                                 fix_refs=False, db=db)
    assert stmts

    stmts = dbc.get_statements_by_gene_role_type('MAP2K1', preassembled=False,
                                                 essentials_only=True, db=db)
    assert stmts


@attr('nonpublic', 'known_failing')
def test_get_statments_grot_wo_evidence():
    num_stmts = 10000
    db = get_prepped_db(num_stmts, with_agents=True)

    stmts = dbc.get_statements_by_gene_role_type('MAP2K1', with_evidence=False,
                                                 db=db)
    assert stmts, stmts


@attr('nonpublic')
def test_get_content_by_refs():
    db = get_prepped_db(100)
    tcid = db.select_one(db.TextContent.id)[0]
    reading_dict = dbc.get_reader_output(db, tcid)
    assert reading_dict


@attr('nonpublic')
def test_readonly_creation():
    ro = get_filled_ro(1000)
    res = ro.select_all(ro.PaStmtSrc)
    assert len(res), res


@attr('nonpublic', 'known_failing')
def test_get_statement_jsons_by_agent():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()

    agents = [(None, 'MEK', 'FPLX'), (None, 'ERK', 'FPLX')]
    stmt_jsons = dbc.get_statement_jsons_from_agents(agents=agents,
                                                     stmt_type='Phosphorylation',
                                                     db=db)
    assert stmt_jsons
    assert stmt_jsons['statements']
    assert stmt_jsons['total_evidence']
    assert stmt_jsons['evidence_returned']
    stmts = stmts_from_json(stmt_jsons['statements'].values())
    assert len(stmts) == len(stmt_jsons['statements'])
    for s in stmts:
        s_agents = [(None, ag_id, ag_ns) for ag in s.agent_list()
                    for ag_ns, ag_id in ag.db_refs.items()]
        for ag_tpl in agents:
            assert ag_tpl in s_agents


@attr('nonpublic', 'known_failing')
def test_get_statement_jsons_options():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()

    options = {'max_stmts': 10, 'ev_limit': 4, 'offset': 5,
               'best_first': False}
    agents = [('SUBJECT', 'MEK', 'FPLX'), ('OBJECT', 'ERK', 'FPLX')]
    option_dicts = [{}]
    for key, value in options.items():
        nd = {key: value}
        new_option_dicts = []
        for option_dict in option_dicts:
            new_option_dicts.append(option_dict)
            new_option_dicts.append({k: v for d in [option_dict, nd]
                                     for k, v in d.items()})
        option_dicts = new_option_dicts

    ev_counts = {}
    total_stmts = None
    for option_dict in option_dicts:
        res = dbc.get_statement_jsons_from_agents(agents=agents,
                                                  stmt_type='Phosphorylation',
                                                  db=db, **option_dict)
        assert res
        assert len(res['statements'])
        stmts = res['statements']
        if 'max_stmts' in option_dict.keys():
            assert len(stmts) == option_dict['max_stmts']
        elif not 'offset' in option_dict.keys():
            if total_stmts:
                assert len(stmts) == total_stmts,\
                    ("Number of statements returned changed incorrectly."
                     "Actual: %d, expected: %d" % (len(stmts), total_stmts))
            else:
                total_stmts = len(stmts)

        my_ev_counts = {}
        for mk_hash, stmt in stmts.items():
            my_ev_counts[mk_hash] = len(stmt['evidence'])
        if 'ev_limit' in option_dict.keys():
            assert all([c <= options['ev_limit']
                        for c in my_ev_counts.values()]),\
                "Evidence limit was not applied: %s." % my_ev_counts
        else:
            my_ev_counts = {}
            for mk_hash, stmt in stmts.items():
                my_ev_counts[mk_hash] = len(stmt['evidence'])

            if ev_counts:
                assert all([ev_counts[h] == c
                            for h, c in my_ev_counts.items()]),\
                    ("Evidence counts changed: %s vs. %s"
                     % (my_ev_counts, ev_counts))
            else:
                ev_counts = my_ev_counts
    return


@attr('nonpublic', 'known_failing')
def test_nfkb_anomaly():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()

    agents = [(None, 'NFkappaB', 'FPLX')]
    res = dbc.get_statement_jsons_from_agents(agents=agents, max_stmts=10,
                                              ev_limit=10, db=db)
    assert res
    assert len(res['statements']) == 10, len(res['statements'])


@attr('nonpublic', 'known_failing')
def test_triple_agent_bug():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()

    agents = [(None, '1834', 'HGNC'), (None, '6769', 'HGNC'),
              (None, '12856', 'HGNC')]
    res = dbc.get_statement_jsons_from_agents(agents=agents, max_stmts=10,
                                              stmt_type='Complex',
                                              ev_limit=5, db=db)
    assert res


@attr('nonpublic')
def test_null_response():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()

    res = dbc.get_statement_jsons_from_hashes([0], db=db)
    assert isinstance(res, dict), type(res)
    assert len(res['statements']) == 0, len(res['statements'])


@attr('nonpublic', 'known_failing')
def test_get_statement_jsons_by_paper_id():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()

    paper_refs = [('pmid', '27769048'),
                  ('doi', '10.3389/FIMMU.2017.00781'),
                  ('pmcid', 'PMC4789553')]
    stmt_jsons = dbc.get_statement_jsons_from_papers(paper_refs, db=db)
    assert stmt_jsons
    assert stmt_jsons['statements']
    assert stmt_jsons['total_evidence']
    stmts = stmts_from_json(stmt_jsons['statements'].values())
    assert len(stmts) == len(stmt_jsons['statements'])
    pmid_set = {ev.pmid for s in stmts for ev in s.evidence}
    assert len(pmid_set) >= len(paper_refs)


@attr('nonpublic', 'known_failing')
def test_get_statement_jsons_by_mk_hash():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()

    res = {h for h, in db.select_all(db.PAStatements.mk_hash)}
    mk_hashes = random.sample(res, 100)

    stmt_jsons = dbc.get_statement_jsons_from_hashes(mk_hashes, db=db)
    assert stmt_jsons
    assert stmt_jsons['statements']
    assert stmt_jsons['total_evidence']
    stmts = stmts_from_json(stmt_jsons['statements'].values())
    assert len(stmts) == len(stmt_jsons['statements'])
    assert len(stmts) == len(mk_hashes)


@attr('nonpublic', 'known_failing')
def test_get_statement_jsons_by_mk_hash_sparser_bug():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()

    mk_hashes = {-26808188314528604}
    stmt_jsons = dbc.get_statement_jsons_from_hashes(mk_hashes, db=db)
    assert stmt_jsons
    assert len(stmt_jsons['statements']) == 1, len(stmt_jsons['statements'])
    stmts = stmts_from_json(stmt_jsons['statements'].values())
    ev_list = [ev for s in stmts for ev in s.evidence]
    assert any([ev.source_api == 'sparser' for ev in ev_list]), \
        'No evidence from sparser.'
    assert all([(ev.source_api not in ['reach', 'sparser']
                 or (hasattr(ev, 'pmid') and ev.pmid is not None))
                for ev in ev_list])


def _get_ref_id_samples(db, num_each):
    id_types = ['trid', 'pmid', 'pmcid']

    ref_data = db.select_all([db.TextRef.id, db.TextRef.pmid,
                              db.TextRef.pmcid])

    ref_id_sets = defaultdict(set)
    for row in ref_data:
        for id_type, id_val in zip(id_types, row):
            ref_id_sets[id_type].add(id_val)

    id_sample_dict = {id_type: random.sample(ids, min(num_each, len(ids)))
                      for id_type, ids in ref_id_sets.items()}

    return id_sample_dict


@attr('nonpublic')
def test_raw_stmt_jsons_from_papers():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()
    id_samples = _get_ref_id_samples(db, 100)

    res_nums = {}
    for id_type, id_list in id_samples.items():
        print(id_type)
        res_dict = dbc.get_raw_stmt_jsons_from_papers(id_list, id_type=id_type,
                                                      db=db)
        res_nums[id_type] = len(res_dict)

    assert all(res_nums.values()),\
        'Failure with %s' % str({id_type: num
                                 for id_type, num in res_nums.items()
                                 if not num})

    return


@attr('nonpublic')
def test_stmts_from_papers():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()
    id_samples = _get_ref_id_samples(db, 100)

    for id_type, id_list in id_samples.items():
        print(id_type)

        # Test pa retrieval
        pa_dict = dbc.get_statements_by_paper(id_list, id_type=id_type, db=db)
        assert len(pa_dict), 'Failure with %s %s' % ('pa', id_type)

        # Test raw retrieval
        raw_dict = dbc.get_statements_by_paper(id_list, id_type=id_type,
                                               preassembled=False, db=db)
        assert len(raw_dict), 'Failure with %s %s' % ('raw', id_type)

    return


@attr('nonpublic')
def test_pa_curation():
    db = get_prepped_db(100, with_pa=True)
    sample = db.select_sample_from_table(2, db.PAStatements)
    mk_hashes = {s.mk_hash for s in sample}
    i = 0
    for pa_hash in mk_hashes:
        dbc.submit_curation(pa_hash, tag='test1',
                            text='This is a test.', curator='tester%d' % i,
                            ip='192.0.2.1', source='test_app', db=db)
        i += 1
        dbc.submit_curation(pa_hash, tag='test2',
                            text='This is a test too.',
                            curator='tester%d' % i, ip='192.0.2.32',
                            source='test_app', db=db)
        i += 1
    res = db.select_all(db.Curation)
    assert len(res) == 4, "Wrong number of curations: %d" % len(res)
    curs = dbc.get_curations(tag='test1', db=db)
    assert len(curs) == 2, len(curs)
    curs2 = dbc.get_curations(curator='tester2', db=db)
    assert len(curs2) == 1, len(curs2)


@attr('nonpublic')
def test_bad_hash_curation():
    db = get_prepped_db(100, with_pa=True)
    try:
        dbc.submit_curation(0, tag='test', text='text',
                            curator='tester', ip='192.0.2.1', db=db)
    except dbc.BadHashError:
        return
    except Exception as e:
        assert False, "Excepted for the wrong reason: %s" % str(e)
    assert False, "Didn't get an error."


@attr('nonpublic')
def test_source_hash():
    db = get_prepped_db(100)
    res = db.select_all(db.RawStatements)
    pairs = [(dbu.get_statement_object(db_raw), db_raw.source_hash)
             for db_raw in res]
    for stmt, sh in pairs:
        sh_rec = stmt.evidence[0].get_source_hash()
        assert sh_rec == sh,\
            "Recreated source hash %s does not match database sourch hash %s."\
            % (sh_rec, sh)


def test_get_raw_statement_jsons_from_agents():
    db = get_prepped_db(100000)
    res = dbc.get_direct_raw_stmt_jsons_from_agents(db=db, agents=[
        ('SUBJECT', 'MEK', 'FPLX'),
    ])
    assert len(res)
    assert isinstance(res, dict)
    stmts = stmts_from_json(res.values())
    assert stmts
    assert all('MEK' == s.agent_list()[0].db_refs['FPLX']
               for s in stmts), stmts


def test_get_raw_statement_json_from_papers():
    db = get_prepped_db(10000)
    pmid, = db.select_one(db.TextRef.pmid, db.TextRef.pmid.isnot(None),
                          *db.link(db.TextRef, db.RawStatements))

    res = dbc.get_raw_stmt_jsons_from_papers([pmid], id_type='pmid', db=db)
    assert len(res) == 1, len(res)
    assert pmid in res.keys(), 'Expected %s in %s' % (pmid, res.keys())
    stmts = stmts_from_json(res[pmid])
    assert len(stmts)
