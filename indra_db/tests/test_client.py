import os
import pickle
import random

from nose.plugins.attrib import attr

from indra.literature import pubmed_client as pubc
from indra.statements import stmts_from_json

from indra_db import util as dbu
from indra_db import client as dbc
from indra_db.tests.util import get_prepped_db, get_db_with_views, get_temp_db

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@attr('nonpublic', 'slow')
def test_get_statements():
    num_stmts = 10000
    db, _ = get_prepped_db(num_stmts)

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
    db, _ = get_prepped_db(num_stmts, with_agents=True)

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


@attr('nonpublic')
def test_get_statments_grot_wo_evidence():
    num_stmts = 1000
    db, _ = get_prepped_db(num_stmts, with_agents=True)

    stmts = dbc.get_statements_by_gene_role_type('MAP2K1', with_evidence=False,
                                                 db=db)
    assert stmts, stmts


@attr('nonpublic')
def test_get_content_by_refs():
    db, _ = get_prepped_db(100)
    tcid = db.select_one(db.TextContent.id)[0]
    reading_dict = dbc.get_reader_output(db, tcid)
    assert reading_dict


def test_materialize_view_creation():
    db = get_db_with_views(10000)
    res = db.select_all(db.PaStmtSrc)
    assert len(res), res


@attr('nonpublic')
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


@attr('nonpublic')
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


@attr('nonpublic')
def test_nfkb_anomaly():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()

    agents = [(None, 'NFkappaB', 'FPLX')]
    res = dbc.get_statement_jsons_from_agents(agents=agents, max_stmts=1000,
                                              ev_limit=10, db=db)
    assert res
    assert len(res['statements']) == 1000, len(res['statements'])


@attr('nonpublic')
def test_triple_agent_bug():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()

    agents = [(None, '1834', 'HGNC'), (None, '6769', 'HGNC'),
              (None, '12856', 'HGNC')]
    res = dbc.get_statement_jsons_from_agents(agents=agents, max_stmts=100,
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


@attr('nonpublic')
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


@attr('nonpublic')
def test_get_statement_jsons_by_mk_hash():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()

    mk_hashes = {-35990550780621697, -34509352007749723, -33762223064440060,
                 -33265410753427801, -33264422871226821, -33006503639209361,
                 -32655830663272427, -32156860839881910, -31266440463655983,
                 -30976459682454095, -30134498128794870, -28378918778758360,
                 -24358695784465547, -24150179679440010, -23629903237028340,
                 -23464686784015252, -23180557592374280, -22224931284906368,
                 -21436209384793545, -20730671023219399, -20628745469039833,
                 -19678219086449831, -19263047787836948, -19233240978956273,
                 -18854520239423344, -18777221295617488, -18371306000768702,
                 -17790680150174704, -17652929178146873, -17157963869106438,
                 -17130129999418301, -16284802852951883, -16037105293100219,
                 -15490761426613291, -14975140226841255, -14082507581435438,
                 -13857723711006775, -12377086298836870, -11313819223154032,
                 -11213416806465629, -10533303510589718, -9966418144787259,
                 -9862339997617041,  -9169838304025767, -7914540609802583,
                 -5761487437008515, -5484899507470794, -4221831802677562,
                 -3843980816183311, -3444432161721189, -2550187846777281,
                 -1690192884583623, -1574988790414009, -776020752709166,
                 -693617835322587, -616115799439746, 58075179102507,
                 1218693303789519, 1858833736757788, 1865941926926838,
                 1891718725870829, 3185457948420843, 3600108659508886,
                 3858621152710053, 4594557398132265, 5499056407723241,
                 6796567607348165, 6828272448940477, 6929632245307987,
                 7584487035784255, 8424911311360927, 8837984832930769,
                 10511090751198119, 10789407105295331, 10924988153490362,
                 11707113199128693, 12528041861567565, 13094138872844955,
                 13166641722496149, 13330125910711684, 13347703252882432,
                 15002261599485956, 16397210433817325, 16975780060710533,
                 17332680694583377, 17888579535249950, 19337587406307012,
                 22774500444258387, 23665225082661845, 23783937267011041,
                 24050979216195140, 24765024299377586, 25290573037450021,
                 29491428193112311, 30289509021065753, 30992174235867673,
                 31766667918079590, 31904387104764159, 34782800852366343,
                 35686927318045812}
    stmt_jsons = dbc.get_statement_jsons_from_hashes(mk_hashes, db=db)
    assert stmt_jsons
    assert stmt_jsons['statements']
    assert stmt_jsons['total_evidence']
    stmts = stmts_from_json(stmt_jsons['statements'].values())
    assert len(stmts) == len(stmt_jsons['statements'])
    assert len(stmts) == len(mk_hashes)


@attr('nonpublic')
def test_get_statement_jsons_by_mk_hash_sparser_bug():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()

    mk_hashes = {7066059628266471, -3738332857231047}
    stmt_jsons = dbc.get_statement_jsons_from_hashes(mk_hashes, db=db)
    assert stmt_jsons
    assert len(stmt_jsons['statements']) == 2, len(stmt_jsons['statements'])
    stmts = stmts_from_json(stmt_jsons['statements'].values())
    ev_list = [ev for s in stmts for ev in s.evidence]
    assert any([ev.source_api == 'sparser' for ev in ev_list]), \
        'No evidence from sparser.'
    assert all([(ev.source_api not in ['reach', 'sparser']
                 or (hasattr(ev, 'pmid') and ev.pmid is not None))
                for ev in ev_list])


@attr('nonpublic')
def test_pa_curation():
    db, key = get_prepped_db(100, with_pa=True)
    sample = db.select_sample_from_table(2, db.PAStatements)
    mk_hashes = {s.mk_hash for s in sample}
    i = 0
    for pa_hash in mk_hashes:
        dbc.submit_curation(pa_hash, api_key=key, tag='test1',
                            text='This is a test.', curator='tester%d' % i,
                            ip='192.0.2.1', source='test_app', db=db)
        i += 1
        dbc.submit_curation(pa_hash, api_key=key, tag='test2',
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
    db, key = get_prepped_db(100, with_pa=True)
    try:
        dbc.submit_curation(0, api_key=key, tag='test', text='text',
                            curator='tester', ip='192.0.2.1', db=db)
    except dbc.BadHashError:
        return
    except Exception as e:
        assert False, "Excepted for the wrong reason: %s" % str(e)
    assert False, "Didn't get an error."


@attr('nonpublic')
def test_source_hash():
    db, _ = get_prepped_db(100)
    res = db.select_all(db.RawStatements)
    pairs = [(dbu.get_statement_object(db_raw), db_raw.source_hash)
             for db_raw in res]
    for stmt, sh in pairs:
        sh_rec = stmt.evidence[0].get_source_hash()
        assert sh_rec == sh,\
            "Recreated source hash %s does not match database sourch hash %s."\
            % (sh_rec, sh)


@attr('nonpublic')
def test_raw_stmt_jsons_from_papers():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()

    with open(os.path.join(THIS_DIR, 'id_sample_lists.pkl'), 'rb') as f:
        id_samples = pickle.load(f)

    for id_type, id_list in id_samples.items():
        print(id_type)
        res_dict = dbc.get_raw_stmt_jsons_from_papers(id_list, id_type=id_type,
                                                      db=db)
        assert len(res_dict), 'Failure with %s' % id_type

    return


@attr('nonpublic')
def test_stmts_from_papers():
    # Note that these tests only work if `test_materialize_view_creation` has
    # passed, and it is assumed the test database remains in a good state.
    db = get_temp_db()

    with open(os.path.join(THIS_DIR, 'id_sample_lists.pkl'), 'rb') as f:
        id_samples = pickle.load(f)

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
