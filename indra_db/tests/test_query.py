import json
import random
from collections import defaultdict
from itertools import combinations, permutations, product

from indra.statements import Agent, get_statement_by_name, get_all_descendants, \
    Complex
from indra.sources.indra_db_rest.query_results import QueryResult
from indra_db.schemas.readonly_schema import ro_type_map, ro_role_map, \
    SOURCE_GROUPS
from indra_db.util import extract_agent_data, get_ro
from indra_db.client.readonly.query import *

from indra_db.tests.util import get_temp_db


def make_agent_from_ref(ref):
    db_refs = ref.copy()
    name = db_refs.pop('NAME')
    return Agent(name, db_refs=db_refs)


def _build_test_set():
    agents = [{'NAME': 'ERK', 'FPLX': 'ERK', 'TEXT': 'MAPK'},
              {'NAME': 'TP53', 'HGNC': '11998'},
              {'NAME': 'MEK', 'FPLX': 'MEK'},
              {'NAME': 'Vemurafenib', 'CHEBI': 'CHEBI:63637'}]
    stypes = ['Phosphorylation', 'Activation', 'Inhibition', 'Complex']
    sources = [('medscan', 'rd'), ('reach', 'rd'), ('pc', 'db'),
               ('signor', 'db')]
    mesh_term_ids = ['D000225', 'D002352', 'D015536', 'D00123413', 'D0000334']
    mesh_concept_ids = ['C0001243', 'C005758']
    all_mesh_ids = mesh_term_ids + mesh_concept_ids

    mesh_combos = []
    for num_mesh in range(0, 3):
        if num_mesh == 1:
            mesh_groups = [[mid] for mid in all_mesh_ids]
        else:
            mesh_groups = combinations(all_mesh_ids, num_mesh)

        mesh_combos.extend(list(mesh_groups))
    random.shuffle(mesh_combos)

    source_data = []
    for num_srcs in range(1, 5):
        if num_srcs == 1:
            src_iter = [[src] for src in sources]
        else:
            src_iter = combinations(sources, num_srcs)

        for src_list in src_iter:
            only_src = None if len(src_list) > 1 else src_list[0][0]
            has_rd = any(t == 'rd' for _, t in src_list)
            if has_rd:
                mesh_ids = mesh_combos[len(source_data) % len(mesh_combos)]
            else:
                mesh_ids = []
            source_data.append({'sources': {src: random.randint(1, 50)
                                            for src, _ in src_list},
                                'has_rd': any(t == 'rd' for _, t in src_list),
                                'has_db': any(t == 'db' for _, t in src_list),
                                'only_src': only_src,
                                'mesh_ids': mesh_ids})
    random.shuffle(source_data)

    stmts = [tuple(tpl) + (None, None)
             for tpl in product(stypes, permutations(agents, 2))]
    stmts += [('ActiveForm', (ref,), activity, is_active)
              for activity, is_active, ref
              in product(['transcription', 'activity'], [True, False], agents)]

    complex_pairs = []

    name_meta_rows = []
    name_meta_cols = ('mk_hash', 'ag_num', 'db_id', 'role_num', 'type_num',
                      'ev_count', 'belief', 'activity', 'is_active',
                      'agent_count')

    text_meta_rows = []
    text_meta_cols = ('mk_hash', 'ag_num', 'db_id', 'role_num', 'type_num',
                      'ev_count', 'belief', 'activity', 'is_active',
                      'agent_count')

    other_meta_rows = []
    other_meta_cols = ('mk_hash', 'ag_num', 'db_name', 'db_id', 'role_num',
                       'type_num', 'ev_count', 'belief', 'activity',
                       'is_active', 'agent_count')

    source_meta_rows = []
    source_meta_cols = ('mk_hash', 'reach', 'medscan', 'pc', 'signor',
                        'ev_count', 'belief', 'type_num', 'activity',
                        'is_active', 'agent_count', 'num_srcs', 'src_json',
                        'only_src', 'has_rd', 'has_db')

    mesh_term_meta_rows = []
    mesh_term_meta_cols = ('mk_hash', 'ev_count', 'belief', 'mesh_num',
                           'type_num', 'activity', 'is_active', 'agent_count')

    mesh_concept_meta_rows = []
    mesh_concept_meta_cols = ('mk_hash', 'ev_count', 'belief', 'mesh_num',
                              'type_num', 'activity', 'is_active',
                              'agent_count')
    for stype, refs, activity, is_active in stmts:
        # Extract agents, and make a Statement.
        StmtClass = get_statement_by_name(stype)
        if stype == 'ActiveForm':
            ag = make_agent_from_ref(refs[0])
            stmt = StmtClass(ag, activity=activity, is_active=is_active)
        else:
            ag1 = make_agent_from_ref(refs[0])
            ag2 = make_agent_from_ref(refs[1])
            if stype == 'Complex':
                if {ag1.name, ag2.name} in complex_pairs:
                    continue
                stmt = StmtClass([ag1, ag2])
                complex_pairs.append({ag1.name, ag2.name})
            else:
                stmt = StmtClass(ag1, ag2)

        # Connect with a source.
        source_dict = source_data[len(source_meta_rows) % len(source_data)]
        ev_count = sum(source_dict['sources'].values())
        belief = random.random()
        src_row = (stmt.get_hash(),)
        for src_name in ['reach', 'medscan', 'pc', 'signor']:
            src_row += (source_dict['sources'].get(src_name),)
        src_row += (ev_count, belief, ro_type_map.get_int(stype), activity,
                    is_active, len(refs), len(source_dict['sources']),
                    json.dumps(source_dict['sources']), source_dict['only_src'],
                    source_dict['has_rd'], source_dict['has_db'])
        source_meta_rows.append(src_row)

        # Add mesh rows
        for mesh_id in source_dict['mesh_ids']:
            if mesh_id[0] == 'D':
                mesh_term_meta_rows.append(
                    (stmt.get_hash(), ev_count, belief, int(mesh_id[1:]),
                     ro_type_map.get_int(stype), activity, is_active, len(refs))
                )
            else:
                mesh_concept_meta_rows.append(
                    (stmt.get_hash(), ev_count, belief, int(mesh_id[1:]),
                     ro_type_map.get_int(stype), activity, is_active, len(refs))
                )

        # Generate agent rows.
        ref_rows, _, _ = extract_agent_data(stmt, stmt.get_hash())
        for row in ref_rows:
            row = row[:4] + (ro_role_map.get_int(row[4]),
                             ro_type_map.get_int(stype),  ev_count, belief,
                             activity, is_active, len(refs))
            if row[2] == 'NAME':
                row = row[:2] + row[3:]
                name_meta_rows.append(row)
            elif row[2] == 'TEXT':
                row = row[:2] + row[3:]
                text_meta_rows.append(row)
            else:
                other_meta_rows.append(row)

    db = get_temp_db(clear=True)
    src_meta_cols = [{'name': col} for col, _ in sources]
    if 'readonly' not in db.get_schemas():
        db.create_schema('readonly')
    db.load_source_meta_cols(src_meta_cols)
    for tbl in [db.SourceMeta, db.MeshTermMeta, db.MeshConceptMeta, db.NameMeta,
                db.TextMeta, db.OtherMeta]:
        db.create_table(tbl)
    db.copy('readonly.source_meta', source_meta_rows, source_meta_cols)
    db.copy('readonly.mesh_term_meta', mesh_term_meta_rows, mesh_term_meta_cols)
    db.copy('readonly.mesh_concept_meta', mesh_concept_meta_rows,
            mesh_concept_meta_cols)
    db.copy('readonly.name_meta', name_meta_rows, name_meta_cols)
    db.copy('readonly.text_meta', text_meta_rows, text_meta_cols)
    db.copy('readonly.other_meta', other_meta_rows, other_meta_cols)
    return db


class Counter:
    def __init__(self):
        self.correct = 0
        self.incorrect = 0
        self.total = 0
        self.section_correct = 0
        self.section_incorrect = 0
        self.section_total = 0

    def up(self, correct):
        if correct:
            self.correct += 1
            self.section_correct += 1
        else:
            self.incorrect += 1
            self.section_incorrect += 1
        self.total += 1
        self.section_total += 1

        if self.section_total % 100 == 0:
            self._print()
        return

    def _print(self):
        print(f'\rCorrect: {self.section_correct}, '
              f'Incorrect: {self.section_incorrect}, '
              f'Total: {self.section_total}', end='', flush=True)

    def mark(self, section):
        self._print()
        print()
        print(section)
        self.section_total = 0
        self.section_incorrect = 0
        self.section_correct = 0


def test_has_sources():
    ro = get_ro('primary')
    q = HasSources(['reach', 'sparser'])
    res = q.get_statements(ro, limit=5, ev_limit=8)
    assert len(res.results) == 5
    stmts = res.statements()
    res_json = res.json()
    assert 'results' in res_json
    assert len(stmts) == len(res.results)
    assert all(sc[r] > 0 for sc in res.source_counts.values()
               for r in ['reach', 'sparser'])


def test_has_only_source():
    ro = get_ro('primary')
    q = HasOnlySource('signor')
    res = q.get_statements(ro, limit=5, ev_limit=8)
    res_json = res.json()
    assert 'results' in res_json
    assert set(res.results.keys()) == set(res.source_counts.keys())
    stmts = res.statements()
    assert len(stmts) == len(res.results)
    assert all(src_cnt > 0 if src == 'signor' else src_cnt == 0
               for sc in res.source_counts.values()
               for src, src_cnt in sc.items())


def test_has_readings():
    ro = get_ro('primary')
    q = HasReadings()
    res = q.get_statements(ro, limit=5, ev_limit=8)
    for sc in res.source_counts.values():
        for src, cnt in sc.items():
            if src in SOURCE_GROUPS['reading'] and cnt > 0:
                break
        else:
            assert False, f"No readings found in: {sc}"
    assert set(res.results.keys()) == set(res.source_counts.keys())
    stmts = res.statements()
    assert len(stmts) == len(res.results)


def test_has_databases():
    ro = get_ro('primary')
    q = HasDatabases()
    res = q.get_statements(ro, limit=5, ev_limit=8)
    for sc in res.source_counts.values():
        for src, cnt in sc.items():
            if src in SOURCE_GROUPS['databases'] and cnt > 0:
                break
        else:
            assert False, f"No databases found in: {sc}"
    assert set(res.results.keys()) == set(res.source_counts.keys())
    stmts = res.statements()
    assert len(stmts) == len(res.results)


def test_has_hash():
    ro = get_ro('primary')
    hashes = {h for h, in ro.session.query(ro.SourceMeta.mk_hash).limit(10)}
    q = HasHash(hashes)
    res = q.get_statements(ro, limit=5, ev_limit=8)
    assert set(res.results.keys()) < hashes
    assert set(res.results.keys()) == set(res.source_counts.keys())


def test_has_agent():
    ro = get_ro('primary')
    q = HasAgent('RAS')
    res = q.get_statements(ro, limit=5, ev_limit=8)
    stmts = res.statements()
    assert all('RAS' in [ag.name for ag in s.agent_list() if ag is not None]
               for s in stmts)

    q = HasAgent('MEK', namespace='FPLX', role='SUBJECT')
    res = q.get_statements(ro, limit=5, ev_limit=8)
    stmts = res.statements()
    assert all('MEK' == s.agent_list(deep_sorted=True)[0].db_refs['FPLX']
               or (isinstance(s, Complex)
                   and 'MEK' in {a.db_refs.get('FPLX') for a in s.agent_list()})
               for s in stmts)

    q = HasAgent('CHEBI:63637', namespace='CHEBI', agent_num=3)
    res = q.get_statements(ro, limit=5, ev_limit=8)
    stmts = res.statements()
    for s in stmts:
        ag = s.agent_list(deep_sorted=True)[3]
        assert ag.name == 'vemurafenib'
        assert ag.db_refs['CHEBI'] == 'CHEBI:63637'


def test_from_papers():
    ro = get_ro('primary')
    pmid = '27014235'
    q = FromPapers([('pmid', pmid)])
    res = q.get_statements(ro, limit=5)
    assert res.statements()
    assert all(any(ev.text_refs.get('PMID') == pmid for ev in s.evidence)
               for s in res.statements())


def test_has_num_agents():
    ro = get_ro('primary')
    q = HasNumAgents((1, 2))
    res = q.get_statements(ro, limit=5, ev_limit=8)
    stmts = res.statements()
    assert all(len(s.agent_list()) in (1, 2) for s in stmts)

    q = HasNumAgents((6,))
    res = q.get_statements(ro, limit=5, ev_limit=8)
    stmts = res.statements()
    assert all(sum([ag is not None for ag in s.agent_list()]) == 6
               for s in stmts)


def test_num_evidence():
    ro = get_ro('primary')
    q = HasNumEvidence(tuple(range(5, 10)))
    res = q.get_statements(ro, limit=5, ev_limit=8)
    assert all(5 <= n < 10 for n in res.evidence_counts.values())
    stmts = res.statements()
    assert all(5 < len(s.evidence) <= 8 for s in stmts)


def test_has_type():
    ro = get_ro('primary')
    q = HasType(['Phosphorylation', 'Activation'])
    res = q.get_statements(ro, limit=5, ev_limit=8)
    stmts = res.statements()
    assert all(s.__class__.__name__ in ('Phosphorylation', 'Activation')
               for s in stmts)

    type_list = ['SelfModification', 'RegulateAmount', 'Translocation']
    q = HasType(type_list, include_subclasses=True)
    res = q.get_statements(ro, limit=5, ev_limit=8)
    stmts = res.statements()
    types = {t for bt in (get_statement_by_name(n) for n in type_list)
             for t in [bt] + get_all_descendants(bt)}
    assert all(type(s) in types for s in stmts)


def test_from_mesh():
    ro = get_ro('primary')
    q = FromMeshIds(['D001943'])
    res = q.get_statements(ro, limit=5, ev_limit=8)
    mm_entries = ro.select_all([ro.MeshTermMeta.mk_hash, ro.MeshTermMeta.mesh_num],
                               ro.MeshTermMeta.mk_hash.in_(set(res.results.keys())))
    mm_dict = defaultdict(list)
    for h, mn in mm_entries:
        mm_dict[h].append(mn)

    assert all(1943 in mn_list for mn_list in mm_dict.values())


def test_is_inverse_of_for_intersections():
    q = FromMeshIds(['D001943']) & HasAgent('MEK')
    nq = ~q
    assert q.is_inverse_of(nq)
    assert nq.is_inverse_of(q)

    q2 = FromMeshIds(['D001943']) | ~HasAgent('MEK')
    assert not q.is_inverse_of(q2)
    assert not q2.is_inverse_of(q)


def test_is_inverse_of_for_unions():
    q = HasHash([1, 2, 3]) | HasDatabases()
    nq = ~q
    assert q.is_inverse_of(nq)
    assert nq.is_inverse_of(q)

    q2 = HasHash([1, 2, 3]) & ~HasDatabases()
    assert not q.is_inverse_of(q2)
    assert not q2.is_inverse_of(q)


def test_query_set_behavior():
    db = _build_test_set()
    all_hashes = {h for h, in db.select_all(db.SourceMeta.mk_hash)}
    all_source_counts = {c for c, in db.select_all(db.SourceMeta.ev_count)}
    print(f"There are {len(all_hashes)} distinct hashes in the database.")
    lookup_hashes = random.sample(all_hashes, 5)
    lookup_src_cnts = random.sample(all_source_counts, 3)

    c = Counter()

    def dq(query):
        # Test string creation and JSON dump and load
        assert str(query), "No string formed."
        q_json = query.to_json()
        json_str = json.dumps(q_json)
        assert json_str, "No JSON created (this is a really weird error)."
        rjq = Query.from_json(q_json)
        assert rjq == query, "Re-formed query does not == the original."
        assert rjq is not query, "Somehow re-composed query IS original query."

        rjq = Query.from_json(json.loads(json_str))
        assert rjq == query, "Re-formed query from re-formed JSON != original."
        assert rjq is not query, \
            "Somehow thoroughly reconstituted query IS original query."

        # Test actually running the query
        res = query.get_hashes(db, with_src_counts=False)
        return set(res.results)

    queries = [
        HasAgent('TP53', role='SUBJECT'),
        HasAgent('ERK', namespace='FPLX', role='OBJECT'),
        FromMeshIds(['D015536']),
        FromMeshIds(['D002352', 'D015536']),
        FromMeshIds(['D0000334', 'C0001243']),
        HasHash(lookup_hashes[:-3]),
        HasHash(lookup_hashes[-1:]),
        HasHash(lookup_hashes[-4:-1]),
        HasOnlySource('pc'),
        HasOnlySource('medscan'),
        HasReadings(),
        HasDatabases(),
        HasSources(['signor', 'reach']),
        HasSources(['medscan']),
        HasType(['Phosphorylation', 'Activation']),
        HasType(['RegulateActivity'], include_subclasses=True),
        HasType(['Complex']),
        HasNumAgents([2, 3]),
        HasNumAgents([1]),
        HasNumEvidence(lookup_src_cnts[:-1]),
        HasNumEvidence(lookup_src_cnts[-1:])
    ]

    failures = []
    results = []
    n_runs = 0
    unfound = []
    collecting_results = True

    def try_query(q, compare=None, md=None):
        nonlocal n_runs

        # Test query logical consistency
        result = None
        try:
            result = dq(q)
            if compare is not None:
                assert result == compare, 'Result mismatch.'
            if not q.empty and not result:
                unfound.append(q)
            if collecting_results:
                results.append((result, q))
            n_runs += 1
            c.up(True)
        except Exception as e:
            failures.append({'query': q, 'error': e, 'result': result,
                             'compare': compare, 'md': md})
            if collecting_results:
                results.append((result, q))
            n_runs += 1
            c.up(False)
            return

        # Test negative query logical consistency
        negative_result = None
        nq = None
        try:
            nq = ~q
            if nq is None:
                assert False, "Inverted query is None."
            negative_result = dq(nq)
            assert negative_result == (all_hashes - result), \
                'Negative result mismatch.'
            assert nq.is_inverse_of(q), "Inverse comparison failed! (nq vs. q)"
            assert q.is_inverse_of(nq), "Inverse comparison failed! (q vs. nq)"
            assert q.empty == nq.full, "q.empty != nq.full."
            assert q.full == nq.empty, "q.full != nq.empty."

            if not nq.empty and not negative_result:
                unfound.append(nq)
            if collecting_results:
                results.append((negative_result, nq))
            n_runs += 1
            c.up(True)
        except Exception as e:
            if md is not None:
                neg_md = 'not (' + md + ')'
            else:
                neg_md = 'not (' + str(q) + ')'
            failures.append({'query': nq, 'error': e, 'result': negative_result,
                             'compare': all_hashes - result, 'md': neg_md})
            if collecting_results:
                results.append((negative_result, nq))
            n_runs += 1
            c.up(False)
            return

        return

    c.mark("Testing individual queries...")
    for q in queries:
        try_query(q)
    original_results = [res for res in results if res[1] is not None]
    collecting_results = False

    c.mark("Testing pairs...")
    for (r1, q1), (r2, q2) in permutations(original_results, 2):
        try_query(q1 & q2, r1 & r2, md=f'{q1} and {q2}')
        try_query(q1 | q2, r1 | r2, md=f'{q1} or {q2}')

    c.mark("Testing simple triples...")
    for (r1, q1), (r2, q2), (r3, q3) in combinations(original_results, 3):
        try_query(q1 & q2 & q3, r1 & r2 & r3, md=f'{q1} and {q2} and {q3}')
        try_query(q1 | q2 | q3, r1 | r2 | r3, md=f'{q1} or {q2} or {q3}')

    c.mark("Testing mixed triples...")
    for (r1, q1), (r2, q2), (r3, q3) in permutations(original_results, 3):
        try_query((q1 & q2) | q3, (r1 & r2) | r3, md=f'({q1} and {q2}) or {q3}')
        try_query((q1 | q2) & q3, (r1 | r2) & r3, md=f'({q1} or {q2}) and {q3}')

    c.mark('Done!')
    print(f"Ran {n_runs} checks...")

    print(f"UNFOUND:")
    for q in unfound[:10]:
        print('-------------------------')
        print(q)
    if len(unfound) > 10:
        print(f"...overall {len(unfound)}...")
    print()

    if failures:
        print("FAILURES:")
        error_groups = {}
        for fd in failures:
            err_str = str(fd['error'])
            if err_str not in error_groups:
                error_groups[err_str] = []
            error_groups[err_str].append({'result': fd['query'],
                                          'input': fd['md']})
        for err_str, examples in error_groups.items():
            print("=================================")
            print(err_str)
            print("=================================")
            for example in examples[:5]:
                print('---------------------------------')
                print('input:', example['input'])
                print('query:', example['result'])
            if len(examples) > 5:
                print(f'...overall {len(examples)} errors...')
            print()

    assert not failures, f"{len(failures)}/{n_runs} checks failed."

    return


def test_get_interactions():
    ro = get_ro('primary')
    query = HasAgent('TP53')
    res = query.get_interactions(ro, limit=10)
    assert isinstance(res, QueryResult)
    assert len(res.results) == 10
    js = res.json()
    assert 'results' in js
    assert len(js['results']) == len(res.results)


def test_get_relations():
    ro = get_ro('primary')
    query = HasAgent('TP53')
    res = query.get_relations(ro, limit=10)
    assert isinstance(res, QueryResult)
    assert len(res.results) <= 10, len(res.results)
    js = res.json()
    assert 'results' in js
    assert len(js['results']) == len(res.results)


def test_get_agents():
    ro = get_ro('primary')
    query = HasAgent('TP53')
    res = query.get_agents(ro, limit=10)
    assert isinstance(res, QueryResult)
    assert len(res.results) <= 10, len(res.results)
    js = res.json()
    assert 'results' in js
    assert len(js['results']) == len(res.results)


def test_evidence_filtering_has_only_source():
    ro = get_ro('primary')
    q1 = HasAgent('TP53')
    q2 = ~HasOnlySource('medscan')
    query = q1 & q2
    res = query.get_statements(ro, limit=2, ev_limit=None,
                               evidence_filter=q2.ev_filter())
    assert isinstance(res, StatementQueryResult)
    stmts = res.statements()
    assert len(stmts) == 2
    assert not any(ev.text_refs.get('READER') == 'medscan' for s in stmts
                   for ev in s.evidence)
    js = res.json()
    assert 'results' in js
    assert len(js['results']) == len(stmts)


def test_evidence_filtering_has_source():
    ro = get_ro('primary')
    q1 = HasAgent('TP53')
    q2 = HasSources(['reach', 'sparser'])
    query = q1 & q2
    res = query.get_statements(ro, limit=2, ev_limit=None,
                               evidence_filter=q2.ev_filter())
    assert isinstance(res, StatementQueryResult)
    stmts = res.statements()
    assert len(stmts) == 2
    assert all(ev.text_refs.get('READER') in ['REACH', 'SPARSER']
               for s in stmts for ev in s.evidence)
    js = res.json()
    assert 'results' in js
    assert len(js['results']) == len(stmts)


def test_evidence_filtering_has_database():
    ro = get_ro('primary')
    q1 = HasAgent('TP53')
    q2 = HasDatabases()
    query = q1 & q2
    res = query.get_statements(ro, limit=2, ev_limit=None,
                               evidence_filter=q2.ev_filter())
    assert isinstance(res, StatementQueryResult)
    stmts = res.statements()
    assert len(stmts) == 2
    assert all(ev.source_api not in SOURCE_GROUPS['reading']
               for s in stmts for ev in s.evidence)
    js = res.json()
    assert 'results' in js
    assert len(js['results']) == len(stmts)


def test_evidence_filtering_has_readings():
    ro = get_ro('primary')
    q1 = HasAgent('TP53')
    q2 = HasReadings()
    query = q1 & q2
    res = query.get_statements(ro, limit=2, ev_limit=10,
                               evidence_filter=q2.ev_filter())
    assert isinstance(res, StatementQueryResult)
    stmts = res.statements()
    assert len(stmts) == 2
    assert all(ev.source_api in SOURCE_GROUPS['reading']
               for s in stmts for ev in s.evidence)
    assert all(len(s.evidence) == 10 for s in stmts)
    js = res.json()
    assert 'results' in js
    assert len(js['results']) == len(stmts)


def test_evidence_filtering_mesh():
    ro = get_ro('primary')
    q1 = HasAgent('TP53')
    q2 = FromMeshIds(['D001943'])
    query = q1 & q2
    res = query.get_statements(ro, limit=2, ev_limit=None,
                               evidence_filter=q2.ev_filter())
    assert isinstance(res, StatementQueryResult)
    stmts = res.statements()
    assert len(stmts) == 2
    assert all(len(s.evidence) < res.evidence_counts[s.get_hash()]
               for s in stmts)
    js = res.json()
    assert 'results' in js
    assert len(js['results']) == len(stmts)


def test_evidence_filtering_pairs():
    ro = get_ro('primary')
    q1 = HasAgent('TP53')
    q_list = [~HasOnlySource('medscan'), HasOnlySource('reach'),
              ~HasSources(['reach', 'sparser']), HasSources(['pc', 'signor']),
              HasDatabases(), ~HasReadings(), FromMeshIds(['D001943'])]
    for q2, q3 in combinations(q_list, 2):
        query = q1 | q2 | q3
        ev_filter = q2.ev_filter() & q3.ev_filter()
        query.get_statements(ro, limit=2, ev_limit=5, evidence_filter=ev_filter)

        ev_filter = q2.ev_filter() | q3.ev_filter()
        query.get_statements(ro, limit=2, ev_limit=5, evidence_filter=ev_filter)


def test_evidence_filtering_trios():
    ro = get_ro('primary')
    q1 = HasAgent('TP53')
    q_list = [~HasOnlySource('medscan'), HasSources(['reach', 'sparser']),
              HasDatabases(), HasReadings(), FromMeshIds(['D001943'])]
    for q2, q3, q4 in combinations(q_list, 3):
        query = q1 | q2 | q3 | q4
        ev_filter = q2.ev_filter() & q3.ev_filter() & q4.ev_filter()
        query.get_statements(ro, limit=2, ev_limit=5, evidence_filter=ev_filter)

        ev_filter = q2.ev_filter() | q3.ev_filter() | q4.ev_filter()
        query.get_statements(ro, limit=2, ev_limit=5, evidence_filter=ev_filter)

    for q2, q3, q4 in permutations(q_list, 3):
        query = q1 | q2 | q3 | q4
        ev_filter = q2.ev_filter() & q3.ev_filter() | q4.ev_filter()
        query.get_statements(ro, limit=2, ev_limit=5, evidence_filter=ev_filter)


def test_evidence_count_is_none():
    ro = get_ro('primary')
    query = HasAgent('TP53') - HasOnlySource('medscan')
    res = query.get_statements(ro, limit=2)
    assert isinstance(res, StatementQueryResult)
    stmts = res.statements()
    assert len(stmts) == 2
    ev_list = stmts[0].evidence
    assert len(ev_list) > 10
    assert all(len(s.evidence) == res.evidence_counts[s.get_hash()]
               for s in stmts)
    assert res.returned_evidence == sum(res.evidence_counts.values())


def test_evidence_count_is_10():
    ro = get_ro('primary')
    query = HasAgent('TP53') - HasOnlySource('medscan')
    res = query.get_statements(ro, limit=2, ev_limit=10)
    assert isinstance(res, StatementQueryResult)
    stmts = res.statements()
    assert len(stmts) == 2
    assert all(len(s.evidence) <= 10 for s in stmts)
    assert res.returned_evidence == 20
    assert sum(res.evidence_counts.values()) > 20


def test_evidence_count_is_0():
    ro = get_ro('primary')
    query = HasAgent('TP53') - HasOnlySource('medscan')
    res = query.get_statements(ro, limit=2, ev_limit=0)
    assert isinstance(res, StatementQueryResult)
    stmts = res.statements()
    assert len(stmts) == 2
    assert all(len(s.evidence) == 0 for s in stmts)
    assert res.returned_evidence == 0, res.returned_evidence
    assert sum(res.evidence_counts.values()) > 20, \
        sum(res.evidence_counts.values())


def test_real_world_examples():
    ro = get_ro('primary')
    query = (HasAgent('MEK', namespace='FPLX', role='SUBJECT')
             & HasAgent('ERK', namespace='FPLX', role='OBJECT')
             & HasType(['Phosphorylation'])
             - HasOnlySource('medscan'))
    ev_filter = HasOnlySource('medscan').invert().ev_filter()
    res = query.get_statements(ro, limit=100, ev_limit=10,
                               evidence_filter=ev_filter)
    assert len(res.results)

    query = HasAgent('RAS') & HasAgent('RAF') & HasNumAgents((3,))
    res = query.get_statements(ro, limit=100, ev_limit=10)
    stmts = res.statements()
    assert len(stmts)


def _check_belief_sorted_result(query):
    ro = get_ro('primary')

    # Test `get_statements`
    res = query.get_statements(ro, sort_by='belief', limit=50, ev_limit=3)
    stmts = res.statements()
    assert len(res.belief_scores) == 50
    assert len(stmts) == 50
    assert all(len(s.evidence) <= 3 for s in stmts)
    assert all(0 <= s2.belief <= s1.belief <= 1
               for s1, s2 in zip(stmts[:-1], stmts[1:]))

    # Test `get_hashes`
    res = query.get_hashes(ro, sort_by='belief', limit=500)
    assert len(res.belief_scores) <= 500, len(res.belief_scores)
    assert len(res.results) == len(res.belief_scores)
    beliefs = list(res.belief_scores.values())
    assert all(b1 >= b2 for b1, b2 in zip(beliefs[:-1], beliefs[1:]))

    # Test `get_agents`
    res = query.get_agents(ro, limit=500, sort_by='belief', with_hashes=True)
    assert len(res.belief_scores) <= 500
    assert len(res.belief_scores) == len(res.results)
    beliefs = list(res.belief_scores.values())
    assert all(b1 >= b2 for b1, b2 in zip(beliefs[:-1], beliefs[1:]))

    # Test AgentJsonExpander First level.
    first_res = next(iter(res.results.values()))
    exp = AgentJsonExpander(first_res['agents'], hashes=first_res['hashes'])
    res = exp.expand(ro, sort_by='belief')
    assert len(res.belief_scores) == len(res.results)
    assert all(e['agents'] == first_res['agents']
               and set(e['hashes']) <= set(first_res['hashes'])
               for e in res.results.values())
    beliefs = list(res.belief_scores.values())
    assert all(b1 >= b2 for b1, b2 in zip(beliefs[:-1], beliefs[1:]))

    # Test AgentJsonExpander Second level.
    first_res = next(iter(res.results.values()))
    exp2 = AgentJsonExpander(first_res['agents'], stmt_type=first_res['type'],
                             hashes=first_res['hashes'])
    res = exp2.expand(ro, sort_by='belief')
    assert len(res.belief_scores) == len(res.results.values())
    assert all(e['agents'] == first_res['agents']
               and e['type'] == first_res['type']
               and e['hash'] in first_res['hashes']
               for e in res.results.values())
    beliefs = list(res.belief_scores.values())
    assert all(b1 >= b2 for b1, b2 in zip(beliefs[:-1], beliefs[1:]))


def test_belief_sorting_simple():
    query = HasAgent('MEK', namespace='NAME')
    _check_belief_sorted_result(query)


def test_belief_sorting_source_search():
    query = HasOnlySource('trips')
    _check_belief_sorted_result(query)


def test_belief_sorting_source_intersection():
    query = HasReadings() & HasDatabases()
    _check_belief_sorted_result(query)


def test_belief_sorting_intersection():
    query = HasAgent('MEK', namespace='NAME') - HasOnlySource('medscan')
    _check_belief_sorted_result(query)


def test_belief_sorting_union():
    q = HasAgent('MEK', namespace='NAME') | HasAgent('MAP2K1', namespace='NAME')
    _check_belief_sorted_result(q)
