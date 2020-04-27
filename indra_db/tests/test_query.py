import json
import random
from itertools import combinations, permutations, product

from indra.statements import Agent, get_statement_by_name
from indra_db.client.readonly.query import QueryResult
from indra_db.schemas.readonly_schema import ro_type_map, ro_role_map
from indra_db.util import extract_agent_data, get_ro, get_db
from indra_db.client.readonly.query import *

from indra_db.tests.util import get_temp_db


def make_agent_from_ref(ref):
    db_refs = ref.copy()
    name = db_refs.pop('NAME')
    return Agent(name, db_refs=db_refs)


def build_test_set():
    agents = [{'NAME': 'ERK', 'FPLX': 'ERK', 'TEXT': 'MAPK'},
              {'NAME': 'TP53', 'HGNC': '11998'},
              {'NAME': 'MEK', 'FPLX': 'MEK'},
              {'NAME': 'Vemurafenib', 'CHEBI': 'CHEBI:63637'}]
    stypes = ['Phosphorylation', 'Activation', 'Inhibition', 'Complex']
    sources = [('medscan', 'rd'), ('reach', 'rd'), ('pc11', 'db'),
               ('signor', 'db')]
    mesh_ids = ['D000225', 'D002352', 'D015536']

    mesh_combos = []
    for num_mesh in range(0, 3):
        if num_mesh == 1:
            mesh_groups = [[mid] for mid in mesh_ids]
        else:
            mesh_groups = combinations(mesh_ids, num_mesh)

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
                      'ev_count', 'activity', 'is_active', 'agent_count')

    text_meta_rows = []
    text_meta_cols = ('mk_hash', 'ag_num', 'db_id', 'role_num', 'type_num',
                      'ev_count', 'activity', 'is_active', 'agent_count')

    other_meta_rows = []
    other_meta_cols = ('mk_hash', 'ag_num', 'db_name', 'db_id', 'role_num',
                       'type_num', 'ev_count', 'activity', 'is_active',
                       'agent_count')

    source_meta_rows = []
    source_meta_cols = ('mk_hash', 'reach', 'medscan', 'pc11', 'signor',
                        'ev_count', 'type_num', 'activity', 'is_active',
                        'agent_count', 'num_srcs', 'src_json', 'only_src',
                        'has_rd', 'has_db')

    mesh_meta_rows = []
    mesh_meta_cols = ('mk_hash', 'ev_count', 'mesh_num', 'type_num',
                      'activity', 'is_active', 'agent_count')
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
        src_row = (stmt.get_hash(),)
        for src_name in ['reach', 'medscan', 'pc11', 'signor']:
            src_row += (source_dict['sources'].get(src_name),)
        src_row += (ev_count, ro_type_map.get_int(stype), activity, is_active,
                    len(refs), len(source_dict['sources']),
                    json.dumps(source_dict['sources']), source_dict['only_src'],
                    source_dict['has_rd'], source_dict['has_db'])
        source_meta_rows.append(src_row)

        # Add mesh rows
        for mesh_id in source_dict['mesh_ids']:
            mesh_meta_rows.append((stmt.get_hash(), ev_count, int(mesh_id[1:]),
                                   ro_type_map.get_int(stype), activity,
                                   is_active, len(refs)))

        # Generate agent rows.
        ref_rows, _, _ = extract_agent_data(stmt, stmt.get_hash())
        for row in ref_rows:
            row = row[:4] + (ro_role_map.get_int(row[4]),
                             ro_type_map.get_int(stype),  ev_count, activity,
                             is_active, len(refs))
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
    db.SourceMeta.load_cols(db.engine, src_meta_cols)
    for tbl in [db.SourceMeta, db.MeshMeta, db.NameMeta, db.TextMeta,
                db.OtherMeta]:
        tbl.__table__.create(db.engine)
    db.copy('readonly.source_meta', source_meta_rows, source_meta_cols)
    db.copy('readonly.mesh_meta', mesh_meta_rows, mesh_meta_cols)
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


def test_query_set_behavior():
    db = build_test_set()
    all_hashes = {h for h, in db.select_all(db.NameMeta.mk_hash)}
    print(f"There are {len(all_hashes)} distinct hashes in the database.")
    lookup_hashes = random.sample(all_hashes, 5)

    c = Counter()

    def dq(query):
        res = query.get_hashes(db)
        return res.results

    queries = [
        HasAgent('TP53', role='SUBJECT'),
        HasAgent('ERK', namespace='FPLX', role='OBJECT'),
        FromMeshId('D015536'),
        HasHash(lookup_hashes[:-3]),
        HasHash(lookup_hashes[-1:]),
        HasHash(lookup_hashes[-4:-1]),
        HasOnlySource('pc11'),
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
        HasNumEvidence([10, 11]),
        HasNumEvidence([1])
    ]

    failures = []
    results = []
    unfound = []

    def try_query(q, compair=None, md=None):
        result = None
        try:
            result = dq(q)
            if compair is not None:
                assert result == compair, 'Result mismatch.'
            if not q.empty and not result:
                unfound.append(q)
            results.append((result, q))
            c.up(True)
        except Exception as e:
            failures.append({'query': q, 'error': e, 'result': result,
                             'compair': compair, 'md': md})
            results.append((result, q))
            c.up(False)
            return

        negative_result = None
        nq = None
        try:
            nq = ~q
            if nq is None:
                assert False, "Inverted query is None."
            negative_result = dq(nq)
            assert negative_result == (all_hashes - result), \
                'Negative result mismatch.'

            if not nq.empty and not negative_result:
                unfound.append(nq)
            results.append((negative_result, nq))
            c.up(True)
        except Exception as e:
            if md is not None:
                neg_md = 'not (' + md + ')'
            else:
                neg_md = 'not (' + str(q) + ')'
            failures.append({'query': nq, 'error': e, 'result': negative_result,
                             'compair': all_hashes - result, 'md': neg_md})
            results.append((negative_result, nq))
            c.up(False)
            return

        return

    c.mark("Testing individual queries...")
    for q in queries:
        try_query(q)
    original_results = [res for res in results if res[1] is not None]

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
        try_query(q1 & q2 | q3, r1 & r2 | r3, md=f'({q1} and {q2}) or {q3}')
        try_query(q1 | q2 & q3, r1 | r2 & r3, md=f'({q1} or {q2}) and {q3}')

    c.mark('Done!')
    print(f"Ran {len(results)} checks...")

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

    assert not failures, f"{len(failures)}/{len(results)} checks failed."

    return results, failures


def test_get_interactions():
    ro = get_db('primary')
    query = HasAgent('TP53') - HasOnlySource('medscan')
    res = query.get_interactions(ro, limit=10, detail_level='relations')
    assert isinstance(res, QueryResult)
    assert len(res.results) == 10


def test_evidence_filtering():
    ro = get_db('primary')
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


def test_evidence_count_is_none():
    ro = get_db('primary')
    query = HasAgent('TP53') - HasOnlySource('medscan')
    res = query.get_statements(ro, limit=2)
    assert isinstance(res, StatementQueryResult)
    stmts = res.statements()
    assert len(stmts) == 2
    ev_list = stmts[0].evidence
    assert len(ev_list) > 10
    assert all(len(s.evidence) == res.evidence_totals[s.get_hash()]
               for s in stmts)
    assert res.returned_evidence == sum(res.evidence_totals.values())


def test_evidence_count_is_10():
    ro = get_db('primary')
    query = HasAgent('TP53') - HasOnlySource('medscan')
    res = query.get_statements(ro, limit=2, ev_limit=10)
    assert isinstance(res, StatementQueryResult)
    stmts = res.statements()
    assert len(stmts) == 2
    assert all(len(s.evidence) <= 10 for s in stmts)
    assert res.returned_evidence == 20
    assert sum(res.evidence_totals.values()) > 20


def test_evidence_count_is_0():
    ro = get_db('primary')
    query = HasAgent('TP53') - HasOnlySource('medscan')
    res = query.get_statements(ro, limit=2, ev_limit=0)
    assert isinstance(res, StatementQueryResult)
    stmts = res.statements()
    assert len(stmts) == 2
    assert all(len(s.evidence) == 0 for s in stmts)
    assert res.returned_evidence == 0
    assert sum(res.evidence_totals.values()) > 20
