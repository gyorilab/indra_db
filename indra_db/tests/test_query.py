import json
import random
from datetime import datetime
from itertools import combinations, permutations, product

from indra.statements import Agent, get_statement_by_name
from indra_db.schemas.readonly_schema import ro_type_map, ro_role_map
from indra_db.util import get_db, extract_agent_data
from indra_db.client.query import *

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


def test_query_set_behavior():
    db = build_test_set()

    def dq(q):
        print('---------------------------')
        print(q)
        print('---------------------------')
        print(q._get_hash_query(db))
        print('---------------------------')
        start = datetime.now()
        # res = q.get_statements(db, limit=10, ev_limit=2)
        res = q.get_hashes(db)
        print(f'Number of hashes: {len(res.results)}')
        end = datetime.now()
        print(f'Duration: {end - start}')
        print('\n================================================\n')
        return res.results

    queries = [
        HasAgent('TP53', role='SUBJECT'),
        HasAgent('ERK', namespace='FPLX', role='OBJECT'),
        FromMeshId('D056910'),
        InHashList([12080991702025131, 12479954161276307, 24255960759225919]),
        InHashList([25663052342435447]),
        HasOnlySource('reach'),
        HasReadings(),
        HasDatabases(),
        HasSources(['sparser', 'reach']),
        HasSources(['medscan']),
        HasAnyType(['Phosphorylation', 'Activation']),
        HasAnyType(['RegulateActivity'], include_subclasses=True),
        HasAnyType(['Complex'])
    ]

    failures = []
    results = []

    def try_query(q, compair=None):
        result = None
        try:
            result = dq(q)
            if compair is not None:
                assert result == compair
        except Exception as e:
            failures.append({'query': q.to_json(), 'error': e, 'result': result,
                             'compair': compair})
            return q, None
        results.append((result, q))
        return q, result

    for q in queries:
        try_query(q)
    original_results = results[:]

    for (r1, q1), (r2, q2) in permutations(original_results, 2):
        try_query(q1 & q2, r1 & r2)
        try_query(q1 | q2, r1 | r2)

    for (r1, q1), (r2, q2), (r3, q3) in combinations(original_results, 3):
        try_query(q1 & q2 & q3, r1 & r2 & r3)
        try_query(q1 | q2 | q3, r1 | r2 | r3)

    for (r1, q1), (r2, q2), (r3, q3) in permutations(original_results, 3):
        try_query(q1 & q2 | q3, r1 & r2 | r3)

    assert not failures, f"{len(failures)}/{len(results)} checks failed."

    return results, failures
