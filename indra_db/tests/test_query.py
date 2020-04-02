from datetime import datetime, timedelta
from itertools import combinations, permutations

from indra_db.util import get_db
from indra_db.client.query import *


db = get_db('primary')


def dq(q):
    print('---------------------------')
    print(q)
    print('---------------------------')
    print(q._get_mk_hashes_query(db))
    print('---------------------------')
    start = datetime.now()
    # res = q.get_statements(db, limit=10, ev_limit=2)
    res = q.get_hashes(db)
    print(f'Number of hashes: {len(res.results)}')
    end = datetime.now()
    print(f'Duration: {end - start}')
    print('\n================================================\n')
    return res.results


def test_query_set_behavior():

    queries = [
        AgentQuery('TP53', role='SUBJECT'),
        AgentQuery('ERK', namespace='FPLX', role='OBJECT'),
        MeshQuery('D056910'),
        HashQuery([12080991702025131, 12479954161276307, 24255960759225919]),
        HashQuery([25663052342435447]),
        OnlySourceQuery('reach'),
        HasReadingsQuery(),
        HasDatabaseQuery(),
        HasSourcesQuery(['sparser', 'reach']),
        HasSourcesQuery(['medscan']),
        TypeQuery(['Phosphorylation', 'Activation']),
        TypeQuery(['RegulateActivity'], include_subclasses=True),
        TypeQuery(['Complex'])
    ]

    results = []
    for q in queries:
        results.append(dq(q))

    for (r1, q1), (r2, q2) in permutations(zip(results, queries), 2):
        r1n2 = dq(q1 & q2)
        assert r1n2 == r1 & r2
        r1r2 = dq(q1 | q2)
        assert r1r2 == r1 | r2

    for (r1, q1), (r2, q2), (r3, q3) in combinations(zip(results, queries), 3):
        r1n2n3 = dq(q1 & q2 & q3)
        assert r1n2n3 == r1 & r2 & r3

        r1r2r3 = dq(q1 | q2 | q3)
        assert r1r2r3 == r1 | r2 | r3

    for (r1, q1), (r2, q2), (r3, q3) in permutations(queries, 3):
        r1n2r3 = dq(q1 & q2 | q3)
        assert r1n2r3 == r1 & r2 | r3
