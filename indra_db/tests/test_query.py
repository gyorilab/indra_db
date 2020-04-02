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

    failures = []

    def try_query(q, compair=None):
        try:
            result = dq(q)
            if compair is not None:
                assert result == compair
        except Exception as e:
            failures.append({'query': q.to_json(), 'error': e, 'result': result,
                             'compair': compair})
            return q, None
        return q, result

    results = []
    for q in queries:
        q, res = try_query(q)
        if res:
            results.append((q, res))

    for (r1, q1), (r2, q2) in permutations(results, 2):
        try_query(q1 & q2, r1 & r2)
        try_query(q1 | q2, r1 | r2)

    for (r1, q1), (r2, q2), (r3, q3) in combinations(results, 3):
        try_query(q1 & q2 & q3, r1 & r2 & r3)
        try_query(q1 | q2 | q3, r1 | r2 | r3)

    for (r1, q1), (r2, q2), (r3, q3) in permutations(results, 3):
        try_query(q1 & q2 | q3, r1 & r2 | r3)

    return results, failures
