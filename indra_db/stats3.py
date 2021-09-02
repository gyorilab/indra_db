import json
from collections import defaultdict
from indra_db import get_db, get_ro
ro = get_ro('primary')
db = get_db('primary')

res = db.session.execute("""
SELECT source, text_type, COUNT(*) FROM text_content GROUP BY source, text_type;
""")
print("Source-type counts:\n", '\n'.join(str(t) for t in res))
res = db.session.execute("""
SELECT reader, reader_version, source, text_type, COUNT(*) 
FROM text_content LEFT JOIN reading ON text_content_id = text_content.id
GROUP BY reader, reader_version, source, text_type;
""")
print("reader rv src type counts:\n", '\n'.join(str(t) for t in res))
resp = ro.session.execute("""
SELECT DISTINCT db_id FROM readonly.other_meta WHERE db_name = 'HGNC';
""")
ids = {db_id for db_id, in resp}
print("Distinct HGNC Ids:", len(ids))
with open('unique_hgnc_ids.json', 'w') as f:
    json.dump(list(ids), f)

res = ro.session.execute("""
SELECT DISTINCT db_id, db_name
FROM readonly.other_meta
""")
ids = {tuple(t) for t in res}
print("Number of all distinct groundings:", len(ids))

raw_cnt = db.session.execute("""
SELECT count(*) FROM raw_statements
""")
print("Raw stmt count:", list(raw_cnt))

pa_cnt = db.session.execute("""
SELECT count(*) FROM pa_statements
""")
print("PA stmt count:", list(pa_cnt))

raw_used_in_pa_cnt = db.session.execute("""
SELECT count(*) FROM raw_unique_links
""")
print("Raw statements used in PA:", list(raw_used_in_pa_cnt))

res = db.session.execute("""
SELECT DISTINCT stmt_mk_hash FROM pa_agents WHERE db_name NOT IN ('TEXT', 'TEXT_NORM', 'NAME')
""")
grounded_hashes = {h for h, in res}
print("Number of grounded hashes:", len(grounded_hashes))

res = ro.session.execute("""
SELECT mk_hash, agent_count, array_agg(ag_num), array_agg(db_name)
FROM readonly.other_meta
WHERE NOT is_complex_dup
GROUP BY mk_hash, agent_count
""")
stmt_vals = [tuple(t) for t in res]
print("Number of hashes, should be close to grounded pa count:", len(stmt_vals))

cnt = 0
for h, n, ag_ids, ag_grnds in stmt_vals:
    ag_dict = defaultdict(set)
    for ag_id, ag_grnd in zip(ag_ids, ag_grnds):
        ag_dict[ag_id].add(ag_grnd)
    if len(ag_dict) == n:
        cnt += 1
print("Number of pa statemenets in ro with all agents grounded:", cnt)

