import json
import click
from collections import defaultdict
from indra_db import get_db, get_ro


@click.group()
def counts():
    """Generate counts from the databases."""


@counts.command()
def run():
    """Run out all the counts. This will take a long time."""
    print("Running counts: this may take a while.")
    get_counts()


def get_counts():
    from humanize import intword
    from tabulate import tabulate

    ro = get_ro('primary')
    db = get_db('primary')

    # Do source and text-type counts.
    res = db.session.execute("""
    SELECT source, text_type, COUNT(*) FROM text_content GROUP BY source, text_type;
    """)
    print("Source-type counts:")
    print(tabulate([(src, tt, intword(n)) for src, tt, n in res],
                   headers=["Source", "Text Type", "Count"]))

    # Do reader counts.
    res = db.session.execute("""
    SELECT reader, reader_version, source, text_type, COUNT(*) 
    FROM text_content LEFT JOIN reading ON text_content_id = text_content.id
    GROUP BY reader, reader_version, source, text_type;
    """)
    print("reader rv src type counts:")
    print(tabulate([t[:-1] + (intword(t[-1]),) for t in res],
                   headers=["Reader", "Reader Version", "Source", "Text Type", "Count"]))

    # Get the list of distinct HGNC IDs (and dump them as JSON).
    resp = ro.session.execute("""
    SELECT DISTINCT db_id FROM readonly.other_meta WHERE db_name = 'HGNC';
    """)
    ids = {db_id for db_id, in resp}
    print("Distinct HGNC Ids:", intword(len(ids)))
    with open('unique_hgnc_ids.json', 'w') as f:
        json.dump(list(ids), f)

    # Count the number of distinct groundings.
    res = ro.session.execute("""
    SELECT DISTINCT db_id, db_name
    FROM readonly.other_meta
    """)
    ids = {tuple(t) for t in res}
    print("Number of all distinct groundings:", intword(len(ids)))

    # Count the number of raw statements.
    raw_cnt, = db.session.execute("""
    SELECT count(*) FROM raw_statements
    """)
    print("Raw stmt count:", intword(raw_cnt))

    # Count the number of preassembled statements.
    pa_cnt, = db.session.execute("""
    SELECT count(*) FROM pa_statements
    """)
    print("PA stmt count:", intword(pa_cnt))

    # Count the number of links between raw and preassembled statements.
    raw_used_in_pa_cnt, = db.session.execute("""
    SELECT count(*) FROM raw_unique_links
    """)
    print("Raw statements used in PA:", intword(raw_used_in_pa_cnt))

    # Get the distinct grounded hashes.
    res = db.session.execute("""
    SELECT DISTINCT stmt_mk_hash FROM pa_agents WHERE db_name NOT IN ('TEXT', 'TEXT_NORM', 'NAME')
    """)
    grounded_hashes = {h for h, in res}
    print("Number of grounded hashes:", intword(len(grounded_hashes)))

    # Get number of hashes with agent counts.
    res = ro.session.execute("""
    SELECT mk_hash, agent_count, array_agg(ag_num), array_agg(db_name)
    FROM readonly.other_meta
    WHERE NOT is_complex_dup
    GROUP BY mk_hash, agent_count
    """)
    stmt_vals = [tuple(t) for t in res]
    print("Number of hashes, should be close to grounded pa count:", intword(len(stmt_vals)))

    # Count up the number of statements with all agents grounded.
    cnt = 0
    for h, n, ag_ids, ag_grnds in stmt_vals:
        ag_dict = defaultdict(set)
        for ag_id, ag_grnd in zip(ag_ids, ag_grnds):
            ag_dict[ag_id].add(ag_grnd)
        if len(ag_dict) == n:
            cnt += 1
    print("Number of pa statemenets in ro with all agents grounded:", intword(cnt))

