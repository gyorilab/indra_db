import json
import boto3
import random

from indra_db.tests.util import get_temp_db
from indra_db.managers.xdd_manager import XddManager


def test_dump():
    db = get_temp_db(clear=True)
    m = XddManager()

    # Enter "old" DOIs
    s3 = boto3.client('s3')
    res = s3.list_objects_v2(**m.bucket.kw())
    dois = set()
    for ref in res['Contents']:
        key = ref['Key']
        if 'bib' not in key:
            continue
        try:
            obj = s3.get_object(Key=key, **m.bucket.kw())
        except Exception:
            print('ack')
            continue
        bibs = json.loads(obj['Body'].read())
        dois |= {bib['identifier'][0]['id'] for bib in bibs
                 if 'identifier' in bib}
    sample_dois = random.sample(dois, len(dois)//2)
    new_trs = [db.TextRef.new(doi=doi) for doi in sample_dois]
    print(f"Adding {len(new_trs)} 'old' text refs.")
    db.session.add_all(new_trs)
    db.session.commit()

    # Run the update.
    m.run(db)

    # Check the result.
    assert db.select_all(db.TextRef)
    assert db.select_all(db.TextContent)
    assert db.select_all(db.Reading)
    assert db.select_all(db.RawStatements)
    assert db.select_all(db.RawAgents)
