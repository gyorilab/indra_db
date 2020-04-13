import json
import boto3
from collections import defaultdict

from indra.statements import Statement
from indra_db.reading.read_db import DatabaseStatementData, generate_reading_id
from indra_db.util import S3Path, get_db


BUCKET = S3Path(bucket='hms-uw-collaboration')
XDD_READER_VERSIONS = {'REACH': '1.3.3-61059a-biores-e9ee36',
                       'SPARSER': 'February2020-linux'}
XDD_INDRA_VERSION = '1.16.0-c439fdbc936f4eac00cafd559927d7ee06c492e8'


def main():
    db = get_db('primary')

    print("Looking for good xdd files.")
    s3 = boto3.client('s3')
    list_res = s3.list_objects_v2(**BUCKET.kw())
    files = [BUCKET.get_element_path(e['Key']) for e in list_res['Contents']]
    good_files = [file for file in files if not file.key.startswith('852')]
    print(f"Found {len(good_files)} good files.")

    print("Pairing up xdd files...")
    file_pairs = defaultdict(dict)
    for file in good_files:
        run_id, file_suffix = file.key.split('_')
        file_type = file_suffix.split('.')[0]
        file_pairs[run_id][file_type] = file
    print(f"Found {len(file_pairs)} pairs.")

    print("Processing statements...")
    run_stmts = defaultdict(lambda: defaultdict(list))
    for run_id, file_pair in file_pairs.items():
        print(f"Processing {run_id}")
        try:
            bib_obj = s3.get_object(**file_pair['bib'].kw())
        except Exception as e:
            print(f'ERROR on bib for {run_id}: {e}')
            continue
        bibs = json.loads(bib_obj['Body'].read())
        doi_lookup = {bib['_xddid']: bib['identifier'][0]['id']
                      for bib in bibs}
        dois = {doi for doi in doi_lookup.values()}

        print(f"Getting trids from database for {run_id}")
        trids = {doi.lower(): trid
                 for trid, doi in db.select_all([db.TextRef.id, db.TextRef.doi],
                                                db.TextRef.doi_in(dois))}

        try:
            stmts_obj = s3.get_object(**file_pair['stmts'].kw())
        except Exception as e:
            print(f'ERROR on stmt for {run_id}: {e}')
            continue
        stmts = json.loads(stmts_obj['Body'].read())
        for sj in stmts:
            ev = sj['evidence'][0]
            xddid = ev['text_refs']['CONTENT_ID']
            ev.pop('pmid', None)
            ev['text_refs']['DOI'] = doi_lookup[xddid]

            trid = trids[doi_lookup[xddid]]
            ev['text_refs']['TRID'] = trid
            ev['text_refs']['XDD_RUN_ID'] = run_id

            run_stmts[trid][ev['text_refs']['READER']].append(sj)

    print("Dumping text content.")
    tc_rows = {(trid, 'xdd', 'xdd', 'fulltext') for trid in run_stmts.keys()}
    tc_cols = ('text_ref_id', 'source', 'format', 'text_type')
    db.copy_lazy('text_content', tc_rows, tc_cols, commit=False)

    print("Looking up new tcids.")
    tcids = db.select_all([db.TextContent.text_ref_id, db.TextContent.id],
                          db.TextContent.text_ref_id.in_(run_stmts.keys()),
                          db.TextContent.source == 'xdd')
    tcid_lookup = {trid: tcid for trid, tcid in tcids}

    print("Compiling reading and statement rows.")
    r_rows = set()
    r_cols = ('id', 'text_content_id', 'reader', 'reader_version', 'format',
              'batch_id')
    s_rows = set()
    for trid, trid_set in run_stmts.items():
        for reader, stmt_list in trid_set.items():
            tcid = tcid_lookup[trid]
            reader_version = XDD_READER_VERSIONS[reader.upper()]
            reading_id = generate_reading_id(tcid, reader, reader_version)
            r_rows.add((reading_id, tcid, reader.upper(), reader_version,
                        'xdd', db.make_copy_batch_id()))
            for sj in stmt_list:
                stmt = Statement._from_json(sj)
                sd = DatabaseStatementData(stmt, reading_id)
                s_rows.add(sd.make_tuple(db.make_copy_batch_id()))

    print("Dumping reading.")
    db.copy_lazy('reading', r_rows, r_cols, commit=False)

    print("Dumping raw statements.")
    db.copy_lazy('raw_statements', s_rows, DatabaseStatementData.get_cols())


if __name__ == '__main__':
    main()
