import json
import boto3
import logging
from collections import defaultdict

from indra.statements import Statement
from indra_db.reading.read_db import DatabaseStatementData, generate_reading_id
from indra_db.util import S3Path, get_db

logger = logging.getLogger(__name__)


class XddManager:
    bucket = S3Path(bucket='hms-uw-collaboration')
    reader_versions = {'REACH': '1.3.3-61059a-biores-e9ee36',
                       'SPARSER': 'February2020-linux'}
    indra_version = '1.16.0-c439fdbc936f4eac00cafd559927d7ee06c492e8'

    def __init__(self):
        self.groups = None

    def get_groups(self, db):
        if self.groups is not None:
            return self.groups

        logger.info("Finding groups that have not been handled yet.")
        s3 = boto3.client('s3')
        groups = _list_s3(s3, self.bucket, delimiter='/')
        previous_groups = \
            {s for s, in db.XddUpdates.select_all(db.XddUpdates.day_str)}

        self.groups = [group for group in groups
                       if group.key not in previous_groups]
        return self.groups

    def get_statements(self, db):
        s3 = boto3.client('s3')

        run_stmts = defaultdict(lambda: defaultdict(list))
        for group in self.groups:
            file_pair_dict = _get_file_pairs_from_group(s3, group)
            for run_id, (bibs, stmts) in file_pair_dict.items():
                doi_lookup = {bib['_xddid']: bib['identifier'][0]['id']
                              for bib in bibs}
                dois = {doi for doi in doi_lookup.values()}
                trids = _get_trids_from_dois(db, dois)

                for sj in stmts:
                    ev = sj['evidence'][0]
                    xddid = ev['text_refs']['CONTENT_ID']
                    ev.pop('pmid', None)
                    ev['text_refs']['DOI'] = doi_lookup[xddid]

                    trid = trids[doi_lookup[xddid]]
                    ev['text_refs']['TRID'] = trid
                    ev['text_refs']['XDD_RUN_ID'] = run_id
                    ev['text_refs']['XDD_GROUP_ID'] = group.key

                    run_stmts[trid][ev['text_refs']['READER']].append(sj)

        return run_stmts


def _list_s3(s3, s3_path, **kwargs):
    kwargs.update(s3_path.kw())
    list_res = s3.list_objects_v2(**kwargs)
    return [s3_path.get_element_path(e['Key']) for e in list_res['Contents']]


def _get_file_pairs_from_group(s3, group):
    files = _list_s3(s3, group)
    file_pairs = defaultdict(dict)
    for file_path in files:
        run_id, file_suffix = file_path.key.split('_')
        file_type = file_suffix.split('.')[0]
        try:
            file_obj = s3.get_object(**file_path.kw())
            file_json = json.loads(file_obj['Body'].read())
            file_pairs[run_id][file_type] = file_json
        except Exception as e:
            logger.error(f"Failed to load {file_path}")
            logger.exception(e)
            if run_id in file_pairs:
                del file_pairs[run_id]
    return {k: (v['bib'], v['stmts']) for k, v in file_pairs.items()}


def _get_trids_from_dois(db, dois):
    trid_doi_res = db.select_all([db.TextRef.id, db.TextRef.doi],
                                 db.TextRef.doi_in(dois))
    return {doi.lower(): trid for trid, doi in trid_doi_res}


def main():
    db = get_db('primary')

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
