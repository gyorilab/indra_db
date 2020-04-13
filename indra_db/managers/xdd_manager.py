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
        self.statements = None

    def load_groups(self, db):
        logger.info("Finding groups that have not been handled yet.")
        s3 = boto3.client('s3')
        groups = _list_s3_prefixes(s3, self.bucket)
        previous_groups = {s for s, in db.select_all(db.XddUpdates.day_str)}

        self.groups = [group for group in groups
                       if group.key not in previous_groups]
        return

    def load_statements(self, db):
        s3 = boto3.client('s3')
        self.statements = defaultdict(lambda: defaultdict(list))
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

                    self.statements[trid][ev['text_refs']['READER']].append(sj)
        return

    def dump_statements(self, db):
        tc_rows = {(trid, 'xdd', 'xdd', 'fulltext')
                   for trid in self.statements.keys()}
        tc_cols = ('text_ref_id', 'source', 'format', 'text_type')
        db.copy_lazy('text_content', tc_rows, tc_cols, commit=False)

        # Look up tcids for newly entered content.
        tcids = db.select_all(
            [db.TextContent.text_ref_id, db.TextContent.id],
            db.TextContent.text_ref_id.in_(self.statements.keys()),
            db.TextContent.source == 'xdd'
        )
        tcid_lookup = {trid: tcid for trid, tcid in tcids}

        # Compile reading and statements into rows.
        r_rows = set()
        r_cols = ('id', 'text_content_id', 'reader', 'reader_version',
                  'format', 'batch_id')
        s_rows = set()
        for trid, trid_set in self.statements.items():
            for reader, stmt_list in trid_set.items():
                tcid = tcid_lookup[trid]
                reader_version = self.reader_versions[reader.upper()]
                reading_id = generate_reading_id(tcid, reader, reader_version)
                r_rows.add((reading_id, tcid, reader.upper(), reader_version,
                            'xdd', db.make_copy_batch_id()))
                for sj in stmt_list:
                    stmt = Statement._from_json(sj)
                    sd = DatabaseStatementData(
                        stmt,
                        reading_id,
                        indra_version=self.indra_version
                    )
                    s_rows.add(sd.make_tuple(db.make_copy_batch_id()))

        print("Dumping reading.")
        db.copy_lazy('reading', r_rows, r_cols, commit=False)

        print("Dumping raw statements.")
        db.copy_lazy('raw_statements', s_rows,
                     DatabaseStatementData.get_cols())

        db.insert(db.XddUpdates, [{'reader_versions': self.reader_versions,
                                   'indra_version': self.indra_version,
                                   'day_str': group.key}
                                  for group in self.groups])
        return

    def run(self, db):
        self.load_groups(db)
        self.load_statements(db)
        self.dump_statements(db)


def _list_s3(s3, s3_path):
    list_res = s3.list_objects_v2(**s3_path.kw())
    return [s3_path.get_element_path(e['Key']) for e in list_res['Contents']]


def _list_s3_prefixes(s3, s3_path):
    list_res = s3.list_objects_v2(Delimiter='/', **s3_path.kw())
    return [s3_path.get_element_path(e['Prefix'])
            for e in list_res['CommonPrefixes']]


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

    m = XddManager()
    m.run(db)


if __name__ == '__main__':
    main()
