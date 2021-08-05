import json
import boto3
import click
import logging
from collections import defaultdict

from indra.statements import Statement
from indra_db.util import S3Path, get_db, insert_raw_agents

logger = logging.getLogger(__name__)


class XddManager:
    bucket = S3Path(bucket='hms-uw-collaboration')
    reader_versions = {'REACH': '1.3.3-61059a-biores-e9ee36',
                       'SPARSER': 'February2020-linux'}
    indra_version = '1.16.0-c439fdbc936f4eac00cafd559927d7ee06c492e8'

    def __init__(self):
        self.groups = None
        self.groups_done = None
        self.statements = None
        self.text_content = None

    def load_groups(self, db):
        logger.info("Finding groups that have not been handled yet.")
        s3 = boto3.client('s3')
        groups = self.bucket.list_prefixes(s3)
        previous_groups = {s for s, in db.select_all(db.XddUpdates.day_str)}

        self.groups = [group for group in groups
                       if group.key[:-1] not in previous_groups]
        return

    def load_statements(self, db):
        logger.info("Loading statements.")
        s3 = boto3.client('s3')
        self.statements = defaultdict(lambda: defaultdict(list))
        self.text_content = {}
        self.groups_done = {}
        for group in self.groups:
            logger.info(f"Processing {group.key}")
            file_pair_dict, got_all = _get_file_pairs_from_group(s3, group)
            self.groups_done[group.key[:-1]] = got_all
            for (run_id, id_src), (bibs, stmts) in file_pair_dict.items():
                logger.info(f"Loading {run_id}")
                doi_lookup = {bib['_xddid']: bib['identifier'][0]['id'].upper()
                              for bib in bibs if 'identifier' in bib}
                pub_lookup = {bib['_xddid']: bib['publisher'] for bib in bibs}
                dois = {doi for doi in doi_lookup.values()}
                trids = _get_trids_from_dois(db, dois)

                for sj in stmts:
                    ev = sj['evidence'][0]
                    xddid = ev['text_refs']['CONTENT_ID']
                    ev.pop('pmid', None)
                    if xddid not in doi_lookup:
                        logger.warning("Skipping statement because bib "
                                       "lacked a DOI.")
                        continue
                    ev['text_refs']['DOI'] = doi_lookup[xddid]

                    trid = trids[doi_lookup[xddid]]
                    ev['text_refs']['TRID'] = trid
                    ev['text_refs']['XDD_RUN_ID'] = run_id
                    ev['text_refs']['XDD_GROUP_ID'] = group.key

                    self.statements[trid][ev['text_refs']['READER']].append(sj)
                    if trid not in self.text_content:
                        if id_src:
                            src = f'xdd-{id_src}'
                        else:
                            src = 'xdd'
                        self.text_content[trid] = \
                            (trid, src, 'xdd', 'fulltext',
                             pub_lookup[xddid] == 'bioRxiv')
        return

    def dump_statements(self, db):
        from indra_db.reading.read_db import DatabaseStatementData, \
            generate_reading_id
        tc_rows = set(self.text_content.values())
        tc_cols = ('text_ref_id', 'source', 'format', 'text_type', 'preprint')
        logger.info(f"Dumping {len(tc_rows)} text content.")
        db.copy_lazy('text_content', tc_rows, tc_cols)

        # Look up tcids for newly entered content.
        tcids = db.select_all(
            [db.TextContent.text_ref_id, db.TextContent.id],
            db.TextContent.text_ref_id.in_(self.statements.keys()),
            db.TextContent.format == 'xdd'
        )
        tcid_lookup = {trid: tcid for trid, tcid in tcids}

        # Compile reading and statements into rows.
        r_rows = set()
        r_cols = ('id', 'text_content_id', 'reader', 'reader_version',
                  'format', 'batch_id')
        s_rows = set()
        rd_batch_id = db.make_copy_batch_id()
        stmt_batch_id = db.make_copy_batch_id()
        stmts = []
        for trid, trid_set in self.statements.items():
            for reader, stmt_list in trid_set.items():
                tcid = tcid_lookup[trid]
                reader_version = self.reader_versions[reader.upper()]
                reading_id = generate_reading_id(tcid, reader, reader_version)
                r_rows.add((reading_id, tcid, reader.upper(), reader_version,
                            'xdd', rd_batch_id))
                for sj in stmt_list:
                    stmt = Statement._from_json(sj)
                    stmts.append(stmt)
                    sd = DatabaseStatementData(
                        stmt,
                        reading_id,
                        indra_version=self.indra_version
                    )
                    s_rows.add(sd.make_tuple(stmt_batch_id))

        logger.info(f"Dumping {len(r_rows)} readings.")
        db.copy_lazy('reading', r_rows, r_cols, commit=False,
                     constraint='reading-uniqueness')

        logger.info(f"Dumping {len(s_rows)} raw statements.")
        skipped = db.copy_report_lazy('raw_statements', s_rows,
                                      DatabaseStatementData.get_cols(),
                                      commit=False)
        skipped_uuids = {t[DatabaseStatementData.get_cols().index('uuid')]
                         for t in skipped}
        new_stmts = [s for s in stmts if s.uuid not in skipped_uuids]
        if len(new_stmts):
            insert_raw_agents(db, stmt_batch_id, new_stmts, verbose=False,
                              commit=False)
        return

    def run(self, db):
        self.load_groups(db)
        self.load_statements(db)
        self.dump_statements(db)
        update_rows = [(json.dumps(self.reader_versions),
                        self.indra_version, grp_key)
                       for grp_key, is_done in self.groups_done.items()
                       if is_done]
        db.copy('xdd_updates', update_rows,
                ('reader_versions', 'indra_version', 'day_str'))
        return


class XDDFileError(Exception):
    pass


def _get_file_pairs_from_group(s3, group: S3Path):
    files = group.list_objects(s3)
    file_pairs = defaultdict(dict)
    got_all = True
    for file_path in files:
        # Get information from the filename, including the cases with and
        # without the id_src label.
        parts = file_path.key.split('_')
        if len(parts) == 2:
            run_id, file_suffix = parts
            id_src = None
        elif len(parts) == 3:
            run_id, id_src, file_suffix = parts
        else:
            raise XDDFileError(f"XDD file does not match known standards: "
                               f"{file_path.key}")
        file_type = file_suffix.split('.')[0]

        # Try getting the file
        try:
            file_obj = s3.get_object(**file_path.kw())
            file_json = json.loads(file_obj['Body'].read())
            file_pairs[(run_id, id_src)][file_type] = file_json
        except Exception as e:
            logger.error(f"Failed to load {file_path}")
            logger.exception(e)
            if run_id in file_pairs:
                del file_pairs[run_id]
            got_all = False

    # Create a dict of tuples from the pairs of files.
    ret = {}
    for batch_id, files in file_pairs.items():
        if len(files) != 2 or 'bib' not in files or 'stmts' not in files:
            logger.warning(f"Run {batch_id} does not have both 'bib' and "
                           f"'stmts' in files: {files.keys()}. Skipping.")
            got_all = False
            continue
        ret[batch_id] = (files['bib'], files['stmts'])
    return ret, got_all


def _get_trids_from_dois(db, dois):
    # Get current relevant text refs (if any)
    tr_list = db.select_all(db.TextRef, db.TextRef.doi_in(dois))

    # Add new dois (if any)
    new_dois = set(dois) - {tr.doi.upper() for tr in tr_list}
    if new_dois:
        new_trs = [db.TextRef.new(doi=doi) for doi in new_dois]
        logger.info(f"Adding {len(new_trs)} new text refs.")
        db.session.add_all(new_trs)
        db.session.commit()
        tr_list += new_trs

    # Make the full mapping table.
    return {tr.doi.upper(): tr.id for tr in tr_list}


@click.group()
def xdd():
    """Manage xDD runs."""


@xdd.command()
def run():
    """Process the latest outputs from xDD."""
    db = get_db('primary')
    XddManager().run(db)
