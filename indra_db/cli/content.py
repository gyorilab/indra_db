import ftplib
import gzip
import os
import random
import re
import csv
import tarfile
from itertools import islice
import click
import zlib
import logging
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict

from io import BytesIO
from ftplib import FTP
from functools import wraps
from datetime import datetime, timedelta
from os import path, remove, rename, listdir
from typing import Tuple

from tqdm import tqdm

from indra.literature.elsevier_client import has_full_text
from indra.util import zip_string, batch_iter
from indra.util import UnicodeXMLTreeBuilder as UTB

from indra_db.databases import texttypes, formats
from indra_db.databases import sql_expressions as sql_exp
from indra_db.util.data_gatherer import DataGatherer, DGContext

from .util import format_date

try:
    from psycopg2 import DatabaseError
except ImportError:
    class DatabaseError(object):
        "Using this in a try-except will catch nothing. (That's the point.)"
        pass

logger = logging.getLogger(__name__)

ftp_blocksize = 33554432  # Chunk size recommended by NCBI
THIS_DIR = path.dirname(path.abspath(__file__))


gatherer = DataGatherer('content', ['refs', 'content'])


class UploadError(Exception):
    pass


class _NihFtpClient(object):
    """High level access to the NIH FTP repositories.

    Parameters
    ----------
    my_path : str
        The path to the subdirectory around which this client operates.
    ftp_url : str
        The url to the ftp site. May be a local directory (see `local`). By
        default this is `'ftp.ncbi.nlm.nih.gov'`.
    local : bool
        These methods may be run on a local directory (intended for testing).
        (default is `False`).
    """
    def __init__(self, my_path, ftp_url='ftp.ncbi.nlm.nih.gov', local=False):
        self.my_path = my_path
        self.is_local = local
        self.ftp_url = ftp_url
        return

    def _path_join(self, *args):
        joined_str = path.join(*args)
        part_list = joined_str.split('/')
        for part in part_list[1:]:
            if part == '..':
                idx = part_list.index(part) - 1
                part_list.pop(idx)
                part_list.pop(idx)
        ret = path.join(*part_list)
        if part_list[0] == '':
            ret = '/' + ret
        return ret

    def get_ftp_connection(self, ftp_path=None):
        if ftp_path is None:
            ftp_path = self.my_path
        # Get an FTP connection
        ftp = FTP(self.ftp_url)
        ftp.login()
        # Change to the manuscripts directory
        ftp.cwd(ftp_path)
        return ftp

    def get_xml_file(self, xml_file):
        "Get the content from an xml file as an ElementTree."
        logger.info("Downloading %s" % (xml_file))
        xml_bytes = self.get_uncompressed_bytes(xml_file, force_str=False)
        logger.info("Parsing XML metadata")
        return ET.XML(xml_bytes, parser=UTB())

    def get_csv_as_dict(self, csv_file, cols=None, infer_header=True,
                        add_fields=None):
        """Get the content from a csv file as a list of dicts.

        Parameters
        ----------
        csv_file : str
            The name of the csv file within the target FTP directory.
        cols : Iterable[str]
            Pass in labels for the columns of the CSV.
        infer_header : bool
            If True, infer the cols from the first line. If False, the cols
            will simply be indexed by integers.
        add_fields : dict
            A dictionary of additional fields to add to the dicts.
        """
        csv_str = self.get_file(csv_file)
        csv_lines = csv_str.splitlines()
        result = []
        for row in csv.reader(csv_lines):
            if not cols:
                if infer_header:
                    cols = row[:]
                    continue
                else:
                    cols = list(range(len(row)))
            row_dict = dict(zip(cols, row))
            if add_fields is not None:
                row_dict.update(add_fields)
            result.append(row_dict)
        return result

    def ret_file(self, f_path, buf):
        "Load the content of a file into the given buffer."
        full_path = self._path_join(self.my_path, f_path)
        if not self.is_local:
            with self.get_ftp_connection() as ftp:
                ftp.retrbinary('RETR /%s' % full_path,
                               callback=lambda s: buf.write(s),
                               blocksize=ftp_blocksize)
                buf.flush()
        else:
            with open(self._path_join(self.ftp_url, full_path), 'rb') as f:
                buf.write(f.read())
                buf.flush()
        return

    def download_file(self, f_path, dest=None):
        "Download a file into a file given by f_path."
        name = path.basename(f_path)
        if dest is not None:
            name = path.join(dest, name)
        with open(name, 'wb') as gzf:
            self.ret_file(f_path, gzf)
        return name

    def get_file(self, f_path, force_str=True, decompress=True):
        "Get the contents of a file as a string."
        gzf_bytes = BytesIO()
        self.ret_file(f_path, gzf_bytes)
        ret = gzf_bytes.getvalue()
        if f_path.endswith('.gz') and decompress:
            ret = zlib.decompress(ret, 16+zlib.MAX_WBITS)
        if force_str and isinstance(ret, bytes):
            ret = ret.decode('utf8')
        return ret

    def get_uncompressed_bytes(self, f_path, force_str=True):
        "Get a file that is gzipped, and return the unzipped string."
        return self.get_file(f_path, force_str=force_str, decompress=True)

    def ftp_ls_timestamped(self, ftp_path=None):
        "Get all contents and metadata in mlsd format from the ftp directory."
        if ftp_path is None:
            ftp_path = self.my_path
        else:
            ftp_path = self._path_join(self.my_path, ftp_path)

        if not self.is_local:
            with self.get_ftp_connection(ftp_path) as ftp:
                raw_contents = ftp.mlsd()
                contents = [(k, meta['modify']) for k, meta in raw_contents
                            if not k.startswith('.')]
        else:
            dir_path = self._path_join(self.ftp_url, ftp_path)
            raw_contents = listdir(dir_path)
            contents = [(fname, path.getmtime(path.join(dir_path, fname)))
                        for fname in raw_contents]
        return contents

    def ftp_ls(self, ftp_path=None):
        "Get a list of the contents in the ftp directory."
        if ftp_path is None:
            ftp_path = self.my_path
        else:
            ftp_path = self._path_join(self.my_path, ftp_path)
        if not self.is_local:
            with self.get_ftp_connection(ftp_path) as ftp:
                contents = ftp.nlst()
        else:
            contents = listdir(self._path_join(self.ftp_url, ftp_path))
        return contents


def get_clean_id(db, id_type, id_val):
    if id_type == 'pmid':
        id_val, _ = db.TextRef.process_pmid(id_val)
    elif id_type == 'pmcid':
        id_val, _, _ = db.TextRef.process_pmcid(id_val)
    elif id_type == 'doi':
        id_val, _, _ = db.TextRef.process_doi(id_val)
    return id_val


class ContentManager(object):
    """Abstract class for all upload/update managers.

    This abstract class provides the api required for any object that is
    used to manage content between the database and the content.
    """
    my_source: str = NotImplemented
    tr_cols: Tuple = NotImplemented
    tc_cols: Tuple = NotImplemented
    primary_col: str = NotImplemented
    err_patt = re.compile(r'.*?constraint "(.*?)".*?Key \((.*?)\)=\((.*?)\).*?',
                          re.DOTALL)

    def __init__(self):
        self.review_fname = None
        return

    def upload_text_refs(self, db, ref_data):
        logger.info("Processing text ref rows.")
        new_cols = []
        for id_type in self.tr_cols:
            if id_type == 'pmid':
                new_cols += ['pmid', 'pmid_num']
            elif id_type == 'pmcid':
                new_cols += ['pmcid', 'pmcid_num', 'pmcid_version']
            elif id_type == 'doi':
                new_cols += ['doi', 'doi_ns', 'doi_id']
            else:
                new_cols.append(id_type)
        cols = tuple(new_cols)

        # Process all the rows.
        new_data = []
        for row in ref_data:
            if len(row) != len(self.tr_cols):
                raise ValueError("Row length does not match column length "
                                 "of labels.")
            new_row = []
            for id_type, id_val in zip(self.tr_cols, row):
                if id_type == 'pmid':
                    pmid, pmid_num = db.TextRef.process_pmid(id_val)
                    new_row += [pmid, pmid_num]
                elif id_type == 'pmcid':
                    pmcid, pmcid_num, pmcid_version = \
                        db.TextRef.process_pmcid(id_val)
                    new_row += [pmcid, pmcid_num, pmcid_version]
                elif id_type == 'doi':
                    doi, doi_ns, doi_id = db.TextRef.process_doi(id_val)
                    new_row += [doi, doi_ns, doi_id]
                else:
                    new_row.append(id_val)
            new_data.append(tuple(new_row))
        ref_data = new_data

        ret_cols = (self.primary_col, 'id')
        existing_tuples, new_tuples, skipped_rows = \
            db.copy_detailed_report_lazy('text_ref', ref_data, cols, ret_cols)
        id_dict = dict(new_tuples + existing_tuples)
        return id_dict, skipped_rows

    def upload_text_content(self, db, data):
        """Insert text content into the database using COPY."""
        return db.copy_report_lazy('text_content', data, self.tc_cols)

    def make_text_ref_str(self, tr):
        """Make a string from a text ref using tr_cols."""
        return str([getattr(tr, id_type) for id_type in self.tr_cols])

    def add_to_review(self, desc, msg):
        """Add an entry to the review document."""
        # NOTE: If this is ever done on AWS or through a
        # container, the review file MUST be loaded somewhere
        # it won't disappear. (such as s3). Perhaps these could
        # be logged on the database?
        logger.warning("Found \"%s\"! Check %s."
                       % (desc, self.review_fname))
        with open(self.review_fname, 'a+') as f:
            f.write(msg + '\n')
        return

    def filter_text_refs(self, db, tr_data_set, primary_id_types=None):
        """Try to reconcile the data we have with what's already on the db.

        Note that this method is VERY slow in general, and therefore should
        be avoided whenever possible.

        The process can be sped up considerably by multiple orders of
        magnitude if you specify a limited set of id types to query to get
        text refs. This does leave some possibility of missing relevant refs.
        """
        logger.info("Beginning to filter %d text refs..." % len(tr_data_set))

        # If there are not actual refs to work with, don't waste time.
        N = len(tr_data_set)
        if not N:
            return set(), []

        # Get all text refs that match any of the id data we have.
        logger.debug("Getting list of existing text refs...")
        or_list = []
        if primary_id_types is not None:
            match_id_types = primary_id_types
        else:
            match_id_types = self.tr_cols

        # This is a helper for accessing the data tuples we create
        def id_idx(id_type):
            return self.tr_cols.index(id_type)

        # Get IDs from the tr_data_set that have one or more of the listed
        # id types.
        for id_type in match_id_types:
            id_list = [entry[id_idx(id_type)] for entry in tr_data_set
                       if entry[id_idx(id_type)] is not None]
            # Add SqlAlchemy filter clause based on ID list for this ID type
            if id_list:
                if id_type == 'pmid':
                    term = db.TextRef.pmid_in(id_list, filter_ids=True)
                    bad_ids = [pmid for pmid in id_list
                               if not all(db.TextRef.process_pmid(pmid))]
                elif id_type == 'pmcid':
                    term = db.TextRef.pmcid_in(id_list, filter_ids=True)
                    bad_ids = [pmcid for pmcid in id_list
                               if not all(db.TextRef.process_pmcid(pmcid)[:2])]
                elif id_type == 'doi':
                    term = db.TextRef.doi_in(id_list, filter_ids=True)
                    bad_ids = [doi for doi in id_list
                               if not all(db.TextRef.process_doi(doi))]
                else:
                    term = getattr(db.TextRef, id_type).in_(id_list)
                    bad_ids = []
                or_list.append(term)
                if bad_ids:
                    logger.info("Handling %d malformed '%s's with separate "
                                "query." % (len(bad_ids), id_type))
                    or_list.append(getattr(db.TextRef, id_type).in_(bad_ids))
        if len(or_list) == 1:
            tr_list = db.select_all(db.TextRef, or_list[0])
        else:
            tr_list = db.select_all(db.TextRef, sql_exp.or_(*or_list))
        logger.debug("Found %d potentially relevant text refs." % len(tr_list))

        # Create an index of tupled data entries for quick lookups by any id
        # type, for example tr_data_idx_dict['pmid'][<a pmid>] will get the
        # tuple with all the id data.
        logger.debug("Building index of new data...")
        tr_data_idx_dict = {}
        for id_type in match_id_types:
            tr_data_idx = {}
            for entry in tr_data_set:
                id_val = entry[id_idx(id_type)]
                try:
                    id_val = get_clean_id(db, id_type, id_val)
                except Exception as err:
                    logger.warning(f"Id of type {id_type} malformed: {id_val}")

                if id_val is not None:
                    tr_data_idx[id_val] = entry
            tr_data_idx_dict[id_type] = tr_data_idx
            del tr_data_idx

        # Look for updates to the existing text refs
        logger.debug("Beginning to iterate over text refs...")
        tr_data_match_list = []
        flawed_tr_data = []
        multi_match_records = set()
        update_dict = {}

        def add_to_found_record_list(record):
            # Adds a record tuple from tr_data_list to tr_data_match_list and
            # return True on success. If the record has multiple matches in
            # the database then we wouldn't know which one to update--hence
            # we record for review and return False for failure.
            if record not in tr_data_match_list:
                tr_data_match_list.append(record)
                added = True
            else:
                self.add_to_review(
                    "tr matching input record matched to another tr",
                    "Input record %s already matched. Matched again to %s."
                    % (record, self.make_text_ref_str(tr))
                    )
                flawed_tr_data.append(('over_match_db', record))
                multi_match_records.add(record)
                added = False
            return added

        for tr in tr_list:
            match_set = set()

            # Find the matches in the data. Multiple distinct matches indicate
            # problems, and are flagged.
            for id_type, tr_data_idx in tr_data_idx_dict.items():
                candidate = tr_data_idx.get(getattr(tr, id_type))
                if candidate is not None:
                    match_set.add(candidate)

            # Every tr MUST have a match, or else something is broken.
            assert match_set, "No matches found, which is impossible."

            # Given a unique match, update any missing ids from the input data.
            if len(match_set) == 1:
                tr_new = match_set.pop()

                # Add this record to the match list, unless there are conflicts
                # If there are conflicts (multiple matches in the DB) then
                # we skip any updates.
                if not add_to_found_record_list(tr_new):
                    continue

                # Tabulate new/updated ID information.
                # Go through all the id_types
                all_good = True
                id_updates = {}
                for i, id_type in enumerate(self.tr_cols):
                    if id_type not in match_id_types:
                        continue
                    # Check if the text ref is missing that id.
                    if getattr(tr, id_type) is None:
                        # If so, and if our new data does have that id, update
                        # the text ref.
                        if tr_new[i] is not None:
                            logger.debug("Will update text ref for %s: %s."
                                         % (id_type, tr_new[i]))
                            id_updates[id_type] = tr_new[i]
                    else:
                        # Check to see that all the ids agree. If not, report
                        # it in the review.txt file.
                        new_id = get_clean_id(db, id_type, tr_new[i])
                        if new_id is None:
                            continue
                        elif new_id != getattr(tr, id_type):
                            self.add_to_review(
                                'conflicting ids',
                                'Got conflicting %s: in db %s vs %s.'
                                % (id_type, self.make_text_ref_str(tr), tr_new)
                                )
                            flawed_tr_data.append((id_type, tr_new))
                            all_good = False

                if all_good and len(id_updates):
                    update_dict[tr.id] = (tr, id_updates, tr_new)
            else:
                # These still matched something in the db, so they shouldn't be
                # uploaded as new refs.
                for tr_new in match_set:
                    add_to_found_record_list(tr_new)
                    flawed_tr_data.append(('over_match_input', tr_new))

                # This condition only occurs if the records we got are
                # internally inconsistent. This is rare, but it can happen.
                self.add_to_review(
                    "multiple matches in records for tex ref",
                    'Multiple matches for %s from %s: %s.'
                    % (self.make_text_ref_str(tr), self.my_source, match_set))

        # Apply ID updates to TextRefs with unique matches in tr_data_set
        logger.info("Applying %d updates." % len(update_dict))
        updated_id_map = {}
        for tr, id_updates, record in update_dict.values():
            if record not in multi_match_records:
                tr.update(**id_updates)
                updated_id_map[tr.pmid] = tr.id
            else:
                logger.warning("Skipping update of text ref %d with %s due "
                               "to multiple matches to record %s."
                               % (tr.id, id_updates, record))

        # This applies all the changes made to the text refs to the db.
        logger.debug("Committing changes...")
        db.commit("Failed to update with new ids.")

        # Now update the text refs with any new refs that were found
        filtered_tr_records = tr_data_set - set(tr_data_match_list) \
            - multi_match_records

        logger.debug("Filtering complete! %d records remaining."
                     % len(filtered_tr_records))
        return filtered_tr_records, flawed_tr_data, updated_id_map

    @classmethod
    def _record_for_review(cls, func):
        @wraps(func)
        def take_action(self, db, *args, **kwargs):
            review_fmt = "review_%s_%s_%%s.txt" % (func.__name__,
                                                   self.my_source)
            self.review_fname = path.join(THIS_DIR, review_fmt % 'in_progress')
            logger.info("Creating review file %s." % self.review_fname)
            open(self.review_fname, 'a+').close()
            completed = func(self, db, *args, **kwargs)
            if completed:
                utcnow = datetime.utcnow()
                is_init_upload = (func.__name__ == 'populate')
                with open(self.review_fname, 'r') as f:
                    conflicts_bytes = zip_string(f.read())
                    db.insert('updates', init_upload=is_init_upload,
                              source=self.my_source,
                              unresolved_conflicts_file=conflicts_bytes)
                rename(self.review_fname,
                       review_fmt % utcnow.strftime('%Y%m%d-%H%M%S'))
            return completed
        return take_action

    @classmethod
    def get_latest_update(cls, db):
        """Get the date of the latest update."""
        last_dt, = (db.session.query(sql_exp.func.max(db.Updates.datetime))
                    .filter(db.Updates.source == cls.my_source).one())
        if not last_dt:
            logger.error("The database has not had an initial upload, or else "
                         "the updates table has not been populated.")
            return None

        return last_dt

    def populate(self, db):
        "A stub for the method used to initially populate the database."
        raise NotImplementedError(
            "`Populate` not implemented for `%s`." % self.__class__.__name__
            )

    def update(self, db):
        "A stub for the method used to update the content on the database."
        raise NotImplementedError(
            "`Update` not implemented for `%s`." % self.__class__.__name__
            )


class _NihManager(ContentManager):
    """Abstract class for all the managers that use the NIH FTP service.

    See `_NihFtpClient` for parameters.
    """
    my_path = NotImplemented

    def update(self, db):
        raise NotImplementedError

    def populate(self, db):
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        self.ftp = _NihFtpClient(self.my_path, *args, **kwargs)
        super(_NihManager, self).__init__()
        return


def summarize_content_labels(label_list):
    archive_stats = {}
    for archive_name, file_num, num_files in label_list:
        if archive_name not in archive_stats:
            archive_stats[archive_name] = {'min': num_files, 'max': 0,
                                           'tot': num_files}
        if file_num < archive_stats[archive_name]['min']:
            archive_stats[archive_name]['min'] = file_num
        if (file_num + 1) > archive_stats[archive_name]['max']:
            archive_stats[archive_name]['max'] = file_num + 1
    return archive_stats


class Pubmed(_NihManager):
    """Manager for the pubmed/medline content.

    For relevant updates from NCBI on the managemetn and upkeep of the PubMed
    Abstract FTP server, see here:

      https://www.nlm.nih.gov/databases/download/pubmed_medline.html
    """
    my_path = 'pubmed'
    my_source = 'pubmed'
    primary_col = 'pmid'
    tr_cols = ('pmid', 'pmcid', 'doi', 'pii', 'pub_year')
    tc_cols = ('text_ref_id', 'source', 'format', 'text_type', 'content',
               'license')

    def __init__(self, *args, categories=None, tables=None,
                 max_annotations=500000, **kwargs):
        super(Pubmed, self).__init__(*args, **kwargs)
        self.deleted_pmids = None
        if categories is None:
            self.categories = [texttypes.TITLE, texttypes.ABSTRACT]
        else:
            self.categories = categories[:]
        assert all(cat in texttypes.values() for cat in self.categories)

        if tables is None:
            self.tables = ['text_ref', 'text_content']
        else:
            self.tables = tables[:]

        self.db_pmids = None
        self.max_annotations = max_annotations
        self.num_annotations = 0
        self.annotations = {}
        return

    def get_deleted_pmids(self):
        if self.deleted_pmids is None:
            del_pmid_str = self.ftp.get_uncompressed_bytes(
                'deleted.pmids.gz'
                )
            pmid_list = [
                line.strip() for line in del_pmid_str.split('\n')
                ]
            self.deleted_pmids = pmid_list
        return self.deleted_pmids[:]

    def get_file_list(self, sub_dir):
        all_files = self.ftp.ftp_ls(sub_dir)
        return [sub_dir + '/' + k for k in all_files if k.endswith('.xml.gz')]

    def get_article_info(self, xml_file, q=None):
        from indra.literature import pubmed_client
        tree = self.ftp.get_xml_file(xml_file)
        article_info = pubmed_client.get_metadata_from_xml_tree(
            tree,
            get_abstracts=True,
            prepend_title=False
            )
        if q is not None:
            q.put((xml_file, article_info))
            return
        else:
            return article_info

    @staticmethod
    def fix_doi(doi):
        """Sometimes the doi is doubled (no idea why). Fix it."""
        if doi is None:
            return
        L = len(doi)
        if L % 2 != 0:
            return doi
        if doi[:L//2] != doi[L//2:]:
            return doi
        logger.info("Fixing doubled doi: %s" % doi)
        return doi[:L//2]

    def load_annotations(self, db, tr_data):
        """Load annotations into the database."""
        for tr_dict in tr_data:
            self.annotations[tr_dict['pmid']] = tr_dict['annotations']
            self.num_annotations += len(tr_dict['annotations'])

        # Add mesh annotations to the db in batches.
        if self.num_annotations > self.max_annotations:
            self.dump_annotations(db)
            self.annotations = {}
            self.num_annotations = 0

        return

    def load_text_refs(self, db, tr_data, update_existing=False):
        """Sanitize, update old, and upload new text refs."""
        # Convert the article_info into a list of tuples for insertion into
        # the text_ref table
        tr_records = set()
        for tr in tr_data:
            row = tuple(tr.get(col) for col in self.tr_cols)
            tr_records.add(row)

        # Check the ids more carefully against what is already in the db.
        if update_existing:
            tr_records, flawed_refs, id_map = \
                self.filter_text_refs(db, tr_records,
                                      primary_id_types=['pmid', 'pmcid'])
            logger.info(f"Found {len(tr_records)} refs to add, {len(id_map)} "
                        f"refs updated.")
        else:
            id_map = {}

        # Remove the pmids from any data entries that failed to copy.
        upload_id_map, skipped_rows = self.upload_text_refs(db, tr_records)
        logger.info(f"{len(skipped_rows)} rows skipped due to constraint "
                    f"violation.")
        id_map.update(upload_id_map)
        gatherer.add('refs', len(tr_records) - len(skipped_rows))
        return id_map

    def load_text_content(self, db, tc_data, pmid_map):
        # Add the text_ref IDs to the content to be inserted
        text_content_records = set()
        for tc in tc_data:
            pmid = tc['pmid']
            if pmid not in pmid_map.keys():
                logger.warning("Found content marked to be uploaded which "
                               "does not have a text ref. Skipping pmid "
                               "%s..." % pmid)
                continue
            trid = pmid_map[pmid]
            text_content_records.add((trid, self.my_source, formats.TEXT,
                                      tc['text_type'], tc['content'], 'pubmed'))

        logger.info("Found %d new text content entries."
                    % len(text_content_records))

        self.upload_text_content(db, text_content_records)
        gatherer.add('content', len(text_content_records))
        return

    def iter_contents(self, archives=None):
        """Iterate over the files in the archive, yielding ref and content data.

        Parameters
        ----------
        archives : Optional[Iterable[str]]
            The names of the archive files from the FTP server to processes. If
            None, all available archives will be iterated over.

        Yields
        ------
        label : tuple
            A key representing the particular XML: (XML File Name, Entry Number,
            Total Entries)
        text_ref_dict : dict
            A dictionary containing the text ref information.
        text_content_dict : dict
            A dictionary containing the text content information.
        """
        from indra.literature.pubmed_client import get_metadata_from_xml_tree

        if archives is None:
            archives = self.get_file_list('baseline') \
                       + self.get_file_list('updatefiles')

        invalid_pmids = set(self.get_deleted_pmids())

        for xml_file in sorted(archives):
            # Parse the XML file.
            tree = self.ftp.get_xml_file(xml_file)
            article_info = get_metadata_from_xml_tree(tree, get_abstracts=True,
                                                      prepend_title=False)

            # Filter out any PMIDs that have been deleted.
            valid_article_pmids = set(article_info.keys()) - invalid_pmids

            # Yield results for each PMID.
            num_records = len(valid_article_pmids)
            for idx, pmid in enumerate(valid_article_pmids):
                data = article_info[pmid]

                # Extract the ref data.
                tr = {'pmid': pmid}
                for id_type in self.tr_cols[1:]:
                    val = None
                    if id_type == 'pub_year':
                        r = data.get('publication_date')
                        if 'year' in r:
                            val = r['year']
                    else:
                        r = data.get(id_type)
                        if id_type == 'doi':
                            r = self.fix_doi(r)
                        if r:
                            val = r.strip().upper()
                    tr[id_type] = val
                    tr['annotations'] = data['mesh_annotations']

                # Extract the content data
                tc_list = []
                for text_type in self.categories:
                    content = article_info[pmid].get(text_type)
                    if content and content.strip():
                        tc = {'pmid': pmid, 'text_type': text_type,
                              'content': zip_string(content)}
                        tc_list.append(tc)
                yield (xml_file, idx, num_records), tr, tc_list

    def load_files(self, db, files, continuing=False, carefully=False,
                   log_update=True):
        """Load the files in subdirectory indicated by ``dirname``."""
        if 'text_ref' not in self.tables:
            logger.info("Loading pmids from the database...")
            self.db_pmids = {pmid for pmid, in db.select_all(db.TextRef.pmid)}

        # If we are picking up where we left off, look for prior files.
        if continuing or log_update:
            sf_list = db.select_all(
                db.SourceFile,
                db.SourceFile.source == self.my_source
                )
            existing_files = {sf.name for sf in sf_list}
            logger.info(f"Found {len(existing_files)} existing files.")
        else:
            existing_files = set()

        # Remove prior files, and quit if there is nothing left to do.
        files = set(files) - existing_files
        if not files:
            logger.info("No files to process, nothing to do.")
            return False
        else:
            logger.info(f"{len(files)} new files to process.")

        batch_size = 10000
        contents = self.iter_contents(files)
        batched_contents = batch_iter(contents, batch_size, lambda g: zip(*g))
        for i, (label_batch, tr_batch, tc_batch) in enumerate(batched_contents):

            # Figure out where we are in the list of archives.
            archive_stats = summarize_content_labels(label_batch)

            # Log where we are.
            logger.info(f"Beginning batch {i}...")
            for file_name, info in archive_stats.items():
                logger.info(f"  - {info['min']}-{info['max']}/{info['tot']} "
                            f"of {file_name}")

            # Update the database records.
            self.load_annotations(db, tr_batch)
            id_map = self.load_text_refs(db, tr_batch,
                                         update_existing=carefully)
            expanded_tcs = (tc for tc_list in tc_batch for tc in tc_list)
            self.load_text_content(db, expanded_tcs, id_map)

            # Check if we finished an xml file.
            for file_name, info in archive_stats.items():
                if info['max'] == info['tot']:
                    if log_update and file_name not in existing_files:
                        db.insert('source_file', source=self.my_source,
                                  name=file_name)
        self.dump_annotations(db)

        return True

    def dump_annotations(self, db):
        """Dump all the annotations that have been saved so far."""
        logger.info("Dumping mesh annotations for %d refs."
                    % len(self.annotations))

        # If there are no annotations, don't waste time.
        if not self.annotations:
            return False

        copy_rows = []
        for pmid, annotation_list in self.annotations.items():
            for annotation in annotation_list:
                # Handle it if supplementary IDs (they start with C and their
                # intified values can overlap with main terms that start with D)
                is_concept = annotation.get('type') == 'supplementary'

                # Format the row.
                copy_row = (int(pmid), int(annotation['mesh'][1:]),
                            annotation['major_topic'], is_concept)

                # Handle the qualifier
                qualifiers = annotation['qualifiers']
                qual = int(qualifiers[0]['mesh'][1:]) if qualifiers else None
                copy_row += (qual,)

                copy_rows.append(copy_row)

        # Copy the results into the database
        db.copy_lazy('mesh_ref_annotations', copy_rows,
                     ('pmid_num', 'mesh_num', 'major_topic', 'is_concept',
                      'qual_num'))
        return True

    @ContentManager._record_for_review
    @DGContext.wrap(gatherer, 'pubmed')
    def populate(self, db, continuing=False):
        """Perform the initial input of the pubmed content into the database.

        Parameters
        ----------
        db : indra.db.DatabaseManager instance
            The database to which the data will be uploaded.
        continuing : bool
            If true, assume that we are picking up after an error, or otherwise
            continuing from an earlier process. This means we will skip over
            source files contained in the database. If false, all files will be
            read and parsed.
        """
        files = self.get_file_list('baseline')
        return self.load_files(db, files, continuing, False)

    @ContentManager._record_for_review
    @DGContext.wrap(gatherer, 'pubmed')
    def update(self, db):
        """Update the contents of the database with the latest articles."""
        files = self.get_file_list('baseline') \
            + self.get_file_list('updatefiles')
        return self.load_files(db, files, True, True)


class PmcManager(_NihManager):
    """Abstract class for uploaders of PMC content: PmcOA and Manuscripts."""
    my_source = NotImplemented
    primary_col = 'pmcid'
    tr_cols = ('pmid', 'pmcid', 'doi', 'manuscript_id', 'pub_year',)
    tc_cols = ('text_ref_id', 'source', 'format', 'text_type',
               'content', 'license')

    def __init__(self, *args, **kwargs):
        super(PmcManager, self).__init__(*args, **kwargs)
        self.file_data = self.get_file_data()

    def update(self, db):
        raise NotImplementedError

    def get_file_data(self):
        raise NotImplementedError

    @staticmethod
    def get_missing_pmids(db, tr_data):
        """Try to get missing pmids using the pmc client."""
        from indra.literature.pmc_client import id_lookup

        logger.info("Getting missing pmids.")

        missing_pmid_entries = []
        for tr_entry in tr_data:
            if tr_entry['pmid'] is None:
                missing_pmid_entries.append(tr_entry)

        num_missing = len(missing_pmid_entries)
        if num_missing == 0:
            logger.debug("No missing pmids.")
            return

        logger.debug('Missing %d pmids.' % num_missing)
        tr_list = db.select_all(
            db.TextRef, db.TextRef.pmcid.in_(
                [tr_entry['pmcid'] for tr_entry in missing_pmid_entries]
                )
            )
        pmids_from_db = {tr.pmcid: tr.pmid for tr in tr_list
                         if tr.pmid is not None}

        logger.debug("Found %d pmids on the databse." % len(pmids_from_db))
        num_found_non_db = 0
        for tr_entry in missing_pmid_entries:
            if tr_entry['pmcid'] not in pmids_from_db.keys():
                ret = id_lookup(tr_entry['pmcid'], idtype='pmcid')
                if 'pmid' in ret.keys() and ret['pmid'] is not None:
                    tr_entry['pmid'] = ret['pmid']
                    num_found_non_db += 1
                    num_missing -= 1
            else:
                tr_entry['pmid'] = pmids_from_db[tr_entry['pmcid']]
                num_missing -= 1
        logger.debug("Found %d more pmids from other sources."
                     % num_found_non_db)
        logger.debug("There are %d missing pmids remaining." % num_missing)
        return

    def filter_text_content(self, db, tc_data):
        """Filter the text content to identify pre-existing records."""
        logger.info("Beginning to filter text content...")
        arc_pmcid_list = [tc['pmcid'] for tc in tc_data]
        if not len(tc_data):
            return []

        logger.info("Getting text refs for pmcid->trid dict..")
        tc_q = (db.session.query(db.TextContent.text_ref_id,
                                 db.TextContent.text_type)
                .filter(db.TextContent.source == self.my_source,
                        db.TextContent.format == formats.XML))
        tc_al = tc_q.subquery().alias('tc')
        q = (db.session.query(db.TextRef.pmcid, db.TextRef.id,
                              tc_al.c.text_type)
             .outerjoin(tc_al)
             .filter(db.TextRef.pmcid_in(arc_pmcid_list)))
        existing_tc_meta = q.all()

        pmcid_trid_dict = {pmcid: trid for pmcid, trid, _ in existing_tc_meta}

        # This should be a very small list, in general.
        existing_tc_records = [
            (trid, self.my_source, formats.XML, text_type)
            for _, trid, text_type in existing_tc_meta if text_type is not None
            ]
        logger.info("Found %d existing records on the db."
                    % len(existing_tc_records))
        tc_records = []
        for tc in tc_data:
            if tc['pmcid'] not in pmcid_trid_dict.keys():
                logger.warning("Found pmcid (%s) among text content data, but "
                               "not in the database. Skipping." % tc['pmcid'])
                continue
            tc_records.append((pmcid_trid_dict[tc['pmcid']], self.my_source,
                               formats.XML, tc['text_type'], tc['content'],
                               tc['license']))
        filtered_tc_records = {rec for rec in tc_records
                               if rec[:-2] not in existing_tc_records}
        logger.info("Finished filtering the text content.")
        return filtered_tc_records

    def upload_batch(self, db, tr_data, tc_data):
        """Add a batch of text refs and text content to the database."""

        # Check for any pmids we can get from the pmc client (this is slow!)
        self.get_missing_pmids(db, tr_data)
        primary_id_types=['pmid', 'pmcid', 'manuscript_id']
        # Turn the list of dicts into a set of tuples
        tr_data_set = {tuple([entry[id_type] for id_type in self.tr_cols])
                       for entry in tr_data}

        filtered_tr_records, flawed_tr_records, updated_id_map = \
            self.filter_text_refs(db, tr_data_set,
                                  primary_id_types=primary_id_types)
        pmcids_to_skip = {rec[primary_id_types.index('pmcid')]
                          for cause, rec in flawed_tr_records
                          if cause in ['pmcid', 'over_match_input',
                                       'over_match_db']}
        if len(pmcids_to_skip) != 0:
            mod_tc_data = [
                tc for tc in tc_data if tc['pmcid'] not in pmcids_to_skip
                ]
        else:
            mod_tc_data = tc_data

        # Upload the text ref data.
        logger.info('Adding %d new text refs...' % len(filtered_tr_records))
        self.upload_text_refs(db, filtered_tr_records)
        gatherer.add('refs', len(filtered_tr_records))

        # Process the text content data
        filtered_tc_records = self.filter_text_content(db, mod_tc_data)

        # Upload the text content data.
        logger.info('Adding %d more text content entries...' %
                    len(filtered_tc_records))
        self.upload_text_content(db, filtered_tc_records)
        gatherer.add('content', len(filtered_tc_records))
        return

    def get_data_from_xml_str(self, xml_str, filename):
        """Get the data out of the xml string."""
        # Load the XML
        try:
            tree = ET.XML(xml_str.encode('utf8'))
        except ET.ParseError:
            logger.info("Could not parse %s. Skipping." % filename)
            return None

        # Get the ID information from the XML.
        id_data = {
            e.get('pub-id-type'): e.text for e in
            tree.findall('.//article-id')
            }
        if 'pmcid' not in id_data.keys():
            if 'pmc' in id_data.keys():
                if id_data['pmc'].startswith('PMC'):
                    pmcid = id_data['pmc']
                else:
                    pmcid = 'PMC' + id_data['pmc']
            else:
                pmcid = filename.split('/')[1].split('.')[0]
            id_data['pmcid'] = pmcid
            logger.info('Processing XML for %s.' % pmcid)
        if 'manuscript' in id_data.keys():
            id_data['manuscript_id'] = id_data['manuscript']

        # Get the year from the XML
        year = None
        for elem in tree.findall('.//article-meta/pub-date'):
            year_elem = elem.find('year')
            if year_elem is not None:
                year = int(year_elem.text)
                break

        # Format the text ref datum.
        tr_datum_raw = {k: id_data.get(k) for k in self.tr_cols}
        tr_datum = {k: val.strip().upper() if val is not None else None
                    for k, val in tr_datum_raw.items()}
        tr_datum['pub_year'] = year

        # Format the text content datum.
        tc_datum = {
            'pmcid': id_data['pmcid'],
            'text_type': texttypes.FULLTEXT,
            'license': self.get_license(id_data['pmcid']),
            'content': zip_string(xml_str)
            }
        return tr_datum, tc_datum

    def get_license(self, pmcid):
        """Get the license for this pmcid."""
        raise NotImplementedError()

    def download_archive(self, archive, continuing=False):
        """Download the archive."""
        # Download the archive if need be.
        logger.info('Downloading archive %s.' % archive)
        archive_local_path = None
        try:
            # This is a guess at the location of the archive.
            archive_local_path = path.join(THIS_DIR, path.basename(archive))

            if continuing and path.exists(archive_local_path):
                logger.info('Archive %s found locally at %s, not loading again.'
                            % (archive, archive_local_path))
            else:
                archive_local_path = self.ftp.download_file(archive,
                                                            dest=THIS_DIR)
                logger.debug("Download successfully completed for %s."
                             % archive)
        except BaseException:
            if archive_local_path is not None:
                logger.error(f"Failed to download {archive}, deleting "
                             f"corrupted file.")
                remove(archive_local_path)
            raise
        return archive_local_path

    def iter_xmls(self, archives=None, continuing=False, pmcid_set=None):
        """Iterate over the xmls in the given archives.

        Parameters
        ----------
        archives : Optional[Iterable[str]]
            The names of the archive files from the FTP server to processes. If
            None, all available archives will be iterated over.
        continuing : Optional[Bool]
            If True, look for locally saved archives to parse, saving the time
            of downloading.
        pmcid_set : Optional[set[str]]
            A set of PMCIDs whose content you want returned from each archive.
            Many archives are massive repositories with 10s of thousands of
            papers in each, and only a fraction may need to be returned.
            Extracting and processing XMLs can be time consuming, so skipping
            those you don't need can really pay off!

        Yields
        ------
        label : Tuple
            A key representing the particular XML: (Archive Name, Entry Number,
            Total Entries)
        xml_name : str
            The name of the XML file.
        xml_str : str
            The extracted XML string.
        """
        # By default, iterate through all the archives.
        if archives is None:
            archives = set(self.get_all_archives())

        # Load the pmcid-to-file-name dict.
        # Note: if a pmcid isn't in the file dict it isn't in the corpus, so
        # it is appropriate to just drop it.
        if pmcid_set is not None:
            file_dict = self.get_pmcid_file_dict()
            desired_files = {file_dict.get(pmcid, '~') for pmcid in pmcid_set}
            desired_files -= {'~'}

        # Yield the contents from each archive.
        for archive in sorted(archives):
            try:
                archive_path = self.download_archive(archive, continuing)
            except ftplib.Error as e:
                logger.warning(f"Failed to download {archive}: {e}")
                logger.info(f"Skipping {archive}...")
                continue

            num_yielded = 0
            with tarfile.open(archive_path, mode='r:gz') as tar:

                # Get names of all the XML files in the tar file, and report.
                try:
                    xml_files = [m for m in tar.getmembers() if m.isfile()
                                 and m.name.endswith('xml')]
                    xml_files.sort(key=lambda m: m.name)
                except zlib.error:
                    logger.warning(f"Failed to parse archive {archive}, likely "
                                   f"fault in FTP download.")
                    logger.info(f"Deleting {archive} before yielding any XMLs.")
                    remove(archive_path)
                    continue

                if len(xml_files) > 1:
                    logger.info(f'Iterating over {len(xml_files)} files in '
                                f'{archive}.')
                else:
                    logger.info(f"Loading file from {archive}")

                # Yield each XML file.
                for n, xml_file in enumerate(xml_files):
                    # Skip the files we don't need.
                    if pmcid_set is not None \
                            and xml_file.name not in desired_files:
                        continue
                    logger.info(f"Extracting {xml_file.name} "
                                f"({n}/{len(xml_files)}).")
                    xml_str = tar.extractfile(xml_file).read().decode('utf8')
                    yield (archive, n, len(xml_files)), xml_file.name, xml_str
                    num_yielded += 1

            # Remove it when we're done (unless there was an exception).
            logger.info(f"Deleting {archive} after yielding {num_yielded} "
                        f"XMLs.")
            remove(archive_path)

    def iter_contents(self, archives=None, continuing=False, pmcid_set=None):
        """Iterate over the files in the archive, yielding ref and content data.

        Parameters
        ----------
        archives : Optional[Iterable[str]]
            The names of the archive files from the FTP server to processes. If
            None, all available archives will be iterated over.
        continuing : Optional[Bool]
            If True, look for locally saved archives to parse, saving the time
            of downloading.
        pmcid_set : Optional[set[str]]
            A set of PMCIDs whose content you want returned from each archive.
            Many archives are massive repositories with 10s of thousands of
            papers in each, and only a fraction may need to be returned.
            Extracting and processing XMLs can be time consuming, so skipping
            those you don't need can really pay off!

        Yields
        ------
        label : tuple
            A key representing the particular XML: (Archive Name, Entry Number,
            Total Entries)
        text_ref_dict : dict
            A dictionary containing the text ref information.
        text_content_dict : dict
            A dictionary containing the text content information.
        """
        xml_iter = self.iter_xmls(archives, continuing, pmcid_set)
        for label, file_name, xml_str in xml_iter:
            logger.info(f"Getting data from {file_name}")
            res = self.get_data_from_xml_str(xml_str, file_name)
            if res is None:
                continue
            tr, tc = res
            logger.info(f"Yielding ref and content for {file_name}.")
            yield label, tr, tc

    def is_archive(self, *args):
        raise NotImplementedError("is_archive must be defined by the child.")

    def get_all_archives(self):
        return [k for k in self.ftp.ftp_ls() if self.is_archive(k)]

    def upload_archives(self, db, archives=None, continuing=False,
                        pmcid_set=None, batch_size=10000):
        """Do the grunt work of downloading and processing a list of archives.

        Parameters
        ----------
        db : :py:class:`PrincipalDatabaseManager <indra_db.databases.PrincipalDatabaseManager>`
            A handle to the principal database.
        archives : Optional[Iterable[str]]
            An iterable of archive names from the FTP server.
        continuing : bool
            If True, best effort will be made to avoid repeating work already
            done using some cached files and downloaded archives. If False, it
            is assumed the caches are empty.
        pmcid_set : set[str]
            A set of PMC Ids to include from this list of archives.
        batch_size : Optional[int]
            Default is 10,000. The number of pieces of content to submit to the
            database at a time.
        """
        # Form a generator over the content in batches.
        contents = self.iter_contents(archives, continuing, pmcid_set)
        batched_contents = batch_iter(contents, batch_size, lambda g: zip(*g))

        # Upload each batch of content into the database.
        for i, (lbls, trs, tcs) in enumerate(batched_contents):

            # Figure out where we are in the list of archives.
            archive_stats = summarize_content_labels(lbls)

            # Log where we are.
            logger.info(f"Beginning batch {i}...")
            for archive_name, info in archive_stats.items():
                logger.info(f"  - {info['min']}-{info['max']}/{info['tot']} "
                            f"of {archive_name}")

            # Do the upload
            self.upload_batch(db, trs, tcs)

            # Check if we finished an archive
            for archive_name, info in archive_stats.items():
                if info['max'] == info['tot']:
                    sf_list = db.select_all(
                        db.SourceFile,
                        db.SourceFile.source == self.my_source,
                        db.SourceFile.name == archive_name
                    )
                    if not sf_list:
                        db.insert('source_file', source=self.my_source,
                                  name=archive_name)

    @ContentManager._record_for_review
    @DGContext.wrap(gatherer)
    def populate(self, db, continuing=False):
        """Perform the initial population of the pmc content into the database.

        Parameters
        ----------
        db : indra.db.DatabaseManager instance
            The database to which the data will be uploaded.
        continuing : bool
            If true, assume that we are picking up after an error, or
            otherwise continuing from an earlier process. This means we will
            skip over source files contained in the database. If false, all
            files will be read and parsed.

        Returns
        -------
        completed : bool
            If True, an update was completed. Othewise, the updload was aborted
            for some reason, often because the upload was already completed
            at some earlier time.
        """
        gatherer.set_sub_label(self.my_source)
        archives = set(self.get_all_archives())

        if continuing:
            sf_list = db.select_all(
                'source_file',
                db.SourceFile.source == self.my_source
                )
            for sf in sf_list:
                logger.info("Skipping %s, already done." % sf.name)
                archives.remove(sf.name)

            # Don't do unnecessary work.
            if not len(archives):
                logger.info("No archives to load. All done.")
                return False

        self.upload_archives(db, archives, continuing=continuing)
        return True

    def get_pmcid_file_dict(self):
        """Get a dict keyed by PMCID mapping them to file names."""
        return {d['AccessionID']: d['Article File'] for d in self.file_data}

    def get_csv_files(self, path):
        """Get a list of CSV files from the FTP server."""
        all_files = self.ftp.ftp_ls(path)
        return [f for f in all_files if f.endswith('filelist.csv')]

class PmcOA(PmcManager):
    """ContentManager for the pmc open access content.

    For further details on the API, see the parent class: PmcManager.
    """
    my_path = 'pub/pmc'
    my_source = 'pmc_oa'

    def __init__(self, *args, **kwargs):
        super(PmcOA, self).__init__(*args, **kwargs)
        self.licenses = {d['AccessionID']: d['License']
                         for d in self.file_data}

    def get_license(self, pmcid):
        return self.licenses[pmcid]

    def is_archive(self, k):
        return k.endswith('.tar.gz')

    def get_all_archives(self):
        """Get a list of PMC OA .tar.gz archives in the known subfolders."""
        archives = []
        for subf in ['oa_comm', 'oa_noncomm', 'oa_other']:
            subpath = self.ftp._path_join('oa_bulk', subf, 'xml')
            all_files = self.ftp.ftp_ls(subpath)
            for f in all_files:
                if self.is_archive(f):
                    archives.append(self.ftp._path_join(subpath, f))
        return archives

    def get_file_data(self):
        """Retrieve the metadata provided by the FTP server for files."""
        ftp_file_list = []
        for subf in ['oa_comm', 'oa_noncomm', 'oa_other']:
            path = self.ftp._path_join('oa_bulk', subf, 'xml')
            csv_files = self.get_csv_files(path)
            for f in csv_files:
                file_root = f.split('.filelist')[0]
                archive = self.ftp._path_join(path, f'{file_root}.tar.gz')
                ftp_file_list += self.ftp.get_csv_as_dict(
                    self.ftp._path_join(path, f),
                    add_fields={'archive': archive})
        return ftp_file_list

    def get_archives_after_date(self, min_date):
        """Get the names of all single-article archives after the given date."""
        logger.info("Getting list of articles that have been uploaded since "
                    "the last update.")
        archive_set = {
            f['archive'] for f in self.file_data
            if 'LastUpdated (YYYY-MM-DD HH:MM:SS)' in f and
            datetime.strptime(f['LastUpdated (YYYY-MM-DD HH:MM:SS)'],
                              '%Y-%m-%d %H:%M:%S') > min_date
        }
        return archive_set

    @ContentManager._record_for_review
    @DGContext.wrap(gatherer, my_source)
    def update(self, db):
        load_pmcid_set = self.find_all_missing_pmcids(db)
        archives_dict = self.get_archives_for_pmcids(load_pmcid_set)

        # Upload these archives.
        logger.info("Beginning to upload archives.")
        for archive, pmcid_set in sorted(archives_dict.items()):
            logging.info(f"Extracting {len(pmcid_set)} articles from "
                         f"{archive}.")
            self.upload_archives(db, [archive], pmcid_set=pmcid_set)
        return True

    def find_all_missing_pmcids(self, db):
        """Find PMCIDs available from the FTP server that are not in the DB."""
        logger.info("Getting list of PMC OA content available.")
        ftp_pmcid_set = {entry['AccessionID'] for entry in self.file_data}

        logger.info("Getting a list of text refs that already correspond to "
                    "PMC OA content.")
        tr_list = db.select_all(
            db.TextRef,
            db.TextRef.id == db.TextContent.text_ref_id,
            db.TextContent.source == self.my_source
            )
        load_pmcid_set = ftp_pmcid_set - {tr.pmcid for tr in tr_list}

        logger.info("There are %d PMC OA articles to load."
                    % (len(load_pmcid_set)))

        return load_pmcid_set

    def get_archives_for_pmcids(self, pmcid_set):
        logger.info("Determining which archives need to be loaded.")
        update_archives = defaultdict(set)
        for file_dict in self.file_data:
            pmcid = file_dict['AccessionID']
            if pmcid in pmcid_set:
                update_archives[file_dict['archive']].add(pmcid)
        return update_archives

    @ContentManager._record_for_review
    @DGContext.wrap(gatherer, my_source)
    def upload_all_missing_pmcids(self, db, archives_to_skip=None):
        """
        This is a special case of update where we upload all missing PMCIDs
        instead of a regular incremental update.

        Parameters
        ----------
        db : indra.db.DatabaseManager instance
            The database to which the data will be uploaded.
        archives_to_skip : list[str] or None
            A list of archives to skip. Processing each archive is
            time-consuming, so we can skip some archives if we have already
            processed them. Note that if 100% of the articles from a given
            archive are already in the database, it will be skipped
            automatically; this parameter is only used to skip archives that
            have some articles that could not be uploaded (e.g. because of
            text ref conflicts, etc.).
        """
        load_pmcid_set = self.find_all_missing_pmcids(db)
        update_archives = self.get_archives_for_pmcids(load_pmcid_set)

        logger.info("Beginning to upload archives.")
        count = 0
        for archive, pmcid_set in sorted(update_archives.items()):
            if archive in archives_to_skip:
                continue
            count += len(pmcid_set)
            logging.info(f"Extracting {len(pmcid_set)} articles from {archive}.")
            self.upload_archives(db, [archive], pmcid_set=pmcid_set,
                                batch_size=500)
        logger.info(f"Processed {count} articles. Done uploading.")
        return True

class Manuscripts(PmcManager):
    """ContentManager for the pmc manuscripts.

    For further details on the API, see the parent class: PmcManager.
    """
    my_path = 'pub/pmc/manuscript'
    my_source = 'manuscripts'

    def get_license(self, pmcid):
        return 'manuscripts'

    def get_file_data(self):
        """Retrieve the metadata provided by the FTP server for files."""
        files = self.get_csv_files('xml')
        ftp_file_list = []
        for f in files:
            file_root = f.split('.filelist')[0]
            archive = self.ftp._path_join('xml', f'{file_root}.tar.gz')
            ftp_file_list += self.ftp.get_csv_as_dict(
                self.ftp._path_join('xml', f),
                add_fields={'archive': archive})
        return ftp_file_list

    def get_tarname_from_filename(self, fname):
        "Get the name of the tar file based on the file name (or a pmcid)."
        re_match = re.match(r'(PMC00\d).*?', fname)
        if re_match is not None:
            tarname = re_match.group(0) + 6*'X' + '.xml.tar.gz'
        else:
            tarname = None
        return tarname

    def is_archive(self, k):
        return k.endswith('.xml.tar.gz')

    def enrich_textrefs(self, db):
        """Method to add manuscript_ids to the text refs."""
        tr_list = db.select_all(db.TextRef,
                                db.TextContent.text_ref_id == db.TextRef.id,
                                db.TextContent.source == self.my_source,
                                db.TextRef.manuscript_id.is_(None))
        file_list = self.ftp.get_csv_as_dict('filelist.csv')
        pmcid_mid_dict = {entry['PMCID']: entry['MID'] for entry in file_list}
        pmid_mid_dict = {entry['PMID']: entry['MID'] for entry in file_list
                         if entry['PMID'] != '0'}
        for tr in tr_list:
            if tr.pmcid is not None:
                tr.manuscript_id = pmcid_mid_dict[tr.pmcid]
            elif tr.pmid is not None and tr.pmid in pmid_mid_dict.keys():
                tr.manuscript_id = pmid_mid_dict[tr.pmid]
        db.commit("Could not update text refs with manuscript ids.")
        return

    @ContentManager._record_for_review
    @DGContext.wrap(gatherer, my_source)
    def update(self, db):
        """Add any new content found in the archives.

        Note that this is very much the same as populating for manuscripts,
        as there are no finer grained means of getting manuscripts than just
        looking through the massive archive files. We do check to see if there
        are any new listings in each files, minimizing the amount of time
        downloading and searching, however this will in general be the slowest
        of the update methods.

        The continuing feature isn't implemented yet.
        """
        logger.info("Getting list of manuscript content available.")
        ftp_pmcid_set = {entry['AccessionID'] for entry in self.file_data}

        logger.info("Getting a list of text refs that already correspond to "
                    "manuscript content.")
        tr_list = db.select_all(
            db.TextRef,
            db.TextRef.id == db.TextContent.text_ref_id,
            db.TextContent.source == self.my_source
            )
        load_pmcid_set = ftp_pmcid_set - {tr.pmcid for tr in tr_list}

        logger.info("There are %d manuscripts to load."
                    % (len(load_pmcid_set)))

        logger.info("Determining which archives need to be loaded.")
        update_archives = defaultdict(set)
        for file_dict in self.file_data:
            pmcid = file_dict['AccessionID']
            if pmcid in load_pmcid_set:
                update_archives[file_dict['archive']].add(pmcid)

        logger.info("Beginning to upload archives.")
        for archive, pmcid_set in sorted(update_archives.items()):
            logging.info(f"Extracting {len(pmcid_set)} articles from "
                         f"{archive}.")
            self.upload_archives(db, [archive], pmcid_set=pmcid_set,
                                 batch_size=500)
        return True


class Elsevier(ContentManager):
    """Content manager for maintaining content from Elsevier."""
    my_source = 'elsevier'
    tc_cols = ('text_ref_id', 'source', 'format', 'text_type',
               'content',)

    def __init__(self, *args, **kwargs):
        super(Elsevier, self).__init__(*args, **kwargs)
        with open(path.join(THIS_DIR, 'elsevier_titles.txt'), 'r') as f:
            self.__journal_set = {self.__regularize_title(t)
                                  for t in f.read().splitlines()}
        self.__found_journal_set = set()
        self.__matched_journal_set = set()
        return

    @staticmethod
    def __regularize_title(title):
        title = title.lower()
        for space_car in [' ', '_', '-', '.']:
            title = title.replace(space_car, '')
        return title

    def __select_elsevier_refs(self, tr_set, max_retries=2):
        """Try to check if this content is available on Elsevier."""
        from indra.literature.pubmed_client import get_metadata_for_ids
        from indra.literature.crossref_client import get_publisher

        elsevier_tr_set = set()
        for tr in tr_set.copy():
            if tr.doi is not None:
                publisher = get_publisher(tr.doi)
                if publisher is not None and\
                   publisher.lower() == "elsevier bv":
                    tr_set.remove(tr)
                    elsevier_tr_set.add(tr)

        if tr_set:
            pmid_set = {tr.pmid for tr in tr_set}
            tr_dict = {tr.pmid: tr for tr in tr_set}
            num_retries = 0
            meta_data_dict = None
            while num_retries < max_retries:
                try:
                    if not pmid_set:
                        logger.info("Empty pmid_id set")
                    meta_data_dict = get_metadata_for_ids(pmid_set)
                    break
                except Exception as e:
                    num_retries += 1
                    if num_retries < max_retries:
                        logger.warning("Caught exception while getting "
                                       "metadata. Retrying...")
                    else:
                        logger.error("No more tries for:\n%s" % str(pmid_set))
                        logger.exception(e)
                        meta_data_dict = None
                        break

            if meta_data_dict is not None:
                titles = {(pmid, meta['journal_title'])
                          for pmid, meta in meta_data_dict.items()}
                for pmid, title in titles:
                    reg_title = self.__regularize_title(title)
                    self.__found_journal_set.add(reg_title)
                    if reg_title in self.__journal_set:
                        self.__matched_journal_set.add(reg_title)
                        elsevier_tr_set.add(tr_dict[pmid])
        return elsevier_tr_set

    def __get_content(self, trs):
        """Get the content."""
        from indra.literature.elsevier_client import download_article_from_ids
        article_tuples = set()
        for tr in trs:
            id_dict = {id_type: getattr(tr, id_type)
                       for id_type in ['doi', 'pmid', 'pii']
                       if getattr(tr, id_type) is not None}
            if id_dict:
                content_str = download_article_from_ids(**id_dict)
                if content_str is not None:
                    if has_full_text(content_str):
                        text_type = texttypes.FULLTEXT
                    else:
                        text_type = texttypes.ABSTRACT
                    content_zip = zip_string(content_str)
                    article_tuples.add((tr.id, self.my_source, formats.TEXT,
                                        text_type, content_zip))
        return article_tuples

    def __process_batch(self, db, tr_batch):
        logger.info("Beginning to load batch of %d text refs." % len(tr_batch))
        elsevier_trs = self.__select_elsevier_refs(tr_batch)
        logger.debug("Found %d elsevier text refs." % len(elsevier_trs))
        article_tuples = self.__get_content(elsevier_trs)
        logger.debug("Got %d elsevier results." % len(article_tuples))
        self.copy_into_db(db, 'text_content', article_tuples, self.tc_cols)
        return

    def _get_elsevier_content(self, db, tr_query, continuing=False):
        """Get the elsevier content given a text ref query object."""
        pickle_stash_fname = path.join(THIS_DIR,
                                       'checked_elsevier_trid_stash.pkl')
        tr_batch = set()
        if continuing and path.exists(pickle_stash_fname):
            with open(pickle_stash_fname, 'rb') as f:
                tr_ids_checked = pickle.load(f)
            logger.info("Continuing; %d text refs already checked."
                        % len(tr_ids_checked))
        else:
            tr_ids_checked = set()
        try:
            batch_num = 0
            for tr in tr_query.yield_per(1000):
                # If we're continuing an earlier upload, don't check id's we've
                # already checked.
                if continuing and tr.id in tr_ids_checked:
                    continue

                tr_batch.add(tr)
                if len(tr_batch) % 200 == 0:
                    batch_num += 1
                    logger.info('Beginning batch %d.' % batch_num)
                    self.__process_batch(db, tr_batch)
                    tr_ids_checked |= {tr.id for tr in tr_batch}
                    tr_batch.clear()
            if tr_batch:
                logger.info('Loading final batch.')
                self.__process_batch(db, tr_batch)
                tr_ids_checked |= {tr.id for tr in tr_batch}
        except BaseException as e:
            logger.error("Caught exception while loading elsevier.")
            logger.exception(e)
            with open(pickle_stash_fname, 'wb') as f:
                pickle.dump(tr_ids_checked, f)
            logger.info("Stashed the set of checked text ref ids in: %s"
                        % pickle_stash_fname)
            return False
        finally:
            with open('journals.pkl', 'wb') as f:
                pickle.dump({'elsevier': self.__journal_set,
                             'found': self.__found_journal_set,
                             'matched': self.__matched_journal_set}, f)
        return True

    def copy_into_db(self, db, table_name, data, columns):
        """Write data to the given table using COPY"""
        logger.info(f"Copying {len(data)} rows into {table_name}.")
        if not data:
            return
        db.copy_report_lazy(table_name, data, columns)

    @ContentManager._record_for_review
    def populate(self, db, n_procs=1, continuing=False):
        """Load all available elsevier content for refs with no pmc content."""
        # Note that we do not implement multiprocessing, because by the nature
        # of the web API's used, we are limited by bandwidth from any one IP.
        tr_w_pmc_q = db.filter_query(
            db.TextRef,
            db.TextRef.id == db.TextContent.text_ref_id,
            db.TextContent.text_type == 'fulltext'
            )
        tr_wo_pmc_q = db.filter_query(db.TextRef).except_(tr_w_pmc_q)
        return self._get_elsevier_content(db, tr_wo_pmc_q, continuing)

    @ContentManager._record_for_review
    def update(self, db, n_procs=1, buffer_days=15):
        """Load all available new elsevier content from new pmids."""
        # There is the possibility that elsevier content will lag behind pubmed
        # updates, so we go back a bit before the last update to make sure we
        # didn't miss anything
        latest_updatetime = self.get_latest_update(db)
        start_datetime = latest_updatetime - timedelta(days=buffer_days)

        # Construct a query for recently added (as defined above) text refs
        # that do not already have text content.
        new_trs = db.filter_query(
            db.TextRef,
            sql_exp.or_(
                db.TextRef.last_updated > start_datetime,
                db.TextRef.create_date > start_datetime,
                )
            )
        tr_w_pmc_q = db.filter_query(
            db.TextRef,
            db.TextRef.id == db.TextContent.text_ref_id,
            db.TextContent.text_type == 'fulltext'
            )
        tr_query = new_trs.except_(tr_w_pmc_q)

        return self._get_elsevier_content(db, tr_query, False)

    def get_elsevier_nlm_ids(self, node_file, journal_set):
        """Extract NLM IDs of Elsevier journals from node.tsv.gz."""
        elsevier_nlm_ids = set()
        logger.info("iterating node.tsv.gz")
        with gzip.open(node_file, "rt", encoding="utf-8") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                nlm_id, label, journal_name = parts[0], parts[1], parts[5]
                if self.__regularize_title(journal_name) in journal_set:
                    elsevier_nlm_ids.add(nlm_id)
        return elsevier_nlm_ids

    def get_elsevier_pmids(self, edge_file, elsevier_nlm_ids):
        """Extract PMIDs that are published in Elsevier journals."""
        pmids = set()
        with gzip.open(edge_file, "rt", encoding="utf-8") as f:
            next(f)
            for line in f:
                pmid, nlm_id, rel_type = line.strip().split("\t")
                if nlm_id in elsevier_nlm_ids:
                    pmid = pmid.replace("pubmed:", "")
                    pmids.add(pmid)
        return pmids

    def get_pmids_without_fulltext(self, db):
        """Retrieve all PMIDs (DOIs) from text_ref that do not have full text in text_content."""
        # If the pmid don't have fulltext,
        # then the leftjoin will make text_ref_id none
        query = (
            db.session.query(db.TextRef)
            .outerjoin(db.TextContent,
                       (db.TextRef.id == db.TextContent.text_ref_id) &
                       (db.TextContent.text_type == 'fulltext'))
            .filter(db.TextContent.text_ref_id.is_(None))
        )
        pmid_id_dict = {tr.pmid: tr.id for tr in query.all() if tr.pmid is not None}
        return pmid_id_dict


    def get_content_from_pmids(self, pmids, pmid_tr_dict):
        from indra.literature.elsevier_client import download_article_from_ids
        """Load PMIDs from a pickle file and retrieve Elsevier article content."""
        fulltext_count = 0
        abstract_count = 0
        article_tuples = set()
        for pmid in pmids:
            id_dict = {"pmid": pmid}
            content_str = download_article_from_ids(**id_dict)
            if content_str is not None:
                if has_full_text(content_str):
                    fulltext_count += 1
                    text_type = texttypes.FULLTEXT
                    content_zip = zip_string(content_str)
                    article_tuples.add((pmid_tr_dict[pmid], self.my_source, formats.TEXT,
                                        text_type, content_zip))
                # else:
                #     abstract_count += 1
                #     text_type = texttypes.ABSTRACT

        logger.info(f"Retrieved {len(article_tuples)} articles with "
              f"{fulltext_count} fulltext and {abstract_count} abstract.")
        return article_tuples

    def filter_existing_articles(self, db, article_tuples):
        """
        Filters out articles that already exist in the database with
         the same text_ref_id and the source is 'elsevier'.

        Parameters:
        - db: Database
        - article_tuples: Set of tuples (text_ref_id, source, format, text_type, content_zip).
        Returns:
        - A filtered set of article tuples
        """
        logger.info("Filtering existing articles...")

        text_ref_ids = {article[0] for article in article_tuples}

        existing_text_ref_ids = set(
            row[0] for row in db.session.query(db.TextContent.text_ref_id)
            .filter(
                db.TextContent.text_ref_id.in_(text_ref_ids),
                db.TextContent.source == "elsevier",
                db.TextContent.text_type == "fulltext"
            )
            .all()
        )

        filtered_articles = {
            article for article in article_tuples
            if article[0] not in existing_text_ref_ids
        }

        logger.info(
            f"Filtered {len(article_tuples) - len(filtered_articles)} existing articles.")
        return filtered_articles

    @ContentManager._record_for_review
    def update_by_cogex(self, db, edge_file=None, node_file=None):
        """Load all available new elsevier content from new pmids.
        Checking against the indra cogex edge and node files"""
        pmid_list_file = os.path.join(THIS_DIR, 'pmids_elsevier.pkl')
        pmid_tr_dict = os.path.join(THIS_DIR, 'pmids_tr_dict_elsevier.pkl')

        if os.path.exists(pmid_list_file) and os.path.exists(pmid_tr_dict):
            with open(pmid_list_file, "rb") as f:
                pmid_list = pickle.load(f)
                logger.info(f"Loaded {len(pmid_list)} PMIDs from {pmid_list_file}")
            with open(pmid_tr_dict, "rb") as f:
                pmid_id_dict = pickle.load(f)
                logger.info(f"Loaded PMIDs Text ref dict from {pmid_tr_dict}")

        else:
            if not (os.path.exists(edge_file) and os.path.exists(node_file)):
                logger.error("Missing edge/node file from indra cogex.")
            elsevier_nlm_ids = self.get_elsevier_nlm_ids(node_file, self.__journal_set)
            logger.info(f"Total Elsevier Journals {len(elsevier_nlm_ids)}")

            elsevier_pmids = self.get_elsevier_pmids(edge_file, elsevier_nlm_ids)
            logger.info(f"Total Elsevier PMIDs: {len(elsevier_pmids)}")

            pmid_id_dict = self.get_pmids_without_fulltext(db)
            all_pmid = set(pmid_id_dict.keys())
            logger.info(f"Number of pmid that don't have fulltext {len(all_pmid)}")
            pmid_set = all_pmid & elsevier_pmids #str
            pmid_list = list(pmid_set)
            random.shuffle(pmid_list)
            with open(pmid_list_file, "wb") as f:
                pickle.dump(pmid_list, f)
            with open(pmid_tr_dict, "wb") as f:
                pickle.dump(pmid_id_dict, f)

            logger.info(f"Saved {len(pmid_list)} PMIDs to {pmid_list_file}")

        total = len(pmid_list)
        batch_size = 100
        with tqdm(total=total, desc="Processing PMIDs", unit="paper") as pbar:
            for i in range(0, total, batch_size):
                batch = pmid_list[i:i + batch_size]
                article_tuples = self.get_content_from_pmids(batch,
                                                             pmid_id_dict)
                filtered_article_tuple = self.filter_existing_articles(db, article_tuples)
                self.copy_into_db(db, 'text_content',
                                  filtered_article_tuple, self.tc_cols)
                pbar.update(len(batch))



@click.group()
def content():
    """Manage the text refs and content on the database."""


@content.command()
@click.argument("task", type=click.Choice(["upload", "update"]))
@click.argument("sources", nargs=-1,
                type=click.Choice(["pubmed", "pmc_oa", "manuscripts"]),
                required=False)
@click.option('-c', '--continuing', is_flag=True,
              help=('Continue uploading or updating, picking up where you left '
                    'off.'))
@click.option('-d', '--debug', is_flag=True,
              help='Run with debugging level output.')
def run(task, sources, continuing, debug):
    """Upload/update text refs and content on the database.

    \b
    Usage tasks are:
     - upload: use if the knowledge bases have not yet been added.
     - update: if they have been added, but need to be updated.

    The currently available sources are "pubmed", "pmc_oa", and "manuscripts".
    """
    # Import what is needed.
    from indra_db.util import get_db

    content_managers = [Pubmed, PmcOA, Manuscripts]
    db = get_db('primary')

    # Set the debug level.
    if debug:
        import logging
        from indra_db.databases import logger as db_logger
        logger.setLevel(logging.DEBUG)
        db_logger.setLevel(logging.DEBUG)

    # Define which sources we touch, just once
    if not sources:
        sources = {cm.my_source for cm in content_managers}
    else:
        sources = set(sources)
    selected_managers = (CM for CM in content_managers
                         if CM.my_source in sources)

    # Perform the task.
    for ContentManager in selected_managers:
        if task == 'upload':
            print(f"Uploading {ContentManager.my_source}.")
            ContentManager().populate(db, continuing)
        elif task == 'update':
            print(f"Updating {ContentManager.my_source}")
            ContentManager().update(db)


@content.command()
@click.option("--edge-file", type=click.Path(exists=True), required=False,
              help="Path to the INDRA CoGEx PubMed Journal edge file.")
@click.option("--node-file", type=click.Path(exists=True), required=False,
              help="Path to the INDRA CoGEx PubMed Journal node file.")
def update_elsevier(edge_file, node_file):
    """Run Elsevier update_by_cogex from the command line.

    The goal is to find articles that are Elsevier-published
    based on the journal's NLM ID and the curated list of relevant
    Elsevier journals. The node and edge files are typically
    inside the cogex data folder under the pubmed/journal path.
    """
    from indra_db.util import get_db
    db = get_db('primary')
    manager = Elsevier()
    manager.update_by_cogex(db=db, edge_file=edge_file, node_file=node_file)



@content.command('list')
@click.option('-l', '--long', is_flag=True,
              help="Include a list of the most recently added content for all "
                   "source types.")
def show_list(long):
    """List the current knowledge sources and their status."""
    # Import what is needed.
    import tabulate
    from sqlalchemy import func
    from indra_db.util import get_db

    content_managers = [Pubmed, PmcOA, Manuscripts]
    db = get_db('primary')

    # Generate the rows.
    source_updates = {cm.my_source: format_date(cm.get_latest_update(db))
                      for cm in content_managers}
    if long:
        print("This may take a while...", end='', flush=True)
        q = (db.session.query(db.TextContent.source,
                              func.max(db.TextContent.insert_date))
                       .group_by(db.TextContent.source))
        rows = [(src, source_updates.get(src, '-'), format_date(last_insert))
                for src, last_insert in q.all()]
        headers = ('Source', 'Last Updated', 'Latest Insert')
        print("\r", end='')
    else:
        rows = [(k, v) for k, v in source_updates.items()]
        headers = ('Source', 'Last Updated')
    print(tabulate.tabulate(rows, headers, tablefmt='simple'))
