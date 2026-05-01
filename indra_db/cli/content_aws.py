import gzip
import os
import re
import csv

import logging
import pickle
import xml.etree.ElementTree as ET



from os import path, remove, rename, listdir

from indra.util import zip_string
from indra.util import UnicodeXMLTreeBuilder as UTB

from indra_db.databases import texttypes, formats
from indra_db.util.data_gatherer import DataGatherer, DGContext
from sqlalchemy import exists, and_
from sqlalchemy.orm import aliased

from .content import ContentManager
import json
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError


try:
    from psycopg2 import DatabaseError
except ImportError:
    class DatabaseError(object):
        "Using this in a try-except will catch nothing. (That's the point.)"
        pass

logger = logging.getLogger(__name__)

THIS_DIR = path.dirname(path.abspath(__file__))


gatherer = DataGatherer('content', ['refs', 'content'])


class UploadError(Exception):
    pass


class _NihAwsClient(object):
    """High level access to the PMC AWS Open Data bucket.

    Parameters
    ----------
    bucket : str
        The S3 bucket name. By default this is 'pmc-oa-opendata'.
    region_name : str
        AWS region for the bucket. By default this is 'us-east-1'.
    local : bool
        These methods may be run on a local directory, intended for testing.
    local_root : str
        Local root directory used when local=True.
    """
    inventory_prefix = 'inventory-reports/pmc-oa-opendata/metadata/'
    article_prefix_patt = re.compile(r'^(PMC\d+)\.(\d+)$')
    metadata_key_patt = re.compile(r'^metadata/(PMC\d+)\.(\d+)\.json$')

    def __init__(self, bucket='pmc-oa-opendata', region_name='us-east-1',
                 local=False, local_root=None):
        self.bucket = bucket
        self.region_name = region_name
        self.is_local = local
        self.local_root = local_root

        if self.is_local:
            if self.local_root is None:
                raise ValueError('local_root must be given when local=True.')
            self.s3 = None
        else:
            self.s3 = boto3.client(
                's3',
                region_name=self.region_name,
                config=Config(signature_version=UNSIGNED)
            )
        return

    def _clean_key(self, key):
        """Normalize an S3 key, URL, or inventory URL into a bucket-relative key."""
        if key.startswith('s3://'):
            # s3://pmc-oa-opendata/PMC123.1/PMC123.1.xml?md5=...
            key = key.split('://', 1)[1]
            parts = key.split('/', 1)
            if len(parts) == 1:
                return ''
            key = parts[1]
        key = key.split('?', 1)[0]
        return key.lstrip('/')

    def _local_path(self, key):
        return path.join(self.local_root, self._clean_key(key))

    def key_exists(self, key):
        """Return True if the object key exists."""
        key = self._clean_key(key)

        if self.is_local:
            return path.exists(self._local_path(key))

        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as err:
            code = err.response.get('Error', {}).get('Code')
            if code in {'404', 'NoSuchKey', 'NotFound'}:
                return False
            raise

    def get_file(self, key, force_str=True, encoding='utf8'):
        """Get the contents of an S3 object as a string or bytes."""
        key = self._clean_key(key)

        if self.is_local:
            with open(self._local_path(key), 'rb') as f:
                ret = f.read()
        else:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            ret = obj['Body'].read()

        if force_str and isinstance(ret, bytes):
            ret = ret.decode(encoding)
        return ret

    def get_json(self, json_key):
        """Get a JSON file as a dict."""
        return json.loads(self.get_file(json_key))

    def get_xml_file(self, xml_key):
        """Get the content from an XML file as an ElementTree."""
        logger.info("Downloading %s" % xml_key)
        xml_bytes = self.get_file(xml_key, force_str=False)
        logger.info("Parsing XML metadata")
        return ET.XML(xml_bytes, parser=UTB())

    def download_file(self, key, dest=None):
        """Download an S3 object into a local file."""
        key = self._clean_key(key)
        name = path.basename(key)
        if dest is not None:
            name = path.join(dest, name)

        if self.is_local:
            with open(self._local_path(key), 'rb') as f_in:
                with open(name, 'wb') as f_out:
                    f_out.write(f_in.read())
        else:
            with open(name, 'wb') as f_out:
                self.s3.download_fileobj(self.bucket, key, f_out)

        return name

    def s3_ls_iter(self, prefix=None):
        """Iterate over object keys under the given prefix."""
        if prefix is None:
            prefix = ''
        prefix = self._clean_key(prefix)

        if self.is_local:
            root = self._local_path(prefix)
            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    full_path = path.join(dirpath, fname)
                    yield path.relpath(full_path, self.local_root)
            return

        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                yield obj['Key']

    def s3_ls(self, prefix=None):
        """Get a list of object keys under the given prefix."""
        return list(self.s3_ls_iter(prefix))

    def s3_ls_prefixes(self, prefix=None, delimiter='/'):
        """Get directory-like prefixes under a given prefix."""
        if prefix is None:
            prefix = ''
        prefix = self._clean_key(prefix)

        if self.is_local:
            dir_path = self._local_path(prefix)
            if not path.isdir(dir_path):
                return []
            return [
                path.join(prefix, k).rstrip('/') + '/'
                for k in listdir(dir_path)
                if path.isdir(path.join(dir_path, k))
            ]

        prefixes = []
        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(
                Bucket=self.bucket,
                Prefix=prefix,
                Delimiter=delimiter):
            prefixes.extend([
                item['Prefix'] for item in page.get('CommonPrefixes', [])
            ])
        return prefixes

    def get_csv_as_dict(self, csv_key, cols=None, infer_header=True,
                        add_fields=None):
        """Get the content from a CSV file as a list of dicts."""
        csv_str = self.get_file(csv_key)
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

    def get_latest_inventory_manifest_key(self):
        """Get the latest inventory manifest.json key."""
        manifest_keys = [
            key for key in self.s3_ls_iter(self.inventory_prefix)
            if key.endswith('/manifest.json')
        ]

        if not manifest_keys:
            raise ValueError('No inventory manifest found.')

        return sorted(manifest_keys)[-1]

    def get_inventory_csv_keys(self, manifest_key=None):
        """Get CSV.gz keys listed in an inventory manifest."""
        if manifest_key is None:
            manifest_key = self.get_latest_inventory_manifest_key()

        manifest = self.get_json(manifest_key)
        return [f['key'] for f in manifest.get('files', [])]

    def iter_inventory_rows(self, csv_key):
        """Iterate over rows from a gzipped inventory CSV.

        Each row is returned as:
            {
                'bucket': ..., e.g pmc-oa-opendata
                'key': ..., e.g metadata/PMC2000230.1.json
                'last_modified': ..., e.g 2026-02-12T02:59:59.000Z
                'etag': ... e.g dc97f78385423933f1cf04ad8c4df6e3
            }
        """
        csv_key = self._clean_key(csv_key)

        if self.is_local:
            body = open(self._local_path(csv_key), 'rb')
        else:
            obj = self.s3.get_object(Bucket=self.bucket, Key=csv_key)
            body = obj['Body']

        try:
            with gzip.GzipFile(fileobj=body) as gz:
                reader = csv.reader(
                    line.decode('utf8', errors='replace') for line in gz
                )
                for row in reader:
                    if len(row) < 4:
                        continue
                    yield {
                        'bucket': row[0],
                        'key': row[1],
                        'last_modified': row[2],
                        'etag': row[3],
                    }
        finally:
            if self.is_local:
                body.close()

    def iter_latest_inventory_rows(self):
        """Iterate over rows from the latest inventory."""
        # the csv files contain all the pmc ids
        csv_keys = self.get_inventory_csv_keys()
        for csv_key in csv_keys:
            for row in self.iter_inventory_rows(csv_key):
                yield row

    def iter_article_versions_from_inventory(self):
        """Iterate over article versions from metadata rows in inventory.

        Yields
        ------
        tuple
            (pmcid, version, article_prefix, metadata_key)
        """
        for row in self.iter_latest_inventory_rows():
            key = row['key']
            m = self.metadata_key_patt.match(key)
            if not m:
                continue
            pmcid = m.group(1)
            version = int(m.group(2))
            article_prefix = '%s.%d' % (pmcid, version)
            yield pmcid, version, article_prefix, key

    def get_latest_article_prefixes(self):
        """Get latest article prefix for each PMCID from inventory.

        Returns
        -------
        dict
            {pmcid: article_prefix}
        """
        latest = {}
        for pmcid, version, article_prefix, _ in \
                self.iter_article_versions_from_inventory():
            if pmcid not in latest or version > latest[pmcid][0]:
                latest[pmcid] = (version, article_prefix)

        return {pmcid: article_prefix
                for pmcid, (_, article_prefix) in latest.items()}

    def get_xml_key(self, article_prefix):
        """Get XML key from article prefix."""
        article_prefix = article_prefix.rstrip('/')
        return '%s/%s.xml' % (article_prefix, article_prefix)

    def get_text_key(self, article_prefix):
        """Get TXT key from article prefix. May use in later case"""
        article_prefix = article_prefix.rstrip('/')
        return '%s/%s.txt' % (article_prefix, article_prefix)

    def get_metadata_key(self, article_prefix):
        """Get metadata JSON key from article prefix."""
        article_prefix = article_prefix.rstrip('/')
        return 'metadata/%s.json' % article_prefix

    def get_versions(self, pmcid):
        """Return available versions of a PMC article.

        This uses S3 prefix listing directly and does not require inventory.
        """
        versions = []
        prefixes = self.s3_ls_prefixes(prefix='%s.' % pmcid)

        for prefix in prefixes:
            article_prefix = prefix.rstrip('/')
            m = self.article_prefix_patt.match(article_prefix)
            if m and m.group(1) == pmcid:
                versions.append(int(m.group(2)))

        return tuple(sorted(versions))

    def get_latest_article_prefix(self, pmcid):
        """Return latest article prefix for a PMCID, or None."""
        versions = self.get_versions(pmcid)
        if not versions:
            return None
        return '%s.%d' % (pmcid, versions[-1])

    def get_xml_key_for_pmcid(self, pmcid):
        """Return latest XML key for a PMCID, or None."""
        article_prefix = self.get_latest_article_prefix(pmcid)
        if article_prefix is None:
            return None
        return self.get_xml_key(article_prefix)

class _NihAwsManager(ContentManager):
    """Abstract class for managers that use the PMC AWS service.

    See `_NihAwsClient` for parameters.
    """
    def update(self, db):
        raise NotImplementedError

    def populate(self, db):
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        self.aws = _NihAwsClient(*args, **kwargs)
        super(_NihAwsManager, self).__init__()
        return


class PMC(_NihAwsManager):
    """Manager for PMC content from AWS.

    This replaces the FTP-based PmcOA and Manuscripts managers. The AWS bucket
    mixes PMC Open Access and Author Manuscript content, so we read the
    metadata JSON for each article and route each record to the old source name:
    `pmc_oa` or `manuscripts`.
    """
    my_source = 'pmc'
    primary_col = 'pmcid'
    tr_cols = ('pmid', 'pmcid', 'doi', 'manuscript_id', 'pub_year',)
    tc_cols = ('text_ref_id', 'source', 'format', 'text_type',
               'content', 'license')

    content_sources = ('pmc_oa', 'manuscripts')

    def __init__(self, *args, include_oa=True, include_manuscripts=True,
                 include_historical_ocr=True, include_retracted=False,
                 **kwargs):
        super(PMC, self).__init__(*args, **kwargs)
        self.include_oa = include_oa
        self.include_manuscripts = include_manuscripts
        self.include_historical_ocr = include_historical_ocr
        self.include_retracted = include_retracted
        return

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

        logger.debug("Found %d pmids on the database." % len(pmids_from_db))
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

    def get_source_from_metadata(self, metadata):
        """Get the old database source name from an AWS metadata JSON dict."""
        if metadata.get('is_retracted') and not self.include_retracted:
            return None

        if metadata.get('is_historical_ocr') and not self.include_historical_ocr:
            return None

        if metadata.get('is_manuscript'):
            if self.include_manuscripts:
                return 'manuscripts'
            return None

        if metadata.get('is_pmc_openaccess'):
            if self.include_oa:
                return 'pmc_oa'
            return None

        return None

    def get_license(self, pmcid, metadata=None):
        """Get the license for this pmcid."""
        if metadata is not None:
            return metadata.get('license_code')
        return None

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

        self.get_missing_pmids(db, tr_data)
        primary_id_types = ['pmid', 'pmcid', 'manuscript_id']

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

        logger.info('Adding %d new text refs...' % len(filtered_tr_records))
        self.upload_text_refs(db, filtered_tr_records)
        gatherer.add('refs', len(filtered_tr_records))

        filtered_tc_records = self.filter_text_content(db, mod_tc_data)

        logger.info('Adding %d more text content entries...' %
                    len(filtered_tc_records))
        new_trids = self.upload_text_content(db, filtered_tc_records)
        gatherer.add('content', len(new_trids))
        return

    def upload_batch_for_source(self, db, source, tr_data, tc_data):
        """Upload a batch using a specific text_content source."""
        old_source = self.my_source
        self.my_source = source
        try:
            return self.upload_batch(db, tr_data, tc_data)
        finally:
            self.my_source = old_source

    def get_data_from_xml_str(self, xml_str, filename, metadata=None):
        """Get the data out of the xml string."""
        try:
            tree = ET.XML(xml_str.encode('utf8'))
        except ET.ParseError:
            logger.info("Could not parse %s. Skipping." % filename)
            return None

        id_data = {
            e.get('pub-id-type'): e.text for e in
            tree.findall('.//article-id')
            }

        if metadata is not None:
            if id_data.get('pmcid') is None:
                id_data['pmcid'] = metadata.get('pmcid')
            if id_data.get('pmid') is None and metadata.get('pmid') is not None:
                id_data['pmid'] = str(metadata.get('pmid'))
            if id_data.get('doi') is None:
                id_data['doi'] = metadata.get('doi')
            if metadata.get('mid') is not None:
                id_data['manuscript_id'] = metadata.get('mid')

        if 'pmcid' not in id_data.keys() or id_data['pmcid'] is None:
            pmcid = filename.split('/')[0].split('.')[0]
            id_data['pmcid'] = pmcid
            logger.info('Processing XML for %s.' % pmcid)

        if 'manuscript' in id_data.keys():
            id_data['manuscript_id'] = id_data['manuscript']

        year = None
        for elem in tree.findall('.//article-meta/pub-date'):
            year_elem = elem.find('year')
            if year_elem is not None:
                try:
                    year = int(year_elem.text)
                except (TypeError, ValueError):
                    year = None
                break

        tr_datum_raw = {k: id_data.get(k) for k in self.tr_cols}
        tr_datum = {k: val.strip().upper() if isinstance(val, str) else val
                    for k, val in tr_datum_raw.items()}
        tr_datum['pub_year'] = year

        tc_datum = {
            'pmcid': id_data['pmcid'],
            'text_type': texttypes.FULLTEXT,
            'license': self.get_license(id_data['pmcid'], metadata),
            'content': zip_string(xml_str)
            }
        return tr_datum, tc_datum

    def get_latest_metadata_keys(self):
        """Get latest metadata JSON key for each PMCID.

        Returns
        -------
        dict
            A dictionary from PMCID to metadata key.
        """
        latest_prefixes = self.aws.get_latest_article_prefixes()
        return {pmcid: self.aws.get_metadata_key(article_prefix)
                for pmcid, article_prefix in latest_prefixes.items()}

    def iter_metadata(self, pmcid_set=None):
        """Iterate over latest metadata JSON objects.

        Parameters
        ----------
        pmcid_set : Optional[set[str]]
            If given, only metadata for these PMCIDs will be returned.

        Yields
        ------
        metadata_key : str
            The S3 key for the metadata JSON.
        metadata : dict
            The parsed metadata JSON.
        """
        latest_metadata_keys = self.get_latest_metadata_keys()

        for pmcid, metadata_key in latest_metadata_keys.items():
            if pmcid_set is not None and pmcid not in pmcid_set:
                continue

            try:
                metadata = self.aws.get_json(metadata_key)
            except Exception as err:
                logger.warning("Could not load metadata %s: %s",
                               metadata_key, err)
                continue

            yield metadata_key, metadata

    def iter_xmls(self, pmcid_set=None):
        """Iterate over the xmls in the AWS bucket.

        Parameters
        ----------
        pmcid_set : Optional[set[str]]
            A set of PMCIDs whose content you want returned. If None, all
            latest PMC article versions found in inventory will be considered.

        Yields
        ------
        label : Tuple
            A key representing the particular XML: (XML Key, Entry Number,
            Total Entries). The total may be 0 when streaming.
        source : str
            The database source name, e.g., 'pmc_oa' or 'manuscripts'.
        xml_name : str
            The S3 key of the XML file.
        xml_str : str
            The XML string.
        metadata : dict
            The parsed AWS metadata JSON.
        """
        for idx, (metadata_key, metadata) in enumerate(
                self.iter_metadata(pmcid_set=pmcid_set)):

            source = self.get_source_from_metadata(metadata)
            if source is None:
                continue

            xml_url = metadata.get('xml_url')
            if xml_url:
                xml_key = self.aws._clean_key(xml_url)
            else:
                article_prefix = '%s.%s' % (metadata['pmcid'],
                                            metadata['version'])
                xml_key = self.aws.get_xml_key(article_prefix)

            try:
                xml_str = self.aws.get_file(xml_key)
            except Exception as err:
                logger.warning("Could not download XML %s from %s: %s",
                               xml_key, metadata_key, err)
                continue

            yield (xml_key, idx, 0), source, xml_key, xml_str, metadata

    def iter_contents(self, pmcid_set=None):
        """Iterate over the files in AWS, yielding ref and content data.

        Parameters
        ----------
        pmcid_set : Optional[set[str]]
            A set of PMCIDs whose content you want returned.

        Yields
        ------
        label : tuple
            A key representing the particular XML: (XML Key, Entry Number,
            Total Entries).
        source : str
            The database source name.
        text_ref_dict : dict
            A dictionary containing the text ref information.
        text_content_dict : dict
            A dictionary containing the text content information.
        """
        xml_iter = self.iter_xmls(pmcid_set=pmcid_set)
        for label, source, file_name, xml_str, metadata in xml_iter:
            logger.info(f"Getting data from {file_name}")
            res = self.get_data_from_xml_str(xml_str, file_name,
                                             metadata=metadata)
            if res is None:
                continue
            tr, tc = res
            logger.info(f"Yielding ref and content for {file_name}.")
            yield label, source, tr, tc

    def upload_xmls(self, db, pmcid_set=None, batch_size=10000):
        """Do the grunt work of downloading and processing XMLs from AWS.

        Parameters
        ----------
        db : :py:class:`PrincipalDatabaseManager <indra_db.databases.PrincipalDatabaseManager>`
            A handle to the principal database.
        pmcid_set : set[str]
            A set of PMC Ids to include.
        batch_size : Optional[int]
            Default is 10,000. The number of pieces of content to submit to the
            database at a time.
        """
        source_batches = {
            source: {'labels': [], 'trs': [], 'tcs': []}
            for source in self.content_sources
            }

        for label, source, tr, tc in self.iter_contents(pmcid_set=pmcid_set):
            batch = source_batches[source]
            batch['labels'].append(label)
            batch['trs'].append(tr)
            batch['tcs'].append(tc)

            if len(batch['trs']) >= batch_size:
                self._flush_source_batch(db, source, batch)

        for source, batch in source_batches.items():
            if batch['trs']:
                self._flush_source_batch(db, source, batch)

        return True

    def _flush_source_batch(self, db, source, batch):
        """Upload one source-specific batch and record processed XML keys."""
        logger.info("Uploading %d %s records.",
                    len(batch['trs']), source)

        self.upload_batch_for_source(db, source, batch['trs'], batch['tcs'])

        for label in batch['labels']:
            xml_key = label[0]
            sf_list = db.select_all(
                db.SourceFile,
                db.SourceFile.source == source,
                db.SourceFile.name == xml_key
                )
            if not sf_list:
                db.insert('source_file', source=source, name=xml_key)

        batch['labels'].clear()
        batch['trs'].clear()
        batch['tcs'].clear()
        return

    def find_all_pmcids_need_update(self, db, scope=None):
        """Find PMC IDs available from AWS that are not in the DB.

        Parameters
        ----------
        db : DatabaseManager
        scope : Optional[str]
            The value is either None (default) to return PMC IDs not present
            at all for PMC sources or "fulltext" to return PMC IDs that
            EXIST but only have ABSTRACT, i.e. are still missing FULLTEXT.
        """
        logger.info("Getting list of PMC content available.")
        aws_pmcid_set = set(self.aws.get_latest_article_prefixes().keys())

        if scope == "fulltext":
            tc_full = aliased(db.TextContent)

            pmcids_needing_fulltext = {
                pmcid for (pmcid,) in (
                    db.session.query(db.TextRef.pmcid)
                    .filter(
                        db.TextRef.pmcid.isnot(None),
                        ~exists().where(
                            and_(
                                tc_full.text_ref_id == db.TextRef.id,
                                tc_full.text_type == texttypes.FULLTEXT
                            )
                        )
                    )
                )
            }

            load_pmcid_set = aws_pmcid_set & pmcids_needing_fulltext
            logger.info("Need FULLTEXT for %d PMCIDs.",
                        len(load_pmcid_set))

        else:
            logger.info("Getting a list of text refs that already correspond "
                        "to PMC content.")
            tr_list = db.select_all(
                db.TextRef,
                db.TextRef.id == db.TextContent.text_ref_id,
                db.TextContent.source.in_(self.content_sources)
                )
            existing_pmcids = {tr.pmcid for tr in tr_list}
            load_pmcid_set = aws_pmcid_set - existing_pmcids

            logger.info("There are %d PMC articles to load."
                        % (len(load_pmcid_set)))

        return load_pmcid_set

    @ContentManager._record_for_review
    @DGContext.wrap(gatherer, 'pmc')
    def update(self, db):
        """Update the contents of the database with the latest PMC articles."""
        logger.info("Finding all the PMC IDs that need to be updated.")
        load_pmcid_set = self.find_all_pmcids_need_update(db)

        logger.info("Beginning to upload XMLs from AWS.")
        self.upload_xmls(db, pmcid_set=load_pmcid_set)
        return True

    @ContentManager._record_for_review
    @DGContext.wrap(gatherer, 'pmc')
    def populate(self, db, continuing=False):
        """Perform the initial population of the pmc content into the database.

        Parameters
        ----------
        db : indra.db.DatabaseManager instance
            The database to which the data will be uploaded.
        continuing : bool
            If true, assume that we are picking up after an error, or otherwise
            continuing from an earlier process. This means we will skip over
            source files contained in the database. If false, all files will be
            read and parsed.

        Returns
        -------
        completed : bool
            If True, an update was completed. Othewise, the upload was aborted
            for some reason, often because the upload was already completed
            at some earlier time.
        """
        pmcid_set = None

        if continuing:
            sf_list = db.select_all(
                db.SourceFile,
                db.SourceFile.source.in_(self.content_sources)
                )
            processed_xml_keys = {sf.name for sf in sf_list}

            latest_prefixes = self.aws.get_latest_article_prefixes()
            pmcid_set = {
                pmcid for pmcid, article_prefix in latest_prefixes.items()
                if self.aws.get_xml_key(article_prefix) not in processed_xml_keys
                }

            if not pmcid_set:
                logger.info("No XMLs to load. All done.")
                return False

        self.upload_xmls(db, pmcid_set=pmcid_set)
        return True