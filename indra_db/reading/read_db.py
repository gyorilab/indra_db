"""This module provides essential tools to run reading using indra's own
database. This may also be run as a script; for details run:
`python read_pmids_db --help`
"""

import json
import pickle
import random
import logging
import re
from datetime import datetime
from math import ceil
from multiprocessing.pool import Pool

from indra.statements import make_hash

from indra.tools.reading.util.script_tools import get_parser
from indra.util.get_version import get_version as get_indra_version
from indra.literature.elsevier_client import extract_text as process_elsevier
from indra.tools.reading.readers import ReadingData, _get_dir, get_reader, \
    Content, Reader, EmptyReader
from indra.util import zip_string, batch_iter

from indra_db import get_primary_db, formats
from indra_db.managers.database_manager import readers, reader_versions
from indra_db.util import insert_raw_agents, unpack

logger = logging.getLogger(__name__)


class ReadDBError(Exception):
    pass


class DatabaseReadingData(ReadingData):
    """This version of ReadingData adds valuable methods for database ops.

    In particular, this adds methods that help in the packaging of the content
    for copy into the database.

    Parameters
    ----------
    tcid : int
        The unique text content id provided by the database.
    reader_name : str
        The name of the reader, consistent with it's `name` attribute, for
        example: 'REACH'
    reader_version : str
        A string identifying the version of the underlying nlp reader.
    content_format : str
        The format of the content. Options are in indra.db.formats.
    content : str or dict
        The content of the reading result. A string in the format given by
        `content_format`.
    reading_id : int
        (optional) The unique integer id given to each reading result. In
        practice, this is often assigned
    """
    def __init__(self, tcid, reader_name, reader_version, content_format,
                 content, reading_id=None):
        super(DatabaseReadingData, self).__init__(tcid, reader_name,
                                                  reader_version,
                                                  content_format, content)
        self.tcid = tcid
        self.reading_id = reading_id
        return

    @classmethod
    def from_db_reading(cls, db_reading):
        """Construct a DatabaseReadingData object from an entry in the database

        As returned by SQL Alchemy.
        """
        if db_reading.format == 'json':
            content = json.loads(unpack(db_reading.bytes))
        else:
            content = unpack(db_reading.bytes)
        return cls(db_reading.text_content_id, db_reading.reader,
                   db_reading.reader_version, db_reading.format,
                   content, db_reading.id)

    @staticmethod
    def get_cols():
        """Get the columns for the tuple returned by `make_tuple`."""
        return ('id', 'text_content_id', 'reader', 'reader_version', 'format',
                'bytes', 'batch_id')

    def zip_content(self):
        """Compress the content, returning bytes."""
        if self.content is None:
            return None

        if self.format == formats.JSON:
            ret = zip_string(json.dumps(self.content))
        elif self.format == formats.TEXT or self.format == formats.EKB:
            ret = zip_string(self.content)
        else:
            raise Exception('Do not know how to zip format %s.' % self.format)
        return ret

    def make_tuple(self, batch_id):
        """Make the tuple expected by the database."""
        return (self.get_id(), self.content_id, self.reader,
                self.reader_version, self.format, self.zip_content(), batch_id)

    def get_id(self):
        if self.reading_id is None:
            self.reading_id = readers[self.reader.upper()]*10e12
            self.reading_id += (reader_versions[self.reader.lower()]
                                .index(self.reader_version)*10e10)
            self.reading_id += self.tcid
        return self.reading_id

    def matches(self, r_entry):
        """Determine if reading data matches the a reading entry from the db.

        Returns True if tcid, reader, reader_version match the corresponding
        elements of a db.Reading instance, else False.
        """
        # Note the temporary fix in clipping the reader version length. This is
        # because the version is for some reason clipped in the database.
        return (r_entry.text_content_id == self.content_id
                and r_entry.reader == self.reader
                and r_entry.reader_version == self.reader_version[:20])


class DatabaseStatementData(object):
    """Contains metadata for statements, as well as the statement itself.

    This, like ReadingData, is primarily designed for use with the database,
    carrying valuable information and methods for such.

    Parameters
    ----------
    statement : an indra Statement instance
        The statement whose extra meta data this object encapsulates.
    reading_id : int or None
        The id number of the entry in the `readings` table of the database.
        None if no such id is available.
    """
    def __init__(self, statement, reading_id=None, db_info_id=None):
        self.reading_id = reading_id
        self.db_info_id = db_info_id
        self.statement = statement
        self.indra_version = get_indra_version()
        self.__text_patt = re.compile('[\W_]+')
        return

    def _get_text_hash(self):
        ev = self.statement.evidence[0]
        simple_text = self.__text_patt.sub('', ev.text)
        if 'coords' in ev.annotations.keys():
            simple_text += str(ev.annotations['coords'])
        return make_hash(simple_text.lower(), 16)

    @staticmethod
    def get_cols():
        """Get the columns for the tuple returned by `make_tuple`."""
        return 'batch_id', 'reading_id', 'db_info_id', 'uuid', 'mk_hash', \
               'source_hash', 'type', 'json', 'indra_version', 'text_hash'

    def make_tuple(self, batch_id):
        """Make a tuple for copying into the database."""
        return (batch_id, self.reading_id, self.db_info_id,
                self.statement.uuid, self.statement.get_hash(),
                self.statement.evidence[0].get_source_hash(),
                self.statement.__class__.__name__,
                json.dumps(self.statement.to_json()), self.indra_version,
                self._get_text_hash())


def get_stmts_safely(reading_data):
    stmt_data_list = []
    try:
        stmts = reading_data.get_statements()
    except Exception as e:
        logger.error("Got exception creating statements for %d."
                     % reading_data.reading_id)
        logger.exception(e)
        return []
    if stmts is not None:
        if not len(stmts):
            logger.debug("Got no statements for %s." % reading_data.reading_id)
        for stmt in stmts:
            stmt.evidence[0].pmid = None
            stmt_data = DatabaseStatementData(stmt, reading_data.reading_id)
            stmt_data_list.append(stmt_data)
    else:
        logger.warning("Got None statements for %s." % reading_data.reading_id)
    return stmt_data_list


def make_statements(reading_data_list, num_proc=1):
    """Convert a list of ReadingData instances into StatementData instances."""
    stmt_data_list = []

    if num_proc is 1:  # Don't use pool if not needed.
        for reading_data in reading_data_list:
            stmt_data_list += get_stmts_safely(reading_data)
    else:
        pool = Pool(num_proc)
        try:
            stmt_data_list_list = pool.map(get_stmts_safely, reading_data_list)
            for stmt_data_sublist in stmt_data_list_list:
                stmt_data_list += stmt_data_sublist
        finally:
            pool.close()
            pool.join()

    logger.info("Found %d statements from %d readings." %
                (len(stmt_data_list), len(reading_data_list)))
    return stmt_data_list


class DatabaseReader(object):
    """An class to run readings utilizing the database.

    Parameters
    ----------
    tcids : iterable of ints
        An iterable (set, list, tuple, generator, etc) of integers referring to
        the primary keys of text content in the database.
    reader : Reader
        An INDRA Reader object.
    verbose : bool
        Optional, default False - If True, log and print the output of the
        commandline reader utilities, if False, don't.
    reading_mode : str : 'all', 'unread', or 'none'
        Optional, default 'undread' - If 'all', read everything (generally
        slow); if 'unread', only read things that were unread, (the cache of old
        readings may still be used if `stmt_mode='all'` to get everything); if
        'none', don't read, and only retrieve existing readings.
    stmt_mode : str : 'all', 'unread', or 'none'
        Optional, default 'all' - If 'all', produce statements for all content
        for all readers. If the readings were already produced, they will be
        retrieved from the database if `read_mode` is 'none' or 'unread'. If
        this option is 'unread', only the newly produced readings will be
        processed. If 'none', no statements will be produced.
    batch_size : int
        Optional, default 1000 - The number of text content entries to be
        yielded by the database at a given time.
    db : indra_db.DatabaseManager instance
        Optional, default is None, in which case the primary database provided
        by `get_primary_db` function is used. Used to interface with a
        different database.
    """
    def __init__(self, tcids, reader, verbose=True, reading_mode='unread',
                 stmt_mode='all', batch_size=1000, db=None, n_proc=1):
        self.tcids = tcids
        self.reader = reader
        self.reader.reset()
        self.verbose = verbose
        self.reading_mode = reading_mode
        if isinstance(reader, EmptyReader):
            self.reading_mode = 'none'
        self.stmt_mode = stmt_mode
        self.batch_size = batch_size
        self.n_proc = n_proc
        if db is None:
            self._db = get_primary_db()
        else:
            self._db = db
        self._tc_rd_link = \
            self._db.TextContent.id == self._db.Reading.text_content_id
        logger.info("Instantiating reading handler for reader %s with version "
                    "%s using reading mode %s and statement mode %s for %d "
                    "tcids." % (reader.name, reader.version, reading_mode,
                                stmt_mode, len(tcids)))

        # To be filled.
        self.extant_readings = []
        self.new_readings = []
        self.statement_outputs = []
        self.starts = {}
        self.stops = {}
        return

    def iter_over_content(self):
        # Get the text content query object
        tc_query = self._db.filter_query(
            self._db.TextContent,
            self._db.TextContent.id.in_(self.tcids)
            )

        if self.reading_mode != 'all':
            logger.debug("Getting content to be read.")
            # Each sub query is a set of content that has been read by one of
            # the readers.
            tc_sub_q = tc_query.filter(
                self._tc_rd_link,
                self._db.Reading.reader == self.reader.name,
                self._db.Reading.reader_version == self.reader.version[:20]
                )

            # Now let's exclude all of those.
            tc_tbr_query = tc_query.except_(tc_sub_q)
        else:
            logger.debug('All content will be read (force_read).')
            tc_tbr_query = tc_query

        for tc in tc_tbr_query.distinct().yield_per(self.batch_size):
            processed_content = process_content(tc)
            if processed_content is not None:
                yield processed_content
        return

    def _make_new_readings(self, **kwargs):
        """Read contents retrieved from the database.

        The content will be retrieved in batches, given by the `batch` arg.
        This prevents the system RAM from being overloaded.

        Keyword arguments are passed to the `read` methods of the readers.

        Returns
        -------
        outputs : list of ReadingData instances
            The results of the readings with relevant metadata.
        """
        logger.info("Creating new readings from the database for %s."
                    % self.reader.name)
        self.starts['new_readings'] = datetime.utcnow()
        # Iterate
        logger.debug("Beginning to iterate.")
        self.reader.read(self.iter_over_content(), **kwargs)
        if self.reader.results:
            self.new_readings.extend(self.reader.results)
        logger.debug("Finished iteration.")

        self.stops['new_readings'] = datetime.utcnow()
        logger.info("Made %d new readings." % len(self.new_readings))
        return

    def _get_prior_readings(self):
        """Get readings from the database."""
        logger.info("Loading pre-existing readings from the database for %s."
                    % self.reader.name)
        self.starts['old_readings'] = datetime.utcnow()
        db = self._db
        if self.tcids:
            logger.info("Looking for content matching reader %s, version %s."
                        % (self.reader.name, self.reader.version[:20]))
            readings_query = db.filter_query(
                db.Reading,
                db.Reading.reader == self.reader.name,
                db.Reading.reader_version == self.reader.version[:20],
                db.Reading.text_content_id.in_(self.tcids)
                )
            for r in readings_query.yield_per(self.batch_size):
                self.extant_readings.append(
                    DatabaseReadingData.from_db_reading(r)
                    )
        logger.info("Found %d pre-existing readings."
                    % len(self.extant_readings))
        self.stops['old_readings'] = datetime.utcnow()
        return

    def dump_readings_to_db(self):
        """Put the reading output on the database."""
        logger.info("Beginning to dump %d readings for %s to the database."
                    % (len(self.new_readings), self.reader.name))
        self.starts['dump_readings_db'] = datetime.utcnow()
        if not self.new_readings:
            logger.info("No new readings to load.")
            self.stops['dump_readings_db'] = datetime.utcnow()
            return

        db = self._db

        # Get the id for this batch of uploads.
        batch_id = db.make_copy_batch_id()

        # Make a list of data to copy, ensuring there are no conflicts.
        upload_list = []
        rd_dict = {}
        for rd in self.new_readings:
            # If there were no conflicts, we can add this to the copy list.
            upload_list.append(rd.make_tuple(batch_id))
            rd_dict[(rd.tcid, rd.reader, rd.reader_version[:20])] = rd

        # Copy into the database.
        logger.info("Adding %d/%d reading entries to the database." %
                    (len(upload_list), len(self.new_readings)))
        if upload_list:
            is_all = self.reading_mode == 'all'
            db.copy('reading', upload_list,
                    DatabaseReadingData.get_cols(), lazy=is_all,
                    push_conflict=is_all)

            # Update the reading_data objects with their reading_ids.
            rdata = db.select_all([db.Reading.id, db.Reading.text_content_id,
                                   db.Reading.reader, db.Reading.reader_version],
                                  db.Reading.batch_id == batch_id)
            for tpl in rdata:
                rd_dict[tuple(tpl[1:])].reading_id = tpl[0]

        self.stops['dump_readings_db'] = datetime.utcnow()
        return

    def dump_readings_to_pickle(self, pickle_file):
        """Dump the reading results into a pickle file."""
        logger.info("Beginning to dump %d readings for %s to %s."
                    % (len(self.new_readings), self.reader.name, pickle_file))
        self.starts['dump_readings_pkl'] = datetime.utcnow()
        with open(pickle_file, 'wb') as f:
            rdata = [output.make_tuple(None)
                     for output in self.new_readings + self.extant_readings]
            pickle.dump(rdata, f)
            logger.info("Reading outputs pickled in: %s" % pickle_file)

        self.stops['dump_readings_pkl'] = datetime.utcnow()
        return

    def get_readings(self):
        """Get the reading output for the given ids."""
        # Get a database instance.
        logger.info("Producing readings for %s in %s mode."
                    % (self.reader.name, self.reading_mode))

        # Handle the cases where I need to retrieve old readings.
        if self.reading_mode != 'all' and self.stmt_mode == 'all':
            self._get_prior_readings()

        # Now produce any new readings that need to be produced.
        if self.reading_mode != 'none':
            self._make_new_readings()

        return

    def dump_statements_to_db(self):
        """Upload the statements to the database."""
        self.starts['dump_statements_db'] = datetime.utcnow()
        logger.info("Uploading %d statements to the database." %
                    len(self.statement_outputs))
        batch_id = self._db.make_copy_batch_id()

        # Find and filter out duplicate statements.
        stmt_tuples = {}
        stmts = []
        dups = {}
        for sd in self.statement_outputs:
            tpl = sd.make_tuple(batch_id)
            key = (tpl[1], tpl[4], tpl[9])
            if key in stmt_tuples.keys():
                logger.warning('Duplicate key found: %s.' % str(key))
                if key in dups.keys():
                    dups[key].append(tpl)
                else:
                    dups[key] = [tpl]
            else:
                stmt_tuples[key] = tpl
                stmts.append(sd.statement)

        # Dump the good statements into the raw statements table.
        self._db.copy('raw_statements', stmt_tuples.values(),
                      DatabaseStatementData.get_cols(), lazy=True,
                      push_conflict=True,
                      constraint='reading_raw_statement_uniqueness',
                      commit=False)

        # Dump the duplicates into a separate to all for debugging.
        self._db.copy('rejected_statements', [tpl for dlist in dups.values()
                                              for tpl in dlist],
                      DatabaseStatementData.get_cols())

        # Add the agents for the accepted statements.
        logger.info("Uploading agents to the database.")
        if len(stmts):
            insert_raw_agents(self._db, batch_id, stmts, verbose=False)
        self.stops['dump_statements_db'] = datetime.utcnow()
        return

    def dump_statements_to_pickle(self, pickle_file):
        """Dump the statements into a pickle file."""
        self.starts['dump_statements_pkl'] = datetime.utcnow()
        with open(pickle_file, 'wb') as f:
            pickle.dump([sd.statement for sd in self.statement_outputs], f)
        print("Statements pickled in %s." % pickle_file)
        self.stops['dump_readings_pkl'] = datetime.utcnow()
        return

    def get_statements(self):
        """Convert the reader output into a list of StatementData instances."""
        self.starts['make_statements'] = datetime.utcnow()
        if self.stmt_mode == 'all':
            all_outputs = self.new_readings + self.extant_readings
            self.statement_outputs = make_statements(all_outputs, self.n_proc)
        elif self.stmt_mode == 'unread':
            self.statement_outputs = make_statements(self.new_readings,
                                                     self.n_proc)
        self.stops['make_statements'] = datetime.utcnow()
        return


# =============================================================================
# Content Retrieval
# =============================================================================
def process_content(text_content):
    """Get the appropriate content object from the text content."""
    if text_content.format == formats.TEXT:
        cont_fmt = 'txt'
    elif (text_content.source in ['pmc_oa', 'manuscripts']
          and text_content.format == formats.XML):
        cont_fmt = 'nxml'
    else:
        cont_fmt = text_content.format
    content = Content.from_string(text_content.id, cont_fmt,
                                  text_content.content, compressed=True,
                                  encoded=True)
    if text_content.source == 'elsevier':
        raw_xml_text = content.get_text()
        elsevier_text = process_elsevier(raw_xml_text)
        if elsevier_text is None:
            logger.warning("Could not extract text from Elsevier xml for "
                           "tcid: %d" % text_content.id)
            return None
        content = Content.from_string(content.get_id(), 'text', elsevier_text)
    return content


# =============================================================================
# High level functions
# =============================================================================

def construct_readers(reader_names, **kwargs):
    """Construct the Reader objects from the names of the readers."""
    readers = []
    for reader_name in reader_names:
        if 'ResultClass' not in kwargs.keys():
            kwargs['ResultClass'] = DatabaseReadingData
        reader = get_reader(reader_name, **kwargs)
        readers.append(reader)
    return readers


def run_reading(readers, tcids, verbose=True, reading_mode='unread',
                stmt_mode='all', batch_size=1000, reading_pickle=None,
                stmts_pickle=None, upload_readings=True, upload_stmts=True,
                db=None):
    """Run the reading with the given readers on the given text content ids."""
    workers = []
    for reader in readers:
        logger.info("Beginning reading for %s." % reader.name)
        db_reader = DatabaseReader(tcids, reader, verbose, stmt_mode=stmt_mode,
                                   reading_mode=reading_mode, db=db,
                                   batch_size=batch_size)
        workers.append(db_reader)
        db_reader.get_readings()
        if upload_readings:
            db_reader.dump_readings_to_db()
        if reading_pickle:
            db_reader.dump_readings_to_pickle(reading_pickle)

        if stmt_mode != 'none':
            db_reader.get_statements()
            if upload_stmts:
                db_reader.dump_statements_to_db()
            if stmts_pickle:
                db_reader.dump_statements_to_pickle(reader.name + '_'
                                                    + stmts_pickle)
    return workers


# =============================================================================
# Main for script use
# =============================================================================
def make_parser():
    parser = get_parser(
        'A tool to read and process content from the database.',
        ('A file containing a list of ids of the form <id_type>:<id>. '
         'Note that besided the obvious id types (pmid, pmcid, doi, etc.), '
         'you may use trid and tcid to indicate text ref and text content '
         'ids, respectively. Note that these are specific to the database, '
         'and should thus be used with care.')
    )
    parser.add_argument(
        '-m', '--reading_mode',
        choices=['all', 'unread', 'none'],
        default='unread',
        help=("Set the reading mode. If 'all', read everything, if "
              "'unread', only read content that does not have pre-existing "
              "readings of the same reader and version, if 'none', only "
              "use pre-existing readings. Default is 'unread'.")
    )
    parser.add_argument(
        '-S', '--stmt_mode',
        choices=['all', 'unread', 'none'],
        default='all',
        help=("Choose which readings should produce statements. If 'all', all "
              "readings that are produced or retrieved will be used to make "
              "statements. If 'unread', only produce statements from "
              "previously unread content. If 'none', do not produce any "
              "statements (only readings will be produced).")
    )
    parser.add_argument(
        '-t', '--temp',
        default='.',
        help='Select the location of the temp file.'
    )
    parser.add_argument(
        '-o', '--output',
        dest='name',
        help=('Pickle all results and save in files labelled as '
              '<NAME>_<output_type>.pkl.'),
        default=None
    )
    parser.add_argument(
        '-b', '--inner_batch',
        dest='b_in',
        help=('Choose the size of the inner batches, which is the number of '
              'text content entires loaded at a given time, and the number of '
              'entries that are read at a time by a reader. The default is '
              '1,000.'),
        default=1000,
        type=int
    )
    parser.add_argument(
        '-B', '--outer_batch',
        dest='b_out',
        default=10000,
        type=int,
        help=('Select the number of ids to read per outer level batch. This '
              'determines the number of readings/statements uploaded/pickled '
              'at a time, and thus also limits the amount of RAM that will be '
              'used. A larger outer batch means more RAM. The default is '
              '10,000.')
    )
    parser.add_argument(
        '--no_reading_upload',
        help='Choose not to upload the reading output to the database.',
        action='store_true'
    )
    parser.add_argument(
        '--no_statement_upload',
        help='Choose not to upload the statements to the database.',
        action='store_true'
    )
    parser.add_argument(
        '--max_reach_space_ratio',
        type=float,
        help='Set the maximum ratio of spaces to non-spaces for REACH input.',
        default=None
    )
    parser.add_argument(
        '--max_reach_input_len',
        type=int,
        help='Set the maximum length of content that REACH will read.',
        default=None
    )
    return parser


def main():
    # Process the arguments. =================================================
    parser = make_parser()
    args = parser.parse_args()
    if args.debug and not args.quiet:
        logger.setLevel(logging.DEBUG)

    # Get the ids.
    with open(args.input_file, 'r') as f:
        input_lines = f.readlines()
    logger.info("Found %d ids." % len(input_lines))

    # Select only a sample of the lines, if sample is chosen.
    if args.n_samp is not None:
        input_lines = random.sample(input_lines, args.n_samp)
    else:
        random.shuffle(input_lines)

    # If a range is specified, only use that range.
    if args.range_str is not None:
        start_idx, end_idx = [int(n) for n in args.range_str.split(':')]
        input_lines = input_lines[start_idx:end_idx]

    # Get the outer batch.
    B = args.b_out
    n_max = int(ceil(float(len(input_lines))/B))

    # Create a single base directory
    base_dir = _get_dir(args.temp, 'run_%s' % ('_and_'.join(args.readers)))

    # Get the readers objects.
    kwargs = {'base_dir': base_dir, 'n_proc': args.n_proc}
    if args.max_reach_space_ratio is not None:
        kwargs['input_character_limit'] = args.max_reach_space_ratio
    if args.max_reach_input_len is not None:
        kwargs['max_space_ratio'] = args.max_reach_input_len
    readers = construct_readers(args.readers, **kwargs)

    # Set the verbosity. The quiet argument overrides the verbose argument.
    verbose = args.verbose and not args.quiet

    # Some combinations of options don't make sense:
    forbidden_combos = [('all', 'unread'), ('none', 'unread'),
                        ('none', 'none')]
    assert (args.reading_mode, args.stmt_mode) not in forbidden_combos, \
        ("The combination of reading mode %s and statement mode %s is not "
         "allowed." % (args.reading_mode, args.stmt_mode))

    for n in range(n_max):
        logger.info("Beginning outer batch %d/%d. ------------" % (n+1, n_max))

        # Get the pickle file names.
        if args.name is not None:
            reading_pickle = args.name + '_readings_%d.pkl' % n
            stmts_pickle = args.name + '_stmts_%d.pkl' % n
        else:
            reading_pickle = None
            stmts_pickle = None

        # Get the dict of ids.
        tcids = [int(tcid_str.strip())
                 for tcid_str in input_lines[B*n:B*(n+1)]]

        # Read everything ====================================================
        run_reading(readers, tcids, verbose, args.reading_mode, args.stmt_mode,
                    args.b_in, reading_pickle, stmts_pickle,
                    not args.no_reading_upload, not args.no_statement_upload)


if __name__ == "__main__":
    main()
