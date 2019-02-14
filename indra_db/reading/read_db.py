"""This module provides essential tools to run reading using indra's own
database. This may also be run as a script; for details run:
`python read_pmids_db --help`
"""
from __future__ import absolute_import, print_function, unicode_literals
from builtins import dict, str

import pickle
import random
import logging
from datetime import datetime
from math import log10, floor, ceil
from uuid import uuid4

from indra.tools.reading.util.script_tools import get_parser, make_statements,\
                                             StatementData
from indra.literature.elsevier_client import extract_text as process_elsevier
from indra.tools.reading.readers import ReadingData, _get_dir, get_reader, \
    Content

from indra_db import get_primary_db, formats, texttypes
from indra_db import sql_expressions as sql
from indra_db.util import insert_raw_agents

logger = logging.getLogger('make_db_readings')


class ReadDBError(Exception):
    pass


class DatabaseReader(object):
    """An class to run readings utilizing the database.

    Parameters
    ----------
    tcids : iterable of ints
        An iterable (set, list, tuple, generator, etc) of integers referring to
        the primary keys of text content in the database.
    readers : list[Reader]
        A list of INDRA Reader objects.
    verbose : bool
        Optional, default False - If True, log and print the output of the
        commandline reader utilities, if False, don't.
    read_mode : str : 'all', 'unread', or 'none'
        Optional, default 'undread' - If 'all', read everything (generally
        slow); if 'unread', only read things that were unread, (the cache of old
        readings may still be used if `stmt_mode='all'` to get everything); if
        'none', don't read, and only retrieve existing readings.
    stmt_mode : bool
        Optional, default 'all' - If 'all', produce statements for all content
        for all readers. If the readings were already produced, they will be
        retrieved from the database if `read_mode` is 'none' or 'unread'. If
        this option is 'unread', only the newly produced readings will be
        processed. If 'none', no statements will be produced.
    batch_size : int
        Optional, default 1000 - The number of text content entries to be
        yielded by the database at a given time.
    no_upload : bool
        Optional, default False - If True, do not upload content to the
        database.
    db : indra_db.DatabaseManager instance
        Optional, default is None, in which case the primary database provided
        by `get_primary_db` function is used. Used to interface with a
        different database.
    """
    def __init__(self, tcids, readers, verbose=True, read_mode='unread',
                 stmt_mode='all', batch_size=1000, no_upload=False, db=None):
        self.tcids = tcids
        self.readers = readers
        self.verbose = verbose
        self.read_mode = read_mode
        self.stmt_mode = stmt_mode
        self.batch_size = batch_size
        self.no_upload = no_upload
        if db is None:
            self._db = get_primary_db()
        else:
            self._db = db
        self._tc_rd_link = \
            self._db.TextContent.id == self._db.Reading.text_content_id
        return

    def run(self):
        if self.read_mode != 'none':
            self.get_content()

        self.get_readings()

        if not self.no_upload:
            self.dump_readings()

        if not self.stmt_mode == 'none':
            self.make_statements()

            if not self.no_upload:
                self.dump_statements()
        pass

    def get_content(self):
        # Get the text content query object
        tc_query = self._db.filter_query(
            self._db.TextContent,
            self._tb.TextContent.id.in_(self.tcids)
            ).distinct()

        if self.read_mode != 'all':
            logger.debug("Getting content to be read.")
            # Each sub query is a set of content that has been read by one of
            # the readers.
            tc_q_subs = [tc_query.filter(self._tc_rd_link,
                                         _get_matches_clause(self._db, r))
                         for r in self.readers]
            tc_tbr_query = tc_query.except_(sql.intersect(*tc_q_subs))
        else:
            logger.debug('All content will be read (force_read).')
            tc_tbr_query = tc_query

        return tc_tbr_query.yield_per(self.batch_size)

    def get_readings(self):
        pass

    def dump_readings(self):
        raise NotImplementedError()

    def make_statements(self):
        pass

    def dump_statements(self):
        raise NotImplementedError()

    def make_new_readings(self, skip_dict=None, db=None, **kwargs):
        """Read contents retrieved from the database.

        The content will be retrieved in batchs, given by the `batch` argument.
        This prevents the system RAM from being overloaded.

        Parameters
        ----------
        skip_dict : dict {<reader> : list [int]}
            A dict containing text content id's to be skipped.
        db : indra_db.DatabaseManager instance
            A handle to a database. Default None; if None, a handle to the primary
            database (see indra_db) is retrieved.

        Other keyword arguments are passed to the `read` methods of the readers.

        Returns
        -------
        outputs : list of ReadingData instances
            The results of the readings with relevant metadata.
        """
        if db is None:
            db = get_primary_db()

        # Iterate
        logger.debug("Begginning to iterate.")
        batch_list_dict = {r.name: [] for r in self.readers}
        new_outputs = []
        for text_content in self.get_content():
            # The get_content function returns an iterator which yields
            # results in batches, so as not to overwhelm RAM. We need to read
            # in batches for much the same reason.
            for r in self.readers:
                if not self.read_mode == 'all':
                    if skip_dict is not None:
                        if text_content.id in skip_dict[r.name]:
                            continue
                    else:
                        # Try to get a previous reading from this reader.
                        reading = db.select_one(
                            db.Reading,
                            db.Reading.text_content_id == text_content.id,
                            _get_matches_clause(db, r)
                        )
                        if reading is not None:
                            continue
                processed_content = process_content(text_content)
                if processed_content is not None:
                    batch_list_dict[r.name].append(processed_content)

                if (len(batch_list_dict[r.name])+1) % self.batch_size is 0:
                    # TODO: this is a bit cludgy...maybe do this better?
                    # Perhaps refactor read_content.
                    logger.debug("Reading batch of files for %s." % r.name)
                    results = r.read(batch_list_dict[r.name], **kwargs)
                    if results is not None:
                        new_outputs += results
                    batch_list_dict[r.name] = []
        logger.debug("Finished iteration.")

        # Pick up any stragglers.
        for r in self.readers:
            if len(batch_list_dict[r.name]) > 0:
                logger.debug("Reading remaining files for %s." % r.name)
                results = r.read(batch_list_dict[r.name], **kwargs)
                if results is not None:
                    new_outputs += results

        return new_outputs

    def get_prior_readings(self):
        """Get readings from the database."""
        db = self._db
        prev_readings = []
        if self.tcids:
            for reader in self.readers:
                readings_query = db.filter_query(
                    db.Reading,
                    db.Reading.reader == reader.name,
                    db.Reading.reader_version == reader.version[:20],
                    db.Reading.text_content_id.in_(self.tcids)
                )
                for r in readings_query.yield_per(self.batch_size):
                    prev_readings.append(ReadingData.from_db_reading(r))
        return prev_readings

    def upload_readings(self, output_list):
        """Put the reading output on the database."""
        db = self._db

        # Get the id for this batch of uploads.
        batch_id = db.make_copy_batch_id()

        # Make a list of data to copy, ensuring there are no conflicts.
        upload_list = []
        rd_dict = {}
        for rd in output_list:
            # If there were no conflicts, we can add this to the copy list.
            upload_list.append(rd.make_tuple(batch_id))
            rd_dict[(rd.tcid, rd.reader, rd.reader_version)] = rd

        # Copy into the database.
        logger.info("Adding %d/%d reading entries to the database." %
                    (len(upload_list), len(output_list)))
        db.copy('reading', upload_list, ReadingData.get_cols())

        # Update the reading_data objects with their reading_ids.
        rdata = db.select_all([db.Reading.id, db.Reading.text_content_id,
                               db.Reading.reader, db.Reading.reader_version],
                              db.Reading.batch_id == batch_id)
        for tpl in rdata:
            rd_dict[tuple(tpl[1:])] = tpl[0]

        return

    def produce_readings(self):
        """Produce the reading output for the given ids, and upload them to db.
        """
        # Get a database instance.
        logger.debug("Producing readings in %s mode." % self.read_mode)

        # Handle the cases where I need to retrieve old readings.
        prev_readings = []
        skip_reader_tcid_dict = None
        if self.read_mode != 'all' and self.stmt_mode == 'all':
            prev_readings = self.get_prior_readings()
            skip_reader_tcid_dict = {r.name: [] for r in self.readers}
            logger.info("Found %d pre-existing readings." % len(prev_readings))
            if self.read_mode != 'none':
                for rd in prev_readings:
                    skip_reader_tcid_dict[rd.reader].append(rd.tcid)

        # Now produce any new readings that need to be produced.
        outputs = []
        if self.read_mode != 'none':
            outputs = self.make_new_readings(skip_dict=skip_reader_tcid_dict)
            logger.info("Made %d new readings." % len(outputs))

        if not self.no_upload:
            try:
                self.upload_readings(outputs)
            except Exception as e:
                logger.exception(e)

        outputs += prev_readings

        return outputs

    def upload_statements(self, stmt_data_list, db=None):
        """Upload the statements to the database."""
        if db is None:
            db = get_primary_db()

        logger.info("Uploading %d statements to the database." %
                    len(stmt_data_list))
        batch_id = db.make_copy_batch_id()
        db.copy('raw_statements', [s.make_tuple(batch_id) for s in stmt_data_list],
                StatementData.get_cols(), lazy=True, push_conflict=True)

        logger.info("Uploading agents to the database.")
        reading_id_set = set([sd.reading_id for sd in stmt_data_list])
        if len(reading_id_set):
            insert_raw_agents(db, batch_id, verbose=True)
        return

    def produce_statements(self, output_list, no_upload=False, pickle_file=None,
                           n_proc=1, db=None):
        """Convert the reader output into a list of StatementData instances."""
        if db is None:
            db = get_primary_db()

        stmt_data_list = make_statements(output_list, n_proc)

        if not no_upload:
            try:
                upload_statements(stmt_data_list, db=db)
            except Exception as e:
                logger.exception(e)
                if pickle_file is None:
                    pickle_file = ("failure_stmt_dump_%s.pkl"
                                   % datetime.now().strftime('%Y%m%d_%H%M%S'))
                logger.error("Could not upload statements. Results pickled in: %s."
                             % pickle_file)
        if pickle_file is not None:
            with open(pickle_file, 'wb') as f:
                pickle.dump([sd.statement for sd in stmt_data_list], f)
            print("Statements pickled in %s." % pickle_file)

        return stmt_data_list


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
            logger.warning("Could not extract text from Elsevier xml for tcid: "
                           "%d" % text_content.id)
            return None
        content = Content.from_string(content.get_id(), 'text', elsevier_text)
    return content


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
              "readings that are produced or retrieved will be used to produce "
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
        help='Choose not to upload the statements to the databse.',
        action='store_true'
    )
    parser.add_argument(
        '--force_fulltext',
        help='Make the reader only read full text from the database.',
        action='store_true'
    )
    parser.add_argument(
        '--use_best_fulltext',
        help='Use only the best full text available.',
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
    special_reach_args_dict = {
        'input_character_limit': args.max_reach_space_ratio,
        'max_space_ratio': args.max_reach_input_len
    }
    readers = []
    for reader_name in args.readers:
        kwargs = {'base_dir': base_dir, 'n_proc': args.n_proc}
        if reader_name == 'REACH':
            for key_name, reach_arg in special_reach_args_dict.items():
                if reach_arg is not None:
                    kwargs[key_name] = reach_arg
        readers.append(get_reader(reader_name, **kwargs))

    # Set the verbosity. The quiet argument overrides the verbose argument.
    verbose = args.verbose and not args.quiet

    # Some combinations of options don't make sense:
    forbidden_combos = [('all', 'unread'), ('none', 'unread'), ('none', 'none')]
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
        tcids = {int(tcid_str) for tcid_str in input_lines[B*n:B*(n+1)]}

        # Read everything ====================================================
        db_reader = DatabaseReader(tcids, readers, verbose, args.reading_mode,
                                   args.stmt_mode, args.batch_size)

        # Convert the outputs to statements ==================================
        if args.stmt_mode != 'none':
            produce_statements(outputs, no_upload=args.no_statement_upload,
                               pickle_file=stmts_pickle, n_proc=args.n_proc)


if __name__ == "__main__":
    main()
