import json
import pickle
import logging
from os import path, remove
from functools import wraps
from datetime import datetime
from collections import defaultdict
from argparse import ArgumentParser
from botocore import errorfactory

from indra.util import batch_iter, clockit
from indra.statements import Statement
from indra.tools import assemble_corpus as ac
from indra.preassembler.sitemapper import logger as site_logger
from indra.preassembler.grounding_mapper.mapper import logger \
    as grounding_logger
from indra.preassembler import Preassembler
from indra.preassembler import logger as ipa_logger
from indra.ontology.bio import bio_ontology
from indra_db.reading.read_db_aws import bucket_name

from indra_db.util.data_gatherer import DataGatherer, DGContext
from indra_db.util import insert_pa_stmts, distill_stmts, get_db, \
    extract_agent_data, insert_pa_agents, hash_pa_agents, S3Path

site_logger.setLevel(logging.INFO)
grounding_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

HERE = path.dirname(path.abspath(__file__))
ipa_logger.setLevel(logging.INFO)


gatherer = DataGatherer('preassembly', ['stmts', 'evidence', 'links'])


def _handle_update_table(func):
    @wraps(func)
    def run_and_record_update(cls, db, *args, **kwargs):
        run_datetime = datetime.utcnow()
        completed = func(cls, db, *args, **kwargs)
        if completed:
            is_corpus_init = (func.__name__ == 'create_corpus')
            db.insert('preassembly_updates', corpus_init=is_corpus_init,
                      run_datetime=run_datetime)
        return completed
    return run_and_record_update


class IndraDBPreassemblyError(Exception):
    pass


class DbPreassembler:
    """Class used to manage the preassembly pipeline

    Parameters
    ----------
    n_proc : int
        Select the number of processes that will be used when performing
        preassembly. Default is 1.
    batch_size : int
        Select the maximum number of statements you wish to be handled at a
        time. In general, a larger batch size will somewhat be faster, but
        require much more memory.
    """
    def __init__(self, n_proc=1, batch_size=10000, s3_cache=None,
                 print_logs=False, stmt_type=None):
        self.n_proc = n_proc
        self.batch_size = batch_size
        if s3_cache is not None:
            # Make the cache specific to the batch size and stmt type. This
            # guards against technical errors resulting from mixing these key
            # parameters.
            if not isinstance(s3_cache, S3Path):
                raise TypeError(f"Expected s3_cache to be type S3Path, but got "
                                f"type {type(s3_cache)}.")
            specifications = f'bs{batch_size}_st{stmt_type}/'
            self.s3_cache = s3_cache.get_element_path(specifications)

            # Report on what caches may already exist. This should hopefully
            # prevent re-doing work just because different batch sizes were
            # used.
            import boto3
            s3 = boto3.client('s3')
            if s3_cache.exists(s3):
                if self.s3_cache.exists(s3):
                    logger.info(f"A prior with these parameters exists in "
                                f"the cache: {s3_cache}.")
                else:
                    logger.info(f"Prior job or jobs with different batch size "
                                f"and/or Statement type exist for the cache: "
                                f"{s3_cache}.")
            else:
                logger.info(f"No prior jobs appear in the cache: {s3_cache}.")
        else:
            self.s3_cache = None
        self.pa = Preassembler(bio_ontology)
        self.__tag = 'Unpurposed'
        self.__print_logs = print_logs
        self.pickle_stashes = None
        self.stmt_type = stmt_type
        return

    def _get_latest_updatetime(self, db):
        """Get the date of the latest update."""
        update_list = db.select_all(db.PreassemblyUpdates)
        if not len(update_list):
            logger.warning("The preassembled corpus has not been initialized, "
                           "or else the updates table has not been populated.")
            return None
        return max([u.run_datetime for u in update_list])

    def _get_cache_path(self, file_name):
        return (self.s3_cache.get_element_path(self.__tag)
                             .get_element_path(file_name))

    def _init_cache(self, continuing):
        if self.s3_cache is None:
            return

        import boto3
        s3 = boto3.client('s3')
        start_file = self._get_cache_path('start.pkl')
        if start_file.exists(s3) and continuing:
            s3_resp = start_file.get(s3)
            start_data = pickle.loads(s3_resp['Body'].read())
            start_time = start_data['start_time']
        else:
            start_time = datetime.utcnow()
            start_data = {'start_time': start_time}
            start_file.put(s3, pickle.dumps(start_data))
        return start_time

    def _clear_cache(self):
        if self.s3_cache is None:
            return
        import boto3
        s3 = boto3.client('s3')
        objects = self._get_cache_path('').list_objects(s3)
        for s3_path in objects:
            s3_path.delete(s3)
        return

    def _run_cached(self, continuing, func, *args, **kwargs):
        # Define the location of this cache.
        import boto3
        s3 = boto3.client('s3')
        result_cache = self._get_cache_path(f'{func.__name__}.pkl')

        # If continuing, try to retrieve the file.
        if continuing and result_cache.exists(s3):
            s3_result = result_cache.get(s3)
            return pickle.loads(s3_result['Body'].read())

        # If not continuing or the file doesn't exist, run the function.
        results = func(*args, **kwargs)
        pickle_data = pickle.dumps(results)
        result_cache.put(s3, pickle_data)
        return results

    def _raw_sid_stmt_iter(self, db, id_set, do_enumerate=False):
        """Return a generator over statements with the given database ids."""
        def _fixed_raw_stmt_from_json(s_json, tr):
            stmt = _stmt_from_json(s_json)
            if tr is not None:
                stmt.evidence[0].pmid = tr.pmid
                stmt.evidence[0].text_refs = {k: v
                                              for k, v in tr.__dict__.items()
                                              if not k.startswith('_')}
            return stmt

        i = 0
        for stmt_id_batch in batch_iter(id_set, self.batch_size):
            subres = (db.filter_query([db.RawStatements.id,
                                       db.RawStatements.json,
                                       db.TextRef],
                                      db.RawStatements.id.in_(stmt_id_batch))
                      .outerjoin(db.Reading)
                      .outerjoin(db.TextContent)
                      .outerjoin(db.TextRef)
                      .yield_per(self.batch_size//10))
            data = [(sid, _fixed_raw_stmt_from_json(s_json, tr))
                    for sid, s_json, tr in subres]
            if do_enumerate:
                yield i, data
                i += 1
            else:
                yield data

    @clockit
    def _extract_and_push_unique_statements(self, db, raw_sids, num_stmts,
                                            mk_done=None):
        """Get the unique Statements from the raw statements."""
        start_date = datetime.utcnow()
        self._log("There are %d distilled raw statement ids to preassemble."
                  % len(raw_sids))

        if mk_done is None:
            mk_done = set()

        new_mk_set = set()
        num_batches = num_stmts/self.batch_size
        for i, stmt_tpl_batch in self._raw_sid_stmt_iter(db, raw_sids, True):
            self._log("Processing batch %d/%d of %d/%d statements."
                      % (i, num_batches, len(stmt_tpl_batch), num_stmts))

            # Get a list of statements and generate a mapping from uuid to sid.
            stmts = []
            uuid_sid_dict = {}
            for sid, stmt in stmt_tpl_batch:
                uuid_sid_dict[stmt.uuid] = sid
                stmts.append(stmt)

            # Map groundings and sequences.
            cleaned_stmts, eliminated_uuids = self._clean_statements(stmts)
            discarded_stmts = [(uuid_sid_dict[uuid], reason)
                               for reason, uuid_set in eliminated_uuids.items()
                               for uuid in uuid_set]
            db.copy('discarded_statements', discarded_stmts,
                    ('stmt_id', 'reason'), commit=False)

            # Use the shallow hash to condense unique statements.
            new_unique_stmts, evidence_links, agent_tuples = \
                self._condense_statements(cleaned_stmts, mk_done, new_mk_set,
                                          uuid_sid_dict)

            # Insert the statements and their links.
            self._log("Insert new statements into database...")
            insert_pa_stmts(db, new_unique_stmts, ignore_agents=True,
                            commit=False)
            gatherer.add('stmts', len(new_unique_stmts))

            self._log("Insert new raw_unique links into the database...")
            ev_links = flatten_evidence_dict(evidence_links)
            db.copy('raw_unique_links', ev_links,
                    ('pa_stmt_mk_hash', 'raw_stmt_id'), commit=False)
            gatherer.add('evidence', len(ev_links))

            db.copy_lazy('pa_agents', hash_pa_agents(agent_tuples),
                         ('stmt_mk_hash', 'ag_num', 'db_name', 'db_id', 'role',
                          'agent_ref_hash'),
                         commit=False)
            insert_pa_agents(db, new_unique_stmts, verbose=True,
                             skip=['agents'])  # This will commit

        self._log("Added %d new pa statements into the database."
                  % len(new_mk_set))
        end_date = datetime.utcnow()
        return {'start': start_date, 'end': end_date, 'mk_set': new_mk_set}

    @clockit
    def _condense_statements(self, cleaned_stmts, mk_done, new_mk_set,
                             uuid_sid_dict):
        self._log("Condense into unique statements...")
        new_unique_stmts = []
        evidence_links = defaultdict(lambda: set())
        agent_tuples = set()
        for s in cleaned_stmts:
            h = s.get_hash(refresh=True)

            # If this statement is new, make it.
            if h not in mk_done and h not in new_mk_set:
                new_unique_stmts.append(s.make_generic_copy())
                new_mk_set.add(h)

            # Add the evidence to the dict.
            evidence_links[h].add(uuid_sid_dict[s.uuid])

            # Add any db refs to the agents.
            ref_data, _, _ = extract_agent_data(s, h)
            agent_tuples |= set(ref_data)

        return new_unique_stmts, evidence_links, agent_tuples

    @_handle_update_table
    @DGContext.wrap(gatherer)
    def create_corpus(self, db, continuing=False):
        """Initialize the table of preassembled statements.

        This method will find the set of unique knowledge represented in the
        table of raw statements, and it will populate the table of preassembled
        statements (PAStatements/pa_statements), while maintaining links between
        the raw statements and their unique (pa) counterparts. Furthermore, the
        refinement/support relationships between unique statements will be found
        and recorded in the PASupportLinks/pa_support_links table.

        For more detail on preassembly, see indra/preassembler/__init__.py
        """
        self.__tag = 'create'
        self._init_cache(continuing)

        if not continuing:
            # Make sure the discarded statements table is cleared.
            db.drop_tables([db.DiscardedStatements])
            db.create_tables([db.DiscardedStatements])
            db.session.close()
            db.grab_session()
        else:
            # Get discarded statements
            skip_ids = {i for i, in db.select_all(db.DiscardedStatements.stmt_id)}
            self._log("Found %d discarded statements from earlier run."
                      % len(skip_ids))

        # Get filtered statement ID's.
        if self.stmt_type is not None:
            clauses = [db.RawStatements.type == self.stmt_type]
        else:
            clauses = []
        stmt_ids = self._run_cached(continuing, distill_stmts, db,
                                    clauses=clauses)

        # Handle the possibility we're picking up after an earlier job...
        done_pa_ids = set()
        if continuing:
            self._log("Getting set of statements already de-duplicated...")
            link_resp = db.select_all([db.RawUniqueLinks.raw_stmt_id,
                                       db.RawUniqueLinks.pa_stmt_mk_hash])
            if link_resp:
                checked_raw_stmt_ids, pa_stmt_hashes = \
                    zip(*db.select_all([db.RawUniqueLinks.raw_stmt_id,
                                        db.RawUniqueLinks.pa_stmt_mk_hash]))
                stmt_ids -= set(checked_raw_stmt_ids)
                self._log("Found %d raw statements without links to unique."
                          % len(stmt_ids))
                stmt_ids -= skip_ids
                self._log("Found %d raw statements that still need to be "
                          "processed." % len(stmt_ids))
                done_pa_ids = set(pa_stmt_hashes)
                self._log("Found %d preassembled statements already done."
                          % len(done_pa_ids))

        # Get the set of unique statements
        self._extract_and_push_unique_statements(db, stmt_ids, len(stmt_ids),
                                                 done_pa_ids)

        # If we are continuing, check for support links that were already found
        if continuing:
            self._log("Getting pre-existing links...")
            db_existing_links = db.select_all([
                db.PASupportLinks.supporting_mk_hash,
                db.PASupportLinks.supporting_mk_hash
            ])
            existing_links = {tuple(res) for res in db_existing_links}
            self._log("Found %d existing links." % len(existing_links))
        else:
            existing_links = set()

        # Now get the support links between all batches.
        support_links = set()
        batching_args = tuple()
        if self.stmt_type is not None:
            batching_args += (db.PAStatements.type == self.stmt_type,)
        outer_iter = db.select_all_batched(self.batch_size,
                                           db.PAStatements.json,
                                           *batching_args,
                                           order_by=db.PAStatements.mk_hash)
        for outer_idx, outer_batch_jsons in outer_iter:
            outer_batch = [_stmt_from_json(sj) for sj, in outer_batch_jsons]
            # Get internal support links
            self._log('Getting internal support links outer batch %d.'
                      % outer_idx)
            some_support_links = self._get_support_links(outer_batch)

            # Get links with all other batches
            inner_iter = db.select_all_batched(self.batch_size,
                                               db.PAStatements.json,
                                               *batching_args,
                                               order_by=db.PAStatements.mk_hash,
                                               skip_idx=outer_idx)
            for inner_idx, inner_batch_jsons in inner_iter:
                inner_batch = [_stmt_from_json(sj) for sj, in inner_batch_jsons]
                split_idx = len(inner_batch)
                full_list = inner_batch + outer_batch
                self._log('Getting support between outer batch %d and inner'
                          'batch %d.' % (outer_idx, inner_idx))
                some_support_links |= \
                    self._get_support_links(full_list, split_idx=split_idx)

            # Add all the new support links
            support_links |= (some_support_links - existing_links)

            # There are generally few support links compared to the number of
            # statements, so it doesn't make sense to copy every time, but for
            # long preassembly, this allows for better failure recovery.
            if len(support_links) >= self.batch_size:
                self._log("Copying batch of %d support links into db."
                          % len(support_links))
                db.copy('pa_support_links', support_links,
                        ('supported_mk_hash', 'supporting_mk_hash'))
                gatherer.add('links', len(support_links))
                existing_links |= support_links
                support_links = set()

        # Insert any remaining support links.
        if support_links:
            self._log("Copying final batch of %d support links into db."
                      % len(support_links))
            db.copy('pa_support_links', support_links,
                    ('supported_mk_hash', 'supporting_mk_hash'))
            gatherer.add('links', len(support_links))

        self._clear_cache()
        return True

    def _get_new_stmt_ids(self, db):
        """Get all the uuids of statements not included in evidence."""
        old_id_q = db.filter_query(
            db.RawStatements.id,
            db.RawStatements.id == db.RawUniqueLinks.raw_stmt_id
        )
        new_sid_q = db.filter_query(db.RawStatements.id).except_(old_id_q)
        if self.stmt_type is not None:
            new_sid_q = new_sid_q.filter(db.RawStatements.type == self.stmt_type)
        all_new_stmt_ids = {sid for sid, in new_sid_q.all()}
        self._log("Found %d new statement ids." % len(all_new_stmt_ids))
        return all_new_stmt_ids

    def _supplement_statements(self, db, continuing=False):
        """Supplement the preassembled statements with the latest content."""

        last_update = self._get_latest_updatetime(db)
        assert last_update is not None, \
            "The preassembly tables have not yet been initialized."
        self._log("Latest update was: %s" % last_update)

        # Get the new statements...
        self._log("Loading info about the existing state of preassembly. "
                  "(This may take a little time)")
        new_ids = self._run_cached(continuing, self._get_new_stmt_ids, db)

        # Weed out exact duplicates.
        if self.stmt_type is not None:
            clauses = [db.RawStatements.type == self.stmt_type]
        else:
            clauses = []
        stmt_ids = self._run_cached(continuing, distill_stmts, db,
                                    get_full_stmts=False, clauses=clauses)

        # Get discarded statements
        skip_ids = {i for i, in db.select_all(db.DiscardedStatements.stmt_id)}

        # Select only the good new statement ids.
        new_stmt_ids = new_ids & stmt_ids - skip_ids

        # Get the set of new unique statements and link to any new evidence.
        old_mk_set = {mk for mk, in db.select_all(db.PAStatements.mk_hash)}
        self._log("Found %d old pa statements." % len(old_mk_set))

        result_dict = self._run_cached(
            continuing,
            self._extract_and_push_unique_statements,
            db, new_stmt_ids, len(new_stmt_ids), old_mk_set
        )
        start_date = result_dict['start']
        end_date = result_dict['end']
        new_mk_set = result_dict['mk_set']

        if continuing:
            self._log("Original old mk set: %d" % len(old_mk_set))
            old_mk_set = old_mk_set - new_mk_set
            self._log("Adjusted old mk set: %d" % len(old_mk_set))

        self._log("Found %d new pa statements." % len(new_mk_set))
        return start_date, end_date

    def _supplement_support(self, db, start_date, end_date, continuing=False):
        """Calculate the support for the given date range of pa statements."""

        # If we are continuing, check for support links that were already found
        new_support_links = set()

        # Now find the new support links that need to be added.
        batching_args = (self.batch_size,
                         db.PAStatements.json,
                         db.PAStatements.create_date >= start_date,
                         db.PAStatements.create_date <= end_date)
        if self.stmt_type is not None:
            batching_args += (db.PAStatements.type == self.stmt_type,)
        npa_json_iter = db.select_all_batched(*batching_args,
                                              order_by=db.PAStatements.mk_hash)
        for outer_idx, npa_json_batch in npa_json_iter:
            # Create the statements from the jsons.
            npa_batch = [_stmt_from_json(s_json) for s_json in npa_json_batch]

            # Compare internally
            self._log("Getting support for new pa batch %d." % outer_idx)
            some_support_links = self._get_support_links(npa_batch)

            # Compare against the other new batch statements.
            other_npa_json_iter = db.select_all_batched(
                *batching_args,
                order_by=db.PAStatements.mk_hash,
                skip_idx=outer_idx
            )
            for inner_idx, other_npa_json_batch in other_npa_json_iter:
                other_npa_batch = [_stmt_from_json(s_json)
                                   for s_json, in other_npa_json_batch]
                split_idx = len(npa_batch)
                full_list = npa_batch + other_npa_batch
                self._log("Comparing outer batch %d to inner batch %d of "
                          "other new statements." % (outer_idx, inner_idx))
                some_support_links |= \
                    self._get_support_links(full_list, split_idx=split_idx)

            # Compare against the existing statements.
            opa_json_iter = db.select_all_batched(
                self.batch_size,
                db.PAStatements.json,
                db.PAStatements.create_date < start_date
            )
            for opa_idx, opa_json_batch in opa_json_iter:
                opa_batch = [_stmt_from_json(s_json)
                             for s_json, in opa_json_batch]
                split_idx = len(npa_batch)
                full_list = npa_batch + opa_batch
                self._log("Comparing new batch %d to batch %d of old "
                          "statements." % (outer_idx, opa_idx))
                some_support_links |= \
                    self._get_support_links(full_list, split_idx=split_idx)

        # Insert any remaining support links.
        if new_support_links:
            self._log("Copying %d support links into db."
                      % len(new_support_links))
            skipped = db.copy_report_lazy(
                'pa_support_links',
                new_support_links,
                ('supported_mk_hash', 'supporting_mk_hash')
            )
            gatherer.add('links', len(new_support_links - set(skipped)))
        return

    @_handle_update_table
    @DGContext.wrap(gatherer)
    def supplement_corpus(self, db, continuing=False):
        """Update the table of preassembled statements.

        This method will take any new raw statements that have not yet been
        incorporated into the preassembled table, and use them to augment the
        preassembled table.

        The resulting updated table is indistinguishable from the result you
        would achieve if you had simply re-run preassembly on _all_ the
        raw statements.
        """
        self.__tag = 'supplement'
        self._init_cache(continuing)

        self.pickle_stashes = []

        start_date, end_date = self._supplement_statements(db, continuing)
        self._supplement_support(db, start_date, end_date, continuing)

        self._clear_cache()
        self.__tag = 'Unpurposed'
        return True

    def _log(self, msg, level='info'):
        """Applies a task specific tag to the log message."""
        if self.__print_logs:
            print("Preassembly Manager [%s] (%s): %s"
                  % (datetime.now(), self.__tag, msg))
        getattr(logger, level)("(%s) %s" % (self.__tag, msg))

    @clockit
    def _clean_statements(self, stmts):
        """Perform grounding, sequence mapping, and find unique set from stmts.

        This method returns a list of statement objects, as well as a set of
        tuples of the form (uuid, matches_key) which represent the links between
        raw (evidence) statements and their unique/preassembled counterparts.
        """
        eliminated_uuids = {}
        all_uuids = {s.uuid for s in stmts}
        self._log("Map grounding...")
        stmts = ac.map_grounding(stmts, use_adeft=True, gilda_mode='local')
        grounded_uuids = {s.uuid for s in stmts}
        eliminated_uuids['grounding'] = all_uuids - grounded_uuids
        self._log("Map sequences...")
        stmts = ac.map_sequence(stmts, use_cache=True)
        seqmapped_and_grounded_uuids = {s.uuid for s in stmts}
        eliminated_uuids['sequence mapping'] = \
            grounded_uuids - seqmapped_and_grounded_uuids
        return stmts, eliminated_uuids

    @clockit
    def _get_support_links(self, unique_stmts, split_idx=None):
        """Find the links of refinement/support between statements."""
        id_maps = self.pa._generate_id_maps(unique_stmts, poolsize=self.n_proc,
                                            split_idx=split_idx)
        ret = set()
        for ix_pair in id_maps:
            if ix_pair[0] == ix_pair[1]:
                assert False, "Self-comparison occurred."
            hash_pair = \
                tuple([shash(unique_stmts[ix]) for ix in ix_pair])
            if hash_pair[0] == hash_pair[1]:
                assert False, "Input list included duplicates."
            ret.add(hash_pair)

        return ret


def _stmt_from_json(stmt_json_bytes):
    return Statement._from_json(json.loads(stmt_json_bytes.decode('utf-8')))


# This is purely for reducing having to type this long thing so often.
def shash(s):
    """Get the shallow hash of a statement."""
    return s.get_hash(shallow=True)


def make_graph(unique_stmts, match_key_maps):
    """Create a networkx graph of the statement and their links."""
    import networkx as nx
    g = nx.Graph()
    link_matches = {m for l in match_key_maps for m in l}
    unique_stmts_dict = {}
    for stmt in unique_stmts:
        if stmt.matches_key() in link_matches:
            g.add_node(stmt)
            unique_stmts_dict[stmt.matches_key()] = stmt

    for k1, k2 in match_key_maps:
        g.add_edge(unique_stmts_dict[k1], unique_stmts_dict[k2])

    return g


def flatten_evidence_dict(ev_dict):
    return {(u_stmt_key, ev_stmt_uuid)
            for u_stmt_key, ev_stmt_uuid_set in ev_dict.items()
            for ev_stmt_uuid in ev_stmt_uuid_set}


def _make_parser():
    parser = ArgumentParser(
        description='Manage preassembly of raw statements into pa statements.'
    )
    parser.add_argument(
        choices=['create', 'update'],
        dest='task',
        help=('Choose whether you want to perform an initial upload or update '
              'the existing content on the database.')
    )
    parser.add_argument(
        '-c', '--continue',
        dest='continuing',
        action='store_true',
        help='Continue uploading or updating, picking up where you left off.'
    )
    parser.add_argument(
        '-n', '--num_procs',
        dest='num_procs',
        type=int,
        default=None,
        help=('Select the number of processors to use during this operation. '
              'Default is 1.')
    )
    parser.add_argument(
        '-b', '--batch',
        type=int,
        default=10000,
        help=("Select the number of statements loaded at a time. More "
              "statements at a time will run faster, but require more memory.")
    )
    parser.add_argument(
        '-d', '--debug',
        dest='debug',
        action='store_true',
        help='Run with debugging level output.'
    )
    parser.add_argument(
        '-D', '--database',
        default='primary',
        help=('Choose a database from the names given in the config or '
              'environment, for example primary is INDRA_DB_PRIMAY in the '
              'config file and INDRADBPRIMARY in the environment. The default '
              'is \'primary\'.')
    )
    parser.add_argument(
        '-C', '--cache',
        default=f's3://{bucket_name}/preassembly_results/temp',
        help=('Choose where on s3 the temp files that allow jobs to continue '
              'after stopping are stored. Value should be given in the form:'
              's3://{bucket_name}/{prefix}. NOTE THAT A "/" WILL *NOT* BE '
              'ADDED TO THE END OF THE PREFIX.')
    )
    return parser


def _main():
    parser = _make_parser()
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        from indra_db.databases import logger as db_logger
        db_logger.setLevel(logging.DEBUG)
    print("Getting %s database." % args.database)
    db = get_db(args.database)
    assert db is not None
    db.grab_session()
    s3_cache = S3Path.from_string(args.cache)
    pa = DbPreassembler(args.num_procs, args.batch, s3_cache)

    desc = 'Continuing' if args.continuing else 'Beginning'
    print("%s to %s preassembled corpus." % (desc, args.task))
    if args.task == 'create':
        pa.create_corpus(db, args.continuing)
    elif args.task == 'update':
        pa.supplement_corpus(db, args.continuing)
    else:
        raise IndraDBPreassemblyError('Unrecognized task: %s.' % args.task)


if __name__ == '__main__':
    _main()
