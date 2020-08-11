import json
import pickle
import logging
from os import path
from functools import wraps
from datetime import datetime
from collections import defaultdict
from argparse import ArgumentParser

from sqlalchemy import or_

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
grounding_logger.setLevel(logging.INFO)
logger = logging.getLogger('preassemble_db')

HERE = path.dirname(path.abspath(__file__))
ipa_logger.setLevel(logging.INFO)


gatherer = DataGatherer('preassembly', ['stmts', 'evidence', 'links'])


def _handle_update_table(func):
    @wraps(func)
    def run_and_record_update(obj, db, *args, **kwargs):
        run_datetime = datetime.utcnow()
        completed = func(obj, db, *args, **kwargs)
        if completed:
            is_corpus_init = (func.__name__ == 'create_corpus')
            db.insert('preassembly_updates', corpus_init=is_corpus_init,
                      run_datetime=run_datetime, stmt_type=obj.stmt_type)
        return completed
    return run_and_record_update


class IndraDBPreassemblyError(Exception):
    pass


class UserQuit(BaseException):
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
    def __init__(self, n_proc=None, batch_size=10000, s3_cache=None,
                 print_logs=False, stmt_type=None, yes_all=False):
        self.n_proc = n_proc
        self.batch_size = batch_size
        if s3_cache is not None:
            # Make the cache specific to stmt type. This guards against
            # technical errors resulting from mixing this key parameter.
            if not isinstance(s3_cache, S3Path):
                raise TypeError(f"Expected s3_cache to be type S3Path, but got "
                                f"type {type(s3_cache)}.")
            specifications = f'st_{stmt_type}/'
            self.s3_cache = s3_cache.get_element_path(specifications)

            # Report on what caches may already exist. This should hopefully
            # prevent re-doing work just because different batch sizes were
            # used.
            import boto3
            s3 = boto3.client('s3')
            if s3_cache.exists(s3):
                if self.s3_cache.exists(s3):
                    logger.info(f"A prior run with these parameters exists in "
                                f"the cache: {s3_cache}.")
                else:
                    logger.info(f"Prior job or jobs with different Statement "
                                f"type exist for the cache: {s3_cache}.")
            else:
                logger.info(f"No prior jobs appear in the cache: {s3_cache}.")
        else:
            self.s3_cache = None
        self.pa = Preassembler(bio_ontology)
        bio_ontology.initialize()
        bio_ontology._build_transitive_closure()
        self.__tag = 'Unpurposed'
        self.__print_logs = print_logs
        self.pickle_stashes = None
        self.stmt_type = stmt_type
        self.yes_all = yes_all
        return

    def _yes_input(self, message, default='yes'):
        if self.yes_all:
            return True

        valid = {'yes': True, 'ye': True, 'y': True, 'no': False, 'n': False}

        if default is None:
            prompt = '[y/n]'
        elif default == 'yes':
            prompt = '[Y/n]'
        elif default == 'no':
            prompt = '[y/N]'
        else:
            raise ValueError(f"Argument 'default' must be 'yes' or 'no', got "
                             f"'{default}'.")

        resp = input(f'{message} {prompt}: ')
        while True:
            if resp == '' and default is not None:
                return valid[default]
            elif resp.lower() in valid:
                return valid[resp.lower()]
            resp = input(f'Please answer "yes" (or "y") or "no" (or "n"). '
                         f'{prompt}: ')

    def _get_latest_updatetime(self, db):
        """Get the date of the latest update."""
        if self.stmt_type is not None:
            st_const = or_(db.PreassemblyUpdates.stmt_type == self.stmt_type,
                           db.PreassemblyUpdates.stmt_type.is_(None))
        else:
            st_const = db.PreassemblyUpdates.stmt_type.is_(None)
        update_list = db.select_all(db.PreassemblyUpdates, st_const)
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
        if start_file.exists(s3):
            s3_resp = start_file.get(s3)
            start_data = pickle.loads(s3_resp['Body'].read())
            start_time = start_data['start_time']
            cache_desc = f"Do you want to %s {self._get_cache_path('')} " \
                         f"started {start_time}?"
            if continuing:
                if self._yes_input(cache_desc % 'continue with'):
                    return start_time
                else:
                    raise UserQuit("Aborting job.")
            elif not self._yes_input(cache_desc % 'overwrite existing',
                                     default='no'):
                raise UserQuit("Aborting job.")

            self._clear_cache()

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
        if self.s3_cache is None:
            return func(*args, **kwargs)

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

    def _put_support_mark(self, outer_idx):
        if self.s3_cache is None:
            return

        import boto3
        s3 = boto3.client('s3')

        supp_file = self._get_cache_path('support_idx.pkl')
        supp_file.put(s3, pickle.dumps(outer_idx*self.batch_size))
        return

    def _get_support_mark(self, continuing):
        if self.s3_cache is None:
            return

        if not continuing:
            return -1

        import boto3
        s3 = boto3.client('s3')

        supp_file = self._get_cache_path('support_idx.pkl')
        if not supp_file.exists(s3):
            return -1
        s3_resp = supp_file.get(s3)
        return pickle.loads(s3_resp['Body'].read()) // self.batch_size

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

    def _make_idx_batches(self, hash_list, continuing):
        N = len(hash_list)
        B = self.batch_size
        idx_batch_list = [(n*B, min((n + 1)*B, N)) for n in range(0, N//B + 1)]
        start_idx = self._get_support_mark(continuing) + 1
        return idx_batch_list, start_idx

    @clockit
    def _extract_and_push_unique_statements(self, db, raw_sids, num_stmts,
                                            mk_done=None):
        """Get the unique Statements from the raw statements."""
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
        return new_mk_set

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

    def _dump_links(self, db, supp_links):
        self._log(f"Copying batch of {len(supp_links)} support links into db.")
        skipped = db.copy_report_lazy('pa_support_links', supp_links,
                                      ('supported_mk_hash',
                                       'supporting_mk_hash'))
        gatherer.add('links', len(supp_links - set(skipped)))
        return

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

        if continuing:
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
        mk_done = set()
        if continuing:
            self._log("Getting set of statements already de-duplicated...")
            link_q = db.filter_query([db.RawUniqueLinks.raw_stmt_id,
                                      db.RawUniqueLinks.pa_stmt_mk_hash])
            if self.stmt_type is not None:
                link_q = (link_q
                          .join(db.RawStatements)
                          .filter(db.RawStatements.type == self.stmt_type))
            link_resp = link_q.all()
            if link_resp:
                checked_raw_stmt_ids, pa_stmt_hashes = zip(*link_resp)
                stmt_ids -= set(checked_raw_stmt_ids)
                self._log("Found %d raw statements without links to unique."
                          % len(stmt_ids))
                stmt_ids -= skip_ids
                self._log("Found %d raw statements that still need to be "
                          "processed." % len(stmt_ids))
                mk_done = set(pa_stmt_hashes)
                self._log("Found %d preassembled statements already done."
                          % len(mk_done))

        # Get the set of unique statements
        new_mk_set = self._run_cached(continuing,
                                      self._extract_and_push_unique_statements,
                                      db, stmt_ids, len(stmt_ids), mk_done)

        # Now get the support links between all batches.
        support_links = set()
        hash_list = list(new_mk_set | mk_done)
        self._log(f"Beginning to find support relations for {len(hash_list)} "
                  f"new statements.")
        hash_list.sort()
        idx_batches, start_idx = self._make_idx_batches(hash_list, continuing)
        for outer_idx, (out_si, out_ei) in enumerate(idx_batches[start_idx:]):
            outer_idx += start_idx
            sj_query = db.filter_query(
                db.PAStatements.json,
                db.PAStatements.mk_hash.in_(hash_list[out_si:out_ei])
            )
            outer_batch = [_stmt_from_json(sj) for sj, in sj_query.all()]

            # Get internal support links
            self._log(f'Getting internal support links outer batch '
                      f'{outer_idx}/{len(idx_batches)-1}.')
            some_support_links = self._get_support_links(outer_batch)

            # Get links with all other batches
            in_start = outer_idx + 1
            for inner_idx, (in_si, in_ei) in enumerate(idx_batches[in_start:]):
                inner_sj_q = db.filter_query(
                    db.PAStatements.json,
                    db.PAStatements.mk_hash.in_(hash_list[in_si:in_ei])
                )
                inner_batch = [_stmt_from_json(sj) for sj, in inner_sj_q.all()]
                split_idx = len(inner_batch)
                full_list = inner_batch + outer_batch
                self._log(f'Getting support between outer batch {outer_idx}/'
                          f'{len(idx_batches)-1} and inner batch {inner_idx}/'
                          f'{len(idx_batches)-in_start-1}.')
                some_support_links |= \
                    self._get_support_links(full_list, split_idx=split_idx)

            # Add all the new support links
            support_links |= some_support_links

            # There are generally few support links compared to the number of
            # statements, so it doesn't make sense to copy every time, but for
            # long preassembly, this allows for better failure recovery.
            if len(support_links) >= self.batch_size:
                self._dump_links(db, support_links)
                self._put_support_mark(outer_idx)
                support_links = set()

        # Insert any remaining support links.
        if support_links:
            self._log('Final (overflow) batch of links.')
            self._dump_links(db, support_links)

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

        new_mk_set = self._run_cached(
            continuing,
            self._extract_and_push_unique_statements,
            db, new_stmt_ids, len(new_stmt_ids), old_mk_set
        )

        if continuing:
            self._log("Original old mk set: %d" % len(old_mk_set))
            old_mk_set = old_mk_set - new_mk_set
            self._log("Adjusted old mk set: %d" % len(old_mk_set))

        self._log("Found %d new pa statements." % len(new_mk_set))
        return new_mk_set

    def _supplement_support(self, db, new_hashes, start_time, continuing=False):
        """Calculate the support for the given date range of pa statements."""
        if not isinstance(new_hashes, list):
            new_hashes = list(new_hashes)
        new_hashes.sort()

        # If we are continuing, check for support links that were already found
        support_links = set()
        idx_batches, start_idx = self._make_idx_batches(new_hashes, continuing)
        for outer_idx, (out_s, out_e) in enumerate(idx_batches[start_idx:]):
            outer_idx += start_idx
            # Create the statements from the jsons.
            npa_json_q = db.filter_query(
                db.PAStatements.json,
                db.PAStatements.mk_hash.in_(new_hashes[out_s:out_e])
            )
            npa_batch = [_stmt_from_json(s_json) for s_json in npa_json_q.all()]

            # Compare internally
            self._log(f"Getting support for new pa batch {outer_idx}/"
                      f"{len(idx_batches)}.")
            some_support_links = self._get_support_links(npa_batch)

            # Compare against the other new batch statements.
            in_start = outer_idx + 1
            for in_idx, (in_s, in_e) in enumerate(idx_batches[in_start:]):
                other_npa_q = db.filter_query(
                    db.PAStatements.json,
                    db.PAStatements.mk_hash.in_(new_hashes[in_s:in_e])
                )
                other_npa_batch = [_stmt_from_json(sj)
                                   for sj, in other_npa_q.all()]
                split_idx = len(npa_batch)
                full_list = npa_batch + other_npa_batch
                self._log(f"Comparing outer batch {outer_idx}/"
                          f"{len(idx_batches)-1} to inner batch {in_idx}/"
                          f"{len(idx_batches)-in_start-1} of other new "
                          f"statements.")
                some_support_links |= \
                    self._get_support_links(full_list, split_idx=split_idx)

            # Compare against the existing statements.
            opa_args = (db.PAStatements.create_date < start_time,)
            if self.stmt_type is not None:
                opa_args += (db.PAStatements.type == self.stmt_type,)

            opa_json_iter = db.select_all_batched(self.batch_size,
                                                  db.PAStatements.json,
                                                  *opa_args)
            for opa_idx, opa_json_batch in opa_json_iter:
                opa_batch = [_stmt_from_json(s_json)
                             for s_json, in opa_json_batch]
                split_idx = len(npa_batch)
                full_list = npa_batch + opa_batch
                self._log(f"Comparing new batch {outer_idx}/"
                          f"{len(idx_batches)-1} to batch {opa_idx} of old pa "
                          f"statements.")
                some_support_links |= \
                    self._get_support_links(full_list, split_idx=split_idx)

            support_links |= some_support_links

            # There are generally few support links compared to the number of
            # statements, so it doesn't make sense to copy every time, but for
            # long preassembly, this allows for better failure recovery.
            if len(support_links) >= self.batch_size:
                self._dump_links(db, support_links)
                self._put_support_mark(outer_idx)
                support_links = set()

        # Insert any remaining support links.
        if support_links:
            self._log("Final (overflow) batch of new support links.")
            self._dump_links(db, support_links)
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
        start_time = self._init_cache(continuing)

        self.pickle_stashes = []

        new_hashes = self._supplement_statements(db, continuing)
        self._supplement_support(db, new_hashes, start_time, continuing)

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
        '-n', '--num-procs',
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
              's3://{bucket_name}/{prefix}.')
    )
    parser.add_argument(
        '-T', '--stmt-type',
        help=('Optionally select a particular statement type on which to run '
              'preassembly on. In general types are not compared so you can '
              'greatly multi-process the task by having separate machines '
              'preassemble different types.')
    )
    parser.add_argument(
        '-Y', '--yes-all',
        action='store_true',
        help='Select the "yes" option for all user options during runtime.'
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
    pa = DbPreassembler(args.num_procs, args.batch, s3_cache,
                        stmt_type=args.stmt_type, yes_all=args.yes_all)

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
