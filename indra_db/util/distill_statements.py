__all__ = ['distill_stmts', 'get_filtered_rdg_stmts', 'get_filtered_db_stmts',
           'delete_raw_statements_by_id', 'get_reading_stmt_dict',
           'reader_versions', 'text_content_sources', 'extract_duplicates',
           'KeyFunc']

import json
import pickle
import logging
from datetime import datetime
from collections import defaultdict

from indra.util import clockit
from indra.statements import Statement
from indra.util.nested_dict import NestedDict
from indra_db.databases import reader_versions

from .helpers import _set_evidence_text_ref

logger = logging.getLogger('util-distill')


class KeyFunc:
    @staticmethod
    def mk(stmt):
        return stmt.matches_key()

    @staticmethod
    def mk_and_one_ev_mk(stmt):
        return stmt.matches_key(), stmt.evidence[0].matches_key()

    @staticmethod
    def mk_and_one_ev_src(stmt):
        return stmt.matches_key(), stmt.evidence[0].get_source_hash()


def extract_duplicates(stmts, key_func=None):
    """A function to remove duplicates using matches keys, or another method.

    To select or define a measure of uniqueness, you can set the `key_func`
    parameter to a function that takes a statement as an argument and returns
    a hashable key (string, int, tuple, anything that could be the key to a
    python dictionary).

    Some common options are available as static methods of the KeyFunc class.
    The default is `mk`, which is a statement `matches_key`.
    """
    if key_func is None:
        key_func = KeyFunc.mk

    dup_dict = {}
    unique_dict = {}
    for stmt in stmts:
        key = key_func(stmt)
        if key not in dup_dict.keys():
            if key in unique_dict.keys():
                dup_dict[key] = [stmt, unique_dict[key]]
            else:
                unique_dict[key] = stmt
        else:
            dup_dict[key].append(stmt)

    return list(unique_dict.values()), list(dup_dict.values())


def get_reading_stmt_dict(db, clauses=None, get_full_stmts=True):
    """Get a nested dict of statements, keyed by ref, content, and reading."""
    # Construct the query for metadata from the database.
    elements = [db.TextRef, db.TextContent.id, db.TextContent.source,
                db.TextContent.text_type, db.Reading.id,
                db.Reading.reader_version, db.RawStatements.id,
                db.RawStatements.mk_hash, db.RawStatements.text_hash]
    if get_full_stmts:
        elements += [db.RawStatements.json]

    q = (db.session.query(*elements)
         .filter(db.RawStatements.reading_id == db.Reading.id,
                 db.Reading.text_content_id == db.TextContent.id,
                 db.TextContent.text_ref_id == db.TextRef.id))
    if clauses:
        q = q.filter(*clauses)

    # Prime some counters.
    num_duplicate_evidence = 0
    num_unique_evidence = 0

    # Populate a dict with all the data.
    stmt_nd = NestedDict()
    for data in q.yield_per(1000):
        if get_full_stmts:
            tr, tcid, src, tt, rid, rv, sid, mk_hash, text_hash, sjson = data
            stmt_json = json.loads(sjson.decode('utf8'))
            stmt = Statement._from_json(stmt_json)
            _set_evidence_text_ref(stmt, tr)
        else:
            tr, tcid, src, tt, rid, rv, sid, mk_hash, text_hash = data

        stmt_hash = (mk_hash, text_hash)

        # Back out the reader name.
        for reader, rv_list in reader_versions.items():
            if rv in rv_list:
                break
        else:
            raise Exception("rv %s not recognized." % rv)

        # For convenience get the endpoint statement dict
        s_dict = stmt_nd[tr.id][(src, tt)][tcid][reader][rv][rid]

        # Initialize the value to a set, and count duplicates
        if stmt_hash not in s_dict.keys():
            s_dict[stmt_hash] = set()
            num_unique_evidence += 1
        else:
            num_duplicate_evidence += 1

        # Either store the statement, or the statement id.
        if get_full_stmts:
            s_dict[stmt_hash].add((sid, stmt))
        else:
            s_dict[stmt_hash].add((sid, None))

    # Report on the results.
    print("Found %d relevant text refs with statements." % len(stmt_nd))
    print("number of statement exact duplicates: %d" % num_duplicate_evidence)
    print("number of unique statements: %d" % num_unique_evidence)
    return stmt_nd


# Specify sources of fulltext content, and order priorities.
text_content_sources = [('pubmed', 'title'),
                        ('cord19_abstract', 'abstract'),
                        ('pubmed', 'abstract'),
                        ('cord19_pdf', 'fulltext'),
                        ('elsevier', 'fulltext'),
                        ('cord19_pmc_xml', 'fulltext'),
                        ('manuscripts', 'fulltext'),
                        ('pmc_oa', 'fulltext')]


def get_filtered_rdg_stmts(stmt_nd, get_full_stmts, linked_sids=None):
    """Get the set of statements/ids from readings minus exact duplicates."""
    logger.info("Filtering the statements from reading.")
    if linked_sids is None:
        linked_sids = set()

    # Now we filter and get the set of statements/statement ids.
    bad_dups = set()
    stmt_tpls = set()
    bettered_duplicate_sids = set()  # Statements with "better" alternatives
    for trid, src_dict in stmt_nd.items():
        some_bettered_duplicate_tpls = set()

        # Filter out the older reader versions
        for reader, rv_list in reader_versions.items():
            simple_src_dict = defaultdict(dict)
            for (src, _, _), rv_dict in src_dict.get_paths(reader):
                best_rv = max(rv_dict, key=lambda x: rv_list.index(x))

                # Record the rest of the statement ids.
                for rv, r_dict in rv_dict.items():
                    if rv != best_rv:
                        some_bettered_duplicate_tpls |= r_dict.get_leaves()
                    else:
                        for h, stmt_set in list(r_dict.values())[0].items():
                            # Sort the statements by their source, and whether
                            # they are new or old, defined as whether they
                            # have yet been included in preassembly. There
                            # should be no overlap in hashes here.
                            sid_set = {sid for sid, _ in stmt_set}
                            if sid_set < linked_sids:
                                simple_src_dict[src][h] = ('old', stmt_set)
                            elif sid_set.isdisjoint(linked_sids):
                                simple_src_dict[src][h] = ('new', stmt_set)
                            else:
                                # If you ever see this pop up, something very
                                # strange has happened. It means that some
                                # statements from a given reading were already
                                # included in preassembly, but others were not.
                                # There is no mechanism that should accept only
                                # some statements from within a reading, so
                                # that should never happen, but Murphy will
                                # always win the day.
                                assert False, \
                                    "Found reading partially included."

            # Choose the statements to propagate
            new_stmt_dict = {}
            for src in reversed(text_content_sources):
                for h, (status, s_set) in simple_src_dict[src].items():
                    # If this error ever comes up, it means that the uniqueness
                    # constraint on raw statements per reading is not
                    # functioning correctly.
                    if len(s_set) > 1:
                        logger.warning("Found exact duplicates from the same "
                                       "reading: %s" % str(s_set))
                        bad_dups.add(tuple(s_set))

                    # Choose whether to keep the statement or not.
                    s_tpl = s_set.pop()
                    if h not in new_stmt_dict:
                        # No conflict, no problem
                        new_stmt_dict[h] = s_tpl
                    elif status == 'old':
                        # The same statement was newly found by a better
                        # version.
                        some_bettered_duplicate_tpls.add(s_tpl)
            stmt_tpls |= set(new_stmt_dict.values())

        # Add the bettered duplicates found in this round.
        bettered_duplicate_sids |= \
            {sid for sid, _ in some_bettered_duplicate_tpls}

    # Dump the bad duplicates, if any
    if bad_dups:
        with open('bad_duplicates_%s.pkl' % datetime.now(), 'wb') as f:
            pickle.dump(bad_dups, f)

    if get_full_stmts:
        stmts = {stmt for _, stmt in stmt_tpls if stmt is not None}
        assert len(stmts) == len(stmt_tpls), \
            ("Some statements were None! The interaction between "
             "_get_reading_statement_dict and _filter_rdg_statements was "
             "probably mishandled.")
    else:
        stmts = {sid for sid, _ in stmt_tpls}

    return stmts, bettered_duplicate_sids


def get_filtered_db_stmts(db, get_full_stmts=False, clauses=None):
    """Get the set of statements/ids from databases minus exact duplicates."""
    # Only get the json if it's going to be used.
    if get_full_stmts:
        tbl_list = [db.RawStatements.json]
    else:
        tbl_list = [db.RawStatements.id]

    db_s_q = db.filter_query(tbl_list, db.RawStatements.db_info_id.isnot(None))

    # Add any other criterion specified at higher levels.
    if clauses:
        db_s_q = db_s_q.filter(*clauses)

    # Produce a generator of statement groups.
    db_stmt_data = db_s_q.yield_per(10000)
    if get_full_stmts:
        return {Statement._from_json(json.loads(s_json.decode('utf-8')))
                for s_json, in db_stmt_data}
    else:
        return {sid for sid, in db_stmt_data}


@clockit
def distill_stmts(db, get_full_stmts=False, clauses=None,
                  handle_duplicates='error'):
    """Get a corpus of statements from clauses and filters duplicate evidence.

    Parameters
    ----------
    db : :py:class:`DatabaseManager`
        A database manager instance to access the database.
    get_full_stmts : bool
        By default (False), only Statement ids (the primary index of Statements
        on the database) are returned. However, if set to True, serialized
        INDRA Statements will be returned. Note that this will in general be
        VERY large in memory, and therefore should be used with caution.
    clauses : None or list of sqlalchemy clauses
        By default None. Specify sqlalchemy clauses to reduce the scope of
        statements, e.g. `clauses=[db.Statements.type == 'Phosphorylation']` or
        `clauses=[db.Statements.uuid.in_([<uuids>])]`.
    handle_duplicates : 'error', 'delete', or a string file path
        Choose whether you want to delete the statements that are found to be
        duplicates ('delete'), or write a pickle file with their ids (at the
        string file path) for later handling, or raise an exception ('error').
        The default behavior is 'error'.

    Returns
    -------
    stmt_ret : set
        A set of either statement ids or serialized statements, depending on
        `get_full_stmts`.
    """
    if handle_duplicates == 'delete' or handle_duplicates == 'error':
        logger.info("Looking for ids from existing links...")
        linked_sids = {sid for sid,
                       in db.select_all(db.RawUniqueLinks.raw_stmt_id)}
    else:
        linked_sids = set()

    # Get de-duplicated Statements, and duplicate uuids, as well as uuid of
    # Statements that have been improved upon...
    logger.info("Sorting reading statements...")
    stmt_nd = get_reading_stmt_dict(db, clauses, get_full_stmts)

    stmts, bettered_duplicate_sids = \
        get_filtered_rdg_stmts(stmt_nd, get_full_stmts, linked_sids)
    logger.info("After filtering reading: %d unique statements, and %d with "
                "results from better resources available."
                % (len(stmts), len(bettered_duplicate_sids)))
    del stmt_nd  # This takes up a lot of memory, and is done being used.

    db_stmts = get_filtered_db_stmts(db, get_full_stmts, clauses)
    stmts |= db_stmts

    # Remove support links for statements that have better versions available.
    bad_link_sids = bettered_duplicate_sids & linked_sids
    if len(bad_link_sids):
        logger.error("Found pre-existing evidence links that were bettered...")
        logger.info("Removing the links...")
        rm_links = db.select_all(
            db.RawUniqueLinks,
            db.RawUniqueLinks.raw_stmt_id.in_(bad_link_sids)
        )
        db.delete_all(rm_links)

    return stmts


def delete_raw_statements_by_id(db, raw_sids, sync_session=False,
                                remove='all'):
    """Delete raw statements, their agents, and their raw-unique links.

    It is best to batch over this function with sets of 1000 or so ids. Setting
    sync_session to False will result in a much faster resolution, but you may
    find some ORM objects have not been updated.
    """
    if remove == 'all':
        remove = ['links', 'agents', 'statements']

    # First, delete the evidence links.
    if 'links' in remove:
        ev_q = db.filter_query(db.RawUniqueLinks,
                               db.RawUniqueLinks.raw_stmt_id.in_(raw_sids))
        logger.info("Deleting any connected evidence links...")
        ev_q.delete(synchronize_session=sync_session)

    # Second, delete the agents.
    if 'agents' in remove:
        ag_q = db.filter_query(db.RawAgents,
                               db.RawAgents.stmt_id.in_(raw_sids))
        logger.info("Deleting all connected agents...")
        ag_q.delete(synchronize_session=sync_session)

    # Now finally delete the statements.
    if 'statements' in remove:
        raw_q = db.filter_query(db.RawStatements,
                                db.RawStatements.id.in_(raw_sids))
        logger.info("Deleting all raw indicated statements...")
        raw_q.delete(synchronize_session=sync_session)
    return
