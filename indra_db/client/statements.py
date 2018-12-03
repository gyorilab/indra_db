import json
import logging
from collections import defaultdict
from sqlalchemy import or_

from indra.statements import Unresolved, Evidence

logger = logging.getLogger('db_statements_client')

from indra.util import batch_iter, clockit
from indra.databases import hgnc_client

from indra_db.util import get_primary_db, get_raw_stmts_frm_db_list, \
    _get_statement_object, regularize_agent_id, _get_trids
from indra_db.client.datasets import get_statement_essentials


def get_statements_by_gene_role_type(agent_id=None, agent_ns='HGNC-SYMBOL',
                                     role=None, stmt_type=None, count=1000,
                                     db=None, do_stmt_count=False,
                                     preassembled=True, fix_refs=True,
                                     with_evidence=True, with_support=True,
                                     essentials_only=False):
    """Get statements from the DB by stmt type, agent, and/or agent role.

    Parameters
    ----------
    agent_id : str
        String representing the identifier of the agent from the given
        namespace. Note: if the agent namespace argument, `agent_ns`, is set
        to 'HGNC-SYMBOL', this function will treat `agent_id` as an HGNC gene
        symbol and perform an internal lookup of the corresponding HGNC ID.
        Default is 'HGNC-SYMBOL'.
    agent_ns : str
        Namespace for the identifier given in `agent_id`.
    role : str
        String corresponding to the role of the agent in the statement.
        Options are 'SUBJECT', 'OBJECT', or 'OTHER' (in the case of `Complex`,
        `SelfModification`, and `ActiveForm` Statements).
    stmt_type : str
        Name of the Statement class.
    count : int
        Number of statements to retrieve in each batch (passed to
        :py:func:`get_statements`).
    db : :py:class:`DatabaseManager`
        Optionally specify a database manager that attaches to something
        besides the primary database, for example a local databse instance.
    do_stmt_count : bool
        Whether or not to perform an initial statement counting step to give
        more meaningful progress messages.
    preassembled : bool
        If true, statements will be selected from the table of pre-assembled
        statements. Otherwise, they will be selected from the raw statements.
        Default is True.
    with_support : bool
        Choose whether to populate the supports and supported_by list
        attributes of the Statement objects. Generally results in slower
        queries.
    with_evidence : bool
        Choose whether or not to populate the evidence list attribute of the
        Statements. As with `with_support`, setting this to True will take
        longer.
    fix_refs : bool
        The paper refs within the evidence objects are not populated in the
        database, and thus must be filled using the relations in the database.
        If True (default), the `pmid` field of each Statement Evidence object
        is set to the correct PMIDs, or None if no PMID is available. If False,
        the `pmid` field defaults to the value populated by the reading
        system.
    essentials_only : bool
        Default is False. If True, retrieve only some metadata regarding the
        statements. Implicitly `with_support`, `with_evidence`, `fix_refs`, and
        `do_stmt_count` are all False, as none of the relevant features apply.

    Returns
    -------
    if essentials_only is False:
        list of Statements from the database corresponding to the query.
    else:
        list of tuples containing basic data from the statements.
    """
    if db is None:
        db = get_primary_db()

    if preassembled:
        Statements = db.PAStatements
        Agents = db.PAAgents
    else:
        Statements = db.RawStatements
        Agents = db.RawAgents

    if not (agent_id or role or stmt_type):
        raise ValueError('At least one of agent_id, role, or stmt_type '
                         'must be specified.')
    clauses = []
    if agent_id and agent_ns == 'HGNC-SYMBOL':
        hgnc_symbol = agent_id
        agent_id = hgnc_client.get_hgnc_id(hgnc_symbol)
        if not agent_id:
            logger.warning('Invalid gene name: %s' % hgnc_symbol)
            return []
        agent_ns = 'HGNC'
    agent_id = regularize_agent_id(agent_id, agent_ns)
    clauses.extend([Agents.db_name.like(agent_ns),
                    Agents.db_id.like(agent_id)])
    if role:
        clauses.append(Agents.role == role)
    if agent_id or role:
        if preassembled:
            clauses.append(Agents.stmt_mk_hash == Statements.mk_hash)
        else:
            clauses.append(Agents.stmt_id == Statements.id)
    if stmt_type:
        clauses.append(Statements.type == stmt_type)

    if essentials_only:
        stmts = get_statement_essentials(clauses, count=count, db=db,
                                         preassembled=preassembled)
    else:
        stmts = get_statements(clauses, count=count,
                               do_stmt_count=do_stmt_count, db=db,
                               preassembled=preassembled, fix_refs=fix_refs,
                               with_evidence=with_evidence,
                               with_support=with_support)
    return stmts


def get_statements_by_paper(id_val, id_type='pmid', count=1000, db=None,
                            do_stmt_count=False, preassembled=True):
    """Get the statements from a particular paper.

    Parameters
    ----------
    id_val : int or str
        The value of the id for the paper whose statements you wish to
        retrieve.
    id_type : str
        The type of id used (default is pmid). Options include pmid, pmcid,
        doi, pii, url, or manuscript_id. Note that pmid is generally the
        best means of getting a paper.
    count : int
        Number of statements to retrieve in each batch (passed to
        :py:func:`get_statements`).
    db : :py:class:`DatabaseManager`
        Optionally specify a database manager that attaches to something
        besides the primary database, for example a local databse instance.
    do_stmt_count : bool
        Whether or not to perform an initial statement counting step to give
        more meaningful progress messages.
    preassembled : bool
        If True, statements will be selected from the table of pre-assembled
        statements. Otherwise, they will be selected from the raw statements.
        Default is True.

    Returns
    -------
    A list of Statements from the database corresponding to the paper id given.
    """
    # TODO: Make this get from multiple papers.
    if db is None:
        db = get_primary_db()

    trid_list = _get_trids(db, id_val, id_type)
    if not trid_list:
        return None

    stmts = []
    for trid in trid_list:
        clauses = [db.TextContent.id == db.Reading.text_content_id,
                   db.Reading.id == db.RawStatements.reading_id,
                   db.TextContent.text_ref_id == trid]
        if preassembled:
            clauses += [
                db.RawStatements.id == db.RawUniqueLinks.raw_stmt_id,
                db.PAStatements.mk_hash == db.RawUniqueLinks.pa_stmt_mk_hash
                ]
        stmts.extend(get_statements(clauses, count=count, db=db,
                                    preassembled=preassembled,
                                    do_stmt_count=do_stmt_count))
    return stmts


@clockit
def get_statements(clauses, count=1000, do_stmt_count=False, db=None,
                   preassembled=True, with_support=False, fix_refs=True,
                   with_evidence=True):
    """Select statements according to a given set of clauses.

    Parameters
    ----------
    clauses : list
        list of sqlalchemy WHERE clauses to pass to the filter query.
    count : int
        Number of statements to retrieve and process in each batch.
    do_stmt_count : bool
        Whether or not to perform an initial statement counting step to give
        more meaningful progress messages.
    db : :py:class:`DatabaseManager`
        Optionally specify a database manager that attaches to something
        besides the primary database, for example a local database instance.
    preassembled : bool
        If true, statements will be selected from the table of pre-assembled
        statements. Otherwise, they will be selected from the raw statements.
        Default is True.
    with_support : bool
        Choose whether to populate the supports and supported_by list
        attributes of the Statement objects. General results in slower queries.
    with_evidence : bool
        Choose whether or not to populate the evidence list attribute of the
        Statements. As with `with_support`, setting this to True will take
        longer.
    fix_refs : bool
        The paper refs within the evidence objects are not populated in the
        database, and thus must be filled using the relations in the database.
        If True (default), the `pmid` field of each Statement Evidence object
        is set to the correct PMIDs, or None if no PMID is available. If False,
        the `pmid` field defaults to the value populated by the reading
        system.

    Returns
    -------
    list of Statements from the database corresponding to the query.
    """
    cnt = count
    if db is None:
        db = get_primary_db()

    stmts_tblname = 'pa_statements' if preassembled else 'raw_statements'

    if not preassembled:
        stmts = []
        q = db.filter_query(stmts_tblname, *clauses)
        if do_stmt_count:
            logger.info("Counting statements...")
            num_stmts = q.count()
            logger.info("Total of %d statements" % num_stmts)
        db_stmts = q.yield_per(cnt)
        for subset in batch_iter(db_stmts, cnt):
            stmts.extend(get_raw_stmts_frm_db_list(db, subset, with_sids=False,
                                                   fix_refs=fix_refs))
            if do_stmt_count:
                logger.info("%d of %d statements" % (len(stmts), num_stmts))
            else:
                logger.info("%d statements" % len(stmts))
    else:
        logger.info("Getting preassembled statements.")
        if with_evidence:
            logger.info("Getting preassembled statements.")
            # Get pairs of pa statements with their linked raw statements
            clauses += [
                db.PAStatements.mk_hash == db.RawUniqueLinks.pa_stmt_mk_hash,
                db.RawStatements.id == db.RawUniqueLinks.raw_stmt_id
                ]
            pa_raw_stmt_pairs = \
                db.select_all([db.PAStatements, db.RawStatements],
                              *clauses, yield_per=cnt)
            stmt_dict = _process_pa_statement_res_wev(db, pa_raw_stmt_pairs,
                                                      count=cnt,
                                                      fix_refs=fix_refs)
        else:
            # Get just pa statements without their supporting raw statement(s).
            pa_stmts = db.select_all(db.PAStatements, *clauses, yield_per=cnt)
            stmt_dict = _process_pa_statement_res_nev(db, pa_stmts, count=cnt)

        # Populate the supports/supported by fields.
        if with_support:
            get_support(stmt_dict, db=db)

        stmts = list(stmt_dict.values())
        logger.info("In all, there are %d pa statements." % len(stmts))

    return stmts


@clockit
def _process_pa_statement_res_wev(db, stmt_iterable, count=1000,
                                  fix_refs=True):
    # Iterate over the batches to create the statement objects.
    stmt_dict = {}
    ev_dict = {}
    raw_stmt_dict = {}
    total_ev = 0
    for stmt_pair_batch in batch_iter(stmt_iterable, count):
        # Instantiate the PA statement objects, and record the uuid
        # evidence (raw statement) links.
        raw_stmt_objs = []
        for pa_stmt_db_obj, raw_stmt_db_obj in stmt_pair_batch:
            k = pa_stmt_db_obj.mk_hash
            if k not in stmt_dict.keys():
                stmt_dict[k] = _get_statement_object(pa_stmt_db_obj)
                ev_dict[k] = [raw_stmt_db_obj.id,]
            else:
                ev_dict[k].append(raw_stmt_db_obj.id)
            raw_stmt_objs.append(raw_stmt_db_obj)
            total_ev += 1

        logger.info("Up to %d pa statements, with %d pieces of "
                    "evidence in all." % (len(stmt_dict), total_ev))

        # Instantiate the raw statements.
        raw_stmt_sid_tpls = get_raw_stmts_frm_db_list(db, raw_stmt_objs,
                                                      fix_refs,
                                                      with_sids=True)
        raw_stmt_dict.update({sid: s for sid, s in raw_stmt_sid_tpls})
        logger.info("Processed %d raw statements."
                    % len(raw_stmt_sid_tpls))

    # Attach the evidence
    logger.info("Inserting evidence.")
    for k, sid_list in ev_dict.items():
        stmt_dict[k].evidence = [raw_stmt_dict[sid].evidence[0]
                                 for sid in sid_list]
    return stmt_dict


@clockit
def _process_pa_statement_res_nev(stmt_iterable, count=1000):
    # Iterate over the batches to create the statement objects.
    stmt_dict = {}
    for stmt_pair_batch in batch_iter(stmt_iterable, count):
        # Instantiate the PA statement objects.
        for pa_stmt_db_obj in stmt_pair_batch:
            k = pa_stmt_db_obj.mk_hash
            if k not in stmt_dict.keys():
                stmt_dict[k] = _get_statement_object(pa_stmt_db_obj)

        logger.info("Up to %d pa statements in all." % len(stmt_dict))
    return stmt_dict


@clockit
def get_evidence(pa_stmt_list, db=None, fix_refs=True, use_views=True):
    """Fill in the evidence for a list of pre-assembled statements.

    Parameters
    ----------
    pa_stmt_list : list[Statement]
        A list of unique statements, generally drawn from the database
        pa_statement table (via `get_statemetns`).
    db : DatabaseManager instance or None
        An instance of a database manager. If None, defaults to the "primary"
        database, as defined in the db_config.ini file in .config/indra.
    fix_refs : bool
        The paper refs within the evidence objects are not populated in the
        database, and thus must be filled using the relations in the database.
        If True (default), the `pmid` field of each Statement Evidence object
        is set to the correct PMIDs, or None if no PMID is available. If False,
        the `pmid` field defaults to the value populated by the reading
        system.

    Returns
    -------
    None - modifications are made to the Statements "in-place".
    """
    if db is None:
        db = get_primary_db()

    # Turn the list into a dict.
    stmt_dict = {s.get_hash(shallow=True): s for s in pa_stmt_list}

    if use_views:
        if fix_refs:
            raw_links = db.select_all(
                [db.FastRawPaLink.mk_hash, db.FastRawPaLink.raw_json,
                 db.FastRawPaLink.reading_id],
                db.FastRawPaLink.mk_hash.in_(stmt_dict.keys())
                )
            rel_refs = ['pmid', 'rid']
            ref_cols = [getattr(db.ReadingRefLink, k) for k in rel_refs]
        else:
            raw_links = db.select_all(
                [db.FastRawPaLink.mk_hash, db.FastRawPaLink.raw_json],
                db.FastRawPaLink.mk_hash.in_(stmt_dict.keys())
                )
        rid_ref_dict = {}
        myst_rid_rs_dict = defaultdict(list)
        for info in raw_links:
            if fix_refs:
                mk_hash, raw_json, rid = info
            else:
                mk_hash, raw_json = info
                rid = None
            json_dict = json.loads(raw_json.decode('utf-8'))
            ev_json = json_dict.get('evidence', [])
            assert len(ev_json) == 1, \
                "Raw statements must have one evidence, got %d." % len(ev_json)
            ev = Evidence._from_json(ev_json[0])
            stmt_dict[mk_hash].evidence.append(ev)
            if fix_refs:
                ref_dict = rid_ref_dict.get(rid)
                if ref_dict is None:
                    myst_rid_rs_dict[rid].append(ev)
                    if len(myst_rid_rs_dict) >= 1000:
                        ref_data_list = db.select_all(
                            ref_cols,
                            db.ReadingRefLink.rid.in_(myst_rid_rs_dict.keys())
                            )
                        for pmid, rid in ref_data_list:
                            rid_ref_dict[rid] = pmid
                            for ev in myst_rid_rs_dict[rid]:
                                ev.pmid = pmid
                        myst_rid_rs_dict.clear()
                else:
                    ev.pmid = rid_ref_dict[rid]
    else:
        # Get the data from the database
        raw_list = db.select_all(
            [db.PAStatements.mk_hash, db.RawStatements],
            db.PAStatements.mk_hash.in_(stmt_dict.keys()),
            db.PAStatements.mk_hash == db.RawUniqueLinks.pa_stmt_mk_hash,
            db.RawUniqueLinks.raw_stmt_id == db.RawStatements.id
        )

        # Note that this step depends on the ordering being maintained.
        mk_hashes, raw_stmt_objs = zip(*raw_list)
        raw_stmts = get_raw_stmts_frm_db_list(db, raw_stmt_objs, fix_refs,
                                              with_sids=False)
        raw_stmt_mk_pairs = zip(mk_hashes, raw_stmts)

        # Now attach the evidence
        for mk_hash, raw_stmt in raw_stmt_mk_pairs:
            # Each raw statement can have just one piece of evidence.
            stmt_dict[mk_hash].evidence.append(raw_stmt.evidence[0])

    return


def get_statements_from_hashes(statement_hashes, preassembled=True, db=None,
                               **kwargs):
    """Retrieve statement objects given only statement hashes."""
    if db is None:
        db = get_primary_db()

    if preassembled:
        DbStatements = db.PAStatements
    else:
        DbStatements = db.RawStatements
    stmts = get_statements([DbStatements.mk_hash.in_(statement_hashes)], db=db,
                           preassembled=preassembled, **kwargs)
    return stmts


def get_support(statements, db=None, recursive=False):
    """Populate the supports and supported_by lists of the given statements."""
    # TODO: Allow recursive mode (argument should probably be an integer level)
    if db is None:
        db = get_primary_db()

    if not isinstance(statements, dict):
        stmt_dict = {s.get_hash(shallow=True): s for s in statements}
    else:
        stmt_dict = statements

    logger.info("Populating support links.")
    support_links = db.select_all(
        [db.PASupportLinks.supported_mk_hash,
         db.PASupportLinks.supporting_mk_hash],
        or_(db.PASupportLinks.supported_mk_hash.in_(stmt_dict.keys()),
            db.PASupportLinks.supporting_mk_hash.in_(stmt_dict.keys()))
    )
    for supped_hash, supping_hash in set(support_links):
        if supped_hash == supping_hash:
            assert False, 'Self-support found on-load.'
        supped_stmt = stmt_dict.get(supped_hash)
        if supped_stmt is None:
            supped_stmt = Unresolved(shallow_hash=supped_hash)
        supping_stmt = stmt_dict.get(supping_hash)
        if supping_stmt is None:
            supping_stmt = Unresolved(shallow_hash=supping_hash)
        supped_stmt.supported_by.append(supping_stmt)
        supping_stmt.supports.append(supped_stmt)
    return
