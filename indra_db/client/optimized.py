import json
import logging
from collections import OrderedDict
from sqlalchemy import or_, desc, true, select

from indra.statements import get_statement_by_name

logger = logging.getLogger('db_optimized_client')

from indra.util import clockit
from indra_db.util import get_primary_db, regularize_agent_id


@clockit
def _get_pa_stmt_jsons_w_mkhash_subquery(db, mk_hashes_q, best_first=True,
                                         max_stmts=None, offset=None,
                                         ev_limit=None):
    # Handle limiting.
    mk_hashes_q = mk_hashes_q.distinct()
    if best_first:
        mk_hashes_q = mk_hashes_q.order_by(desc(db.PaMeta.ev_count))
    if max_stmts is not None:
        mk_hashes_q = mk_hashes_q.limit(max_stmts)
    if offset is not None:
        mk_hashes_q = mk_hashes_q.offset(offset)

    # Create the link
    mk_hashes_al = mk_hashes_q.subquery('mk_hashes')
    raw_json_c = db.FastRawPaLink.raw_json.label('raw_json')
    pa_json_c = db.FastRawPaLink.pa_json.label('pa_json')
    reading_id_c = db.FastRawPaLink.reading_id.label('rid')
    cont_q = db.session.query(raw_json_c, pa_json_c, reading_id_c)
    cont_q = cont_q.filter(db.FastRawPaLink.mk_hash == mk_hashes_al.c.mk_hash)

    if ev_limit is not None:
        cont_q = cont_q.limit(ev_limit)

    # TODO: Only make a lateral-joined query when evidence is limited.
    json_content_al = cont_q.subquery().lateral('json_content')

    stmts_q = (mk_hashes_al
               .outerjoin(json_content_al, true())
               .outerjoin(db.ReadingRefLink,
                          db.ReadingRefLink.rid == json_content_al.c.rid))

    selection = (select([mk_hashes_al.c.mk_hash, mk_hashes_al.c.ev_count,
                         json_content_al.c.raw_json, json_content_al.c.pa_json,
                         db.ReadingRefLink.pmid, db.ReadingRefLink.source])
                 .select_from(stmts_q))
    logger.debug("Executing sql to get statements:\n%s" % str(selection))

    proxy = db.session.connection().execute(selection)
    res = proxy.fetchall()

    stmts_dict = OrderedDict()
    ev_totals = OrderedDict()
    total_evidence = 0
    returned_evidence = 0
    for mk_hash, ev_count, raw_json_bts, pa_json_bts, pmid, src in res:
        returned_evidence += 1
        raw_json = json.loads(raw_json_bts.decode('utf-8'))
        ev_json = raw_json['evidence'][0]

        # Add a new statements if the hash is new
        if mk_hash not in stmts_dict.keys():
            total_evidence += ev_count
            ev_totals[mk_hash] = ev_count
            stmts_dict[mk_hash] = json.loads(pa_json_bts.decode('utf-8'))
            stmts_dict[mk_hash]['evidence'] = []

        # Fix the pmid
        if pmid:
            ev_json['pmid'] = pmid

        # Add agents' raw text to annotations.
        raw_text = []
        for ag_name in get_statement_by_name(raw_json['type'])._agent_order:
            ag_value = raw_json.get(ag_name, None)
            if isinstance(ag_value, dict):
                raw_text.append(ag_value['db_refs'].get('TEXT'))
            elif ag_value is None:
                raw_text.append(None)
            else:
                for ag in ag_value:
                    raw_text.append(ag['db_refs'].get('TEXT'))
        if 'annotations' not in ev_json.keys():
            ev_json['annotations'] = {}
        ev_json['annotations']['agents'] = {'raw_text': raw_text}
        if 'prior_uuids' not in ev_json['annotations'].keys():
            ev_json['annotations']['prior_uuids'] = []
        ev_json['annotations']['prior_uuids'].append(raw_json['id'])

        if src:
            ev_json['annotations']['content_source'] = src

        stmts_dict[mk_hash]['evidence'].append(ev_json)

    ret = {'statements': stmts_dict,
           'evidence_totals': ev_totals,
           'total_evidence': total_evidence,
           'evidence_returned': returned_evidence}
    return ret


@clockit
def get_statement_jsons_from_agents(agents=None, stmt_type=None, db=None,
                                    **kwargs):
    """Get json's for statements given agent refs and Statement type.

    Parameters
    ----------
    agents : list[(<role>, <id>, <namespace>)]
        A list of agents, each specified by a tuple of information including:
        the `role`, which can be 'subject', 'object', or None, an `id`, such as
        the HGNC id, a CHEMBL id, or a FPLX id, etc, and the
        `namespace` which specifies which of the above is given in `id`.

        Some examples:
            (None, 'MEK', 'FPLX')
            ('object', '11998', 'HGNC')
            ('subject', 'MAP2K1', 'TEXT')

        Note that you will get the logical AND of the conditions given, in
        other words, each Statement will satisfy all constraints.
    stmt_type : str or None
        The type of statement to retrieve, e.g. 'Phosphorylation'. If None, no
        type restriction is imposed.
    db : :py:class:`DatabaseManager`
        Optionally specify a database manager that attaches to something
        besides the primary database, for example a local database instance.

    Some keyword arguments are passed directly to a lower level function:

    Other Parameters (kwargs)
    -------------------------
    max_stmts : int or None
        Limit the number of statements queried. If None, no restriction is
        applied.
    offset : int or None
        Start reading statements by a given offset. If None, no offset is
        applied. Most commonly used in conjunction with `max_stmts`.
    ev_limit : int or None
        Limit the amount of evidence returned per Statement.
    best_first : bool
        If True, the preassembled statements will be sorted by the amount of
        evidence they have, and those with the most evidence will be
        prioritized. When using `max_stmts`, this means you will get the "best"
        statements. If False, statements will be queried in arbitrary order.

    Returns
    -------
    A dictionary data structure containing, among other metadata, a dict of
    statement jsons under the key 'statements', themselves keyed by their
    shallow matches-key hashes.
    """
    # First look for statements matching the role'd agents.
    if db is None:
        db = get_primary_db()

    # TODO: Extend this to allow retrieval of raw statements.
    mk_hashes_q = None
    mk_hash_c = db.PaMeta.mk_hash.label('mk_hash')
    ev_count_c = db.PaMeta.ev_count.label('ev_count')
    for role, ag_dbid, ns in agents:
        # Make the id match paradigms for the database.
        ag_dbid = regularize_agent_id(ag_dbid, ns)

        # Create this query (for this agent)
        q = (db.session
             .query(mk_hash_c, ev_count_c)
             .filter(db.PaMeta.db_id.like(ag_dbid),
                     db.PaMeta.db_name.like(ns)))
        if stmt_type is not None:
            q = q.filter(db.PaMeta.type.like(stmt_type))

        if role is not None:
            q = q.filter(db.PaMeta.role == role.upper())

        # Intersect with the previous query.
        if mk_hashes_q:
            mk_hashes_q = mk_hashes_q.intersect(q)
        else:
            mk_hashes_q = q
    assert mk_hashes_q, "No conditions imposed."

    return _get_pa_stmt_jsons_w_mkhash_subquery(db, mk_hashes_q, **kwargs)


@clockit
def get_statement_jsons_from_papers(paper_refs, db=None, **kwargs):
    """Get the statements from a list of papers.

    Parameters
    ----------
    paper_refs : list[(<id_type>, <paper_id>)]
        A list of tuples, where each tuple indicates and id-type (e.g. 'pmid')
        and an id value for a particular paper.
    db : :py:class:`DatabaseManager`
        Optionally specify a database manager that attaches to something
        besides the primary database, for example a local databse instance.

    Some keyword arguments are passed directly to a lower level function:

    Other Parameters (kwargs)
    -------------------------
    max_stmts : int or None
        Limit the number of statements queried. If None, no restriction is
        applied.
    offset : int or None
        Start reading statements by a given offset. If None, no offset is
        applied. Most commonly used in conjunction with `max_stmts`.
    ev_limit : int or None
        Limit the amount of evidence returned per Statement.
    best_first : bool
        If True, the preassembled statements will be sorted by the amount of
        evidence they have, and those with the most evidence will be
        prioritized. When using `max_stmts`, this means you will get the "best"
        statements. If False, statements will be queried in arbitrary order.

    Returns
    -------
    A dictionary data structure containing, among other metadata, a dict of
    statement jsons under the key 'statements', themselves keyed by their
    shallow matches-key hashes.
    """
    if db is None:
        db = get_primary_db()

    # Create a sub-query on the reading metadata
    q = db.session.query(db.ReadingRefLink.rid.label('rid'))
    conditions = []
    for id_type, paper_id in paper_refs:
        tbl_attr = getattr(db.ReadingRefLink, id_type)
        if id_type in ['trid', 'tcid']:
            conditions.append(tbl_attr == paper_id)
        else:
            conditions.append(tbl_attr.like(paper_id))
    q = q.filter(or_(*conditions))
    sub_al = q.subquery('reading_ids')

    # Map the reading metadata query to mk_hashes with statement counts.
    mk_hashes_q = (db.session
                   .query(db.PaMeta.mk_hash.label('mk_hash'),
                          db.PaMeta.ev_count.label('ev_count'))
                   .filter(db.PaMeta.mk_hash == db.FastRawPaLink.mk_hash,
                           db.FastRawPaLink.reading_id == sub_al.c.rid))

    return _get_pa_stmt_jsons_w_mkhash_subquery(db, mk_hashes_q, **kwargs)


@clockit
def get_statement_jsons_from_hashes(mk_hashes, db=None, **kwargs):
    """Get statement jsons using the appropriate hashes."""
    if db is None:
        db = get_primary_db()
    mk_hashes_q = (db.session.query(db.PaMeta.mk_hash, db.PaMeta.ev_count)
                   .filter(db.PaMeta.mk_hash.in_(mk_hashes)))
    return _get_pa_stmt_jsons_w_mkhash_subquery(db, mk_hashes_q, **kwargs)
