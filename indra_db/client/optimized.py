import json
import logging
from collections import OrderedDict, defaultdict
from sqlalchemy import or_, desc, true, select, intersect_all

logger = logging.getLogger(__file__)

from indra.util import clockit
from indra.statements import get_statement_by_name, make_hash
from indra_db.util import get_primary_db, regularize_agent_id


def _get_id_col(tr, id_type):
    if id_type == 'trid':
        id_attr = tr.id
    else:
        try:
            id_attr = getattr(tr, id_type)
        except AttributeError:
            raise ValueError("Invalid id_type: %s" % id_type)
    return id_attr


@clockit
def get_raw_stmt_jsons_from_papers(id_list, id_type='pmid', db=None):
    """Get raw statement jsons for a given list of papers.

    Parameters
    ----------
    id_list : list
        A list of ints or strs that are ids of papers of type `id_type`.
    id_type : str
        Default is 'pmid'. The type of ids given in id_list, e.g. 'pmid',
        'pmcid', 'trid'.
    db : :py:class:`DatabaseManager`
        Optionally specify a database manager that attaches to something
        besides the primary database, for example a local database instance.

    Returns
    -------
    result_dict : dict
        A dictionary keyed by id (of `id_type`) with a list of raw statement
        json objects as each value. Ids for which no statements are found will
        not be included in the dict.
    """
    if db is None:
        db = get_primary_db()

    # Get the attribute for this id type.
    id_attr = _get_id_col(db.TextRef, id_type)

    # Get the results.
    res = db.select_all([db.TextRef, db.RawStatements.json],
                        id_attr.in_(id_list),
                        *db.link(db.RawStatements, db.TextRef))

    # Organized the results into a dict of lists keyed by id value.
    # Fix pmids along the way.
    result_dict = defaultdict(list)
    for tr, rjson_bytes in res:
        id_val = _get_id_col(tr, id_type)

        # Decode and unpack the json
        rjson = json.loads(rjson_bytes.decode('utf-8'))

        # Fix the pmids in this json.
        rjson['evidence'][0]['pmid'] = tr.pmid

        # Set the text_refs in this json
        for idt in ['trid', 'pmid', 'pmcid', 'doi']:
            rjson['evidence'][0]['text_refs'][idt] = _get_id_col(tr, idt)

        # Add this to the results.
        result_dict[id_val].append(rjson)

    return result_dict


@clockit
def _get_pa_stmt_jsons_w_mkhash_subquery(db, mk_hashes_q, best_first=True,
                                         max_stmts=None, offset=None,
                                         ev_limit=None, mk_hashes_alias=None,
                                         ev_count_obj=None):
    # Handle limiting.
    mk_hashes_q = mk_hashes_q.distinct()
    if best_first:
        if ev_count_obj is not None:
            mk_hashes_q.order_by(desc(ev_count_obj))
        elif mk_hashes_alias is not None:
            mk_hashes_q = mk_hashes_q.order_by(desc(mk_hashes_alias.c.ev_count))
        else:
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

    ref_link_keys = [k for k in db.ReadingRefLink.__dict__.keys()
                     if not k.startswith('_')]
    selection = (select([mk_hashes_al.c.mk_hash, mk_hashes_al.c.ev_count,
                         json_content_al.c.raw_json, json_content_al.c.pa_json]
                        + [getattr(db.ReadingRefLink, k) for k in ref_link_keys])
                 .select_from(stmts_q))
    logger.debug("Executing sql to get statements:\n%s" % str(selection))

    proxy = db.session.connection().execute(selection)
    res = proxy.fetchall()

    stmts_dict = OrderedDict()
    ev_totals = OrderedDict()
    total_evidence = 0
    returned_evidence = 0
    if res:
        logger.debug("res is %d row by %d cols." % (len(res), len(res[0])))
    else:
        logger.debug("res is empty.")

    for row in res:
        mk_hash, ev_count, raw_json_bts, pa_json_bts = row[:4]
        ref_dict = {ref_link_keys[i]: row[4+i] for i in range(len(ref_link_keys))}
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
        if ref_dict['pmid']:
            ev_json['pmid'] = ref_dict['pmid']

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
        if 'text_refs' not in ev_json.keys():
            ev_json['text_refs'] = {}
        ev_json['text_refs'].update({k.upper(): v
                                     for k, v in ref_dict.items()
                                     if v is not None})

        if ref_dict['source']:
            ev_json['annotations']['content_source'] = ref_dict['source']

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
    def labelled_hash_and_count(meta):
        return meta.mk_hash.label('mk_hash'), meta.ev_count.label('ev_count')

    queries = []
    for role, ag_dbid, ns in agents:
        # Make the id match paradigms for the database.
        ag_dbid = regularize_agent_id(ag_dbid, ns)

        # Create this query (for this agent)
        # If we are looking for a name...
        if ns == 'NAME':
            q = (db.session
                 .query(*labelled_hash_and_count(db.NameMeta))
                 .filter(db.NameMeta.db_id.like(ag_dbid)))
            meta = db.NameMeta
        elif ns == 'TEXT':
            q = (db.session
                 .query(*labelled_hash_and_count(db.TextMeta))
                 .filter(db.TextMeta.db_id.like(ag_dbid)))
            meta = db.TextMeta
        else:
            q = (db.session
                 .query(*labelled_hash_and_count(db.OtherMeta))
                 .filter(db.OtherMeta.db_id.like(ag_dbid),
                         db.OtherMeta.db_name.like(ns)))
            meta = db.OtherMeta

        if stmt_type is not None:
            q = q.filter(meta.type.like(stmt_type))

        if role is not None:
            q = q.filter(meta.role == role.upper())

        # Intersect with the previous query.
        queries.append(q)
    assert queries, "No conditions imposed."

    mk_hashes_al = intersect_all(*queries).alias('intersection')
    mk_hashes_q = db.session.query(mk_hashes_al)

    return _get_pa_stmt_jsons_w_mkhash_subquery(db, mk_hashes_q,
                                                mk_hashes_alias=mk_hashes_al,
                                                **kwargs)


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
            conditions.append(tbl_attr == int(paper_id))
        else:
            conditions.append(tbl_attr.like(paper_id))
    q = q.filter(or_(*conditions))
    sub_al = q.subquery('reading_ids')

    # Map the reading metadata query to mk_hashes with statement counts.
    mk_hashes_al = db.EvidenceCounts.mk_hash.label('mk_hash')
    mk_hashes_q = (db.session
                   .query(mk_hashes_al,
                          db.EvidenceCounts.ev_count.label('ev_count'))
                   .filter(db.EvidenceCounts.mk_hash == db.FastRawPaLink.mk_hash,
                           db.FastRawPaLink.reading_id == sub_al.c.rid))

    return _get_pa_stmt_jsons_w_mkhash_subquery(db, mk_hashes_q,
                                                ev_count_obj=db.EvidenceCounts.ev_count,
                                                **kwargs)


@clockit
def get_statement_jsons_from_hashes(mk_hashes, db=None, **kwargs):
    """Get statement jsons using the appropriate hashes."""
    if db is None:
        db = get_primary_db()
    mk_hash_ints = [int(h) for h in mk_hashes]
    mk_hashes_q = (db.session.query(db.PaMeta.mk_hash, db.PaMeta.ev_count)
                   .filter(db.PaMeta.mk_hash.in_(mk_hash_ints)))
    return _get_pa_stmt_jsons_w_mkhash_subquery(db, mk_hashes_q, **kwargs)
