__all__ = ['get_direct_raw_stmt_jsons_from_agents',
           'get_raw_stmt_jsons_from_papers', 'get_raw_jsons']

import json
from collections import defaultdict

from sqlalchemy import intersect_all

from indra.util import clockit

from indra_db import get_db
from indra_db.util import regularize_agent_id

# ====
# API
# ====


@clockit
def get_raw_stmt_jsons_from_papers(id_list, id_type='pmid', db=None,
                                   max_stmts=None, offset=None):
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
        db = get_db('primary')

    # Get the attribute for this id type.
    if id_type == 'pmid':
        id_constraint = db.TextRef.pmid_in(id_list, filter_ids=True)
    elif id_type == 'pmcid':
        id_constraint = db.TextRef.pmcid_in(id_list, filter_ids=True)
    elif id_type == 'doi':
        id_constraint = db.TextRef.doi_in(id_list, filter_ids=True)
    else:
        id_constraint = _get_id_col(db.TextRef, id_type).in_(id_list)

    # Get the results.
    res = db.select_all([db.TextRef, db.RawStatements.json], id_constraint,
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
        ev = rjson['evidence'][0]
        if 'text_refs' not in ev.keys():
            ev['text_refs'] = {}
        for idt in ['trid', 'pmid', 'pmcid', 'doi']:
            ev['text_refs'][idt.upper()] = _get_id_col(tr, idt)

        # Add this to the results.
        result_dict[id_val].append(rjson)

    return result_dict


@clockit
def get_raw_stmt_jsons_from_agents(agents=None, stmt_type=None, db=None,
                                   max_stmts=None, offset=None):
    """Get Raw statement jsons from a list of agent refs and Statement type."""
    if db is None:
        db = get_db('primary')

    # Turn the agents parameters into an intersection of queries for stmt ids.
    entity_queries = []
    for role, ag_dbid, ns in agents:
        # Make the id match paradigms for the database.
        ag_dbid = regularize_agent_id(ag_dbid, ns)

        # Sanitize wildcards.
        for char in ['%', '_']:
            ag_dbid = ag_dbid.replace(char, '\%s' % char)

        # Generate the query
        q = db.session.query(
            db.RawAgents.stmt_id.label('stmt_id')
        ).filter(
            db.RawAgents.db_id.like(ag_dbid)
        )

        if ns is not None:
            q = q.filter(db.RawAgents.db_name.like(ns))

        if role is not None:
            q = q.filter(db.RawAgents.role == role.upper())

        entity_queries.append(q)

    # Add a constraint for the statement type.
    if stmt_type is not None:
        q = db.session.query(
            db.RawStatements.id.label('stmt_id')
        ).filter(
            db.RawStatements.type == stmt_type
        )
        entity_queries.append(q)

    # Generate the sub-query.
    ag_query_al = intersect_all(*entity_queries).alias('intersection')
    ag_query = db.session.query(ag_query_al).distinct().subquery('ag_stmt_ids')

    # Get the raw statement JSONs from the database.
    res = get_raw_stmt_jsons([db.RawStatements.id == ag_query.c.stmt_id], db=db,
                             max_stmts=max_stmts, offset=offset)
    return res


def get_raw_stmt_jsons(clauses=None, db=None, max_stmts=None, offset=None):
    """Get Raw Statements from the principle database, given arbitrary clauses."""
    if db is None:
        db = get_db('primary')

    if clauses is None:
        clauses = []

    q = db.session.query(
        db.RawStatements.id,
        db.RawStatements.json,
        db.Reading.id,
        db.TextContennt.id,
        db.TextRef
    ).filter(
        *clauses
    ).outerjoind(
        db.Reading,
        db.Reading.id == db.RawStatements.reading_id
    ).join(
        db.TextContent,
        db.TextContent.id == db.Reading.text_content_id
    ).join(
        db.TextRef,
        db.TextRef.id == db.TextContent.text_ref_id
    )

    if max_stmts is not None:
        q = q.limit(max_stmts)

    if offset is not None:
        q = q.offset(offset)

    raw_stmt_jsons = {}
    for sid, json_bytes, rid, tcid, tr in q.all():
        raw_j = json.loads(json_bytes)
        ev = raw_j['evidence'][0]
        ev['text_refs'] = tr.get_ref_dict()
        ev['text_refs']['TCID'] = tcid
        ev['text_refs']['READING_ID'] = rid
        if tr.pmid:
            ev['pmid'] = tr.pmid
        raw_stmt_jsons[sid] = raw_j

    return raw_stmt_jsons


# ======
# Tools
# ======


def _get_id_col(tr, id_type):
    if id_type == 'trid':
        id_attr = tr.id
    else:
        try:
            id_attr = getattr(tr, id_type)
        except AttributeError:
            raise ValueError("Invalid id_type: %s" % id_type)
    return id_attr

