__all__ = ["get_pa_stmt_jsons"]

import json
from collections import defaultdict

from sqlalchemy import func, cast, String, null
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.orm import aliased

from indra_db.util.constructors import get_db
from indra_db.client.principal.raw_statements import _fix_evidence


def get_pa_stmt_jsons(clauses=None, with_evidence=True, db=None, limit=1000):
    """Load preassembled Statements from the principal database."""
    if db is None:
        db = get_db('primary')

    if clauses is None:
        clauses = []

    # Construct the core query.
    if with_evidence:
        core_q = db.session.query(
            db.PAStatements.mk_hash.label('mk_hash'),
            db.PAStatements.json.label('json'),
            func.array_agg(db.RawStatements.json).label("raw_jsons"),
            func.array_agg(db.Reading.id).label("rids"),
            func.array_agg(db.TextContent.id).label("tcids"),
            func.array_agg(db.TextRef).label("text_refs")
        ).outerjoin(
            db.RawUniqueLinks,
            db.RawUniqueLinks.pa_stmt_mk_hash == db.PAStatements.mk_hash
        ).join(
            db.RawStatements,
            db.RawStatements.id == db.RawUniqueLinks.raw_stmt_id
        ).outerjoin(
            db.Reading,
            db.Reading.id == db.RawStatements.reading_id
        ).join(
            db.TextContent,
            db.TextContent.id == db.Reading.text_content_id
        ).join(
            db.TextRef,
            db.TextRef.id == db.TextContent.text_ref_id
        )
    else:
        core_q = db.session.query(
            db.PAStatements.mk_hash.label('mk_hash'),
            db.PAStatements.json.label('json'),
            null().label('raw_jsons'),
            null().label('rids'),
            null().label('tcids'),
            null().label('text_refs')
        )
    core_q = core_q.filter(
        *clauses
    ).group_by(
        db.PAStatements.mk_hash,
        db.PAStatements.json
    )
    if limit:
        core_q = core_q.limit(limit)
    core_sq = core_q.subquery.alias('core')

    # Construct the layer of the query that gathers agent info.
    agent_tuple = (cast(db.PAAgents.ag_num, String),
                   db.PAAgents.db_name,
                   db.PAAgents.db_id)
    at_sq = db.session.query(
        core_sq.c.mk_hash,
        core_sq.c.json,
        core_sq.c.raw_jsons,
        core_sq.c.rids,
        core_sq.c.tcids,
        core_sq.c.text_refs,
        func.array_agg(array(agent_tuple)).label('db_refs')
    ).filter(
        db.PAAgents.stmt_mk_hash == core_sq.c.mk_hash
    ).group_by(
        core_sq.c.mk_hash,
        core_sq.c.json,
        core_sq.c.raw_jsons
    ).subquery().alias('agent_tuples')

    # Construct the layer of the query that gathers supports/supported by.
    sup_from = aliased(db.PASupportLinks, name='sup_from')
    sup_to = aliased(db.PASupportLinks, name='sup_to')
    q = db.session.query(
        at_sq.c.mk_hash,
        at_sq.c.json,
        at_sq.c.raw_jsons,
        at_sq.c.rids,
        at_sq.c.tcids,
        at_sq.c.text_refs,
        at_sq.c.db_refs,
        func.array_agg(sup_from.supporting_mk_hash).label('supporting_hashes'),
        func.array_agg(sup_to.supported_mk_hash).label('supported_hashes')
    ).outerjoin(
        sup_from,
        sup_from.supported_mk_hash == at_sq.c.mk_hash
    ).outerjoin(
        sup_to,
        sup_to.supporting_mk_hash == at_sq.c.mk_hash
    ).group_by(
        at_sq.c.mk_hash,
        at_sq.c.json,
        at_sq.c.raw_jsons,
        at_sq.c.db_refs
    )

    # Run and parse the query.
    stmt_jsons = {}
    stmts_by_hash = {}
    for h, sj, rjs, rids, tcids, text_refs, db_refs, supping, supped in q.all():
        # Gather the agent refs.
        db_ref_dicts = defaultdict(lambda: defaultdict(list))
        for ag_num, db_name, db_id in db_refs:
            db_ref_dicts[int(ag_num)][db_name].append(db_id)
        db_ref_dicts = {k: dict(v) for k, v in db_ref_dicts.items()}

        # Parse the JSON bytes into JSON.
        stmt_json = json.loads(sj)

        # Load the evidence.
        if rjs is not None:
            for rj, rid, tcid, tr in zip(rjs, rids, tcids, text_refs):
                raw_json = json.loads(rj)
                ev = raw_json['evidence'][0]
                _fix_evidence(ev, rid, tcid, tr)
                stmt_json['evidence'].append(ev)

        # Resolve supports supported-by, as much as possible.
        stmts_by_hash[h] = stmt_json
        for h in (h for h in supping if h in stmts_by_hash):
            stmt_json['supports'].append(stmts_by_hash[h])
            stmts_by_hash[h]['supported_by'].append(stmt_json)
        for h in (h for h in supped if h in stmts_by_hash):
            stmt_json['supported_by'].append(stmts_by_hash[h])
            stmts_by_hash[h]['supports'].append(stmt_json)

        # Put it together in a dictionary.
        result_dict = {
            "mk_hash": h,
            "stmt": stmt_json,
            "db_refs": db_ref_dicts,
            "supports_hashes": supping,
            "supported_by_hashes": supped
        }
        stmt_jsons[h] = result_dict
    return stmt_jsons
