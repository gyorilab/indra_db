__all__ = ["get_pa_statements"]

import json
from collections import defaultdict

from sqlalchemy import func, cast, String, null
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.orm import aliased

from indra.statements import Statement
from indra_db.util.constructors import get_db


def get_pa_statements(clauses=None, with_evidence=True, db=None, limit=1000):
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
            func.array_agg(db.RawStatements.json).label("raw_jsons")
        ).filter(
            *db.link(db.PAAgents, db.RawStatements)
        )
    else:
        core_q = db.session.query(
            db.PAStatements.mk_hash.label('mk_hash'),
            db.PAStatements.json.label('json'),
            null().label('raw_jsons')
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
    result_dicts = []
    stmts_by_hash = {}
    for mk_hash, sj, raw_jsons, db_refs, supporting, supported in q.all():
        # Gather the agent refs.
        db_ref_dicts = defaultdict(lambda: defaultdict(list))
        for ag_num, db_name, db_id in db_refs:
            db_ref_dicts[int(ag_num)][db_name].append(db_id)
        db_ref_dicts = {k: dict(v) for k, v in db_ref_dicts.items()}

        # Construct the Statement object.
        stmt = Statement._from_json(json.loads(sj))
        ev_list = [Statement._from_json(json.loads(rj)).evidence[0]
                   for rj in raw_jsons]
        stmt.evidence = ev_list

        # Resolve supports supported-by, as much as possible.
        stmts_by_hash[mk_hash] = stmt
        for h in (h for h in supporting if h in stmts_by_hash):
            stmt.supports.append(stmts_by_hash[h])
            stmts_by_hash[h].supported_by.append(stmt)
        for h in (h for h in supported if h in stmts_by_hash):
            stmt.supported_by.append(stmts_by_hash[h])
            stmts_by_hash[h].supports.append(stmt)

        # Put it together in a dictionary.
        result_dict = {
            "mk_hash": mk_hash,
            "stmt": stmt,
            "db_refs": db_ref_dicts,
            "supports_hashes": supporting,
            "supported_by_hashes": supported
        }
        result_dicts.append(result_dict)
    return result_dicts
