from sqlalchemy import desc


def _apply_limits(db, mk_hashes_q, best_first, max_stmts, offset,
                  ev_count_obj=None, mk_hashes_alias=None):
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

    return mk_hashes_q


