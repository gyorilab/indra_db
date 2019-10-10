from sqlalchemy import desc


def _apply_limits(db, mk_hashes_q, best_first, max_stmts, offset,
                  mk_hash_obj=None, ev_count_obj=None, mk_hashes_alias=None,
                  censured_sources=None):

    mk_hashes_q = mk_hashes_q.distinct()
    if best_first:
        if ev_count_obj is not None:
            mk_hashes_q.order_by(desc(ev_count_obj))
        elif mk_hashes_alias is not None:
            mk_hashes_q = (mk_hashes_q
                           .order_by(desc(mk_hashes_alias.c.ev_count)))
        else:
            mk_hashes_q = mk_hashes_q.order_by(desc(db.PaMeta.ev_count))
    if censured_sources:
        if mk_hashes_alias is not None:
            mk_hashes_q = mk_hashes_q.filter(
                db.PaSourceLookup.mk_hash == mk_hashes_alias.c.mk_hash
            )
        elif mk_hash_obj is not None:
            mk_hashes_q = mk_hashes_q.filter(
                db.PaSourceLookup.mk_hash == mk_hash_obj
            )
        else:
            mk_hashes_q = mk_hashes_q.filter(
                db.PaSourceLookup.mk_hash == db.PaMeta.mk_hash
            )

        for src in censured_sources:
            mk_hashes_q = (mk_hashes_q
                           .filter(db.PaSourceLookup.only_src.notlike(src)))
    if max_stmts is not None:
        mk_hashes_q = mk_hashes_q.limit(max_stmts)
    if offset is not None:
        mk_hashes_q = mk_hashes_q.offset(offset)

    return mk_hashes_q


