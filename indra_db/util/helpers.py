__all__ = ['unpack', '_get_trids', '_fix_evidence_refs',
           'get_raw_stmts_frm_db_list', '_set_evidence_text_ref',
           'get_statement_object']

import json
import zlib
import logging

from indra.util import clockit
from indra.statements import Statement

logger = logging.getLogger('util-helpers')


def get_statement_object(db_stmt):
    """Get an INDRA Statement object from a db_stmt."""
    if isinstance(db_stmt, bytes):
        jb = db_stmt
    else:
        jb = db_stmt.json
    return Statement._from_json(json.loads(jb.decode('utf-8')))


def _set_evidence_text_ref(stmt, tr):
    # This is a separate function because it is likely to change, and this is a
    # critical process that is executed in multiple places.
    for ev in stmt.evidence:
        ev.pmid = tr.pmid
        ev.text_refs = tr.get_ref_dict()


@clockit
def _fix_evidence_refs(db, rid_stmt_trios):
    """Get proper id data for a raw statement from the database.

    Alterations are made to the Statement objects "in-place", so this function
    itself returns None.
    """
    rid_set = {rid for rid, _, _ in rid_stmt_trios if rid is not None}
    logger.info("Getting text refs for %d readings." % len(rid_set))
    if rid_set:
        rid_tr_pairs = db.select_all(
            [db.Reading.id, db.TextRef],
            db.Reading.id.in_(rid_set),
            db.Reading.text_content_id == db.TextContent.id,
            db.TextContent.text_ref_id == db.TextRef.id
        )
        rid_tr_dict = {rid: tr for rid, tr in rid_tr_pairs}
        for rid, sid, stmt in rid_stmt_trios:
            if rid is None:
                # This means this statement came from a database, not reading.
                continue
            assert len(stmt.evidence) == 1, \
                "Only raw statements can have their refs fixed."
            _set_evidence_text_ref(stmt, rid_tr_dict[rid])
    return


@clockit
def get_raw_stmts_frm_db_list(db, db_stmt_objs, fix_refs=True, with_sids=True):
    """Convert table objects of raw statements into INDRA Statement objects."""
    rid_stmt_sid_trios = [(db_stmt.reading_id, db_stmt.id,
                           get_statement_object(db_stmt))
                          for db_stmt in db_stmt_objs]
    if fix_refs:
        _fix_evidence_refs(db, rid_stmt_sid_trios)
    # Note: it is important that order is maintained here (hence not a set or
    # dict).
    if with_sids:
        return [(sid, stmt) for _, sid, stmt in rid_stmt_sid_trios]
    else:
        return [stmt for _, _, stmt in rid_stmt_sid_trios]


def unpack(bts, decode=True):
    ret = zlib.decompress(bts, zlib.MAX_WBITS+16)
    if decode:
        ret = ret.decode('utf-8')
    return ret


def _get_trids(db, id_val, id_type):
    """Return text ref IDs corresponding to any ID type and value."""
    # Get the text ref id(s)
    if id_type in ['trid']:
        trids = [int(id_val)]
    else:
        id_types = ['pmid', 'pmcid', 'doi', 'pii', 'url', 'manuscript_id']
        if id_type not in id_types:
            raise ValueError('id_type must be one of: %s' % str(id_types))
        constraint = (getattr(db.TextRef, id_type) == id_val)
        trids = [trid for trid, in db.select_all(db.TextRef.id, constraint)]
    return trids
