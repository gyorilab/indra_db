__all__ = ['get_statement_jsons_from_papers']

import json
import logging
from collections import OrderedDict

from indra.util import clockit
from indra.statements import get_statement_by_name, stmts_from_json

from sqlalchemy import intersect_all, true, select, or_

from indra_db.util import regularize_agent_id, get_ro
from indra_db.client.tools import _apply_limits

logger = logging.getLogger(__name__)

# =============================================================================
# The API
# =============================================================================


snowflakes = ['Complex', 'Translocation', 'ActiveForm', 'Conversion',
              'Autophosphorylation']


def _iter_agents(stmt_json, agent_order):
    for i, ag_key in enumerate(agent_order):
        ag = stmt_json.get(ag_key)
        if ag is None:
            continue
        if isinstance(ag, list):
            # Like a complex
            for ag_obj in ag:
                if stmt_json['type'] in snowflakes:
                    yield None, ag_obj
                else:
                    yield ['subject', 'object'][i], ag_obj
        else:
            if stmt_json['type'] in snowflakes:
                yield None, ag
            else:
                yield ['subject', 'object'][i], ag


@clockit
def get_statement_jsons_from_papers(paper_refs, ro=None, **kwargs):
    """Get the statements from a list of papers.

    Parameters
    ----------
    paper_refs : list[(<id_type>, <paper_id>)]
        A list of tuples, where each tuple indicates and id-type (e.g. 'pmid')
        and an id value for a particular paper.
    ro : :py:class:`ReadonlyDatabaseManager`
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
    if ro is None:
        ro = get_ro('primary')

    # Create a sub-query on the reading metadata
    q = ro.session.query(ro.ReadingRefLink.rid.label('rid'))
    conditions = []
    for id_type, paper_id in paper_refs:
        if paper_id is None:
            logger.warning("Got paper with id None.")
            continue

        tbl_attr = getattr(ro.ReadingRefLink, id_type)
        if id_type in ['trid', 'tcid']:
            conditions.append(tbl_attr == int(paper_id))
        else:
            conditions.append(tbl_attr.like(paper_id))
    q = q.filter(or_(*conditions))
    sub_al = q.subquery('reading_ids')

    # Map the reading metadata query to mk_hashes with statement counts.
    mk_hashes_al = ro.EvidenceCounts.mk_hash.label('mk_hash')
    mk_hashes_q = (ro.session
                   .query(mk_hashes_al,
                          ro.EvidenceCounts.ev_count.label('ev_count'))
                   .filter(ro.EvidenceCounts.mk_hash == ro.FastRawPaLink.mk_hash,
                           ro.FastRawPaLink.reading_id == sub_al.c.rid))

    return _get_pa_stmt_jsons_w_mkhash_subquery(ro, mk_hashes_q,
                                                mk_hash_obj=ro.EvidenceCounts.mk_hash,
                                                ev_count_obj=ro.EvidenceCounts.ev_count,
                                                **kwargs)


# =============================================================================
# Tools
# =============================================================================

@clockit
def _get_pa_stmt_jsons_w_mkhash_subquery(ro, mk_hashes_q, best_first=True,
                                         max_stmts=None, offset=None,
                                         ev_limit=None, mk_hashes_alias=None,
                                         mk_hash_obj=None, ev_count_obj=None,
                                         source_specs=None,
                                         censured_sources=None):
    # Handle the limiting
    mk_hashes_q = _apply_limits(ro, mk_hashes_q, best_first, max_stmts, offset,
                                mk_hash_obj, ev_count_obj, mk_hashes_alias,
                                censured_sources)

    # Create the link
    mk_hashes_al = mk_hashes_q.subquery('mk_hashes')
    raw_json_c = ro.FastRawPaLink.raw_json.label('raw_json')
    pa_json_c = ro.FastRawPaLink.pa_json.label('pa_json')
    reading_id_c = ro.FastRawPaLink.reading_id.label('rid')
    cont_q = ro.session.query(raw_json_c, pa_json_c, reading_id_c)
    cont_q = cont_q.filter(ro.FastRawPaLink.mk_hash == mk_hashes_al.c.mk_hash)

    if censured_sources is not None:
        cont_q = cont_q.filter(ro.RawStmtSrc.sid == ro.FastRawPaLink.id)
        for src in censured_sources:
            cont_q = cont_q.filter(ro.RawStmtSrc.src.is_distinct_from(src))

    if ev_limit is not None:
        cont_q = cont_q.limit(ev_limit)

    # TODO: Only make a lateral-joined query when evidence is limited.
    json_content_al = cont_q.subquery().lateral('json_content')

    stmts_q = (mk_hashes_al
               .outerjoin(json_content_al, true())
               .outerjoin(ro.ReadingRefLink,
                          ro.ReadingRefLink.rid == json_content_al.c.rid)
               .outerjoin(ro.PaSourceLookup,
                          ro.PaSourceLookup.mk_hash == mk_hashes_al.c.mk_hash))

    ref_link_keys = [k for k in ro.ReadingRefLink.__dict__.keys()
                     if not k.startswith('_')]

    cols = [mk_hashes_al.c.mk_hash, ro.PaSourceLookup.src_json,
            mk_hashes_al.c.ev_count, json_content_al.c.raw_json,
            json_content_al.c.pa_json]
    cols += [getattr(ro.ReadingRefLink, k) for k in ref_link_keys]

    selection = select(cols).select_from(stmts_q)

    # Build up the source condition.
    if source_specs:
        cond = None
        for source_name, relation, value in source_specs:
            # Get the column object, check if None
            src = getattr(ro.PaStmtSrc, source_name)
            if src is None:
                raise ValueError("'%s' is not a valid source"
                                 % source_name)

            # Encode the relation.
            if relation == 'eq':
                new_cond = src == value
            elif relation == 'lt':
                new_cond = src < value
            elif relation == 'gt':
                new_cond = src > value
            elif relation == 'is not':
                new_cond = src.isnot(value)
            elif relation == 'is':
                new_cond = src.is_(value)
            else:
                raise ValueError("Invalid relation: %s" % relation)

            # Add the new condition.
            if cond is None:
                cond = new_cond
            else:
                cond &= new_cond

        selection = selection.where(cond)

    logger.debug("Executing sql to get statements:\n%s" % str(selection))

    proxy = ro.session.connection().execute(selection)
    res = proxy.fetchall()

    stmts_dict = OrderedDict()
    ev_totals = OrderedDict()
    source_counts = OrderedDict()
    total_evidence = 0
    returned_evidence = 0
    if res:
        logger.debug("res is %d row by %d cols." % (len(res), len(res[0])))
    else:
        logger.debug("res is empty.")

    src_list = ro.get_column_names(ro.PaStmtSrc)[1:]
    for row in res:
        row_gen = iter(row)

        mk_hash = next(row_gen)
        src_dict = dict.fromkeys(src_list, 0)
        src_dict.update(next(row_gen))
        ev_count = next(row_gen)
        raw_json_bts = next(row_gen)
        pa_json_bts = next(row_gen)
        ref_dict = dict(zip(ref_link_keys, row_gen))

        returned_evidence += 1
        raw_json = json.loads(raw_json_bts.decode('utf-8'))
        ev_json = raw_json['evidence'][0]

        # Add a new statements if the hash is new
        if mk_hash not in stmts_dict.keys():
            total_evidence += ev_count
            source_counts[mk_hash] = src_dict
            ev_totals[mk_hash] = ev_count
            stmts_dict[mk_hash] = json.loads(pa_json_bts.decode('utf-8'))
            stmts_dict[mk_hash]['evidence'] = []

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

        # Fix the pmid
        if ref_dict['pmid']:
            ev_json['pmid'] = ref_dict['pmid']
        elif 'PMID' in ev_json['text_refs']:
            del ev_json['text_refs']['PMID']

        # Add text refs
        ev_json['text_refs'].update({k.upper(): v
                                     for k, v in ref_dict.items()
                                     if v is not None})

        if ref_dict['source']:
            ev_json['annotations']['content_source'] = ref_dict['source']

        stmts_dict[mk_hash]['evidence'].append(ev_json)

    ret = {'statements': stmts_dict,
           'evidence_totals': ev_totals,
           'total_evidence': total_evidence,
           'evidence_returned': returned_evidence,
           'source_counts': source_counts}
    return ret


def _labelled_hash_and_count(meta):
    return meta.mk_hash.label('mk_hash'), meta.ev_count.label('ev_count')
