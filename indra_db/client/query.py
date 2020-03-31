__all__ = ['StatementQueryResult', 'StatementQuery', 'IntersectionQuery',
           'UnionQuery', 'MergeQuery', 'AgentQuery', 'MeshQuery', 'HashQuery']

import json
import logging
from collections import OrderedDict

from sqlalchemy import desc, true, select, intersect_all, union_all

from indra.statements import stmts_from_json, get_statement_by_name
from indra_db.schemas.readonly_schema import ro_role_map
from indra_db.util import regularize_agent_id

logger = logging.getLogger(__name__)


class StatementQueryResult(object):
    """The result of a statement query.

    This class encapsulates the results of a search for statements in the
    database. This standardizes the results of such searches.

    Attributes
    ----------
    results : dict
        The results of the query keyed by unique IDs (mk_hash for PA Statements,
        IDs for Raw Statements, etc.)
    limit : int
        The limit that was applied to this query.
    query_json : dict
        A description of the query that was used.
    """
    def __init__(self, results: dict, limit: int, offset: int,
                 evidence_totals: dict, total_evidence: int,
                 returned_evidence: int, source_counts: dict,
                 query_json: dict):
        self.results = results
        self.limit = limit
        self.offset = offset
        self.evidence_totals = evidence_totals
        self.total_evidence = total_evidence
        self.returned_evidence = returned_evidence
        self.source_counts = source_counts
        self.query_json = query_json

    def json(self):
        return {'results': self.results, 'limit': self.limit,
                'offset': self.offset, 'query': self.query_json,
                'evidence_totals': self.evidence_totals,
                'total_evidence': self.total_evidence,
                'returned_evidence': self.returned_evidence,
                'source_counts': self.source_counts}

    def statements(self):
        return stmts_from_json(list(self.results.values()))


class StatementQuery(object):
    def __init__(self):
        self.limit = None
        self.offset = None

    def run(self, ro, limit=None, offset=None, best_first=True, ev_limit=None):
        mk_hashes_q = self._get_mk_hashes_query(ro)
        return self._get_stmt_jsons_from_hashes_query(ro, mk_hashes_q, limit,
                                                      offset, best_first,
                                                      ev_limit)

    def _return_result(self, results: dict, evidence_totals: dict,
                       total_evidence: int, returned_evidence: int,
                       source_counts: dict):
        return StatementQueryResult(results, self.limit, self.offset,
                                    evidence_totals, total_evidence,
                                    returned_evidence, source_counts,
                                    self.to_json())

    def to_json(self) -> dict:
        return {'limit': self.limit, 'offset': self.offset,
                'constraint': self._get_constraint_json()}

    def _get_constraint_json(self) -> dict:
        raise NotImplementedError()

    @staticmethod
    def _hash_count_pair(ro) -> tuple:
        raise NotImplementedError()

    def _get_mk_hashes_query(self, ro):
        raise NotImplementedError()

    def _get_stmt_jsons_from_hashes_query(self, ro, mk_hashes_q, limit, offset,
                                          best_first, ev_limit):
        mk_hashes_q = mk_hashes_q.distinct()

        mk_hash_obj, ev_count_obj = self._hash_count_pair(ro)

        # Apply the general options.
        if best_first:
            mk_hashes_q = mk_hashes_q.order_by(desc(ev_count_obj))
        if limit is not None:
            mk_hashes_q = mk_hashes_q.limit(limit)
        if offset is not None:
            mk_hashes_q = mk_hashes_q.offset(offset)

        # Create the link
        mk_hashes_al = mk_hashes_q.subquery('mk_hashes')
        raw_json_c = ro.FastRawPaLink.raw_json.label('raw_json')
        pa_json_c = ro.FastRawPaLink.pa_json.label('pa_json')
        reading_id_c = ro.FastRawPaLink.reading_id.label('rid')
        cont_q = ro.session.query(raw_json_c, pa_json_c, reading_id_c)
        cont_q = cont_q.filter(ro.FastRawPaLink.mk_hash == mk_hashes_al.c.mk_hash)

        if ev_limit is not None:
            cont_q = cont_q.limit(ev_limit)

        # TODO: Only make a lateral-joined query when evidence is limited.
        json_content_al = cont_q.subquery().lateral('json_content')

        stmts_q = (mk_hashes_al
                   .outerjoin(json_content_al, true())
                   .outerjoin(ro.ReadingRefLink,
                              ro.ReadingRefLink.rid == json_content_al.c.rid)
                   .outerjoin(ro.SourceMeta,
                              ro.SourceMeta.mk_hash == mk_hashes_al.c.mk_hash))

        ref_link_keys = [k for k in ro.ReadingRefLink.__dict__.keys()
                         if not k.startswith('_')]

        cols = [mk_hashes_al.c.mk_hash, ro.SourceMeta.src_json,
                mk_hashes_al.c.ev_count, json_content_al.c.raw_json,
                json_content_al.c.pa_json]
        cols += [getattr(ro.ReadingRefLink, k) for k in ref_link_keys]

        selection = select(cols).select_from(stmts_q)

        logger.debug("Executing sql to get statements:\n%s" % str(selection))

        proxy = ro.session.connection().execute(selection)
        res = proxy.fetchall()
        if res:
            logger.debug("res is %d row by %d cols." % (len(res), len(res[0])))
        else:
            logger.debug("res is empty.")

        # Unpack the statements.
        stmts_dict = OrderedDict()
        ev_totals = OrderedDict()
        source_counts = OrderedDict()
        total_evidence = 0
        returned_evidence = 0
        src_list = ro.get_column_names(ro.PaStmtSrc)[1:]
        for row in res:
            returned_evidence += 1

            # Unpack the row
            row_gen = iter(row)

            mk_hash = next(row_gen)
            src_dict = dict.fromkeys(src_list, 0)
            src_dict.update(next(row_gen))
            ev_count = next(row_gen)
            raw_json_bts = next(row_gen)
            pa_json_bts = next(row_gen)
            ref_dict = dict(zip(ref_link_keys, row_gen))

            # Break out the evidence JSON
            raw_json = json.loads(raw_json_bts.decode('utf-8'))
            ev_json = raw_json['evidence'][0]

            # Add a new statement if the hash is new.
            if mk_hash not in stmts_dict.keys():
                total_evidence += ev_count
                source_counts[mk_hash] = src_dict
                ev_totals[mk_hash] = ev_count
                stmts_dict[mk_hash] = json.loads(pa_json_bts.decode('utf-8'))
                stmts_dict[mk_hash]['evidence'] = []

            # Add annotations if not present.
            if 'annotations' not in ev_json.keys():
                ev_json['annotations'] = {}

            # Add agents' raw text to annotations.
            raw_text = []
            agent_names = get_statement_by_name(raw_json['type'])._agent_order
            for ag_name in agent_names:
                ag_value = raw_json.get(ag_name, None)
                if isinstance(ag_value, dict):
                    raw_text.append(ag_value['db_refs'].get('TEXT'))
                elif ag_value is None:
                    raw_text.append(None)
                else:
                    for ag in ag_value:
                        raw_text.append(ag['db_refs'].get('TEXT'))
            ev_json['annotations']['agents'] = {'raw_text': raw_text}

            # Add prior UUIDs to the annotations
            if 'prior_uuids' not in ev_json['annotations'].keys():
                ev_json['annotations']['prior_uuids'] = []
            ev_json['annotations']['prior_uuids'].append(raw_json['id'])

            # Add and/or update text refs.
            if 'text_refs' not in ev_json.keys():
                ev_json['text_refs'] = {}
            if ref_dict['pmid']:
                ev_json['pmid'] = ref_dict['pmid']
            elif 'PMID' in ev_json['text_refs']:
                del ev_json['text_refs']['PMID']
            ev_json['text_refs'].update({k.upper(): v
                                         for k, v in ref_dict.items()
                                         if v is not None})

            # Add the source dictionary.
            if ref_dict['source']:
                ev_json['annotations']['content_source'] = ref_dict['source']

            # Add the evidence JSON to the list.
            stmts_dict[mk_hash]['evidence'].append(ev_json)

        return self._return_result(stmts_dict, ev_totals, total_evidence,
                                   returned_evidence, source_counts)

    def __merge_queries(self, other, MergeClass):
        if isinstance(self, MergeClass):
            if isinstance(other, MergeClass):
                return MergeClass(self.queries + other.queries)
            else:
                return MergeClass(self.queries + [other])
        elif isinstance(other, MergeClass):
            return MergeClass(other.queries + [self])
        else:
            return MergeClass([other, self])

    def __and__(self, other):
        return self.__merge_queries(other, IntersectionQuery)

    def __or__(self, other):
        return self.__merge_queries(other, UnionQuery)


class HashQuery(StatementQuery):
    def __init__(self, stmt_hashes):
        if len(stmt_hashes) == 0:
            raise ValueError("You must include at least one hash.")
        self.stmt_hashes = tuple(stmt_hashes)
        super(HashQuery, self).__init__()

    def _get_constraint_json(self) -> dict:
        return {"hash_query": self.stmt_hashes}

    def __and__(self, other):
        if isinstance(self, HashQuery) and isinstance(other, HashQuery):
            return HashQuery(self.stmt_hashes + other.stmt_hashes)
        return super(HashQuery, self).__and__(other)

    def __str__(self):
        return f"hash in list of {len(self.stmt_hashes)}"

    @staticmethod
    def _hash_count_pair(ro) -> tuple:
        return ro.SourceMeta.mk_hash, ro.SourceMeta.ev_count

    def _get_mk_hashes_query(self, ro):
        mk_hash, ev_count = self._hash_count_pair(ro)
        if len(self.stmt_hashes) == 1:
            mk_hashes_q = (ro.session.query(mk_hash.label('mk_hash'),
                                            ev_count.label('ev_count'))
                             .filter(mk_hash == self.stmt_hashes[0]))
        else:
            mk_hashes_q = (ro.session.query(mk_hash.label('mk_hash'),
                                            ev_count.label('ev_count'))
                             .filter(mk_hash.in_(self.stmt_hashes)))
        return mk_hashes_q


class AgentQuery(StatementQuery):

    def __init__(self, agent_id, namespace='NAME', role=None, agent_num=None):
        self.agent_id = agent_id
        self.namespace = namespace

        if role is not None and agent_num is not None:
            raise ValueError("Only specify role OR agent_num, not both.")

        self.role = role
        self.agent_num = agent_num

        # Regularize ID based on Database optimization (e.g. striping prefixes)
        self.regularized_id = regularize_agent_id(agent_id, namespace)
        super(AgentQuery, self).__init__()

    def __str__(self):
        s = f"an agent has {self.namespace} = {self.agent_id}"
        if self.role is not None:
            s += f" with role={self.role}"
        elif self.agent_num is not None:
            s += f" with agent_num={self.agent_num}"
        return s

    def _get_constraint_json(self) -> dict:
        return {'agent_query': {'agent_id': self.agent_id,
                                'namespace': self.namespace,
                                'regularized_id': self.regularized_id,
                                'role': self.role,
                                'agent_num': self.agent_num}}

    def _choose_table(self, ro):
        if self.namespace == 'NAME':
            meta = ro.NameMeta
        elif self.namespace == 'TEXT':
            meta = ro.TextMeta
        else:
            meta = ro.OtherMeta
        return meta

    def _hash_count_pair(self, ro) -> tuple:
        meta = self._choose_table(ro)
        return meta.mk_hash, meta.ev_count

    def _get_mk_hashes_query(self, ro):
        mk_hash, ev_count = self._hash_count_pair(ro)
        meta = self._choose_table(ro)
        mk_hashes_q = (ro.session.query(mk_hash.label('mk_hash'),
                                        ev_count.label('ev_count'))
                         .filter(meta.db_id.like(self.regularized_id)))

        if self.namespace not in ['NAME', 'TEXT', None]:
            mk_hashes_q = mk_hashes_q.filter(meta.db_name.like(self.namespace))

        if self.role is not None:
            role_num = ro_role_map.get_int(self.role)
            mk_hashes_q = mk_hashes_q.filter(meta.role_num == role_num)
        elif self.agent_num is not None:
            mk_hashes_q = mk_hashes_q.filter(meta.agent_num == self.agent_num)
        return mk_hashes_q


class MeshQuery(StatementQuery):
    def __init__(self, mesh_id):
        if not mesh_id.startswith('D') and not mesh_id[1:].is_digit():
            raise ValueError("Invalid MeSH ID: %s. Must begin with 'D' and "
                             "the rest must be a number." % mesh_id)
        self.mesh_id = mesh_id
        self.mesh_num = int(mesh_id[1:])
        super(MeshQuery, self).__init__()

    def __str__(self):
        return f"MeSH ID = {self.mesh_id}"

    def _get_constraint_json(self) -> dict:
        return {'mesh_query': {'mesh_id': self.mesh_id,
                               'mesh_num': self.mesh_num}}

    @staticmethod
    def _hash_count_pair(ro):
        return ro.MeshMeta.mk_hash, ro.MeshMeta.ev_count

    def _get_mk_hashes_query(self, ro):
        mk_hash, ev_count = self._hash_count_pair(ro)
        mk_hashes_q = (ro.session.query(mk_hash.label('mk_hash'),
                                        ev_count.label('ev_count'))
                         .filter(ro.MeshMeta.mesh_num == self.mesh_num))
        return mk_hashes_q


class MergeQuery(StatementQuery):
    join_word = NotImplemented
    name = NotImplemented

    @staticmethod
    def _merge(*queries):
        raise NotImplementedError()

    def __init__(self, query_list):
        # Make the collection of queries immutable.
        self.queries = tuple(query_list)

        # Because of the derivative nature of the "tables" involved, some more
        # dynamism is required to get, for instance, the hash and count pair.
        self._mk_hashes_al = None
        super(MergeQuery, self).__init__()

    def __str__(self):
        query_strs = []
        for q in self.queries:
            if isinstance(q, MergeQuery) and not isinstance(q, self.__class__):
                query_strs.append(f"({q})")
            else:
                query_strs.append(str(q))
        return f' {self.join_word} '.join(query_strs)

    def _get_constraint_json(self) -> dict:
        return {f'{self.name}_query': [q._get_constraint_json()
                                       for q in self.queries]}

    def _hash_count_pair(self, ro) -> tuple:
        mk_hashes_al = self._get_mk_hashes_al(ro)
        return mk_hashes_al.c.mk_hash, mk_hashes_al.c.ev_count

    def _get_mk_hashes_al(self, ro):
        if self._mk_hashes_al is None:
            mk_hashes_q_list = [q._get_mk_hashes_query(ro)
                                for q in self.queries]
            self._mk_hashes_al = (self._merge(*mk_hashes_q_list)
                                      .alias(self.name))
        return self._mk_hashes_al

    def _get_mk_hashes_query(self, ro):
        mk_hash, ev_count = self._hash_count_pair(ro)
        mk_hashes_q = ro.session.query(mk_hash.label('mk_hash'),
                                       ev_count.label('ev_count'))
        return mk_hashes_q


class IntersectionQuery(MergeQuery):
    @staticmethod
    def _merge(*queries):
        return intersect_all(*queries)

    name = 'intersection'
    join_word = 'and'


class UnionQuery(MergeQuery):
    @staticmethod
    def _merge(*queries):
        return union_all(*queries)

    name = 'union'
    join_word = 'or'
