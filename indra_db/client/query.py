from itertools import combinations

__all__ = ['StatementQueryResult', 'QueryCore', 'Intersection', 'Union',
           'MergeQueryCore', 'HasAgent', 'FromMeshId', 'InHashList',
           'HasSources', 'HasOnlySource', 'HasReadings', 'HasDatabases',
           'SourceCore', 'SourceIntersection', 'HasAnyType']

import json
import logging
from collections import OrderedDict, Iterable, defaultdict

from sqlalchemy import desc, true, select, intersect_all, union_all, or_

from indra.statements import stmts_from_json, get_statement_by_name, \
    get_all_descendants
from indra_db.schemas.readonly_schema import ro_role_map, ro_type_map
from indra_db.util import regularize_agent_id

logger = logging.getLogger(__name__)


class QueryResult(object):
    def __init__(self, results, limit: int, offset: int,
                 evidence_totals: dict, query_json: dict):
        if not isinstance(results, Iterable) or isinstance(results, str):
            raise ValueError("Input `results` is expected to be an iterable, "
                             "and not a string.")
        self.results = results
        self.evidence_totals = evidence_totals
        self.total_evidence = sum(self.evidence_totals.values())
        self.limit = limit
        self.offset = offset
        self.query_json = query_json

    def json(self):
        if not isinstance(self.results, dict) \
                and not isinstance(self.results, list):
            json_results = list(self.results)
        else:
            json_results = self.results
        return {'results': json_results, 'limit': self.limit,
                'offset': self.offset, 'query': self.query_json,
                'evidence_totals': self.evidence_totals,
                'total_evidence': self.total_evidence}


class StatementQueryResult(QueryResult):
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
                 evidence_totals: dict, returned_evidence: int,
                 source_counts: dict, query_json: dict):
        super(StatementQueryResult, self).__init__(results, limit, offset,
                                                   evidence_totals, query_json)
        self.returned_evidence = returned_evidence
        self.source_counts = source_counts

    def json(self):
        json_dict = super(StatementQueryResult, self).json()
        json_dict.update({'returned_evidence': self.returned_evidence,
                          'source_counts': self.source_counts})
        return json_dict

    def statements(self):
        return stmts_from_json(list(self.results.values()))


class QueryCore(object):
    def __init__(self, empty=False):
        self.empty = empty
        self._inverted = False

    def __invert__(self):
        raise NotImplementedError()

    def __hash__(self):
        return hash(str(self))

    def invert(self):
        return self.__invert__()

    def _do_invert(self, *args, **kwargs):
        new_obj = self.__class__(*args, **kwargs)
        new_obj._inverted = not self._inverted
        return new_obj

    def get_statements(self, ro, limit=None, offset=None, best_first=True,
                       ev_limit=None):
        if self.empty:
            return StatementQueryResult({}, limit, offset, {}, 0, {},
                                        self.to_json())

        mk_hashes_q = self._get_hash_query(ro)
        mk_hashes_q = self._apply_limits(ro, mk_hashes_q, limit, offset,
                                         best_first)

        stmt_dict, ev_totals, returned_evidence, source_counts = \
            self._get_stmt_jsons_from_hashes_query(ro, mk_hashes_q, ev_limit)
        return StatementQueryResult(stmt_dict, limit, offset, ev_totals,
                                    returned_evidence, source_counts,
                                    self.to_json())

    def get_hashes(self, ro, limit=None, offset=None, best_first=True):
        if self.empty:
            return QueryResult(set(), limit, offset, {}, self.to_json())

        mk_hashes_q = self._get_hash_query(ro)
        mk_hashes_q = self._apply_limits(ro, mk_hashes_q, limit, offset,
                                         best_first)
        result = mk_hashes_q.all()
        evidence_totals = {h: cnt for h, cnt in result}
        return QueryResult(set(evidence_totals.keys()), limit, offset,
                           evidence_totals, self.to_json())

    def _apply_limits(self, ro, mk_hashes_q, limit=None, offset=None,
                      best_first=True):
        mk_hashes_q = mk_hashes_q.distinct()

        mk_hash_obj, ev_count_obj = self._hash_count_pair(ro)

        # Apply the general options.
        if best_first:
            mk_hashes_q = mk_hashes_q.order_by(desc(ev_count_obj))
        if limit is not None:
            mk_hashes_q = mk_hashes_q.limit(limit)
        if offset is not None:
            mk_hashes_q = mk_hashes_q.offset(offset)
        return mk_hashes_q

    def to_json(self) -> dict:
        return {'constraint': self._get_constraint_json(),
                'inverted': self._inverted}

    def _get_constraint_json(self) -> dict:
        raise NotImplementedError()

    def _get_table(self, ro):
        raise NotImplementedError()

    def _base_query(self, ro):
        mk_hash, ev_count = self._hash_count_pair(ro)
        return ro.session.query(mk_hash.label('mk_hash'),
                                ev_count.label('ev_count'))

    def _hash_count_pair(self, ro) -> tuple:
        meta = self._get_table(ro)
        return meta.mk_hash, meta.ev_count

    def _get_hash_query(self, ro):
        raise NotImplementedError()

    @staticmethod
    def _get_stmt_jsons_from_hashes_query(ro, mk_hashes_q, ev_limit):
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

        return stmts_dict, ev_totals, returned_evidence, source_counts

    def __merge_queries(self, other, MergeClass):
        if not isinstance(other, QueryCore):
            raise ValueError(f"{self.__class__.__name__} cannot operate with "
                             f"{type(other)}")

        if isinstance(self, MergeClass):
            if isinstance(other, MergeClass):
                return MergeClass(self.queries + other.queries)
            else:
                return MergeClass(self.queries + (other,))
        elif isinstance(other, MergeClass):
            return MergeClass(other.queries + (self,))
        else:
            return MergeClass([other, self])

    def __and__(self, other):
        return self.__merge_queries(other, Intersection)

    def __or__(self, other):
        return self.__merge_queries(other, Union)

    def __sub__(self, other):
        return self.__merge_queries(~other, Intersection)


class SourceCore(QueryCore):

    def _get_constraint_json(self) -> dict:
        raise NotImplementedError()

    def __and__(self, other):
        if isinstance(other, SourceCore):
            return SourceIntersection([self, other])
        elif isinstance(other, SourceIntersection):
            return SourceIntersection(other.source_queries + (self,))
        return super(SourceCore, self).__and__(other)

    def __invert__(self):
        raise NotImplementedError()

    def _get_table(self, ro):
        return ro.SourceMeta

    def _apply_filter(self, ro, query, invert=False):
        raise NotImplementedError()

    def _get_hash_query(self, ro):
        q = self._base_query(ro)
        q = self._apply_filter(ro, q)
        return q


class SourceIntersection(QueryCore):
    def __init__(self, source_queries):
        # There are several points at which we could realize this query is by
        # definition empty.
        empty = False

        # Look through all the queries, picking out special cases and grouping
        # the rest by class.
        add_sources = set()
        rem_sources = set()
        add_hashes = None
        rem_hashes = set()
        class_groups = defaultdict(list)
        for sq in source_queries:
            if isinstance(sq, HasSources):
                # Carefully track which sources to add and which to remove.
                if not sq._inverted:
                    add_sources |= set(sq.sources)
                else:
                    rem_sources |= set(sq.sources)
            elif isinstance(sq, InHashList):
                # Collect all hashes to include and those to exclude.
                if not sq._inverted:
                    # This is a part of an intersection, so intersection is
                    # appropriate.
                    if add_hashes is None:
                        add_hashes = set(sq.stmt_hashes)
                    else:
                        add_hashes &= set(sq.stmt_hashes)
                else:
                    # This follows form De Morgan's Law
                    rem_hashes |= set(sq.stmt_hashes)
            else:
                # We will need to check other class groups for inversion, so
                # group them now for efficiency.
                class_groups[sq.__class__].append(sq)

        # Start building up the true set of queries.
        filtered_queries = set()

        # Add the source queries. Net empty queries will be dropped.
        if add_sources:
            filtered_queries.add(HasSources(add_sources - rem_sources))
        if rem_sources:
            filtered_queries.add(~HasSources(rem_sources - add_sources))

        # Add the hash queries. Net empty queries will be dropped.
        if add_hashes is not None:
            filtered_queries.add(InHashList(add_hashes - rem_hashes))
            rem_hashes -= add_hashes
        elif not add_hashes:
            empty = True

        if rem_hashes:
            filtered_queries.add(~InHashList(rem_hashes))

        # Now add in all the other queries, removing those that cancel out.
        for q_list in class_groups.values():
            if len(q_list) == 1:
                filtered_queries.add(q_list[0])
            else:
                filtered_queries |= set(q_list)
                for q1, q2 in combinations(q_list, 2):
                    if q1._get_constraint_json() == q2._get_constraint_json() \
                            and q1._inverted != q2._inverted:
                        filtered_queries -= {q1, q2}

        inv_classes = {(sq.__class__, sq._inverted) for sq in filtered_queries}
        if len(inv_classes) != len(filtered_queries):
            raise ValueError(f"Multiple instances of the same class: "
                             f"{[sq.__class__ for sq in filtered_queries]}")
        self.source_queries = tuple(filtered_queries)
        empty |= any(q.empty for q in self.source_queries)
        super(SourceIntersection, self).__init__(empty)

    def __invert__(self):
        return self._do_invert(self.source_queries)

    def __and__(self, other):
        if isinstance(other, SourceIntersection):
            return SourceIntersection(self.source_queries
                                      + other.source_queries)
        elif isinstance(other, SourceCore):
            return SourceIntersection(self.source_queries + (other,))
        return super(SourceIntersection, self).__and__(other)

    def __str__(self):
        str_list = [str(sq) for sq in self.source_queries]
        if not self._inverted:
            return ' and '.join(str_list)
        else:
            return 'not (' + ' and not '.join(str_list) + ')'

    def _get_constraint_json(self) -> dict:
        info_dict = {}
        for q in self.source_queries:
            q_info = q._get_constraint_json()
            info_dict.update(q_info)
        return {'multi_source_query': info_dict}

    def _get_table(self, ro):
        return ro.SourceMeta

    def _get_hash_query(self, ro):
        query = self._base_query(ro)
        for sq in self.source_queries:
            query = sq._apply_filter(ro, query, self._inverted)
        return query


class HasOnlySource(SourceCore):
    def __init__(self, only_source):
        self.only_source = only_source
        super(HasOnlySource, self).__init__()

    def __str__(self):
        invert_mod = 'not ' if self._inverted else ''
        return f"is {invert_mod}only from {self.only_source}"

    def __invert__(self):
        return self._do_invert(self.only_source)

    def _get_constraint_json(self) -> dict:
        return {'only_source_query': {'only_source': self.only_source}}

    def _apply_filter(self, ro, query, invert=False):
        inverted = self._inverted ^ invert
        meta = self._get_table(ro)
        if not inverted:
            clause = meta.only_src.like(self.only_source)
        else:
            clause = or_(meta.only_src.notlike(self.only_source),
                         meta.only_src.is_(None))
        return query.filter(clause)


class HasSources(SourceCore):
    def __init__(self, sources):
        empty = False
        if len(sources) == 0:
            empty = True
            logger.warning("No sources specified, query is by default empty.")
        self.sources = tuple(set(sources))
        super(HasSources, self).__init__(empty)

    def __and__(self, other):
        if isinstance(other, HasSources) and self._inverted == other._inverted:
            ret = HasSources(set(self.sources) | set(other.sources))
            ret._inverted = self._inverted
            return ret
        return super(HasSources, self).__and__(other)

    def __invert__(self):
        return self._do_invert(self.sources)

    def __str__(self):
        if not self._inverted:
            return f"is from all of {self.sources}"
        else:
            return f"is not from one of {self.sources}"

    def _get_constraint_json(self) -> dict:
        return {'has_source_query': {'sources': self.sources}}

    def _apply_filter(self, ro, query, invert=False):
        inverted = self._inverted ^ invert
        meta = self._get_table(ro)
        clauses = []
        for src in self.sources:
            if not inverted:
                clauses.append(getattr(meta, src) > 0)
            else:
                clauses.append(getattr(meta, src).is_(None))
        if not inverted:
            query = query.filter(*clauses)
        else:
            query = query.filter(or_(*clauses))
        return query


class SourceTypeCore(SourceCore):
    name = NotImplemented
    col = NotImplemented

    def __str__(self):
        if not self._inverted:
            return f"has {self.name}"
        else:
            return f"has no {self.name}"

    def __invert__(self):
        return self._do_invert()

    def _get_constraint_json(self) -> dict:
        return {f'has_{self.name}_query': {f'has_{self.name}': True}}

    def _apply_filter(self, ro, query, invert=False):
        inverted = self._inverted ^ invert
        meta = self._get_table(ro)
        if not inverted:
            clause = getattr(meta, self.col) == True
        else:
            clause = getattr(meta, self.col) == False
        return query.filter(clause)


class HasReadings(SourceTypeCore):
    name = 'readings'
    col = 'has_rd'


class HasDatabases(SourceTypeCore):
    name = 'databases'
    col = 'has_db'


class InHashList(SourceCore):
    def __init__(self, stmt_hashes):
        empty = False
        if len(stmt_hashes) == 0:
            empty = True
            logger.warning("No hashes given, query is by default empty.")
        self.stmt_hashes = tuple(stmt_hashes)
        super(InHashList, self).__init__(empty)

    def __invert__(self):
        return self._do_invert(self.stmt_hashes)

    def __or__(self, other):
        if isinstance(other, InHashList) and self._inverted == other._inverted:
            res = InHashList(set(self.stmt_hashes) | set(other.stmt_hashes))
            res._inverted = self._inverted
            return res
        return super(InHashList, self).__or__(other)

    def __and__(self, other):
        if isinstance(other, InHashList) and self._inverted == other._inverted:
            res = InHashList(set(self.stmt_hashes) & set(other.stmt_hashes))
            res._inverted = self._inverted
            return res
        return super(InHashList, self).__and__(other)

    def __str__(self):
        return f"hash {'not ' if self._inverted else ''}in {self.stmt_hashes}"

    def _get_constraint_json(self) -> dict:
        return {"hash_query": self.stmt_hashes}

    def _apply_filter(self, ro, query, invert=False):
        inverted = self._inverted ^ invert
        mk_hash, _ = self._hash_count_pair(ro)
        if len(self.stmt_hashes) == 1:
            if not inverted:
                clause = mk_hash == self.stmt_hashes[0]
            else:
                clause = mk_hash != self.stmt_hashes[0]
        else:
            if not inverted:
                clause = mk_hash.in_(self.stmt_hashes)
            else:
                clause = mk_hash.notin_(self.stmt_hashes)
        return query.filter(clause)


class HasAgent(QueryCore):
    def __init__(self, agent_id, namespace='NAME', role=None, agent_num=None):
        self.agent_id = agent_id
        self.namespace = namespace

        if role is not None and agent_num is not None:
            raise ValueError("Only specify role OR agent_num, not both.")

        self.role = role
        self.agent_num = agent_num

        # Regularize ID based on Database optimization (e.g. striping prefixes)
        self.regularized_id = regularize_agent_id(agent_id, namespace)
        super(HasAgent, self).__init__()

    def __invert__(self):
        return self._do_invert(self.agent_id, self.namespace, self.role,
                               self.agent_num)

    def __str__(self):
        s = 'not ' if self._inverted else ''
        s += f"an agent where {self.namespace} = {self.agent_id}"
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

    def _get_table(self, ro):
        if self.namespace == 'NAME':
            meta = ro.NameMeta
        elif self.namespace == 'TEXT':
            meta = ro.TextMeta
        else:
            meta = ro.OtherMeta
        return meta

    def _get_hash_query(self, ro):
        meta = self._get_table(ro)
        qry = self._base_query(ro).filter(meta.db_id.like(self.regularized_id))
        if self.namespace not in ['NAME', 'TEXT', None]:
            qry = qry.filter(meta.db_name.like(self.namespace))
        if self.role is not None:
            role_num = ro_role_map.get_int(self.role)
            qry = qry.filter(meta.role_num == role_num)
        elif self.agent_num is not None:
            qry = qry.filter(meta.agent_num == self.agent_num)

        if self._inverted:
            qry = self._base_query(ro).except_(qry)

        return qry


class HasAnyType(QueryCore):
    def __init__(self, stmt_types, include_subclasses=False):
        empty = False
        if len(stmt_types) == 0:
            empty = True
            logger.warning('No statement types indicated, query is empty.')

        st_set = set(stmt_types)
        if include_subclasses:
            for stmt_type in stmt_types:
                stmt_class = get_statement_by_name(stmt_type)
                sub_classes = get_all_descendants(stmt_class)
                st_set |= {c.__name__ for c in sub_classes}
        self.stmt_types = tuple(st_set)
        super(HasAnyType, self).__init__(empty)

    def __invert__(self):
        return self._do_invert(self.stmt_types)

    def __or__(self, other):
        if isinstance(other, HasAnyType) and self._inverted == other._inverted:
            return HasAnyType(set(self.stmt_types) | set(other.stmt_types))
        return super(HasAnyType, self).__or__(other)

    def __and__(self, other):
        if isinstance(other, HasAnyType) and self._inverted == other._inverted:
            return HasAnyType(set(self.stmt_types) & set(other.stmt_types))
        return super(HasAnyType, self).__and__(other)

    def __str__(self):
        invert_word = 'not ' if self._inverted else ''
        return f"type {invert_word}in {self.stmt_types}"

    def _get_constraint_json(self) -> dict:
        return {'type_query': {'types': self.stmt_types}}

    def _get_table(self, ro):
        return ro.SourceMeta

    def _get_type_nums(self):
        return [ro_type_map.get_int(st) for st in self.stmt_types]

    def _apply_filter(self, meta, query):
        type_nums = self._get_type_nums()
        if len(type_nums) == 1:
            if not self._inverted:
                clause = meta.type_num == type_nums[0]
            else:
                clause = meta.type_num != type_nums[0]
        else:
            if not self._inverted:
                clause = meta.type_num.in_(type_nums)
            else:
                clause = meta.type_num.notin_(type_nums)
        return query.filter(clause)

    def _get_hash_query(self, ro):
        return self._apply_filter(self._get_table(ro), self._base_query(ro))


class FromMeshId(QueryCore):
    def __init__(self, mesh_id):
        if not mesh_id.startswith('D') and not mesh_id[1:].is_digit():
            raise ValueError("Invalid MeSH ID: %s. Must begin with 'D' and "
                             "the rest must be a number." % mesh_id)
        self.mesh_id = mesh_id
        self.mesh_num = int(mesh_id[1:])
        super(FromMeshId, self).__init__()

    def __str__(self):
        invert_char = '!' if self._inverted else ''
        return f"MeSH ID {invert_char}= {self.mesh_id}"

    def __invert__(self):
        return self._do_invert(self.mesh_id)

    def _get_constraint_json(self) -> dict:
        return {'mesh_query': {'mesh_id': self.mesh_id,
                               'mesh_num': self.mesh_num}}

    def _get_table(self, ro):
        return ro.MeshMeta

    def _get_hash_query(self, ro):
        meta = self._get_table(ro)
        qry = self._base_query(ro).filter(meta.mesh_num == self.mesh_num)
        if self._inverted:
            qry = (ro.session.query(ro.SourceMeta.mk_hash.label('mk_hash'),
                                    ro.SourceMeta.ev_count.label('ev_count'))
                     .except_(qry))
        return qry


class MergeQueryCore(QueryCore):
    join_word = NotImplemented
    name = NotImplemented

    def __init__(self, query_list, *args, **kwargs):
        # Make the collection of queries immutable.
        self.queries = tuple(query_list)

        # Because of the derivative nature of the "tables" involved, some more
        # dynamism is required to get, for instance, the hash and count pair.
        self._mk_hashes_al = None
        super(MergeQueryCore, self).__init__(*args, **kwargs)

    def __invert__(self):
        raise NotImplementedError()

    def _get_table(self, ro):
        raise NotImplementedError()

    @staticmethod
    def _merge(*queries):
        raise NotImplementedError()

    def __str__(self):
        query_strs = []
        for q in self.queries:
            if isinstance(q, MergeQueryCore) and not q._inverted:
                query_strs.append(f"({q})")
            else:
                query_strs.append(str(q))
        ret = f' {self.join_word} '.join(query_strs)
        if self._inverted:
            ret = f'not ({ret})'
        return ret

    def _get_constraint_json(self) -> dict:
        return {f'{self.name}_query': [q._get_constraint_json()
                                       for q in self.queries]}

    def _hash_count_pair(self, ro) -> tuple:
        mk_hashes_al = self._get_table(ro)
        return mk_hashes_al.c.mk_hash, mk_hashes_al.c.ev_count

    def _get_hash_query(self, ro):
        return self._base_query(ro)


class Intersection(MergeQueryCore):
    name = 'intersection'
    join_word = 'and'

    def __init__(self, query_list):
        mergeable_query_types = [SourceIntersection, SourceCore]
        mergeable_groups = {C: [] for C in mergeable_query_types}
        other_groups = defaultdict(list)
        other_queries = set()
        self.type_query = None
        self.not_type_query = None
        empty = False
        for query in query_list:
            if query.empty:
                empty = True
            for C in mergeable_query_types:
                if isinstance(query, C):
                    mergeable_groups[C].append(query)
                    break
            else:
                if isinstance(query, HasAnyType):
                    if not query._inverted:
                        if self.type_query is None:
                            self.type_query = query
                        else:
                            self.type_query &= query
                    else:
                        if self.not_type_query is None:
                            self.not_type_query = query
                        else:
                            self.not_type_query |= query
                else:
                    other_groups[query.__class__].append(query)
                other_queries.add(query)

        # Add mergeable queries into the final set.
        for queries in mergeable_groups.values():
            if len(queries) == 1:
                other_queries.add(queries[0])
            elif len(queries) > 1:
                # Merge the queries based on their inversion.
                pos_query = None
                neg_query = None
                for q in queries:
                    if not q._inverted:
                        if pos_query is None:
                            pos_query = q
                        else:
                            pos_query &= q
                    else:
                        if neg_query is None:
                            neg_query = q
                        else:
                            neg_query |= q

                # Add the merged queries to the final set.
                for q in [neg_query, pos_query]:
                    if q is not None:
                        other_queries.add(q)
                        other_groups[q.__class__].append(q)

        # Look for exact contradictions.
        for q_list in other_groups.values():
            if len(q_list) > 1:
                for q1, q2 in combinations(q_list, 2):
                    if q1._get_constraint_json() == q2._get_constraint_json() \
                            and q1._inverted != q2._inverted:
                        empty = True
        super(Intersection, self).__init__(other_queries, empty)

    def __invert__(self):
        new_obj = Union([~q for q in self.queries])
        return new_obj

    @staticmethod
    def _merge(*queries):
        return intersect_all(*queries)

    def _get_table(self, ro):
        if self._mk_hashes_al is not None:
            return self._mk_hashes_al
        mkhq_list = []
        for query in self.queries:
            if query == self.type_query:
                continue
            mkhq = query._get_hash_query(ro)
            if self.type_query is not None:
                mkhq = self.type_query._apply_filter(query._get_table(ro),
                                                     mkhq)
            if self.not_type_query is not None:
                mkhq = self.not_type_query._apply_filter(query._get_table(ro),
                                                         mkhq)
            mkhq_list.append(mkhq)
        self._mk_hashes_al = self._merge(*mkhq_list).alias(self.name)
        return self._mk_hashes_al


class Union(MergeQueryCore):
    name = 'union'
    join_word = 'or'

    def __init__(self, query_list):
        other_queries = set()
        pos_hash_queries = []
        neg_hash_queries = []
        all_empty = True
        for query in query_list:
            if not query.empty:
                all_empty = False

            if isinstance(query, InHashList):
                if not query._inverted:
                    pos_hash_queries.append(query)
                else:
                    neg_hash_queries.append(query)
            else:
                other_queries.add(query)

        for hash_query_group in [pos_hash_queries, neg_hash_queries]:
            if len(hash_query_group) == 1:
                other_queries.add(hash_query_group[0])
            elif len(hash_query_group) > 1:
                query = hash_query_group[0]
                for other_query in hash_query_group[1:]:
                    if not query._inverted:
                        query |= other_query
                    else:
                        query &= other_query
                other_queries.add(query)

        super(Union, self).__init__(other_queries, all_empty)

    def __invert__(self):
        new_obj = Intersection([~q for q in self.queries])
        return new_obj

    @staticmethod
    def _merge(*queries):
        return union_all(*queries)

    def _get_table(self, ro):
        if self._mk_hashes_al is None:
            mk_hashes_q_list = [q._get_hash_query(ro)
                                for q in self.queries if not q.empty]
            if len(mk_hashes_q_list) == 1:
                self._mk_hashes_al = (mk_hashes_q_list[0].subquery()
                                                         .alias(self.name))
            else:
                self._mk_hashes_al = (self._merge(*mk_hashes_q_list)
                                          .alias(self.name))
        return self._mk_hashes_al
