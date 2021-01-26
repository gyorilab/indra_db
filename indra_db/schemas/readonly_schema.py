__all__ = ['get_schema']

import logging

from sqlalchemy import Column, Integer, String, BigInteger, Boolean,\
    SmallInteger
from sqlalchemy.dialects.postgresql import BYTEA, JSON, JSONB, REAL

from indra.statements import get_all_descendants, Statement

from .mixins import ReadonlyTable, NamespaceLookup, SpecialColumnTable, \
    IndraDBTable, IndraDBRefTable
from .indexes import *


logger = logging.getLogger(__name__)

CREATE_ORDER = [
    'raw_stmt_src',
    'fast_raw_pa_link',
    'pa_agent_counts',
    'pa_stmt_src',
    'evidence_counts',
    'reading_ref_link',
    'pa_ref_link',
    'mesh_terms',
    'mesh_concepts',
    'hash_pmid_counts',
    'mesh_term_ref_counts',
    'mesh_concept_ref_counts',
    'raw_stmt_mesh_terms',
    'raw_stmt_mesh_concepts',
    'pa_meta',
    'text_meta',
    'name_meta',
    'other_meta',
    'source_meta',
    'mesh_term_meta',
    'mesh_concept_meta',
    'agent_interactions'
]


class StringIntMapping(object):
    arg = NotImplemented

    def __init__(self):
        self._int_to_str = NotImplemented
        self._str_to_int = NotImplemented
        raise NotImplementedError("__init__ must be defined in sub class.")

    def get_str(self, num: int) -> str:
        return self._int_to_str[num]

    def get_int(self, val: str) -> int:
        return self._str_to_int[val]

    def get_with_clause(self) -> str:
        values = ',\n'.join('(%d, \'%s\')' % (num, val)
                            for num, val in self._int_to_str.items())
        return "{arg}_map({arg}_num, {arg}) AS (values\n{values}\n)".format(
            arg=self.arg, values=values
        )


class StatementTypeMapping(StringIntMapping):
    arg = 'type'

    def __init__(self):
        all_stmt_classes = get_all_descendants(Statement)
        stmt_class_names = [sc.__name__ for sc in all_stmt_classes]
        stmt_class_names.sort()

        self._int_to_str = {}
        self._str_to_int = {}
        for stmt_type_num, stmt_type in enumerate(stmt_class_names):
            self._int_to_str[stmt_type_num] = stmt_type
            self._str_to_int[stmt_type] = stmt_type_num


ro_type_map = StatementTypeMapping()


class RoleMapping(StringIntMapping):
    arg = 'role'

    def __init__(self):
        self._int_to_str = {-1: 'SUBJECT', 0: 'OTHER', 1: 'OBJECT'}
        self._str_to_int = {v: k for k, v in self._int_to_str.items()}


ro_role_map = RoleMapping()


def get_schema(Base):
    """Return the schema for the reading view of the database.

    We use a readonly database to allow fast and efficient load of data,
    and to add a layer of separation between the processes of updating
    the content of the database and accessing the content of the
    database. However, it is not practical to have the views created
    through sqlalchemy: instead they are generated and updated manually
    (or by other non-sqlalchemy scripts).

    Before building these tables, the `belief` table must already have been
    loaded into the readonly database.

    The following views must be built in this specific order (_temp_):
      1. raw_stmt_src
      2. fast_raw_pa_link
      3. pa_agent_counts
      4. _pa_stmt_src_
      5. evidence_counts
      6. reading_ref_link
      7. _pa_ref_link_
      8. _mesh_terms_
      9. _mesh_concepts_
      10. _hash_pmid_counts_
      11. mesh_term_ref_counts
      12. mesh_concept_ref_counts
      13. raw_stmt_mesh_terms
      14. raw_stmt_mesh_concepts
      15. _pa_meta_
      16. source_meta
      17. text_meta
      18. name_meta
      19. other_meta
      20. mesh_term_meta
      21. mesh_concept_meta
      22. agent_interactions
    Note that the order of views below is determined not by the above
    order but by constraints imposed by use-case.

    Parameters
    ----------
    Base : type
        The base class for database tables
    """
    ro_tables = {}

    class Belief(Base, IndraDBTable):
        __tablename__ = 'belief'
        __table_args__ = {'schema': 'readonly'}
        _indices = [BtreeIndex('belief_mk_hash_idx', 'mk_hash')]
        _temp = False
        mk_hash = Column(BigInteger, primary_key=True)
        belief = Column(REAL)
    ro_tables[Belief.__tablename__] = Belief

    class EvidenceCounts(Base, ReadonlyTable):
        __tablename__ = 'evidence_counts'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('SELECT count(id) AS ev_count, mk_hash '
                          'FROM readonly.fast_raw_pa_link '
                          'GROUP BY mk_hash')
        _indices = [BtreeIndex('evidence_counts_mk_hash_idx', 'mk_hash')]
        mk_hash = Column(BigInteger, primary_key=True)
        ev_count = Column(Integer)
    ro_tables[EvidenceCounts.__tablename__] = EvidenceCounts

    class ReadingRefLink(Base, ReadonlyTable, IndraDBRefTable):
        __tablename__ = 'reading_ref_link'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('SELECT pmid, pmid_num, pmcid, pmcid_num, '
                          'pmcid_version, doi, doi_ns, doi_id, tr.id AS trid,'
                          'pii, url, manuscript_id, tc.id AS tcid, '
                          'source, r.id AS rid, reader '
                          'FROM text_ref AS tr JOIN text_content AS tc '
                          'ON tr.id = tc.text_ref_id JOIN reading AS r '
                          'ON tc.id = r.text_content_id')
        _indices = [BtreeIndex('rrl_rid_idx', 'rid'),
                    StringIndex('rrl_pmid_idx', 'pmid'),
                    BtreeIndex('rrl_pmid_num_idx', 'pmid_num'),
                    StringIndex('rrl_pmcid_idx', 'pmcid'),
                    BtreeIndex('rrl_pmcid_num_idx', 'pmcid_num'),
                    StringIndex('rrl_doi_idx', 'doi'),
                    BtreeIndex('rrl_doi_ns_idx', 'doi_ns'),
                    StringIndex('rrl_doi_id_idx', 'doi_id'),
                    StringIndex('rrl_manuscript_id_idx', 'manuscript_id'),
                    BtreeIndex('rrl_tcid_idx', 'tcid'),
                    BtreeIndex('rrl_trid_idx', 'trid')]
        trid = Column(Integer)
        pmid = Column(String(20))
        pmid_num = Column(Integer)
        pmcid = Column(String(20))
        pmcid_num = Column(Integer)
        pmcid_version = Column(Integer)
        doi = Column(String(100))
        doi_ns = Column(Integer)
        doi_id = Column(String)
        pii = Column(String(250))
        url = Column(String(250))
        manuscript_id = Column(String(100))
        tcid = Column(Integer)
        source = Column(String(250))
        rid = Column(Integer, primary_key=True)
        reader = Column(String(20))
    ro_tables[ReadingRefLink.__tablename__] = ReadingRefLink

    class FastRawPaLink(Base, ReadonlyTable):
        __tablename__ = 'fast_raw_pa_link'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('WITH %s\n'
                          'SELECT raw.id AS id,\n'
                          '       raw.json AS raw_json,\n'
                          '       raw.reading_id\n,'
                          '       raw.db_info_id,\n'
                          '       pa.mk_hash,\n'
                          '       pa.json AS pa_json,\n'
                          '       type_num,\n'
                          '       raw_src.src\n'
                          'FROM raw_statements AS raw,\n'
                          '     pa_statements AS pa,\n'
                          '     raw_unique_links AS link,\n'
                          '     readonly.raw_stmt_src as raw_src,\n'
                          '     type_map\n'
                          'WHERE link.raw_stmt_id = raw.id\n'
                          '  AND link.pa_stmt_mk_hash = pa.mk_hash\n'
                          '  AND raw_src.sid = raw.id\n'
                          '  AND pa.type = type_map.type')
        _skip_disp = ['raw_json', 'pa_json']
        _indices = [BtreeIndex('hash_index', 'mk_hash'),
                    BtreeIndex('frp_reading_id_idx', 'reading_id'),
                    BtreeIndex('frp_db_info_id_idx', 'db_info_id'),
                    StringIndex('frp_src_idx', 'src')]

        @classmethod
        def get_definition(cls):
            return cls.__definition__ % ro_type_map.get_with_clause()

        id = Column(Integer, primary_key=True)
        raw_json = Column(BYTEA)
        reading_id = Column(BigInteger)
        db_info_id = Column(Integer)
        mk_hash = Column(BigInteger)
        pa_json = Column(BYTEA)
        type_num = Column(SmallInteger)
    ro_tables[FastRawPaLink.__tablename__] = FastRawPaLink

    class PAAgentCounts(Base, ReadonlyTable):
        __tablename__ = 'pa_agent_counts'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ("SELECT count(distinct ag_num) as agent_count,"
                          "       stmt_mk_hash as mk_hash\n"
                          "FROM pa_agents GROUP BY stmt_mk_hash")
        _indices = [BtreeIndex('pa_agent_counts_mk_hash_idx', 'mk_hash')]
        mk_hash = Column(BigInteger, primary_key=True)
        agent_count = Column(Integer)
    ro_tables[PAAgentCounts.__tablename__] = PAAgentCounts

    class RawStmtSrc(Base, ReadonlyTable):
        __tablename__ = 'raw_stmt_src'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('SELECT raw_statements.id AS sid, '
                          'lower(reading.reader) AS src '
                          'FROM raw_statements, reading '
                          'WHERE reading.id = raw_statements.reading_id '
                          'UNION '
                          'SELECT raw_statements.id AS sid, '
                          'lower(db_info.db_name) AS src '
                          'FROM raw_statements, db_info '
                          'WHERE db_info.id = raw_statements.db_info_id')
        _indices = [BtreeIndex('raw_stmt_src_sid_idx', 'sid'),
                    StringIndex('raw_stmt_src_src_idx', 'src')]
        sid = Column(Integer, primary_key=True)
        src = Column(String)
    ro_tables[RawStmtSrc.__tablename__] = RawStmtSrc

    class _PaStmtSrc(Base, SpecialColumnTable):
        __tablename__ = 'pa_stmt_src'
        __table_args__ = {'schema': 'readonly'}
        __definition_fmt__ = ("SELECT * FROM crosstab("
                              "'SELECT mk_hash, src, count(id) "
                              "  FROM readonly.fast_raw_pa_link "
                              "  GROUP BY (mk_hash, src)', "
                              "$$SELECT unnest('{%s}'::text[])$$"
                              " ) final_result(mk_hash bigint, %s)")
        _indices = [BtreeIndex('pa_stmt_src_mk_hash_idx', 'mk_hash')]
        _temp = True
        loaded = False

        mk_hash = Column(BigInteger, primary_key=True)

        @classmethod
        def definition(cls, db):
            db.grab_session()

            # Make sure the necessary extension is installed.
            with db.get_conn() as conn:
                conn.execute('CREATE EXTENSION IF NOT EXISTS tablefunc;')

            logger.info("Discovering the possible sources...")
            src_list = db.session.query(db.RawStmtSrc.src).distinct().all()
            logger.info("Found the following sources: %s"
                        % [src for src, in src_list])
            entries = []
            cols = []
            for src, in src_list:
                if not cls.loaded:
                    setattr(cls, src, Column(BigInteger))
                cols.append(src)
                entries.append('%s bigint' % src)
            sql = cls.__definition_fmt__ % (', '.join(cols),
                                            ', '.join(entries))
            return sql

        def get_sources(self, include_none=False):
            src_dict = {}
            for k, v in self.__dict__.items():
                if k not in {'mk_hash'} and not k.startswith('_'):
                    if include_none or v is not None:
                        src_dict[k] = v
            return src_dict
    ro_tables[_PaStmtSrc.__tablename__] = _PaStmtSrc

    class _PaRefLink(Base, ReadonlyTable):
        __tablename__ = 'pa_ref_link'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('SELECT mk_hash, trid, pmid_num, pmcid_num, source,\n'
                          '       reader\n'
                          'FROM readonly.fast_raw_pa_link\n'
                          '  JOIN readonly.reading_ref_link '
                          '  ON reading_id = rid')
        _indices = [BtreeIndex('pa_ref_link_mk_hash_idx', 'mk_hash'),
                    BtreeIndex('pa_ref_link_trid_idx', 'trid'),
                    BtreeIndex('pa_ref_link_pmid_num_idx', 'pmid_num')]
        _temp = True
        mk_hash = Column(BigInteger, primary_key=True)
        trid = Column(Integer, primary_key=True)
        pmid_num = Column(String)
        pmcid_num = Column(String)
        source = Column(String)
        reader = Column(String)
    ro_tables[_PaRefLink.__tablename__] = _PaRefLink

    class _MeshTerms(Base, ReadonlyTable):
        __tablename__ = 'mesh_terms'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('SELECT pmid_num, mesh_num FROM mesh_ref_annotations\n'
                          '  WHERE is_concept IS NOT true\n'
                          'UNION\n'
                          'SELECT pmid_num, mesh_num FROM mti_ref_annotations_test\n'
                          '  WHERE NOT is_concept')
        _temp = True
        _indices = [BtreeIndex('mt_pmid_num_idx', 'pmid_num')]
        mesh_num = Column(Integer, primary_key=True)
        pmid_num = Column(Integer, primary_key=True)
    ro_tables[_MeshTerms.__tablename__] = _MeshTerms

    class _MeshConcepts(Base, ReadonlyTable):
        __tablename__ = 'mesh_concepts'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('SELECT pmid_num, mesh_num FROM mesh_ref_annotations\n'
                          '  WHERE is_concept IS true\n'
                          'UNION\n'
                          'SELECT pmid_num, mesh_num FROM mti_ref_annotations_test\n'
                          '  WHERE is_concept IS true')
        _temp = True
        _indices = [BtreeIndex('mc_pmid_num_idx', 'pmid_num')]
        mesh_num = Column(Integer, primary_key=True)
        pmid_num = Column(Integer, primary_key=True)
    ro_tables[_MeshConcepts.__tablename__] = _MeshConcepts

    class _HashPmidCounts(Base, ReadonlyTable):
        __tablename__ = 'hash_pmid_counts'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('SELECT mk_hash,\n'
                          '       count(distinct pmid_num)::integer as pmid_count\n'
                          'FROM readonly.pa_ref_link GROUP BY mk_hash')
        _temp = True
        _indices = [BtreeIndex('hpc_mk_hash_idx', 'mk_hash')]
        mk_hash = Column(BigInteger, primary_key=True)
        pmid_count = Column(Integer)
    ro_tables[_HashPmidCounts.__tablename__] = _HashPmidCounts

    class MeshTermRefCounts(Base, ReadonlyTable):
        __tablename__ = 'mesh_term_ref_counts'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('WITH mesh_hash_pmids AS (\n'
                          '    SELECT readonly.mesh_terms.pmid_num, mk_hash,\n'
                          '           mesh_num\n'
                          '    FROM readonly.mesh_terms, readonly.pa_ref_link\n'
                          '    WHERE readonly.mesh_terms.pmid_num = readonly.pa_ref_link.pmid_num\n'
                          '), mesh_ref_counts_proto AS (\n'
                          '    SELECT mk_hash, mesh_num,\n'
                          '           COUNT(DISTINCT pmid_num)::integer AS ref_count\n'
                          '    FROM mesh_hash_pmids GROUP BY mk_hash, mesh_num\n'
                          ')\n'
                          'SELECT readonly.hash_pmid_counts.mk_hash, mesh_num,\n'
                          '       ref_count, pmid_count\n'
                          'FROM mesh_ref_counts_proto, readonly.hash_pmid_counts\n'
                          'WHERE mesh_ref_counts_proto.mk_hash = readonly.hash_pmid_counts.mk_hash')
        _indices = [BtreeIndex('mtrc_mesh_num_idx', 'mesh_num', cluster=True),
                    BtreeIndex('mtrc_mk_hash_idx', 'mk_hash')]
        mk_hash = Column(BigInteger, primary_key=True)
        mesh_num = Column(Integer, primary_key=True)
        ref_count = Column(Integer)
        pmid_count = Column(Integer)
    ro_tables[MeshTermRefCounts.__tablename__] = MeshTermRefCounts

    class MeshConceptRefCounts(Base, ReadonlyTable):
        __tablename__ = 'mesh_concept_ref_counts'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('WITH mesh_hash_pmids AS (\n'
                          '    SELECT readonly.mesh_concepts.pmid_num, mk_hash,\n'
                          '           mesh_num\n'
                          '    FROM readonly.mesh_concepts, readonly.pa_ref_link\n'
                          '    WHERE readonly.mesh_concepts.pmid_num = readonly.pa_ref_link.pmid_num\n'
                          '), mesh_ref_counts_proto AS (\n'
                          '    SELECT mk_hash, mesh_num,\n'
                          '           COUNT(DISTINCT pmid_num)::integer AS ref_count\n'
                          '    FROM mesh_hash_pmids GROUP BY mk_hash, mesh_num\n'
                          ')\n'
                          'SELECT readonly.hash_pmid_counts.mk_hash, mesh_num,\n'
                          '       ref_count, pmid_count\n'
                          'FROM mesh_ref_counts_proto, readonly.hash_pmid_counts\n'
                          'WHERE mesh_ref_counts_proto.mk_hash = readonly.hash_pmid_counts.mk_hash')
        _indices = [BtreeIndex('mcrc_mesh_num_idx', 'mesh_num'),
                    BtreeIndex('mcrc_mk_hash_idx', 'mk_hash')]
        mk_hash = Column(BigInteger, primary_key=True)
        mesh_num = Column(Integer, primary_key=True)
        ref_count = Column(Integer)
        pmid_count = Column(Integer)
    ro_tables[MeshConceptRefCounts.__tablename__] = MeshConceptRefCounts

    class RawStmtMeshTerms(Base, ReadonlyTable):
        __tablename__ = 'raw_stmt_mesh_terms'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('SELECT DISTINCT raw_statements.id as sid,\n'
                          '       mesh_num\n'
                          'FROM text_ref\n'
                          '  JOIN readonly.mesh_terms AS mra\n'
                          '    ON text_ref.pmid_num = mra.pmid_num\n'
                          '  JOIN text_content ON text_ref.id = text_ref_id\n'
                          '  JOIN reading\n'
                          '    ON text_content.id = text_content_id\n'
                          '  JOIN raw_statements ON reading.id = reading_id\n')
        _indices = [BtreeIndex('rsmd_mesh_num_idx', 'mesh_num'),
                    BtreeIndex('rsmd_sid_idx', 'sid')]

        sid = Column(Integer, primary_key=True)
        mesh_num = Column(Integer, primary_key=True)
    ro_tables[RawStmtMeshTerms.__tablename__] = RawStmtMeshTerms

    class RawStmtMeshConcepts(Base, ReadonlyTable):
        __tablename__ = 'raw_stmt_mesh_concepts'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('SELECT DISTINCT raw_statements.id as sid,\n'
                          '       mesh_num\n'
                          'FROM text_ref\n'
                          '  JOIN readonly.mesh_concepts AS mra\n'
                          '    ON text_ref.pmid_num = mra.pmid_num\n'
                          '  JOIN text_content ON text_ref.id = text_ref_id\n'
                          '  JOIN reading\n'
                          '    ON text_content.id = text_content_id\n'
                          '  JOIN raw_statements ON reading.id = reading_id\n')
        _indices = [BtreeIndex('rsmc_mesh_num_idx', 'mesh_num'),
                    BtreeIndex('rsmc_sid_idx', 'sid')]

        sid = Column(Integer, primary_key=True)
        mesh_num = Column(Integer, primary_key=True)
    ro_tables[RawStmtMeshConcepts.__tablename__] = RawStmtMeshConcepts

    class _PaMeta(Base, ReadonlyTable):
        __tablename__ = 'pa_meta'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = (
            'SELECT pa_agents.db_name, pa_agents.db_id,\n'
            '       pa_agents.id AS ag_id, role_num, pa_agents.ag_num,\n'
            '       type_num, pa_statements.mk_hash,\n'
            '       readonly.evidence_counts.ev_count, readonly.belief.belief,\n'
            '       activity, is_active, agent_count, false AS is_complex_dup\n'
            'FROM pa_agents, pa_statements, readonly.pa_agent_counts, type_map,'
            '  role_map, readonly.belief, readonly.evidence_counts'
            '  LEFT JOIN pa_activity'
            '  ON readonly.evidence_counts.mk_hash = pa_activity.stmt_mk_hash\n'
            'WHERE pa_agents.stmt_mk_hash = pa_statements.mk_hash\n'
            '  AND pa_statements.mk_hash = readonly.evidence_counts.mk_hash\n'
            '  AND pa_statements.mk_hash = readonly.belief.mk_hash\n'
            '  AND readonly.pa_agent_counts.mk_hash = pa_agents.stmt_mk_hash\n'
            '  AND pa_statements.type = type_map.type\n'
            '  AND pa_agents.role = role_map.role\n'
            '  AND LENGTH(pa_agents.db_id) < 2000'
        )
        _temp = True
        _indices = [StringIndex('pa_meta_db_name_idx', 'db_name'),
                    BtreeIndex('pa_meta_hash_idx', 'mk_hash')]
        ag_id = Column(Integer, primary_key=True)
        ag_num = Column(Integer)
        db_name = Column(String)
        db_id = Column(String)
        role_num = Column(SmallInteger)
        type_num = Column(SmallInteger)
        mk_hash = Column(BigInteger, primary_key=True)
        ev_count = Column(Integer)
        belief = Column(REAL)
        activity = Column(String)
        is_active = Column(Boolean)
        agent_count = Column(Integer)
        is_complex_dup = Column(Boolean)

        @classmethod
        def create(cls, db, commit=True):
            sql = cls.__create_table_fmt__ \
                  % (cls.full_name(force_schema=True),
                     cls.get_definition())
            sql += '\n'
            sql += (f'INSERT INTO readonly.pa_meta \n'
                    f'SELECT db_name, db_id, ag_id,\n '
                    f'  generate_series(-1, 1, 2) AS role_num,\n'
                    f'  generate_series(0, 1) AS ag_num,\n'
                    f'  type_num, mk_hash, ev_count, belief, activity,\n'
                    f'  is_active, agent_count, true AS is_complex_dup\n'
                    f'FROM readonly.pa_meta\n'
                    f'WHERE type_num = {ro_type_map.get_int("Complex")}\n')
            if commit:
                cls.execute(db, sql)
            return sql

        @classmethod
        def get_definition(cls):
            with_clause = 'WITH\n'
            with_clause += ro_type_map.get_with_clause() + ','
            with_clause += ro_role_map.get_with_clause() + '\n'
            return with_clause + cls.__definition__

    ro_tables[_PaMeta.__tablename__] = _PaMeta

    class SourceMeta(Base, SpecialColumnTable):
        __tablename__ = 'source_meta'
        __table_args__ = {'schema': 'readonly'}
        __definition_fmt__ = (
            'WITH jsonified AS (\n'
            '    SELECT mk_hash, \n'
            '           json_strip_nulls(json_build_object({all_sources})) \n'
            '           AS src_json \n'
            '    FROM readonly.pa_stmt_src\n'
            '),'
            'meta AS ('
            '    SELECT distinct mk_hash, type_num, activity, is_active,\n'
            '                    ev_count, belief, agent_count'
            '    FROM readonly.name_meta'
            '    WHERE NOT is_complex_dup'
            ')\n'
            'SELECT readonly.pa_stmt_src.*, \n'
            '       meta.ev_count, \n'
            '       meta.belief, \n'
            '       meta.type_num, \n'
            '       meta.activity, \n'
            '       meta.is_active,\n'
            '       meta.agent_count,\n'
            '       diversity.num_srcs, \n'
            '       jsonified.src_json, \n'
            '       CASE WHEN diversity.num_srcs = 1 \n'
            '            THEN (ARRAY(\n'
            '              SELECT json_object_keys(jsonified.src_json)\n'
            '              ))[1]\n'
            '            ELSE null \n'
            '            END as only_src,\n'
            '       ARRAY(\n'
            '         SELECT json_object_keys(jsonified.src_json)\n'
            '         ) \n'
            '         &&\n'
            '         ARRAY[{reading_sources}]\n'
            '         as has_rd,\n'
            '       ARRAY(\n'
            '         SELECT json_object_keys(jsonified.src_json)\n'
            '         ) \n'
            '         &&\n'
            '        ARRAY[{db_sources}]\n'
            '        as has_db\n'
            'FROM (\n'
            '  SELECT mk_hash, count(src) AS num_srcs \n'
            '  FROM jsonified, json_object_keys(jsonified.src_json) as src \n'
            '  GROUP BY mk_hash\n'
            ') AS diversity\n'
            'JOIN jsonified \n'
            '  ON jsonified.mk_hash = diversity.mk_hash\n'
            'JOIN readonly.pa_stmt_src \n'
            '  ON diversity.mk_hash = readonly.pa_stmt_src.mk_hash\n'
            'JOIN meta \n'
            '  ON diversity.mk_hash = meta.mk_hash'
        )
        _indices = [BtreeIndex('source_meta_mk_hash_idx', 'mk_hash'),
                    StringIndex('source_meta_only_src_idx', 'only_src'),
                    StringIndex('source_meta_activity_idx', 'activity'),
                    BtreeIndex('source_meta_type_num_idx', 'type_num'),
                    BtreeIndex('source_meta_num_srcs_idx', 'num_srcs')]
        loaded = False

        mk_hash = Column(BigInteger, primary_key=True)
        ev_count = Column(Integer)
        belief = Column(REAL)
        num_srcs = Column(Integer)
        src_json = Column(JSON)
        only_src = Column(String)
        has_rd = Column(Boolean)
        has_db = Column(Boolean)
        type_num = Column(SmallInteger)
        activity = Column(String)
        is_active = Column(Boolean)
        agent_count = Column(Integer)

        @classmethod
        def definition(cls, db):
            db.grab_session()
            srcs = set(db.get_column_names(db._PaStmtSrc)) - {'mk_hash'}
            all_sources = ', '.join(s for src in srcs
                                    for s in (repr(src), src))
            rd_sources = ', '.join(repr(src)
                                   for src in SOURCE_GROUPS['reading'])
            db_sources = ', '.join(repr(src)
                                   for src in SOURCE_GROUPS['databases'])
            sql = cls.__definition_fmt__.format(all_sources=all_sources,
                                                reading_sources=rd_sources,
                                                db_sources=db_sources)
            return sql
    ro_tables[SourceMeta.__tablename__] = SourceMeta

    class TextMeta(Base, NamespaceLookup):
        __tablename__ = 'text_meta'
        __table_args__ = {'schema': 'readonly'}
        __dbname__ = 'TEXT'
        _indices = [StringIndex('text_meta_db_id_idx', 'db_id'),
                    BtreeIndex('text_meta_type_num_idx', 'type_num'),
                    StringIndex('text_meta_activity_idx', 'activity'),
                    BtreeIndex('text_meta_mk_hash_idx', 'mk_hash')]
        ag_id = Column(Integer, primary_key=True)
        ag_num = Column(Integer)
        db_id = Column(String)
        role_num = Column(SmallInteger)
        type_num = Column(SmallInteger)
        mk_hash = Column(BigInteger)
        ev_count = Column(Integer)
        belief = Column(REAL)
        activity = Column(String)
        is_active = Column(Boolean)
        agent_count = Column(Integer)
        is_complex_dup = Column(Boolean)
    ro_tables[TextMeta.__tablename__] = TextMeta

    class NameMeta(Base, NamespaceLookup):
        __tablename__ = 'name_meta'
        __table_args__ = {'schema': 'readonly'}
        __dbname__ = 'NAME'
        _indices = [StringIndex('name_meta_db_id_idx', 'db_id'),
                    BtreeIndex('name_meta_type_num_idx', 'type_num'),
                    StringIndex('name_meta_activity_idx', 'activity'),
                    BtreeIndex('name_meta_mk_hash_idx', 'mk_hash')]
        ag_id = Column(Integer, primary_key=True)
        ag_num = Column(Integer)
        db_id = Column(String)
        role_num = Column(SmallInteger)
        type_num = Column(SmallInteger)
        mk_hash = Column(BigInteger)
        ev_count = Column(Integer)
        belief = Column(REAL)
        activity = Column(String)
        is_active = Column(Boolean)
        agent_count = Column(Integer)
        is_complex_dup = Column(Boolean)
    ro_tables[NameMeta.__tablename__] = NameMeta

    class OtherMeta(Base, ReadonlyTable):
        __tablename__ = 'other_meta'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ("SELECT db_name, db_id, ag_id, role_num, ag_num,\n"
                          "       type_num, mk_hash, ev_count, belief,\n"
                          "       activity, is_active, agent_count,\n"
                          "       is_complex_dup\n"
                          "FROM readonly.pa_meta\n"
                          "WHERE db_name NOT IN ('NAME', 'TEXT')")
        _indices = [StringIndex('other_meta_db_id_idx', 'db_id'),
                    BtreeIndex('other_meta_type_num_idx', 'type_num'),
                    StringIndex('other_meta_db_name_idx', 'db_name'),
                    StringIndex('other_meta_activity_idx', 'activity'),
                    BtreeIndex('other_meta_mk_hash_idx', 'mk_hash')]
        ag_id = Column(Integer, primary_key=True)
        ag_num = Column(Integer)
        db_name = Column(String)
        db_id = Column(String)
        role_num = Column(SmallInteger)
        type_num = Column(SmallInteger)
        mk_hash = Column(BigInteger)
        ev_count = Column(Integer)
        belief = Column(REAL)
        activity = Column(String)
        is_active = Column(Boolean)
        agent_count = Column(Integer)
        is_complex_dup = Column(Boolean)
    ro_tables[OtherMeta.__tablename__] = OtherMeta

    class MeshTermMeta(Base, ReadonlyTable):
        __tablename__ = 'mesh_term_meta'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ("SELECT DISTINCT meta.mk_hash, meta.ev_count,\n"
                          "       meta.belief, mesh_num, type_num, activity,\n"
                          "       is_active, agent_count\n"
                          "FROM readonly.raw_stmt_mesh_terms AS rsmt,\n"
                          "     readonly.source_meta AS meta,\n"
                          "     raw_unique_links AS link\n"
                          "WHERE rsmt.sid = link.raw_stmt_id\n"
                          "  AND meta.mk_hash = link.pa_stmt_mk_hash")
        _indices = [BtreeIndex('mesh_term_meta_mesh_num_idx', 'mesh_num',
                               cluster=True),
                    BtreeIndex('mesh_term_meta_mk_hash_idx', 'mk_hash'),
                    BtreeIndex('mesh_term_meta_type_num_idx', 'type_num'),
                    StringIndex('mesh_term_meta_activity_idx', 'activity')]
        mk_hash = Column(BigInteger, primary_key=True)
        mesh_num = Column(Integer, primary_key=True)
        tr_count = Column(Integer)
        ev_count = Column(Integer)
        belief = Column(REAL)
        type_num = Column(SmallInteger)
        activity = Column(String)
        is_active = Column(Boolean)
        agent_count = Column(Integer)
    ro_tables[MeshTermMeta.__tablename__] = MeshTermMeta

    class MeshConceptMeta(Base, ReadonlyTable):
        __tablename__ = 'mesh_concept_meta'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ("SELECT DISTINCT meta.mk_hash, meta.ev_count,\n"
                          "       meta.belief, mesh_num, type_num, activity,\n"
                          "       is_active, agent_count\n"
                          "FROM readonly.raw_stmt_mesh_concepts AS rsmc,\n"
                          "     readonly.source_meta AS meta,\n"
                          "     raw_unique_links AS link\n"
                          "WHERE rsmc.sid = link.raw_stmt_id\n"
                          "  AND meta.mk_hash = link.pa_stmt_mk_hash")
        _indices = [BtreeIndex('mesh_concept_meta_mesh_num_idx', 'mesh_num'),
                    BtreeIndex('mesh_concept_meta_mk_hash_idx', 'mk_hash'),
                    BtreeIndex('mesh_concept_meta_type_num_idx', 'type_num'),
                    StringIndex('mesh_concept_meta_activity_idx', 'activity')]
        mk_hash = Column(BigInteger, primary_key=True)
        mesh_num = Column(Integer, primary_key=True)
        tr_count = Column(Integer)
        ev_count = Column(Integer)
        belief = Column(REAL)
        type_num = Column(SmallInteger)
        activity = Column(String)
        is_active = Column(Boolean)
        agent_count = Column(Integer)
    ro_tables[MeshConceptMeta.__tablename__] = MeshConceptMeta

    class AgentInteractions(Base, ReadonlyTable):
        __tablename__ = 'agent_interactions'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ("SELECT\n" 
                          "  low_level_names.mk_hash AS mk_hash, \n"
                          "  jsonb_object(\n"
                          "    array_agg(\n"
                          "      CAST(\n"
                          "        low_level_names.ag_num AS VARCHAR)),\n"
                          "    array_agg(low_level_names.db_id)\n"
                          "  ) AS agent_json, \n"
                          "  low_level_names.type_num AS type_num, \n"
                          "  low_level_names.agent_count AS agent_count, \n"
                          "  low_level_names.ev_count AS ev_count, \n"
                          "  low_level_names.belief AS belief, \n"
                          "  low_level_names.activity AS activity, \n"
                          "  low_level_names.is_active AS is_active, \n"
                          "  CAST(\n"
                          "    low_level_names.src_json AS JSONB\n"
                          "  ) AS src_json, \n"
                          "  false AS is_complex_dup\n"
                          "FROM \n"
                          "  (\n"
                          "    SELECT \n"
                          "      readonly.name_meta.mk_hash AS mk_hash, \n"
                          "      readonly.name_meta.db_id AS db_id, \n"
                          "      readonly.name_meta.ag_num AS ag_num, \n"
                          "      readonly.name_meta.type_num AS type_num, \n"
                          "      readonly.name_meta.agent_count AS agent_count, \n"
                          "      readonly.name_meta.ev_count AS ev_count, \n"
                          "      readonly.name_meta.belief AS belief, \n"
                          "      readonly.name_meta.activity AS activity, \n"
                          "      readonly.name_meta.is_active AS is_active, \n"
                          "      readonly.source_meta.src_json AS src_json \n"
                          "    FROM \n"
                          "      readonly.name_meta, \n"
                          "      readonly.source_meta\n"
                          "    WHERE \n"
                          "      readonly.name_meta.mk_hash \n"
                          "        = readonly.source_meta.mk_hash\n"
                          "      AND NOT readonly.name_meta.is_complex_dup"
                          "  ) AS low_level_names \n"
                          "GROUP BY \n"
                          "  low_level_names.mk_hash, \n"
                          "  low_level_names.type_num, \n"
                          "  low_level_names.agent_count, \n"
                          "  low_level_names.ev_count, \n"
                          "  low_level_names.belief, \n"
                          "  low_level_names.activity, \n"
                          "  low_level_names.is_active, \n"
                          "  CAST(low_level_names.src_json AS JSONB)")
        _indices = [BtreeIndex('agent_interactions_mk_hash_idx', 'mk_hash'),
                    BtreeIndex('agent_interactions_agent_json_idx', 'agent_json'),
                    BtreeIndex('agent_interactions_type_num_idx', 'type_num')]
        _always_disp = ['mk_hash', 'agent_json']

        @classmethod
        def create(cls, db, commit=True):
            super(AgentInteractions, cls).create(db, commit)
            from itertools import permutations
            interactions = db.select_all(
                db.AgentInteractions,
                db.AgentInteractions.type_num == ro_type_map.get_int('Complex')
            )
            new_interactions = []
            for interaction in interactions:
                if interaction.agent_count < 2:
                    continue
                for pair in permutations(interaction.agent_json, 2):
                    if interaction.agent_count == 2 and pair == ('0', '1'):
                        continue
                    new_agent_json = {str(i): interaction.agent_json[j]
                                      for i, j in enumerate(pair)}
                    new_interactions.append(
                        (interaction.mk_hash, interaction.ev_count,
                         interaction.belief, interaction.type_num, 2,
                         new_agent_json, interaction.src_json, True)
                    )
            db.copy('readonly.agent_interactions', new_interactions,
                    ('mk_hash', 'ev_count', 'belief', 'type_num', 'agent_count',
                     'agent_json', 'src_json', 'is_complex_dup'))
            return

        mk_hash = Column(BigInteger, primary_key=True)
        ev_count = Column(Integer)
        belief = Column(REAL)
        type_num = Column(SmallInteger)
        activity = Column(String)
        is_active = Column(Boolean)
        agent_count = Column(Integer)
        agent_json = Column(JSONB)
        src_json = Column(JSONB)
        is_complex_dup = Column(Boolean)
    ro_tables[AgentInteractions.__tablename__] = AgentInteractions

    return ro_tables


SOURCE_GROUPS = {'databases': ['phosphosite', 'cbn', 'pc11', 'biopax',
                               'bel_lc', 'signor', 'biogrid', 'tas',
                               'lincs_drug', 'hprd', 'trrust'],
                 'reading': ['geneways', 'tees', 'isi', 'trips', 'rlimsp',
                             'medscan', 'sparser', 'reach', 'eidos', 'mti']}
