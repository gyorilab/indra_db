__all__ = ['get_schema']

import logging

from sqlalchemy import Column, Integer, String, ForeignKey, BigInteger, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import BYTEA, JSON

from .mixins import ReadonlyTable, NamespaceLookup, SpecialColumnTable
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
    'pa_meta',
    'mesh_ref_lookup',
    'source_meta',
    'text_meta',
    'name_meta',
    'other_meta',
    'mesh_meta',
]
CREATE_UNORDERED = {}


def get_schema(Base):
    """Return the schema for the reading view of the database.

    We use a readonly database to allow fast and efficient load of data,
    and to add a layer of separation between the processes of updating
    the content of the database and accessing the content of the
    database. However, it is not practical to have the views created
    through sqlalchemy: instead they are generated and updated manually
    (or by other non-sqlalchemy scripts).

    The following views must be built in this specific order:
      1. raw_stmt_src
      2. fast_raw_pa_link
      3. pa_agent_counts
      4. pa_stmt_src
      5. evidence_counts
      6. reading_ref_link
      7. pa_ref_link
      8. pa_meta
      9. mesh_ref_lookup
     10. source_meta
     11. text_meta
     12. name_meta
     13. other_meta
     14. mesh_meta
    The following can be built at any time and in any order:
        (None currently)
    Note that the order of views below is determined not by the above
    order but by constraints imposed by use-case.

    Parameters
    ----------
    Base : type
        The base class for database tables
    """
    read_views = {}

    class EvidenceCounts(Base, ReadonlyTable):
        __tablename__ = 'evidence_counts'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('SELECT count(id) AS ev_count, mk_hash '
                          'FROM readonly.fast_raw_pa_link '
                          'GROUP BY mk_hash')
        _indices = [BtreeIndex('evidence_counts_mk_hash_idx', 'mk_hash')]
        mk_hash = Column(BigInteger, primary_key=True)
        ev_count = Column(Integer)
    read_views[EvidenceCounts.__tablename__] = EvidenceCounts

    class ReadingRefLink(Base, ReadonlyTable):
        __tablename__ = 'reading_ref_link'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('SELECT pmid, pmcid, tr.id AS trid, doi, '
                          'pii, url, manuscript_id, tc.id AS tcid, '
                          'source, r.id AS rid, reader '
                          'FROM text_ref AS tr JOIN text_content AS tc '
                          'ON tr.id = tc.text_ref_id JOIN reading AS r '
                          'ON tc.id = r.text_content_id')
        _indices = [BtreeIndex('rrl_rid_idx', 'rid'),
                    StringIndex('rrl_pmid_idx', 'pmid'),
                    StringIndex('rrl_pmcid_idx', 'pmcid'),
                    StringIndex('rrl_doi_idx', 'doi'),
                    StringIndex('rrl_manuscript_id_idx', 'manuscript_id'),
                    BtreeIndex('rrl_tcid_idx', 'tcid'),
                    BtreeIndex('rrl_trid_idx', 'trid')]
        trid = Column(Integer)
        pmid = Column(String(20))
        pmcid = Column(String(20))
        doi = Column(String(100))
        pii = Column(String(250))
        url = Column(String(250))
        manuscript_id = Column(String(100))
        tcid = Column(Integer)
        source = Column(String(250))
        rid = Column(Integer, primary_key=True)
        reader = Column(String(20))
    read_views[ReadingRefLink.__tablename__] = ReadingRefLink

    class FastRawPaLink(Base, ReadonlyTable):
        __tablename__ = 'fast_raw_pa_link'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('SELECT raw.id AS id, raw.json AS raw_json, '
                          'raw.reading_id, raw.db_info_id, '
                          'pa.mk_hash, pa.json AS pa_json, pa.type,'
                          'raw_src.src '
                          'FROM raw_statements AS raw, '
                          'pa_statements AS pa, '
                          'raw_unique_links AS link,'
                          'readonly.raw_stmt_src as raw_src '
                          'WHERE link.raw_stmt_id = raw.id '
                          'AND link.pa_stmt_mk_hash = pa.mk_hash '
                          'AND raw_src.sid = raw.id')
        _skip_disp = ['raw_json', 'pa_json']
        _indices = [BtreeIndex('hash_index', 'mk_hash'),
                    BtreeIndex('frp_reading_id_idx', 'reading_id'),
                    BtreeIndex('frp_db_info_id_idx', 'db_info_id'),
                    StringIndex('frp_src_idx', 'src')]
        id = Column(Integer, primary_key=True)
        raw_json = Column(BYTEA)
        reading_id = Column(BigInteger,
                            ForeignKey('readonly.reading_ref_link.rid'))
        reading_ref = relationship(ReadingRefLink)
        db_info_id = Column(Integer)
        mk_hash = Column(BigInteger,
                         ForeignKey('readonly.evidence_counts.mk_hash'))
        ev_counts = relationship(EvidenceCounts)
        pa_json = Column(BYTEA)
        type = Column(String)
    read_views[FastRawPaLink.__tablename__] = FastRawPaLink

    class PAAgentCounts(Base, ReadonlyTable):
        __tablename__ = 'pa_agent_counts'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ("SELECT count(distinct ag_num) as agent_count,"
                          "       mk_hash\n"
                          "FROM pa_agents GROUP BY mk_hash")
        _indices = [BtreeIndex('pa_agent_counts_mk_hash_idx', 'mk_hash')]
        mk_hash = Column(BigInteger, primary_key=True)
        agent_count = Column(Integer)
    read_views[PAAgentCounts.__tablename__] = PAAgentCounts

    class PaMeta(Base, ReadonlyTable):
        __tablename__ = 'pa_meta'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = (
            'SELECT pa_agents.db_name, pa_agents.db_id,\n'
            '       pa_agents.id AS ag_id, pa_agents.role, pa_agents.ag_num,\n'
            '       pa_statements.type, pa_statements.mk_hash,\n'
            '       readonly.evidence_counts.ev_count, activity, is_active,\n'
            '       agent_count\n'
            'FROM pa_agents, pa_statements, readonly.pa_agent_counts,\n'
            '  readonly.evidence_counts'
            '  LEFT JOIN pa_activity'
            '  ON readonly.evidence_counts.mk_hash = pa_activity.stmt_mk_hash\n'
            'WHERE pa_agents.stmt_mk_hash = pa_statements.mk_hash\n'
            '  AND pa_statements.mk_hash = readonly.evidence_counts.mk_hash\n'
            '  AND readonly.pa_agent_counts.mk_hash = pa_agents.stmt_mk_hash'
        )
        _indices = [StringIndex('pa_meta_db_name_idx', 'db_name'),
                    StringIndex('pa_meta_db_id_idx', 'db_id'),
                    BtreeIndex('pa_meta_hash_idx', 'mk_hash')]
        ag_id = Column(Integer, primary_key=True)
        ag_num = Column(Integer)
        db_name = Column(String)
        db_id = Column(String)
        role = Column(String(20))
        type = Column(String(100))
        mk_hash = Column(BigInteger,
                         ForeignKey('readonly.fast_raw_pa_link.mk_hash'))
        raw_pa_link = relationship(FastRawPaLink)
        ev_count = Column(Integer)
        activity = Column(String)
        is_active = Column(Boolean)
        agent_count = Column(Integer)
    read_views[PaMeta.__tablename__] = PaMeta

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
    read_views[RawStmtSrc.__tablename__] = RawStmtSrc

    class PaStmtSrc(Base, SpecialColumnTable):
        __tablename__ = 'pa_stmt_src'
        __table_args__ = {'schema': 'readonly'}
        __definition_fmt__ = ("SELECT * FROM crosstab("
                              "'SELECT mk_hash, src, count(id) "
                              "  FROM readonly.fast_raw_pa_link "
                              "  GROUP BY (mk_hash, src)', "
                              "$$SELECT unnest('{%s}'::text[])$$"
                              " ) final_result(mk_hash bigint, %s)")
        _indices = [BtreeIndex('pa_stmt_src_mk_hash_idx', 'mk_hash')]
        loaded = False

        mk_hash = Column(BigInteger, primary_key=True)

        @classmethod
        def definition(cls, db):
            db.grab_session()

            # Make sure the necessary extension is installed.
            with db.engine.connect() as conn:
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
    read_views[PaStmtSrc.__tablename__] = PaStmtSrc

    class PaRefLink(Base, ReadonlyTable):
        __tablename__ = 'pa_ref_link'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ('SELECT mk_hash, trid, pmid, pmcid, source, reader '
                          'FROM readonly.fast_raw_pa_link '
                          '  JOIN readonly.reading_ref_link '
                          '  ON reading_id = rid')
        _indices = [BtreeIndex('pa_ref_link_mk_hash_idx', 'mk_hash'),
                    BtreeIndex('pa_ref_link_trid_idx', 'trid'),
                    BtreeIndex('pa_ref_link_pmid_idx', 'pmid')]
        mk_hash = Column(BigInteger, primary_key=True)
        trid = Column(Integer, primary_key=True)
        pmid = Column(String)
        pmcid = Column(String)
        source = Column(String)
        reader = Column(String)
    read_views[PaRefLink.__tablename__] = PaRefLink

    class MeshRefLookup(Base, ReadonlyTable):
        __tablename__ = 'mesh_ref_lookup'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ("SELECT text_ref.id AS trid,\n"
                          "       reading.id AS rid,\n"
                          "       raw_statements.id AS sid,\n"
                          "       pa_stmt_mk_hash AS mk_hash,\n"
                          "       SUBSTRING(mesh_id, 2)::int as mesh_num\n"
                          "FROM text_ref\n"
                          "  JOIN mesh_ref_annotations\n"
                          "    ON text_ref.pmid = mesh_ref_annotations.pmid\n"
                          "  JOIN text_content ON text_ref.id = text_ref_id\n"
                          "  JOIN reading \n"
                          "    ON text_content.id = text_content_id\n"
                          "  JOIN raw_statements ON reading.id = reading_id\n"
                          "  JOIN raw_unique_links \n"
                          "    ON raw_statements.id = raw_stmt_id")
        _indices = [BtreeIndex('mrl_mesh_num_idx', 'mesh_num'),
                    BtreeIndex('mrl_mk_hash_idx', 'mk_hash'),
                    BtreeIndex('mrl_sid_idx', 'sid')]

        trid = Column(Integer, primary_key=True)
        rid = Column(BigInteger)
        mesh_num = Column(Integer, primary_key=True)
        sid = Column(Integer, primary_key=True)
        mk_hash = Column(BigInteger)
    read_views[MeshRefLookup.__tablename__] = MeshRefLookup

    class SourceMeta(Base, SpecialColumnTable):
        __tablename__ = 'pa_source_meta'
        __table_args__ = {'schema': 'readonly'}
        __definition_fmt__ = (
            'WITH jsonified AS (\n'
            '    SELECT mk_hash, \n'
            '           json_strip_nulls(json_build_object({all_sources})) \n'
            '           AS src_json \n'
            '    FROM readonly.pa_stmt_src\n'
            ')\n'
            'SELECT readonly.pa_stmt_src.*, \n'
            '       readonly.pa_meta.ev_count, \n'
            '       readonly.pa_meta.activity, \n'
            '       readonly.pa_meta.is_active,\n'
            '       readonly.pa_meta.agent_count,\n'
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
            'JOIN readonly.pa_meta \n'
            '  ON diversity.mk_hash = readonly.pa_meta.mk_hash'
        )
        _indices = [BtreeIndex('source_meta_mk_hash_idx', 'mk_hash'),
                    StringIndex('source_meta_only_src_idx', 'only_src'),
                    StringIndex('source_meta_activity_idx', 'activity'),
                    StringIndex('source_meta_type_idx', 'type'),
                    BtreeIndex('source_meta_num_srcs_idx', 'num_srcs')]
        loaded = False

        mk_hash = Column(BigInteger, primary_key=True)
        ev_count = Column(Integer)
        num_srcs = Column(Integer)
        src_json = Column(JSON)
        only_src = Column(String)
        has_rd = Column(Boolean)
        has_db = Column(Boolean)
        activity = Column(String)
        is_active = Column(Boolean)
        agent_count = Column(Integer)

        @classmethod
        def definition(cls, db):
            db.grab_session()
            srcs = set(db.get_column_names(db.PaStmtSrc)) - {'mk_hash'}
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
    read_views[SourceMeta.__tablename__] = SourceMeta

    class TextMeta(Base, NamespaceLookup):
        __tablename__ = 'text_meta'
        __table_args__ = {'schema': 'readonly'}
        __dbname__ = 'TEXT'
        _indices = [StringIndex('text_meta_db_id_idx', 'db_id'),
                    StringIndex('text_meta_type_idx', 'type'),
                    StringIndex('text_meta_activity_idx', 'activity')]
        ag_id = Column(Integer, primary_key=True)
        ag_num = Column(Integer)
        db_id = Column(String)
        role = Column(String(20))
        type = Column(String(100))
        mk_hash = Column(BigInteger,
                         ForeignKey('readonly.fast_raw_pa_link.mk_hash'))
        raw_pa_link = relationship(FastRawPaLink)
        ev_count = Column(Integer)
        activity = Column(String)
        is_active = Column(Boolean)
        agent_count = Column(Integer)
    read_views[TextMeta.__tablename__] = TextMeta

    class NameMeta(Base, NamespaceLookup):
        __tablename__ = 'name_meta'
        __table_args__ = {'schema': 'readonly'}
        __dbname__ = 'NAME'
        _indices = [StringIndex('name_meta_db_id_idx', 'db_id'),
                    StringIndex('name_meta_type_idx', 'type'),
                    StringIndex('name_meta_activity_idx', 'activity')]
        ag_id = Column(Integer, primary_key=True)
        ag_num = Column(Integer)
        db_id = Column(String)
        role = Column(String(20))
        type = Column(String(100))
        mk_hash = Column(BigInteger,
                         ForeignKey('readonly.fast_raw_pa_link.mk_hash'))
        raw_pa_link = relationship(FastRawPaLink)
        ev_count = Column(Integer)
        activity = Column(String)
        is_active = Column(Boolean)
        agent_count = Column(Integer)
    read_views[NameMeta.__tablename__] = NameMeta

    class OtherMeta(Base, ReadonlyTable):
        __tablename__ = 'other_meta'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ("SELECT db_name, db_id, ag_id, role, ag_num,\n"
                          "       type, mk_hash, ev_count, activity,\n"
                          "       is_active, agent_count\n"
                          "FROM readonly.pa_meta\n"
                          "WHERE db_name NOT IN ('NAME', 'TEXT')")
        _indices = [StringIndex('other_meta_db_id_idx', 'db_id'),
                    StringIndex('other_meta_type_idx', 'type'),
                    StringIndex('other_meta_db_name_idx', 'db_name'),
                    StringIndex('other_meta_activity_idx', 'activity')]
        ag_id = Column(Integer, primary_key=True)
        ag_num = Column(Integer)
        db_name = Column(String)
        db_id = Column(String)
        role = Column(String(20))
        type = Column(String(100))
        mk_hash = Column(BigInteger,
                         ForeignKey('readonly.fast_raw_pa_link.mk_hash'))
        raw_pa_link = relationship(FastRawPaLink)
        ev_count = Column(Integer)
        activity = Column(String)
        is_active = Column(Boolean)
        agent_count = Column(Integer)
    read_views[OtherMeta.__tablename__] = OtherMeta

    class MeshMeta(Base, ReadonlyTable):
        __tablename__ = 'mesh_meta'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ("SELECT count(distinct trid) AS tr_count,\n"
                          "       count(distinct sid) AS ev_count,\n"
                          "       mk_hash, mesh_num, type,\n"
                          "       activity, is_active, agent_count\n"
                          "FROM readonly.mesh_ref_lookup,"
                          "  readonly.pa_agent_counts\n"
                          "WHERE readonly.mesh_ref_lookup.mk_hash "
                          "  = readonly.pa_agent_counts.mk_hash\n"
                          "GROUP BY mk_hash, mesh_num")
        _indices = [BtreeIndex('mesh_meta_mesh_num_idx', 'mesh_num'),
                    BtreeIndex('mesh_meta_mk_hash_idx', 'mk_hash'),
                    StringIndex('mesh_meta_type_idx', 'type'),
                    StringIndex('mesh_meta_activity_idx', 'activity')]
        mk_hash = Column(BigInteger,
                         ForeignKey('readonly.fast_raw_pa_link.mk_hash'),
                         primary_key=True)
        mesh_num = Column(Integer, primary_key=True)
        tr_count = Column(Integer)
        ev_count = Column(Integer)
        activity = Column(String)
        is_active = Column(Boolean)
        agent_count = Column(Integer)
    read_views[MeshMeta.__tablename__] = MeshMeta

    return read_views


SOURCE_GROUPS = {'databases': ['phosphosite', 'cbn', 'pc11', 'biopax',
                               'bel_lc', 'signor', 'biogrid', 'tas',
                               'lincs_drug', 'hprd', 'trrust'],
                 'reading': ['geneways', 'tees', 'isi', 'trips', 'rlimsp',
                             'medscan', 'sparser', 'reach']}
