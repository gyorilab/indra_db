__all__ = ['get_schema']

import logging

from sqlalchemy import Column, Integer, String, ForeignKey, BigInteger, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import BYTEA, JSON

from .mixins import ReadonlyTable, NamespaceLookup, SpecialColumnTable
from .indexes import *


logger = logging.getLogger(__name__)

CREATE_ORDER = ['raw_stmt_src', 'fast_raw_pa_link', 'pa_stmt_src',
                'evidence_counts', 'pa_source_lookup', 'pa_meta', 'text_meta',
                'name_meta']
CREATE_UNORDERED = {'reading_ref_link'}


def get_schema(Base):
    '''Return the schema for the reading view of the database.

    We use a readonly database to allow fast and efficient load of data,
    and to add a layer of separation between the processes of updating
    the content of the database and accessing the content of the
    database. However, it is not practical to have the views created
    through sqlalchemy: instead they are generated and updated manually
    (or by other non-sqlalchemy scripts).

    The following views must be built in this specific order:
      1. raw_stmt_src
      2. fast_raw_pa_link
      3. pa_stmt_src
      4. evidence_counts
      5. pa_source_lookup
      6. pa_meta
      7. text_meta
      8. name_meta
    The following can be built at any time and in any order:
      - reading_ref_link
    Note that the order of views below is determined not by the above
    order but by constraints imposed by use-case.
    '''
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

    class PaMeta(Base, ReadonlyTable):
        __tablename__ = 'pa_meta'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = (
            'SELECT pa_agents.db_name, pa_agents.db_id, '
            'pa_agents.id AS ag_id, pa_agents.role, pa_agents.ag_num,'
            'pa_statements.type, pa_statements.mk_hash, '
            'readonly.evidence_counts.ev_count '
            'FROM pa_agents, pa_statements, readonly.evidence_counts '
            'WHERE pa_agents.stmt_mk_hash = pa_statements.mk_hash '
            'AND pa_statements.mk_hash = readonly.evidence_counts.mk_hash'
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
    read_views[PaMeta.__tablename__] = PaMeta

    class TextMeta(Base, NamespaceLookup):
        __tablename__ = 'text_meta'
        __table_args__ = {'schema': 'readonly'}
        __dbname__ = 'TEXT'
        _indices = [StringIndex('text_meta_db_id_idx', 'db_id'),
                    StringIndex('text_meta_type_idx', 'type')]
        ag_id = Column(Integer, primary_key=True)
        ag_num = Column(Integer)
        db_id = Column(String)
        role = Column(String(20))
        type = Column(String(100))
        mk_hash = Column(BigInteger,
                         ForeignKey('readonly.fast_raw_pa_link.mk_hash'))
        raw_pa_link = relationship(FastRawPaLink)
        ev_count = Column(Integer)
    read_views[TextMeta.__tablename__] = TextMeta

    class NameMeta(Base, NamespaceLookup):
        __tablename__ = 'name_meta'
        __table_args__ = {'schema': 'readonly'}
        __dbname__ = 'NAME'
        _indices = [StringIndex('name_meta_db_id_idx', 'db_id'),
                    StringIndex('name_meta_type_idx', 'type')]
        ag_id = Column(Integer, primary_key=True)
        ag_num = Column(Integer)
        db_id = Column(String)
        role = Column(String(20))
        type = Column(String(100))
        mk_hash = Column(BigInteger,
                         ForeignKey('readonly.fast_raw_pa_link.mk_hash'))
        raw_pa_link = relationship(FastRawPaLink)
        ev_count = Column(Integer)
    read_views[NameMeta.__tablename__] = NameMeta

    class OtherMeta(Base, ReadonlyTable):
        __tablename__ = 'other_meta'
        __table_args__ = {'schema': 'readonly'}
        __definition__ = ("SELECT db_name, db_id, ag_id, role, ag_num, "
                          "type, mk_hash, ev_count FROM readonly.pa_meta "
                          "WHERE db_name NOT IN ('NAME', 'TEXT')")
        _indices = [StringIndex('other_meta_db_id_idx', 'db_id'),
                    StringIndex('other_meta_type_idx', 'type'),
                    StringIndex('other_meta_db_name_idx', 'db_name')]
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
    read_views[OtherMeta.__tablename__] = OtherMeta

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

    class PaSourceLookup(Base, SpecialColumnTable):
        __tablename__ = 'pa_source_lookup'
        __table_args__ = {'schema': 'readonly'}
        __definition_fmt__ = (
            'WITH jsonified AS (\n'
            '    SELECT mk_hash, \n'
            '           json_strip_nulls(json_build_object({all_sources})) \n'
            '           AS src_json \n'
            '    FROM readonly.pa_stmt_src\n'
            ')\n'
            'SELECT readonly.pa_stmt_src.*, \n'
            '       readonly.evidence_counts.ev_count, \n'
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
            'JOIN readonly.evidence_counts \n'
            '  ON diversity.mk_hash = readonly.evidence_counts.mk_hash'
        )
        _indices = [BtreeIndex('pa_source_lookup_mk_hash_idx', 'mk_hash'),
                    StringIndex('pa_source_lookup_only_src', 'only_src'),
                    BtreeIndex('pa_source_lookup_num_srcs', 'num_srcs')]
        loaded = False

        mk_hash = Column(BigInteger, primary_key=True)
        ev_count = Column(Integer)
        num_srcs = Column(Integer)
        src_json = Column(JSON)
        only_src = Column(String)
        has_rd = Column(Boolean)
        has_db = Column(Boolean)

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
    read_views[PaSourceLookup.__tablename__] = PaSourceLookup

    return read_views


SOURCE_GROUPS = {'databases': ['phosphosite', 'cbn', 'pc11', 'biopax',
                               'bel_lc', 'signor', 'biogrid', 'tas',
                               'lincs_drug', 'hprd', 'trrust'],
                 'reading': ['geneways', 'tees', 'isi', 'trips', 'rlimsp',
                             'medscan', 'sparser', 'reach']}