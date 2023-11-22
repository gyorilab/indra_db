__all__ = ['ReadonlySchema']

import logging
from collections import defaultdict

from sqlalchemy import Column, Integer, String, BigInteger, Boolean,\
    SmallInteger
from sqlalchemy.dialects.postgresql import BYTEA, JSON, JSONB, REAL

from indra.statements import get_all_descendants, Statement
from indra.sources import SOURCE_INFO
from indra.util.statement_presentation import internal_source_mappings

from .mixins import ReadonlyTable, NamespaceLookup, SpecialColumnTable, \
    IndraDBTable, IndraDBRefTable, Schema
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


class ReadonlySchema(Schema):
    """Schema for the Readonly database.

    We use a readonly database to allow fast and efficient load of data,
    and to add a layer of separation between the processes of updating
    the content of the database and accessing the content of the
    database. However, it is not practical to have the views created
    through sqlalchemy: instead they are generated and updated manually
    (or by other non-sqlalchemy scripts).

    Before building these tables, the :func:`belief <belief>` table must already
    have been loaded into the readonly database.

    The following views must be built in this specific order (temp):

      1. :func:`raw_stmt_src <raw_stmt_src>`
      2. :func:`fast_raw_pa_link <fast_raw_pa_link>`
      3. :func:`pa_agent_counts <pa_agent_counts>`
      4. (:func:`pa_stmt_src <pa_stmt_src>`)
      5. :func:`evidence_counts <evidence_counts>`
      6. :func:`reading_ref_link <reading_ref_link>`
      7. (:func:`pa_ref_link <pa_ref_link>`)
      8. (:func:`mesh_terms <mesh_terms>`)
      9. (:func:`mesh_concepts <mesh_concepts>`)
      10. (:func:`hash_pmid_counts <hash_pmid_counts>`)
      11. :func:`mesh_term_ref_counts <mesh_term_ref_counts>`
      12. :func:`mesh_concept_ref_counts <mesh_concept_ref_counts>`
      13. :func:`raw_stmt_mesh_terms <raw_stmt_mesh_terms>`
      14. :func:`raw_stmt_mesh_concepts <raw_stmt_mesh_concepts>`
      15. (:func:`pa_meta <pa_meta>`)
      16. :func:`source_meta <source_meta>`
      17. :func:`text_meta <text_meta>`
      18. :func:`name_meta <name_meta>`
      19. :func:`other_meta <other_meta>`
      20. :func:`mesh_term_meta <mesh_term_meta>`
      21. :func:`mesh_concept_meta <mesh_concept_meta>`
      22. :func:`agent_interaction <agent_interaction>`

    Note that the order of views below is determined not by the above
    order but by constraints imposed by use-case.

    **Meta Tables**

    Any table that has "meta" in the name is intended as a primary lookup table.
    This means it will have both the data indicated in the name of the table,
    such at (agent) "text", (agent) "name", or "source", but also a collection
    of columns with metadata essential for sorting and grouping of hashes:

    - Sorting:

      - **belief**
      - **ev_count**
      - **agent_count**

    - Grouping:

      - **type_num**
      - **activity**
      - **is_active**


    **Temporary Tables**

    There are some intermediate results that it is worthwhile to calculate and
    store for future table construction. Sometimes these were once permanent
    tables but are no longer used for their own sake, and it was simply simpler
    to delete them after their derivatives were completed. In other cases the
    temporary tables are more principled: created because many future tables
    draw on them and using a "with" clause for each one would be impractical.

    Whatever the reason, deleting the temporary tables greatly reduces the
    size of the readonly database. Such tables are marked in with "(temp)" at
    the beginning of their doc string.
    """

    def belief(self):
        """The belief of preassembled statements, keyed by hash.

        **Columns**

        - **mk_hash** ``bigint``
        - **belief** ``real``

        **Indices**

        - **mk_hash**
        """
        class Belief(self.base, IndraDBTable):
            __tablename__ = 'belief'
            __table_args__ = {'schema': 'readonly'}
            _indices = [BtreeIndex('belief_mk_hash_idx', 'mk_hash')]
            _temp = False
            mk_hash = Column(BigInteger, primary_key=True)
            belief = Column(REAL)
        return Belief

    def evidence_counts(self):
        """The evidence counts of pa statements, keyed by hash.

        **Columns**

        - **mk_hash** ``bigint``
        - **ev_count** ``integer``

        **Indices**

        - **mk_hash**
        """
        class EvidenceCounts(self.base, ReadonlyTable):
            __tablename__ = 'evidence_counts'
            __table_args__ = {'schema': 'readonly'}
            __definition__ = ('SELECT count(id) AS ev_count, mk_hash '
                              'FROM readonly.fast_raw_pa_link '
                              'GROUP BY mk_hash')
            _indices = [BtreeIndex('evidence_counts_mk_hash_idx', 'mk_hash')]
            mk_hash = Column(BigInteger, primary_key=True)
            ev_count = Column(Integer)
        return EvidenceCounts

    def reading_ref_link(self):
        """The source metadata for readings, keyed by reading ID.

        **Columns**

        - **trid** ``integer``
        - **pmid** ``varchar(20)``
        - **pmid_num** ``integer``
        - **pmcid** ``varchar(20)``
        - **pmcid_num** ``integer``
        - **pmcid_version** ``integer``
        - **doi** ``varchar(100)``
        - **doi_ns** ``integer``
        - **doi_id** ``varchar``
        - **pii** ``varchar(250)``
        - **url** ``varchar(250)``
        - **manuscript_id** ``varchar(100)``
        - **tcid** ``integer``
        - **source** ``varchar(250)``
        - **rid** ``integer``
        - **reader** ``varchar(20)``

        **Indices**

        - **rid**
        - **pmid**
        - **pmid_num**
        - **pmcid**
        - **pmcid_num**
        - **doi**
        - **doi_ns**
        - **doi_id**
        - **manuscript_id**
        - **tcid**
        - **trid**
        """
        class ReadingRefLink(self.base, ReadonlyTable, IndraDBRefTable):
            __tablename__ = 'reading_ref_link'
            __table_args__ = {'schema': 'readonly'}
            # Columns:
            # pmid, pmid_num, pmcid, pmcid_num, pmcid_version, doi, doi_ns, doi_id, tex_ref_id, pii, url, manuscript_id, text_content_id, source, reading_id, reader
            __definition__ = (
                'SELECT pmid, pmid_num, pmcid, pmcid_num, '
                'pmcid_version, doi, doi_ns, doi_id, tr.id AS trid,'
                'pii, url, manuscript_id, tc.id AS tcid, '
                'source, r.id AS rid, reader '
                'FROM text_ref AS tr JOIN text_content AS tc '
                'ON tr.id = tc.text_ref_id JOIN reading AS r '
                'ON tc.id = r.text_content_id'
            )
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
            rid = Column(BigInteger, primary_key=True)
            reader = Column(String(20))
        return ReadingRefLink

    def fast_raw_pa_link(self):
        """Join of PA JSONs and Raw JSONs for faster lookup.

        **Columns**

        - **id** ``integer``
        - **raw_json** ``bytea``
        - **reading_id** ``bigint``
        - **db_info_id** ``integer``
        - **mk_hash** ``bigint``
        - **pa_json** ``bytea``
        - **type_num** ``smallint``
        - **src** ``varchar``

        **Indices**

        - **mk_hash**
        - **reading_id**
        - **db_info_id**
        - **src**
        """
        class FastRawPaLink(self.base, ReadonlyTable):
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
            src = Column(String)

        return FastRawPaLink

    def pa_agent_counts(self):
        """The number of agents for each Statement, keyed by hash.

        **Columns**

        - **mk_hash** ``bigint``
        - **agent_count** ``integer``

        **Indices**

        - **mk_hash**
        """
        class PAAgentCounts(self.base, ReadonlyTable):
            __tablename__ = 'pa_agent_counts'
            __table_args__ = {'schema': 'readonly'}
            __definition__ = ("SELECT count(distinct ag_num) as agent_count,"
                              "       stmt_mk_hash as mk_hash\n"
                              "FROM pa_agents GROUP BY stmt_mk_hash")
            _indices = [BtreeIndex('pa_agent_counts_mk_hash_idx', 'mk_hash')]
            mk_hash = Column(BigInteger, primary_key=True)
            agent_count = Column(Integer)
        return PAAgentCounts

    def raw_stmt_src(self):
        """The source (e.g. reach, pc) of each raw statement, keyed by SID.

        **Columns**

        - **sid** ``integer``
        - **src** ``varchar``

        **Indices**

        - **sid**
        - **src**
        """
        class RawStmtSrc(self.base, ReadonlyTable):
            # columns:
            # raw_statement_id, source (reader/db_name)
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
        return RawStmtSrc

    def pa_stmt_src(self):
        """(temp) The number of evidence from each source for a PA Statement.

        This table is constructed by forming a column for every source short
        name present in the :func:`raw_stmt_src <raw_stmt_src>`.

        **Columns**

        - **mk_hash** ``bigint``
        - ...one column for each source... ``integer``

        **Indices**

        - **mk_hash**
        """
        class _PaStmtSrc(self.base, SpecialColumnTable):
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

        return _PaStmtSrc

    def pa_ref_link(self):
        """(temp) A quick-lookup from mk_hash to basic text ref data.

        **Columns**

        - **mk_hash** ``bigint``
        - **trid** ``integer``
        - **pmid_num** ``varchar``
        - **pmcid_num** ``varchar``
        - **source** ``varchar``
        - **reader** ``varchar``

        **Indices**

        - **mk_hash**
        - **trid**
        - **pmid_num**
        """
        class _PaRefLink(self.base, ReadonlyTable):
            __tablename__ = 'pa_ref_link'
            __table_args__ = {'schema': 'readonly'}
            __definition__ = ('SELECT mk_hash, trid, pmid_num, pmcid_num, \n'
                              '       source, reader\n'
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
        return _PaRefLink

    def mesh_terms(self):
        """(temp) All mesh annotations with D prefix, keyed by PMID int.

        **Columns**

        - **mesh_num** ``integer``
        - **pmid_num** ``integer``

        **Indices**

        - **pmid_num**
        """
        class _MeshTerms(self.base, ReadonlyTable):
            __tablename__ = 'mesh_terms'
            __table_args__ = {'schema': 'readonly'}
            __definition__ = ('SELECT pmid_num, mesh_num\n'
                              '  FROM mesh_ref_annotations\n'
                              '  WHERE is_concept IS NOT true\n'
                              'UNION\n'
                              'SELECT pmid_num, mesh_num\n'
                              '  FROM mti_ref_annotations_test\n'
                              '  WHERE NOT is_concept')
            _temp = True
            _indices = [BtreeIndex('mt_pmid_num_idx', 'pmid_num')]
            mesh_num = Column(Integer, primary_key=True)
            pmid_num = Column(Integer, primary_key=True)
        return _MeshTerms

    def mesh_concepts(self):
        """(temp) All mesh annotations with C prefix, keyed by PMID int.

        **Columns**

        - **mesh_num** ``integer``
        - **pmid_num** ``integer``

        **Indices**

        - **pmid_num**
        """
        class _MeshConcepts(self.base, ReadonlyTable):
            __tablename__ = 'mesh_concepts'
            __table_args__ = {'schema': 'readonly'}
            __definition__ = ('SELECT pmid_num, mesh_num\n'
                              '  FROM mesh_ref_annotations\n'
                              '  WHERE is_concept IS true\n'
                              'UNION\n'
                              'SELECT pmid_num, mesh_num\n'
                              '  FROM mti_ref_annotations_test\n'
                              '  WHERE is_concept IS true')
            _temp = True
            _indices = [BtreeIndex('mc_pmid_num_idx', 'pmid_num')]
            mesh_num = Column(Integer, primary_key=True)
            pmid_num = Column(Integer, primary_key=True)
        return _MeshConcepts

    def hash_pmid_counts(self):
        """(temp) The number of pmids for each PA Statement, keyed by hash.

        **Columns**

        - **mk_hash** ``bigint``
        - **pmid_count** ``integer``

        **Indices**

        - **mk_hash**
        """
        class _HashPmidCounts(self.base, ReadonlyTable):
            __tablename__ = 'hash_pmid_counts'
            __table_args__ = {'schema': 'readonly'}
            __definition__ = ('SELECT mk_hash,\n'
                              '       count(distinct pmid_num)::integer\n'
                              '       as pmid_count\n'
                              'FROM readonly.pa_ref_link GROUP BY mk_hash')
            _temp = True
            _indices = [BtreeIndex('hpc_mk_hash_idx', 'mk_hash')]
            mk_hash = Column(BigInteger, primary_key=True)
            pmid_count = Column(Integer)
        return _HashPmidCounts

    def mesh_term_ref_counts(self):
        """The D-type mesh IDs with pmid and ref counts, keyed by hash and mesh.

        **Columns**

        - **mk_hash** ``bigint``
        - **mesh_num** ``integer``
        - **ref_count** ``integer``
        - **pmid_count** ``integer``

        **Indices**

        - **mesh_num**
        - **mk_hash**
        """
        class MeshTermRefCounts(self.base, ReadonlyTable):
            __tablename__ = 'mesh_term_ref_counts'
            __table_args__ = {'schema': 'readonly'}
            __definition__ = ('WITH mesh_hash_pmids AS (\n'
                              '    SELECT readonly.mesh_terms.pmid_num,\n'
                              '           mk_hash, mesh_num\n'
                              '    FROM readonly.mesh_terms,\n'
                              '         readonly.pa_ref_link\n'
                              '    WHERE readonly.mesh_terms.pmid_num\n'
                              '        = readonly.pa_ref_link.pmid_num\n'
                              '), mesh_ref_counts_proto AS (\n'
                              '    SELECT mk_hash, mesh_num,\n'
                              '           COUNT(DISTINCT pmid_num)::integer\n'
                              '           AS ref_count\n'
                              '    FROM mesh_hash_pmids\n'
                              '    GROUP BY mk_hash, mesh_num\n'
                              ')\n'
                              'SELECT readonly.hash_pmid_counts.mk_hash,\n'
                              '       mesh_num, ref_count, pmid_count\n'
                              'FROM mesh_ref_counts_proto,\n'
                              '     readonly.hash_pmid_counts\n'
                              'WHERE mesh_ref_counts_proto.mk_hash\n'
                              '      = readonly.hash_pmid_counts.mk_hash')
            _indices = [BtreeIndex('mtrc_mesh_num_idx', 'mesh_num',
                                   cluster=True),
                        BtreeIndex('mtrc_mk_hash_idx', 'mk_hash')]
            mk_hash = Column(BigInteger, primary_key=True)
            mesh_num = Column(Integer, primary_key=True)
            ref_count = Column(Integer)
            pmid_count = Column(Integer)
        return MeshTermRefCounts

    def mesh_concept_ref_counts(self):
        """The C-type mesh IDs with pmid and ref counts, keyed by hash and mesh.

        **Columns**

        - **mk_hash** ``bigint``
        - **mesh_num** ``integer``
        - **ref_count** ``integer``
        - **pmid_count** ``integer``

        **Indices**

        - **mesh_num**
        - **mk_hash**
        """
        class MeshConceptRefCounts(self.base, ReadonlyTable):
            __tablename__ = 'mesh_concept_ref_counts'
            __table_args__ = {'schema': 'readonly'}
            __definition__ = (
                'WITH mesh_hash_pmids AS (\n'
                '    SELECT readonly.mesh_concepts.pmid_num, mk_hash,\n'
                '           mesh_num\n'
                '    FROM readonly.mesh_concepts, readonly.pa_ref_link\n'
                '    WHERE readonly.mesh_concepts.pmid_num '
                '      = readonly.pa_ref_link.pmid_num\n'
                '), mesh_ref_counts_proto AS (\n'
                '    SELECT mk_hash, mesh_num,\n'
                '           COUNT(DISTINCT pmid_num)::integer AS ref_count\n'
                '    FROM mesh_hash_pmids GROUP BY mk_hash, mesh_num\n'
                ')\n'
                'SELECT readonly.hash_pmid_counts.mk_hash, mesh_num,\n'
                '       ref_count, pmid_count\n'
                'FROM mesh_ref_counts_proto, readonly.hash_pmid_counts\n'
                'WHERE mesh_ref_counts_proto.mk_hash '
                '  = readonly.hash_pmid_counts.mk_hash'
            )
            _indices = [BtreeIndex('mcrc_mesh_num_idx', 'mesh_num'),
                        BtreeIndex('mcrc_mk_hash_idx', 'mk_hash')]
            mk_hash = Column(BigInteger, primary_key=True)
            mesh_num = Column(Integer, primary_key=True)
            ref_count = Column(Integer)
            pmid_count = Column(Integer)
        return MeshConceptRefCounts

    def raw_stmt_mesh_terms(self):
        """The D-type mesh number raw statement ID mapping.

        **Columns**

        - **sid** ``integer``
        - **mesh_num** ``integer``

        **Indices**

        - **sid**
        - **mesh_num**
        """
        class RawStmtMeshTerms(self.base, ReadonlyTable):
            __tablename__ = 'raw_stmt_mesh_terms'
            __table_args__ = {'schema': 'readonly'}
            # Columns:
            # raw_statement_id, mesh_num
            __definition__ = (
                'SELECT DISTINCT raw_statements.id as sid,\n'
                '       mesh_num\n'
                'FROM text_ref\n'
                '  JOIN readonly.mesh_terms AS mra\n'
                '    ON text_ref.pmid_num = mra.pmid_num\n'
                '  JOIN text_content ON text_ref.id = text_ref_id\n'
                '  JOIN reading\n'
                '    ON text_content.id = text_content_id\n'
                '  JOIN raw_statements ON reading.id = reading_id\n'
            )
            _indices = [BtreeIndex('rsmd_mesh_num_idx', 'mesh_num'),
                        BtreeIndex('rsmd_sid_idx', 'sid')]

            sid = Column(Integer, primary_key=True)
            mesh_num = Column(Integer, primary_key=True)
        return RawStmtMeshTerms

    def raw_stmt_mesh_concepts(self):
        """The C-type mesh number raw statement ID mapping.

        **Columns**

        - **sid** ``integer``
        - **mesh_num** ``integer``

        **Indices**

        - **sid**
        - **mesh_num**
        """
        class RawStmtMeshConcepts(self.base, ReadonlyTable):
            __tablename__ = 'raw_stmt_mesh_concepts'
            __table_args__ = {'schema': 'readonly'}
            # raw_statement_id, mesh_num
            __definition__ = ('SELECT DISTINCT raw_statements.id as sid,\n'
                              '       mesh_num\n'
                              'FROM text_ref\n'
                              '  JOIN readonly.mesh_concepts AS mra\n'
                              '    ON text_ref.pmid_num = mra.pmid_num\n'
                              '  JOIN text_content\n'
                              '    ON text_ref.id = text_ref_id\n'
                              '  JOIN reading\n'
                              '    ON text_content.id = text_content_id\n'
                              '  JOIN raw_statements\n'
                              '    ON reading.id = reading_id\n')
            _indices = [BtreeIndex('rsmc_mesh_num_idx', 'mesh_num'),
                        BtreeIndex('rsmc_sid_idx', 'sid')]

            sid = Column(Integer, primary_key=True)
            mesh_num = Column(Integer, primary_key=True)
        return RawStmtMeshConcepts

    def pa_meta(self):
        """(temp) The metadata most valuable for querying PA Statements.

        This table is used to generate the more scope-limited
        :func:`name_meta <name_meta>`, :func:`text_meta <text_meta>`, and
        :func:`other_meta <other_meta>`. The reason is that NAME and TEXT (in
        particular) agent groundings are vastly overrepresented.

        **Columns**

        - **ag_id** ``integer``
        - **ag_num** ``integer``
        - **db_name** ``varchar``
        - **db_id** ``varchar``
        - **role_num** ``smallint``
        - **type_num** ``smallint``
        - **mk_hash** ``bigint``
        - **ev_count** ``integer``
        - **belief** ``real``
        - **activity** ``varchar``
        - **is_active** ``boolean``
        - **agent_count** ``integer``
        - **is_complex_dup** ``boolean``

        **Indices**

        - **db_name**
        - **mk_hash**
        """
        class _PaMeta(self.base, ReadonlyTable):
            # Depends on:
            #  - PaActivity (from principal schema)
            #  - PaAgents (from principal schema)
            #  - PAStatements (from principal schema)
            #  - Belief (from readonly schema)
            #  - EvidenceCounts (from readonly schema)
            #  - PaAgentCounts (from readonly schema)
            #  - TypeMap (from ???)
            #  - RoleMap (from ???)
            # To get the needed info from the principal schema, select from
            # the tables on the principal schema, joined by mk_hash and then
            # read in belief, agent count and evidence count from the
            # belief dump, unique statement file and source counts file.
            __tablename__ = 'pa_meta'
            __table_args__ = {'schema': 'readonly'}
            __definition__ = (
                'SELECT pa_agents.db_name, pa_agents.db_id,\n'
                '       pa_agents.id AS ag_id, role_num, pa_agents.ag_num,\n'
                '       type_num, pa_statements.mk_hash,\n'
                '       readonly.evidence_counts.ev_count,\n'
                '       readonly.belief.belief, activity, is_active,\n'
                '       agent_count, false AS is_complex_dup\n'
                'FROM pa_agents, pa_statements, readonly.pa_agent_counts,\n'
                '  type_map, role_map, readonly.belief,\n'
                '  readonly.evidence_counts LEFT JOIN pa_activity\n'
                '  ON readonly.evidence_counts.mk_hash\n'
                '    = pa_activity.stmt_mk_hash\n'
                'WHERE pa_agents.stmt_mk_hash = pa_statements.mk_hash\n'
                '  AND pa_statements.mk_hash\n'
                '    = readonly.evidence_counts.mk_hash\n'
                '  AND pa_statements.mk_hash = readonly.belief.mk_hash\n'
                '  AND readonly.pa_agent_counts.mk_hash\n'
                '    = pa_agents.stmt_mk_hash\n'
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

        return _PaMeta

    def source_meta(self):
        """All the source-related metadata condensed using JSONB, keyed by hash.

        **Columns**

        - **mk_hash** ``bigint``
        - **ev_count** ``integer``
        - **belief** ``real``
        - **num_srcs** ``integer``
        - **src_json** ``json``
        - **only_src** ``varchar``
        - **has_rd** ``boolean``
        - **has_db** ``boolean``
        - **type_num** ``smallint``
        - **activity** ``varchar``
        - **is_active** ``boolean``
        - **agent_count** ``integer``

        **Indices**

        - **mk_hash**
        - **only_src**
        - **activity**
        - **type_num**
        - **num_srcs**
        """
        class SourceMeta(self.base, SpecialColumnTable):
            __tablename__ = 'source_meta'
            __table_args__ = {'schema': 'readonly'}
            # Columns:
            # mk_hash, ev_count, belief, type_num, activity, is_active,
            # agent_count, num_srcs, src_json, only_src, has_rd, has_db
            __definition_fmt__ = (
                'WITH jsonified AS (\n'
                '    SELECT mk_hash, \n'
                '          json_strip_nulls(json_build_object({all_sources}))\n'
                '          AS src_json \n'
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
                '  FROM jsonified, json_object_keys(jsonified.src_json)'
                '    as src \n'
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
                                       for src in SOURCE_GROUPS['reader'])
                db_sources = ', '.join(repr(src)
                                       for src in SOURCE_GROUPS['database'])
                sql = cls.__definition_fmt__.format(all_sources=all_sources,
                                                    reading_sources=rd_sources,
                                                    db_sources=db_sources)
                return sql
        return SourceMeta

    def text_meta(self):
        """The metadata most valuable for querying PA Statements by agent TEXT.

        This table is generated from :func:`pa_meta <pa_meta>`, because TEXT
        is extremely overrepresented among agent groundings. Removing these and
        NAMEs from the "OTHER" efficiently narrows the search very rapidly, and
        for the larger sets of NAME and TEXT removes an index-search.

        **Columns**

        - **ag_id** ``integer``
        - **ag_num** ``integer``
        - **db_id** ``varchar``
        - **role_num** ``smallint``
        - **type_num** ``smallint``
        - **mk_hash** ``bigint``
        - **ev_count** ``integer``
        - **belief** ``real``
        - **activity** ``varchar``
        - **is_active** ``boolean``
        - **agent_count** ``integer``
        - **is_complex_dup** ``boolean``

        **Indices**

        - **mk_hash**
        - **db_id**
        - **type_num**
        - **activity**
        """
        class TextMeta(self.base, NamespaceLookup):
            # Depends on pa_meta
            # Selects only pa_meta.db_name = 'TEXT'
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
        return TextMeta

    def name_meta(self):
        """The metadata most valuable for querying PA Statements by agent NAME.

        This table is generated from :func:`pa_meta <pa_meta>`, because NAME
        is overrepresented among agent groundings. Removing these and NAMEs from
        the "OTHER" efficiently narrows the search very rapidly, and for the
        larger sets of NAME and TEXT removes an index-search.

        **Columns**

        - **ag_id** ``integer``
        - **ag_num** ``integer``
        - **db_id** ``varchar``
        - **role_num** ``smallint``
        - **type_num** ``smallint``
        - **mk_hash** ``bigint``
        - **ev_count** ``integer``
        - **belief** ``real``
        - **activity** ``varchar``
        - **is_active** ``boolean``
        - **agent_count** ``integer``
        - **is_complex_dup** ``boolean``

        **Indices**

        - **mk_hash**
        - **db_id**
        - **type_num**
        - **activity**
        """
        class NameMeta(self.base, NamespaceLookup):
            # Depends on pa_meta
            # Selects only pa_meta.db_name = 'NAME'
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
        return NameMeta

    def other_meta(self):
        """The metadata most valuable for querying PA Statements.

        This table is a copy of :func:`pa_meta <pa_meta>` with rows with agent
        groundings besides NAME and TEXT removed.

        **Columns**

        - **ag_id** ``integer``
        - **ag_num** ``integer``
        - **db_name** ``varchar``
        - **db_id** ``varchar``
        - **role_num** ``smallint``
        - **type_num** ``smallint``
        - **mk_hash** ``bigint``
        - **ev_count** ``integer``
        - **belief** ``real``
        - **activity** ``varchar``
        - **is_active** ``boolean``
        - **agent_count** ``integer``
        - **is_complex_dup** ``boolean``

        **Indices**

        - **mk_hash**
        - **db_name**
        - **db_id**
        - **type_num**
        - **activity**
        """
        class OtherMeta(self.base, ReadonlyTable):
            __tablename__ = 'other_meta'
            __table_args__ = {'schema': 'readonly'}
            __definition__ = ("SELECT db_name, db_id, ag_id, role_num,\n"
                              "       ag_num, type_num, mk_hash, ev_count,\n"
                              "       belief, activity, is_active,\n"
                              "       agent_count, is_complex_dup\n"
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
        return OtherMeta

    def mesh_term_meta(self):
        """A lookup for hashes by D-type mesh IDs.

        **Columns**

        - **mk_hash** ``bigint``
        - **mesh_num** ``integer``
        - **tr_count** ``integer``
        - **ev_count** ``integer``
        - **belief** ``real``
        - **type_num** ``smallint``
        - **activity** ``varchar``
        - **is_active** ``boolean``
        - **agent_count** ``integer``

        **Indices**

        - **mk_hash**
        - **type_num**
        - **activity**
        """
        class MeshTermMeta(self.base, ReadonlyTable):
            __tablename__ = 'mesh_term_meta'
            __table_args__ = {'schema': 'readonly'}
            # Column order:
            # mk_hash, ev_count, belief, mesh_num, type_num, activity, is_active, agent_count
            __definition__ = ("SELECT DISTINCT meta.mk_hash, meta.ev_count,\n"
                              "       meta.belief, mesh_num, type_num,\n"
                              "       activity, is_active, agent_count\n"
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
        return MeshTermMeta

    def mesh_concept_meta(self):
        """A lookup for hashes by C-type mesh IDs.

        **Columns**

        - **mk_hash** ``bigint``
        - **mesh_num** ``integer``
        - **tr_count** ``integer``
        - **ev_count** ``integer``
        - **belief** ``real``
        - **type_num** ``smallint``
        - **activity** ``varchar``
        - **is_active** ``boolean``
        - **agent_count** ``integer``

        **Indices**

        - **mk_hash**
        - **type_num**
        - **activity**
        """
        class MeshConceptMeta(self.base, ReadonlyTable):
            __tablename__ = 'mesh_concept_meta'
            __table_args__ = {'schema': 'readonly'}
            # Columns:
            # mk_hash, tr_count, ev_count, belief, mesh_num, type_num, activity, is_active, agent_count
            __definition__ = ("SELECT DISTINCT meta.mk_hash, meta.ev_count,\n"
                              "       meta.belief, mesh_num, type_num,\n"
                              "       activity, is_active, agent_count\n"
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
        return MeshConceptMeta

    def agent_interactions(self):
        """Agent and type data in simple JSONs for rapid lookup, keyed by hash.

        This table is used for retrieving interactions, agent pairs, and
        relations (any kind of return that is more generic than full
        Statements).

        **Columns**

        - **mk_hash** ``bigint``
        - **ev_count** ``integer``
        - **belief** ``real``
        - **type_num** ``smallint``
        - **activity** ``varchar``
        - **is_active** ``boolean``
        - **agent_count** ``integer``
        - **agent_json** ``jsonb``
        - **src_json** ``jsonb``
        - **is_complex_dup** ``boolean``

        **Indices**

        - **mk_hash**
        - **agent_json**
        - **type_num**
        """
        class AgentInteractions(self.base, ReadonlyTable):
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
                              "      readonly.name_meta.mk_hash AS mk_hash,\n"
                              "      readonly.name_meta.db_id AS db_id,\n"
                              "      readonly.name_meta.ag_num AS ag_num,\n"
                              "      readonly.name_meta.type_num AS type_num,\n"
                              "      readonly.name_meta.agent_count\n"
                              "        AS agent_count,\n"
                              "      readonly.name_meta.ev_count AS ev_count,\n"
                              "      readonly.name_meta.belief AS belief,\n"
                              "      readonly.name_meta.activity AS activity,\n"
                              "      readonly.name_meta.is_active\n"
                              "        AS is_active,\n"
                              "      readonly.source_meta.src_json\n"
                              "        AS src_json\n"
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
                        BtreeIndex('agent_interactions_agent_json_idx',
                                   'agent_json'),
                        BtreeIndex('agent_interactions_type_num_idx',
                                   'type_num')]
            _always_disp = ['mk_hash', 'agent_json']

            @classmethod
            def create(cls, db, commit=True):
                super(AgentInteractions, cls).create(db, commit)
                from itertools import permutations
                # Select all rows in the table that are complexes
                interactions = db.select_all(
                    db.AgentInteractions,
                    db.AgentInteractions.type_num
                    == ro_type_map.get_int('Complex')
                )
                new_interactions = []

                # For each row, create a new row for each pair of agents, if
                # the interaction is not a self-interaction (i.e., if there
                # are more than one agent in the interaction)
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
                        ('mk_hash', 'ev_count', 'belief', 'type_num',
                         'agent_count', 'agent_json', 'src_json',
                         'is_complex_dup'))
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
        return AgentInteractions



def _get_source_groups():
    # Initially a set because we get duplicates due to mappings
    source_groups = defaultdict(set)
    for source_key, info in SOURCE_INFO.items():
        mapped_source = internal_source_mappings.get(source_key, source_key)
        source_groups[info['type']].add(mapped_source)
    return {k: list(v) for k, v in source_groups.items()}

SOURCE_GROUPS = _get_source_groups()
"""The source short-names grouped by "database" or "reader".

It is worth noting that "database" and "knowledge base" are not equivalent here.
There are several entries in the "reader" category that we gather from other
knowledge bases, but are based on machine readers.

This is used in the formation of the sources table, as well as in the display
of content retrieved from the readonly database.
"""
