"""
The Principal Schema
====================

The Principal database is the core representation of our data, the ultimate
authority on what we know. It is heavily optimized for the *input* and
maintenance of our data.
"""

__all__ = ['PrincipalSchema', 'foreign_key_map']

import logging

from sqlalchemy import Column, Integer, String, UniqueConstraint, ForeignKey, \
     Boolean, DateTime, func, BigInteger, or_, tuple_
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import BYTEA, INET, JSONB

from indra_db.schemas.mixins import IndraDBTable, IndraDBRefTable, Schema
from indra_db.schemas.indexes import StringIndex, BtreeIndex

foreign_key_map = [
    ('pa_agents', 'pa_statements'),
    ('raw_unique_links', 'pa_statements'),
    ('raw_unique_links', 'raw_statements'),
    ('raw_statements', 'reading'),
    ('raw_agents', 'raw_statements'),
    ('raw_statements', 'db_info'),
    ('reading', 'text_content'),
    ('text_content', 'text_ref')
]

logger = logging.getLogger(__name__)


class PrincipalSchema(Schema):
    """The Principal schema class organizes the table constructors.

    The tables can be divided into various groups, with a clear order of
    creation for many of them.

    **Core Tables**

    First are the core tables representing our knowledge:

      1. :func:`text_ref <text_ref>`
      2. :func:`text_content <text_content>`
      3. :func:`reading <reading>`
      4. :func:`db_info <db_info>`
      5. :func:`raw_statements <raw_statements>`
      6. :func:`raw_unique_links <raw_unique_links>`
      7. :func:`pa_statements <pa_statements>`
      8. :func:`pa_support_links <pa_support_links>`

    **Statement Attribute Tables**

    Then there are the tables that represent attributes of statements. The set
    of tables is identical for the raw statements:

      - :func:`raw_activity <raw_activity>`
      - :func:`raw_agents <raw_agents>`
      - :func:`raw_muts <raw_muts>`
      - :func:`raw_mods <raw_mods>`

    and the preassembled statements:

      - :func:`pa_activity <pa_activity>`
      - :func:`pa_agents <pa_agents>`
      - :func:`pa_muts <pa_muts>`
      - :func:`pa_mods <pa_mods>`

    **Curation Table**

    This table is where we record the curations submitted by ourselves and our
    users, which we use to improve our results.

      - :func:`curations <curations>`

    **Ancillary Tables**

    We also have several tables that we use to keep track of processing
    metadata, and some artifacts useful in that processing.

      - :func:`updates <updates>`
      - :func:`source_file <source_file>`
      - :func:`reading_updates <reading_updates>`
      - :func:`preassembly_updates <preassembly_updates>`
      - :func:`xdd_updates <xdd_updates>`
      - :func:`rejected_statements <rejected_statements>`
      - :func:`discarded_statements <discarded_statements>`
    """

    def text_ref(self):
        """Represent a piece of text, as per its identifiers.

        Each piece of text will be made available in different forms through
        different services, most commonly abstracts through pubmed and full text
        through pubmed central. However they are from the same *paper*, which
        has various different identifiers, such as pmids, pmcids, and dois.

        We do our best to merge the different identifiers and for the most part
        each paper has exactly one text ref. Where that is not the case it is
        mostly impossible to automatically reconcile the different identifiers
        (this often has to do with inconsistent versioning of a paper and mixups
        over what is IDed).

        **Size**: medium

        **Basic Columns**

        These are the core columns representing the different IDs we use to
        represent a paper.

        - **id** [``integer PRIMARY KEY``]: The primary key of the TextRef
          entry. Elsewhere this is often referred to as a "text ref ID" or
          "trid" for short.
        - **pmid** [``varchar(20)``]: The identifier from pubmed.
        - **pmcid** [``varchar(20)``]: The identifier from PubMed Central (e.g.
          "PMC12345")
        - **doi** [``varchar(100)``]: The ideally universal identifier.
        - **pii** [``varchar(250)``]: The identifier used by Springer.
        - **url** [``varchar UNIQUE``]: For sources found exclusively online
          (e.g. wikipedia) use their URL.
        - **manuscript_id** [``varchar(100) UNIQUE``]: The ID assigned documents
          given to PMC author manuscripts.

        **Metadata Columns**

        In addition we also track some basic metadata about the entry and
        updates to the data in the table.

        - create_date [``timestamp without time zone``]: The date the record was
          added.
        - last_updated [``timestamp without time zone``]: The most recent time
          the record was edited.

        **Constraints**

        Postgres is extremely efficient at detecting conflicts, and we use this
        to help ensure our entries do not have any duplicates.

        - **pmid-doi**: ``UNIQUE(pmid, doi)``
        - **pmid-pmcid**: ``UNIQUE(pmid, pmcid)``
        - **pmcid-doi**: ``UNIQUE(pmcid, doi)``

        **Lookup Columns**

        Some columns are hard to look up when they are in their native string
        format, so they are processed and broken down into integer parts, as
        far as possible.

        - **pmid_num** [``integer``]: the int-ified pmid, faster for lookup.
        - **pmcid_num** [``integer``]: the int-portion of the PMCID, so
          "PMC12345" would here be 12345.
        - **pmcid_version** [``integer``]: although rarely used, occasionally a
          PMC ID will have a version, indicated by a dot, e.g. PMC12345.3, in
          which case the "3" would be stored in this column.
        - **doi_ns** [``integer``]: The DOI system works by assigning
          organizations (such as a journal) namespace IDs, and that organization
          is then responsible for maintaining a unique ID system internally.
          These namespaces are always numbers, and are stored here as such.
        - **doi_id** [``varchar``]: The custom ID given by the publishing
          organization.
        """
        class TextRef(self.base, IndraDBTable, IndraDBRefTable):
            """The class representing the TextRef table.

            The preferred constructor for this class is the ``new`` classmethod.
            This is because many of the IDs are processed in addition to the
            standard SQLAlchemy handling.

            To modify a column value, please use the ``update`` method, again to
            ensure that IDs are properly processed.

            The best way to construct a clause with a :py:class:`TextRef` is to
            use the ``has_ref`` method, inherited from
            :py:class:`IndraDBRefTable`, or else one of ``pmid_in``,
            ``pmcid_in``, or ``doi_in`` for those specific IDs.

            A dictionary of references can be easily generated using the
            ``get_ref_dict`` method.
            """
            __tablename__ = 'text_ref'
            _ref_cols = ['pmid', 'pmcid', 'doi', 'pii', 'url', 'manuscript_id']
            _always_disp = ['id', 'pmid', 'pmcid']
            _indices = [StringIndex('text_ref_pmid_idx', 'pmid'),
                        StringIndex('text_ref_pmcid_idx', 'pmcid'),
                        BtreeIndex('text_ref_pmid_num_idx', 'pmid_num'),
                        BtreeIndex('text_ref_pmcid_num_idx', 'pmcid_num'),
                        BtreeIndex('text_ref_doi_ns_idx', 'doi_ns'),
                        BtreeIndex('text_ref_doi_id_idx', 'doi_id'),
                        StringIndex('text_ref_doi_idx', 'doi')]

            id = Column(Integer, primary_key=True)
            pmid = Column(String(20))
            pmid_num = Column(Integer)
            pmcid = Column(String(20))
            pmcid_num = Column(Integer)
            pmcid_version = Column(Integer)
            doi = Column(String(100))
            doi_ns = Column(Integer)
            doi_id = Column(String)
            pii = Column(String(250))
            url = Column(String, unique=True)
            manuscript_id = Column(String(100), unique=True)
            create_date = Column(DateTime, default=func.now())
            last_updated = Column(DateTime, onupdate=func.now())

            __table_args__ = (
                UniqueConstraint('pmid', 'doi', name='pmid-doi'),
                UniqueConstraint('pmid', 'pmcid', name='pmid-pmcid'),
                UniqueConstraint('pmcid', 'doi', name='pmcid-doi')
            )

            def __repr__(self):
                terms = [f'id={self.id}']
                for col in ['pmid', 'pmcid', 'doi', 'pii', 'url', 'manuscript_id']:
                    if getattr(self, col) is not None:
                        terms.append(f'{col}={getattr(self, col)}')
                    if len(terms) > 2:
                        break
                return f'{self.__class__.__name__}({", ".join(terms)})'

            @classmethod
            def new(cls, pmid=None, pmcid=None, doi=None, pii=None, url=None,
                    manuscript_id=None):
                """The preferred TextRef constructor: create a new TextRef."""
                pmid, pmid_num = cls.process_pmid(pmid)
                pmcid, pmcid_num, pmcid_version = cls.process_pmcid(pmcid)
                doi, doi_ns, doi_id = cls.process_doi(doi)
                return cls(pmid=pmid, pmid_num=pmid_num, pmcid=pmcid,
                           pmcid_num=pmcid_num, pmcid_version=pmcid_version,
                           doi=doi, doi_ns=doi_ns, doi_id=doi_id, pii=pii,
                           url=url, manuscript_id=manuscript_id)

            def update(self, **updates):
                """Update the value of an ID, processing it as necessary."""
                for id_type, id_val in updates.items():
                    if not hasattr(self, id_type):
                        raise ValueError(f"Invalid ID type: {id_type}")
                    if id_type == 'pmid':
                        self.pmid, self.pmid_num = self.process_pmid(id_val)
                    elif id_type == 'pmcid':
                        self.pmcid, self.pmcid_num, self.pmcid_version = \
                            self.process_pmcid(id_val)
                    elif id_type == 'doi':
                        self.doi, self.doi_ns, self.doi_id = \
                            self.process_doi(id_val)
                    else:
                        setattr(self, id_type, id_val)
                return
        return TextRef

    def mesh_ref_annotations(self):
        """Represent the MeSH annotations of papers provided by PubMed.

        Each abstract/entry in PubMed is accompanied by human-curated
        MeSH IDs indicating the topics of the paper. Each paper will have many
        IDs in general, so a separate table is used, liked to the
        :func:`text_ref <text_ref>` table by an un-constrained PMID. This make
        insertion of the data easier because the custom TRIDs need not be
        retrieved to dump the mesh refs.

        **Size**: large

        **Columns**

        - **id** [``integer PRIMARY KEY``]: The primary database-assigned ID of
          the row.
        - **pmid_num** [``integer NOT NULL``]: The int-ified pmid that is used
          to link entries in this table with those in the
          :func:`text_ref <text_ref>` table.
        - **mesh_num** [```integer NOT NULL``]: The intified MeSH ID (with the
          prefix removed). The **is_concept** column indicates whether the
          prefix was D (False) or C (True).
        - **qual_num** [``integer``]: The qualifier number that is sometimes
          included with the annotation (Prefix Q).
        - **major_topic** [``boolean DEFAULT false``]: The major topic flag
          indicates whether the ID describes a primary purpose of the paper.
        - **is_concept** [``boolean DEFAUL false``]: Indicate whether the prefix
          was C (true) or D (false).

        **Constraints**

        Postgres is extremely efficient at detecting conflicts, and we use this
        to help ensure our entries do not have any duplicates.

        - **mesh-uniqueness**: ``UNIQUE(pmid_num, mesh_num, qual_num,
          is_concept)``
        """
        class MeshRefAnnotations(self.base, IndraDBTable):
            __tablename__ = 'mesh_ref_annotations'
            _always_disp = ['pmid_num', 'mesh_num', 'qual_num']
            _indices = [BtreeIndex('mesh_ref_annotations_pmid_idx', 'pmid_num'),
                        BtreeIndex('mesh_ref_annotations_mesh_id_idx', 'mesh_num'),
                        BtreeIndex('mesh_ref_annotations_qual_id_idx', 'qual_num')]
            id = Column(Integer, primary_key=True)
            pmid_num = Column(Integer, nullable=False)
            mesh_num = Column(Integer, nullable=False)
            qual_num = Column(Integer)
            major_topic = Column(Boolean, default=False)
            is_concept = Column(Boolean, default=False)
            __table_args__ = (
                UniqueConstraint('pmid_num', 'mesh_num', 'qual_num', 'is_concept',
                                 name='mesh-uniqueness'),
            )
        return MeshRefAnnotations

    def mti_ref_annotaions_test(self):
        """Represent the MeSH annotations of abstracts as inferred by MTI.

        MTI is a machine learned model that attempts to predict MeSH annotations
        on new un-annotated abstracts after training on the existing
        annotations.

        **Size**: medium

        **Columns**

        - **id** [``integer PRIMARY KEY``]: The primary database-assigned ID of
          the row.
        - **pmid_num** [``integer NOT NULL``]: The int-ified pmid that is used
          to link entries in this table with those in the
          :func:`text_ref <text_ref>` table.
        - **mesh_num** [```integer NOT NULL``]: The intified MeSH ID (with the
          prefix removed). The **is_concept** column indicates whether the
          prefix was D (False) or C (True).
        - **qual_num** [``integer``]: The qualifier number that is sometimes
          included with the annotation (Prefix Q).
        - **major_topic** [``boolean DEFAULT false``]: The major topic flag
          indicates whether the ID describes a primary purpose of the paper.
        - **is_concept** [``boolean DEFAUL false``]: Indicate whether the prefix
          was C (true) or D (false).

        **Constraints**

        Postgres is extremely efficient at detecting conflicts, and we use this
        to help ensure our entries do not have any duplicates.

        - **mesh-uniqueness**: ``UNIQUE(pmid_num, mesh_num, qual_num,
          is_concept)``
        """
        class MtiRefAnnotationsTest(self.base, IndraDBTable):
            __tablename__ = 'mti_ref_annotations_test'
            _always_disp = ['pmid_num', 'mesh_num', 'qual_num']
            _indices = [BtreeIndex('mti_ref_annotations_test_pmid_idx', 'pmid_num'),
                        BtreeIndex('mti_ref_annotations_test_mesh_id_idx', 'mesh_num'),
                        BtreeIndex('mti_ref_annotations_test_qual_id_idx', 'qual_num')]
            id = Column(Integer, primary_key=True)
            pmid_num = Column(Integer, nullable=False)
            mesh_num = Column(Integer, nullable=False)
            qual_num = Column(Integer)
            major_topic = Column(Boolean, default=False)
            is_concept = Column(Boolean, default=False)
            __table_args__ = (
                UniqueConstraint('pmid_num', 'mesh_num', 'qual_num', 'is_concept',
                                    name='mti-uniqueness'),
            )
        return MtiRefAnnotationsTest

    def text_content(self):
        """Represent the content of a text retrieved from a particular source.

        For each paper as a logical entity, there are many places where you can
        acquire the actual article or parts of it. For example you can get an
        abstract from PubMed for most content, and for a minority subset you can
        get full text from PubMed Central, either their Open-Access corpus or
        their author's Manuscripts.

        Both the text itself and the metadata for the source of the text are
        represented in this table.

        **Size**: large

        **Basic Columns**

        - **id** [``integer PRIMARY KEY``]: The auto-generated primary key of
          the table. These are elsewhere called Text Content IDs, or TCIDs.
        - **text_ref_id** [``integer NOT NULL``]: A foreign-key constrained
          reference to the appropriate entry in the :func:`text_ref <text_ref>`
          table.
        - **source** [``varchar(250) NOT NULL``]: The name of the source, e.g.
          "pubmed" or "pmc_oa". The list of content names can be found in the
          class attributes in content managers.
        - **format** [``varchar(250) NOT NULL``]: The file format of the
          content, e.g. "XML" or "TEXT".
        - **text_type** [``varchar(250) NOT NULL``]: The type of the text, e.g.
          "abstract" of "fulltext".
        - **preprint** [``boolean``]: Indicate whether the content is from
          a preprint.
        - **content** [``bytea``]: The raw compressed bytes of the content.

        **Metadata Columns**

        - **insert_data** [``timestamp without time zone``]: The date the record
          was added.
        - **last_updated** [``timestamp without time zone``]: The most recent
          time the record was edited.

        **Constraints**

        Postgres is extremely efficient at detecting conflicts, and we use this
        to help ensure our entries do not have any duplicates.

        - **content-uniqueness**: ``UNIQUE(text_ref_id, source, format,
          text_type)``
        """
        class TextContent(self.base, IndraDBTable):
            __tablename__ = 'text_content'
            _skip_disp = ['content']
            _always_disp = ['id', 'text_ref_id', 'source', 'format', 'text_type']
            id = Column(Integer, primary_key=True)
            text_ref_id = Column(Integer, ForeignKey('text_ref.id'),
                                 nullable=False)
            text_ref = relationship(self.table_dict['text_ref'])
            source = Column(String(250), nullable=False)
            format = Column(String(250), nullable=False)
            text_type = Column(String(250), nullable=False)
            content = Column(BYTEA)
            insert_date = Column(DateTime, default=func.now())
            last_updated = Column(DateTime, onupdate=func.now())
            preprint = Column(Boolean)
            __table_args__ = (
                UniqueConstraint('text_ref_id', 'source', 'format',
                                 'text_type', name='content-uniqueness'),
            )
        return TextContent

    def reading(self):
        """Represent a reading of a piece of text.

        We have multiple readers and of course many thousands of pieces of text
        content. Each entry in this table applies to a given reader applied to
        a given pieces of content.

        As such, the primary ID is a hash constructed from the text content ID
        prepended with integers that are assigned to each reader-reader version
        pair. The function
        :func:`generate_reading_id <indra_db.reading.read_db.generate_reading_id>`
        implements the particular process used. The reader numbers are assigned
        in the :data:`readers <indra_db.databases.readers>` global, and the
        reader version number is the index of the version listed for the given
        reader in the :data:`reader_versions <indra_db.databases.reader_versions>`
        dictionary in the same module.

        **Size**: very large

        **Basic Columns**

        - **id** [``bigint PRIMARY KEY``]: A hash ID constructed from a reader
          number, reader version number, and the text content ID of the content
          that was read.
        - **text_content_id** [``integer NOT NULL``]: A foreign-key constrained
          reference to the appropriate entry in the :func:`text_content
          <text_content>` table.
        - **batch_id** [``integer NOT NULL``]: A simple random integer (not
          unique) that is assigned each batch of inserted readings. It is used
          in the moments after the insert to easily retrieve the content that
          was just added, potentially plus some extra.
        - **reader** [``varchar(20) NOT NULL``]: The name of the reader, e.g.
          "REACH" or "SPARSER".
        - **reader_version** [``varchar(20) NOT NULL``]: The version of the
          reader, which may be any arbitrary string in principle. This allows
          each reader to define its own versioning scheme.
        - **format** [``varchar(20) NOT NULL``]: The file format of the
          reading result, e.g. "XML" or "JSON".
        - **bytes** [``bytea``]: The raw compressed bytes of the reading result.

        **Metadata Columns**

        - **create_date** [``timestamp without time zone``]: The date the record
          was added.
        - **last_updated** [``timestamp without time zone``]: The most recent
          time the record was edited.

        **Constraints**

        Postgres is extremely efficient at detecting conflicts, and we use this
        to help ensure our entries do not have any duplicates.

        - **reading-uniqeness**: ``UNIQUE(text_content_id, reader,
          reader_version)``
        """
        class Reading(self.base, IndraDBTable):
            __tablename__ = 'reading'
            _skip_disp = ['bytes']
            _always_disp = ['id', 'text_content_id', 'reader', 'reader_version']
            id = Column(BigInteger, primary_key=True, default=None)
            text_content_id = Column(Integer,
                                     ForeignKey('text_content.id'),
                                     nullable=False)
            batch_id = Column(Integer, nullable=False)
            text_content = relationship(self.table_dict['text_content'])
            reader = Column(String(20), nullable=False)
            reader_version = Column(String(20), nullable=False)
            format = Column(String(20), nullable=False)  # xml, json, etc.
            bytes = Column(BYTEA)
            create_date = Column(DateTime, default=func.now())
            last_updated = Column(DateTime, onupdate=func.now())
            __table_args__ = (
                UniqueConstraint('text_content_id', 'reader', 'reader_version',
                                 name='reading-uniqueness'),
            )
        return Reading

    def db_info(self):
        """Represent the provenance and metadata for an external knowledge base.

        INDRA DB takes content not just from our own readings but also merges
        that with many pre-existing knowledge bases, many of them human
        curated. These knowledge bases are defined and managed by classes
        contained in :mod:`knowledgebase_manager
        <indra_db.managers.knowledgebase_manager>`.

        No real data is contained in this column, simply records of which
        knowledge bases have been added, updated, and when.

        **Size**: very small

        **Basic Columns**

        - **id** [``integer PRIMARY KEY``]: A database-assigned integer unique
          ID for each database entry. These are elsewhere refered to as
          `db_info_ids` or `dbids`.
        - **db_name** [``varchar NOT NULL``]: A short lowercase string that is
          used internally to identify the knowledge base, e.g. "pc" for Pathway
          Commons.
        - **db_full_name** [``varchar NOT NULL``]: The full name of the
          knowledge base, neatly formatted, e.g. "Pathway Commons".
        - **source_api** [``varchar NOT NULL``]: The indra source API that was
          used to extract Statements from the knowledge base, e.g. "biopax".

        **Metadata Columns**

        - **create_date** [``timestamp without time zone``]: The date the record
          was added.
        - **last_updated** [``timestamp without time zone``]: The most recent
          time the record was edited.
        """
        class DBInfo(self.base, IndraDBTable):
            __tablename__ = 'db_info'
            _always_disp = ['id', 'db_name', 'source_api']
            id = Column(Integer, primary_key=True)
            db_name = Column(String, nullable=False)
            db_full_name = Column(String, nullable=False)
            source_api = Column(String, nullable=False)
            create_date = Column(DateTime, default=func.now())
            last_updated = Column(DateTime, onupdate=func.now())
        return DBInfo

    def raw_statements(self):
        """Represent Statements exactly as extracted their sources."""
        class RawStatements(self.base, IndraDBTable):
            __tablename__ = 'raw_statements'
            _skip_disp = ['json']
            _always_disp = ['id', 'db_info_id', 'reading_id', 'type']
            id = Column(Integer, primary_key=True)
            uuid = Column(String(40), unique=True, nullable=False)
            batch_id = Column(Integer, nullable=False)
            mk_hash = Column(BigInteger, nullable=False)
            text_hash = Column(BigInteger)
            source_hash = Column(BigInteger, nullable=False)
            db_info_id = Column(Integer, ForeignKey('db_info.id'))
            db_info = relationship(self.table_dict['db_info'])
            reading_id = Column(BigInteger, ForeignKey('reading.id'))
            reading = relationship(self.table_dict['reading'])
            type = Column(String(100), nullable=False)
            indra_version = Column(String(100), nullable=False)
            json = Column(BYTEA, nullable=False)
            create_date = Column(DateTime, default=func.now())
            __table_args__ = (
                UniqueConstraint('mk_hash', 'text_hash', 'reading_id',
                                 name='reading_raw_statement_uniqueness'),
                UniqueConstraint('mk_hash', 'source_hash', 'db_info_id',
                                 name='db_info_raw_statement_uniqueness'),
            )
        return RawStatements

    def raw_activity(self):
        """Represent the activity of a raw statement (an ActiveForm)."""
        class RawActivity(self.base, IndraDBTable):
            __tablename__ = 'raw_activity'
            _always_disp = ['stmt_id', 'activity', 'is_active']
            id = Column(Integer, primary_key=True)
            stmt_id = Column(Integer,
                             ForeignKey('raw_statements.id'),
                             nullable=False)
            statements = relationship(self.table_dict['raw_statements'])
            activity = Column(String)
            is_active = Column(Boolean)
        return RawActivity

    def raw_agents(self):
        """Represent an identifier for an agent of a raw statement."""
        class RawAgents(self.base, IndraDBTable):
            __tablename__ = 'raw_agents'
            _always_disp = ['stmt_id', 'db_name', 'db_id', 'ag_num']
            id = Column(Integer, primary_key=True)
            stmt_id = Column(Integer,
                             ForeignKey('raw_statements.id'),
                             nullable=False)
            statements = relationship(self.table_dict['raw_statements'])
            db_name = Column(String(40), nullable=False)
            db_id = Column(String, nullable=False)
            ag_num = Column(Integer, nullable=False)
            role = Column(String(20), nullable=False)
        return RawAgents

    def raw_mods(self):
        """Represent a modification of an agent of a raw statement."""
        class RawMods(self.base, IndraDBTable):
            __tablename__ = 'raw_mods'
            _always_disp = ['stmt_id', 'type', 'position', 'residue', 'modified',
                            'ag_num']
            id = Column(Integer, primary_key=True)
            stmt_id = Column(Integer, ForeignKey('raw_statements.id'),
                             nullable=False)
            statements = relationship(self.table_dict['raw_statements'])
            type = Column(String, nullable=False)
            position = Column(String(10))
            residue = Column(String(5))
            modified = Column(Boolean)
            ag_num = Column(Integer, nullable=False)
        return RawMods

    def raw_muts(self):
        """Represent a mutation of an agent of a raw statement."""
        class RawMuts(self.base, IndraDBTable):
            __tablename__ = 'raw_muts'
            _always_disp = ['stmt_id', 'position', 'residue_from', 'residue_to',
                            'ag_num']
            id = Column(Integer, primary_key=True)
            stmt_id = Column(Integer, ForeignKey('raw_statements.id'),
                             nullable=False)
            statements = relationship(self.table_dict['raw_statements'])
            position = Column(String(10))
            residue_from = Column(String(5))
            residue_to = Column(String(5))
            ag_num = Column(Integer, nullable=False)
        return RawMuts

    def raw_unique_links(self):
        """Represent links between raw statements and preassembled statements.

        Each preassembled statement is constructed from multiple raw statements,
        in general. This table is how the list of evidence can be built for
        each preassembled statement.
        """
        class RawUniqueLinks(self.base, IndraDBTable):
            __tablename__ = 'raw_unique_links'
            _always_disp = ['raw_stmt_id', 'pa_stmt_mk_hash']
            _indices = [BtreeIndex('raw_unique_links_raw_stmt_id_idx',
                                   'raw_stmt_id'),
                        BtreeIndex('raw_unique_links_pa_stmt_mk_hash_idx',
                                   'pa_stmt_mk_hash')]
            id = Column(Integer, primary_key=True)
            raw_stmt_id = Column(Integer, ForeignKey('raw_statements.id'),
                                 nullable=False)
            pa_stmt_mk_hash = Column(BigInteger,
                                     ForeignKey('pa_statements.mk_hash'),
                                     nullable=False)
            __table_args = (
                UniqueConstraint('raw_stmt_id', 'pa_stmt_mk_hash',
                                 name='stmt-link-uniqueness'),
            )
        return RawUniqueLinks

    def pa_statements(self):
        """Represent preassembled statements.

         Preassmebled Statements are a cleaned up and unique corpus of
         statements generated from the raw statements.
         """
        class PAStatements(self.base, IndraDBTable):
            __tablename__ = 'pa_statements'
            _skip_disp = ['json']
            _always_disp = ['mk_hash', 'type']
            _default_order_by = 'create_date'

            mk_hash = Column(BigInteger, primary_key=True)
            matches_key = Column(String, nullable=False)
            uuid = Column(String(40), unique=True, nullable=False)
            type = Column(String(100), nullable=False)
            indra_version = Column(String(100), nullable=False)
            json = Column(BYTEA, nullable=False)
            create_date = Column(DateTime, default=func.now())
        return PAStatements

    def pa_support_links(self):
        """Represent the links of support calculated during preassembly.

        In INDRA, we look for cases where more specific Statements may lend
        support to more general Statements, and potentially vice versa, to
        better gauge whether an extraction is reliable.
        """
        class PASupportLinks(self.base, IndraDBTable):
            __tablename__ = 'pa_support_links'
            _always_disp = ['supporting_mk_hash', 'supported_mk_hash']
            id = Column(Integer, primary_key=True)
            supporting_mk_hash = Column(BigInteger,
                                        ForeignKey('pa_statements.mk_hash'),
                                        nullable=False)
            supported_mk_hash = Column(BigInteger,
                                       ForeignKey('pa_statements.mk_hash'),
                                       nullable=False)
            __table_args__ = (
                UniqueConstraint('supporting_mk_hash', 'supported_mk_hash',
                                 name='pa_support_links_link_uniqueness'),
            )
        return PASupportLinks

    def pa_activity(self):
        """Represent the activity of a preassembled Statement."""
        class PAActivity(self.base, IndraDBTable):
            __tablename__ = 'pa_activity'
            __always_disp__ = ['stmt_mk_hash', 'activity', 'is_active']
            id = Column(Integer, primary_key=True)
            stmt_mk_hash = Column(BigInteger,
                                  ForeignKey('pa_statements.mk_hash'),
                                  nullable=False)
            statements = relationship(self.table_dict['pa_statements'])
            activity = Column(String)
            is_active = Column(Boolean)
        return PAActivity

    def pa_agents(self):
        """Represent an identifier for an agent of a preassembled statement."""
        class PAAgents(self.base, IndraDBTable):
            __tablename__ = 'pa_agents'
            _always_disp = ['stmt_mk_hash', 'db_name', 'db_id', 'ag_num']
            id = Column(Integer, primary_key=True)
            stmt_mk_hash = Column(BigInteger,
                                  ForeignKey('pa_statements.mk_hash'),
                                  nullable=False)
            statements = relationship(self.table_dict['pa_statements'])
            db_name = Column(String(40), nullable=False)
            db_id = Column(String, nullable=False)
            role = Column(String(20), nullable=False)
            ag_num = Column(Integer, nullable=False)
            agent_ref_hash = Column(BigInteger, unique=True, nullable=False)
        return PAAgents

    def pa_mods(self):
        """Represent a modification of an agent of a preassembled statement."""
        class PAMods(self.base, IndraDBTable):
            __tablename__ = 'pa_mods'
            _always_disp = ['stmt_mk_hash', 'type', 'position', 'residue',
                            'modified', 'ag_num']
            id = Column(Integer, primary_key=True)
            stmt_mk_hash = Column(BigInteger,
                                  ForeignKey('pa_statements.mk_hash'),
                                  nullable=False)
            statements = relationship(self.table_dict['pa_statements'])
            type = Column(String, nullable=False)
            position = Column(String(10))
            residue = Column(String(5))
            modified = Column(Boolean)
            ag_num = Column(Integer, nullable=False)
        return PAMods

    def pa_muts(self):
        """Represent a mutation of an agent of a preassembled statement."""
        class PAMuts(self.base, IndraDBTable):
            __tablename__ = 'pa_muts'
            _always_disp = ['stmt_mk_hash', 'position', 'residue_from',
                            'residue_to', 'ag_num']
            id = Column(Integer, primary_key=True)
            stmt_mk_hash = Column(BigInteger,
                                  ForeignKey('pa_statements.mk_hash'),
                                  nullable=False)
            statements = relationship(self.table_dict['pa_statements'])
            position = Column(String(10))
            residue_from = Column(String(5))
            residue_to = Column(String(5))
            ag_num = Column(Integer, nullable=False)
        return PAMuts

    def curations(self):
        """Represent the curations of our content."""
        class Curation(self.base, IndraDBTable):
            __tablename__ = 'curation'
            _always_disp = ['pa_hash', 'source_hash', 'tag', 'curator', 'date']
            id = Column(Integer, primary_key=True)
            pa_hash = Column(BigInteger, ForeignKey('pa_statements.mk_hash'))
            pa_statements = relationship(self.table_dict['pa_statements'])
            source_hash = Column(BigInteger)
            tag = Column(String)
            text = Column(String)
            curator = Column(String, nullable=False)
            auth_id = Column(Integer)
            source = Column(String)
            ip = Column(INET)
            date = Column(DateTime, default=func.now())
            pa_json = Column(JSONB)
            ev_json = Column(JSONB)

            def to_json(self):
                return {attr: getattr(self, attr)
                        for attr in ['id', 'pa_hash', 'source_hash', 'tag', 'text',
                                     'date', 'curator', 'source', 'pa_json',
                                     'ev_json']}
        return Curation

    def source_file(self):
        """Record the pubmed source file that was processed."""
        class SourceFile(self.base, IndraDBTable):
            __tablename__ = 'source_file'
            _always_disp = ['source', 'name']
            id = Column(Integer, primary_key=True)
            source = Column(String(250), nullable=False)
            name = Column(String(250), nullable=False)
            load_date = Column(DateTime, default=func.now())
            __table_args__ = (
                UniqueConstraint('source', 'name', name='source-name'),
            )
        return SourceFile

    def updates(self):
        """Record when text ref and content updates were performed."""
        class Updates(self.base, IndraDBTable):
            __tablename__ = 'updates'
            _skip_disp = ['unresolved_conflicts_file']
            _always_disp = ['source', 'datetime']
            id = Column(Integer, primary_key=True)
            init_upload = Column(Boolean, nullable=False)
            source = Column(String(250), nullable=False)
            unresolved_conflicts_file = Column(BYTEA)
            datetime = Column(DateTime, default=func.now())
        return Updates

    def reading_updates(self):
        """Record runs of the readers on the content we have found."""
        class ReadingUpdates(self.base, IndraDBTable):
            __tablename__ = 'reading_updates'
            _always_disp = ['reader', 'reader_version', 'run_datetime']
            id = Column(Integer, primary_key=True)
            complete_read = Column(Boolean, nullable=False)
            reader = Column(String(250), nullable=False)
            reader_version = Column(String(250), nullable=False)
            run_datetime = Column(DateTime, default=func.now())
            earliest_datetime = Column(DateTime)
            latest_datetime = Column(DateTime, nullable=False)
        return ReadingUpdates

    def xdd_updates(self):
        """Record the times we process dumps from xDD."""
        class XddUpdates(self.base, IndraDBTable):
            __tablename__ = 'xdd_updates'
            _always_disp = ['day_str']
            id = Column(Integer, primary_key=True)
            reader_versions = Column(JSONB)
            indra_version = Column(String)
            day_str = Column(String, nullable=False, unique=True)
            processed_date = Column(DateTime, default=func.now())
        return XddUpdates

    def rejected_statements(self):
        """Represent raw statements that were rejected."""
        class RejectedStatements(self.base, IndraDBTable):
            __tablename__ = 'rejected_statements'
            _skip_disp = ['json']
            _always_disp = ['id', 'db_info_id', 'reading_id', 'type']
            id = Column(Integer, primary_key=True)
            uuid = Column(String(40), unique=True, nullable=False)
            batch_id = Column(Integer, nullable=False)
            mk_hash = Column(BigInteger, nullable=False)
            text_hash = Column(BigInteger)
            source_hash = Column(BigInteger, nullable=False)
            db_info_id = Column(Integer, ForeignKey('db_info.id'))
            db_info = relationship(self.table_dict['db_info'])
            reading_id = Column(BigInteger, ForeignKey('reading.id'))
            reading = relationship(self.table_dict['reading'])
            type = Column(String(100), nullable=False)
            indra_version = Column(String(100), nullable=False)
            json = Column(BYTEA, nullable=False)
            create_date = Column(DateTime, default=func.now())
        return RejectedStatements

    def discarded_statements(self):
        """Record the reasons for which some statements were discarded."""
        class DiscardedStatements(self.base, IndraDBTable):
            __tablename__ = 'discarded_statements'
            _always_disp = ['stmt_id', 'reason']
            id = Column(Integer, primary_key=True)
            stmt_id = Column(Integer, ForeignKey('raw_statements.id'),
                             nullable=False)
            reason = Column(String, nullable=False)
            insert_date = Column(DateTime, default=func.now())
        return DiscardedStatements

    def preassembly_updates(self):
        """Record updates of the preassembled corpus."""
        class PreassemblyUpdates(self.base, IndraDBTable):
            __tablename__ = 'preassembly_updates'
            _always_disp = ['corpus_init', 'run_datetime']
            id = Column(Integer, primary_key=True)
            corpus_init = Column(Boolean, nullable=False)
            run_datetime = Column(DateTime, default=func.now())
            stmt_type = Column(String)
        return PreassemblyUpdates
