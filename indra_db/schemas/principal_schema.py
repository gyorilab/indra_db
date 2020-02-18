__all__ = ['get_schema', 'foreign_key_map']

from sqlalchemy import Column, Integer, String, UniqueConstraint, ForeignKey, \
     Boolean, DateTime, func, BigInteger
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import BYTEA, INET

from indra_db.schemas.mixins import IndraDBTable
from indra_db.schemas.indexes import StringIndex

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


def get_schema(Base):
    table_dict = {}

    class TextRef(Base, IndraDBTable):
        __tablename__ = 'text_ref'
        _ref_cols = ['pmid', 'pmcid', 'doi', 'pii', 'url', 'manuscript_id']
        _always_disp = ['id', 'pmid', 'pmcid']
        _indices = [StringIndex('text_ref_pmid_idx', 'pmid')]

        id = Column(Integer, primary_key=True)
        pmid = Column(String(20))
        pmcid = Column(String(20))
        doi = Column(String(100))
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

        def get_ref_dict(self):
            ref_dict = {}
            for ref in self._ref_cols:
                val = getattr(self, ref, None)
                if val:
                    ref_dict[ref.upper()] = val
            ref_dict['TRID'] = self.id
            return ref_dict

    table_dict[TextRef.__tablename__] = TextRef

    class MeshRefAnnotations(Base, IndraDBTable):
        __tablename__ = 'mesh_ref_annotations'
        _always_disp = ['mesh_text', 'qual_text', 'text_ref_id']
        _indices = [StringIndex('mesh_ref_annotations_pmid_idx', 'pmid'),
                    StringIndex('mesh_ref_annotations_mesh_id_idx', 'mesh_id'),
                    StringIndex('mesh_ref_annotations_mtext_idx', 'mesh_text'),
                    StringIndex('mesh_ref_annotations_qual_id_idx', 'qual_id'),
                    StringIndex('mesh_ref_annotations_qtext_idx', 'qual_text')]

        id = Column(Integer, primary_key=True)
        mesh_id = Column(String, nullable=False)
        mesh_text = Column(String, nullable=False)
        pmid = Column(String, nullable=False)
        major_topic = Column(Boolean, default=False)
        qual_id = Column(String)
        qual_text = Column(String)

    table_dict[MeshRefAnnotations.__tablename__] = MeshRefAnnotations

    class SourceFile(Base, IndraDBTable):
        __tablename__ = 'source_file'
        _always_disp = ['source', 'name']
        id = Column(Integer, primary_key=True)
        source = Column(String(250), nullable=False)
        name = Column(String(250), nullable=False)
        load_date = Column(DateTime, default=func.now())
        __table_args__ = (
            UniqueConstraint('source', 'name', name='source-name'),
        )
    table_dict[SourceFile.__tablename__] = SourceFile

    class Updates(Base, IndraDBTable):
        __tablename__ = 'updates'
        _skip_disp = ['unresolved_conflicts_file']
        _always_disp = ['source', 'datetime']
        id = Column(Integer, primary_key=True)
        init_upload = Column(Boolean, nullable=False)
        source = Column(String(250), nullable=False)
        unresolved_conflicts_file = Column(BYTEA)
        datetime = Column(DateTime, default=func.now())
    table_dict[Updates.__tablename__] = Updates

    class TextContent(Base, IndraDBTable):
        __tablename__ = 'text_content'
        _skip_disp = ['content']
        _always_disp = ['id', 'text_ref_id', 'source', 'format', 'text_type']
        id = Column(Integer, primary_key=True)
        text_ref_id = Column(Integer, ForeignKey('text_ref.id'),
                             nullable=False)
        text_ref = relationship(TextRef)
        source = Column(String(250), nullable=False)
        format = Column(String(250), nullable=False)
        text_type = Column(String(250), nullable=False)
        content = Column(BYTEA, nullable=False)
        insert_date = Column(DateTime, default=func.now())
        last_updated = Column(DateTime, onupdate=func.now())
        __table_args__ = (
            UniqueConstraint('text_ref_id', 'source', 'format',
                             'text_type', name='content-uniqueness'),
        )
    table_dict[TextContent.__tablename__] = TextContent

    class Reading(Base, IndraDBTable):
        __tablename__ = 'reading'
        _skip_disp = ['bytes']
        _always_disp = ['id', 'text_content_id', 'reader', 'reader_version']
        id = Column(BigInteger, primary_key=True, default=None)
        text_content_id = Column(Integer,
                                 ForeignKey('text_content.id'),
                                 nullable=False)
        batch_id = Column(Integer, nullable=False)
        text_content = relationship(TextContent)
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
    table_dict[Reading.__tablename__] = Reading

    class ReadingUpdates(Base, IndraDBTable):
        __tablename__ = 'reading_updates'
        _always_disp = ['reader', 'reader_version', 'run_datetime']
        id = Column(Integer, primary_key=True)
        complete_read = Column(Boolean, nullable=False)
        reader = Column(String(250), nullable=False)
        reader_version = Column(String(250), nullable=False)
        run_datetime = Column(DateTime, default=func.now())
        earliest_datetime = Column(DateTime)
        latest_datetime = Column(DateTime, nullable=False)
    table_dict[ReadingUpdates.__tablename__] = ReadingUpdates

    class DBInfo(Base, IndraDBTable):
        __tablename__ = 'db_info'
        _always_disp = ['id', 'db_name', 'source_api']
        id = Column(Integer, primary_key=True)
        db_name = Column(String(20), nullable=False)
        source_api = Column(String, nullable=False)
        create_date = Column(DateTime, default=func.now())
        last_updated = Column(DateTime, onupdate=func.now())
    table_dict[DBInfo.__tablename__] = DBInfo

    class RawStatements(Base, IndraDBTable):
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
        db_info = relationship(DBInfo)
        reading_id = Column(BigInteger, ForeignKey('reading.id'))
        reading = relationship(Reading)
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
    table_dict[RawStatements.__tablename__] = RawStatements

    class RejectedStatements(Base, IndraDBTable):
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
        db_info = relationship(DBInfo)
        reading_id = Column(BigInteger, ForeignKey('reading.id'))
        reading = relationship(Reading)
        type = Column(String(100), nullable=False)
        indra_version = Column(String(100), nullable=False)
        json = Column(BYTEA, nullable=False)
        create_date = Column(DateTime, default=func.now())
    table_dict[RejectedStatements.__tablename__] = RejectedStatements

    class DiscardedStatements(Base, IndraDBTable):
        __tablename__ = 'discarded_statements'
        _always_disp = ['stmt_id', 'reason']
        id = Column(Integer, primary_key=True)
        stmt_id = Column(Integer, ForeignKey('raw_statements.id'),
                         nullable=False)
        reason = Column(String, nullable=False)
        insert_date = Column(DateTime, default=func.now())
    table_dict[DiscardedStatements.__tablename__] = DiscardedStatements

    class RawAgents(Base, IndraDBTable):
        __tablename__ = 'raw_agents'
        _always_disp = ['stmt_id', 'db_name', 'db_id', 'ag_num']
        id = Column(Integer, primary_key=True)
        stmt_id = Column(Integer,
                         ForeignKey('raw_statements.id'),
                         nullable=False)
        statements = relationship(RawStatements)
        db_name = Column(String(40), nullable=False)
        db_id = Column(String, nullable=False)
        ag_num = Column(Integer, nullable=False)
        role = Column(String(20), nullable=False)
        __table_args = (
            UniqueConstraint('stmt_id', 'db_name', 'db_id', 'role',
                             name='raw-agents-uniqueness'),
        )
    table_dict[RawAgents.__tablename__] = RawAgents

    class RawMods(Base, IndraDBTable):
        __tablename__ = 'raw_mods'
        _always_disp = ['stmt_id', 'type', 'position', 'residue', 'modified',
                        'ag_num']
        id = Column(Integer, primary_key=True)
        stmt_id = Column(Integer, ForeignKey('raw_statements.id'),
                         nullable=False)
        statements = relationship(RawStatements)
        type = Column(String, nullable=False)
        position = Column(String(10))
        residue = Column(String(5))
        modified = Column(Boolean)
        ag_num = Column(Integer, nullable=False)
    table_dict[RawMods.__tablename__] = RawMods

    class RawMuts(Base, IndraDBTable):
        __tablename__ = 'raw_muts'
        _always_disp = ['stmt_id', 'position', 'residue_from', 'residue_to',
                        'ag_num']
        id = Column(Integer, primary_key=True)
        stmt_id = Column(Integer, ForeignKey('raw_statements.id'),
                         nullable=False)
        statements = relationship(RawStatements)
        position = Column(String(10))
        residue_from = Column(String(5))
        residue_to = Column(String(5))
        ag_num = Column(Integer, nullable=False)
    table_dict[RawMuts.__tablename__] = RawMuts

    class RawUniqueLinks(Base, IndraDBTable):
        __tablename__ = 'raw_unique_links'
        _always_disp = ['raw_stmt_id', 'pa_stmt_mk_hash']
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
    table_dict[RawUniqueLinks.__tablename__] = RawUniqueLinks

    class PreassemblyUpdates(Base, IndraDBTable):
        __tablename__ = 'preassembly_updates'
        _always_disp = ['corpus_init', 'run_datetime']
        id = Column(Integer, primary_key=True)
        corpus_init = Column(Boolean, nullable=False)
        run_datetime = Column(DateTime, default=func.now())
    table_dict[PreassemblyUpdates.__tablename__] = PreassemblyUpdates

    class PAStatements(Base, IndraDBTable):
        __tablename__ = 'pa_statements'
        _skip_disp = ['json']
        _always_disp = ['mk_hash', 'type']
        _default_order_by = 'create_date'

        mk_hash = Column(BigInteger, primary_key=True)
        matches_key = Column(String, unique=True, nullable=False)
        uuid = Column(String(40), unique=True, nullable=False)
        type = Column(String(100), nullable=False)
        indra_version = Column(String(100), nullable=False)
        json = Column(BYTEA, nullable=False)
        create_date = Column(DateTime, default=func.now())
    table_dict[PAStatements.__tablename__] = PAStatements

    class PAAgents(Base, IndraDBTable):
        __tablename__ = 'pa_agents'
        _always_disp = ['stmt_mk_hash', 'db_name', 'db_id', 'ag_num']
        id = Column(Integer, primary_key=True)
        stmt_mk_hash = Column(BigInteger,
                              ForeignKey('pa_statements.mk_hash'),
                              nullable=False)
        statements = relationship(PAStatements)
        db_name = Column(String(40), nullable=False)
        db_id = Column(String, nullable=False)
        role = Column(String(20), nullable=False)
        ag_num = Column(Integer, nullable=False)
        __table_args__ = (
            UniqueConstraint('stmt_mk_hash', 'db_name', 'db_id', 'role',
                             name='pa-agent-uniqueness'),
        )
    table_dict[PAAgents.__tablename__] = PAAgents

    class PAMods(Base, IndraDBTable):
        __tablename__ = 'pa_mods'
        _always_disp = ['stmt_mk_hash', 'type', 'position', 'residue',
                        'modified', 'ag_num']
        id = Column(Integer, primary_key=True)
        stmt_mk_hash = Column(BigInteger,
                              ForeignKey('pa_statements.mk_hash'),
                              nullable=False)
        statements = relationship(PAStatements)
        type = Column(String, nullable=False)
        position = Column(String(10))
        residue = Column(String(5))
        modified = Column(Boolean)
        ag_num = Column(Integer, nullable=False)
    table_dict[PAMods.__tablename__] = PAMods

    class PAMuts(Base, IndraDBTable):
        __tablename__ = 'pa_muts'
        _always_disp = ['stmt_mk_hash', 'position', 'residue_from',
                        'residue_to', 'ag_num']
        id = Column(Integer, primary_key=True)
        stmt_mk_hash = Column(BigInteger,
                              ForeignKey('pa_statements.mk_hash'),
                              nullable=False)
        statements = relationship(PAStatements)
        position = Column(String(10))
        residue_from = Column(String(5))
        residue_to = Column(String(5))
        ag_num = Column(Integer, nullable=False)
    table_dict[PAMuts.__tablename__] = PAMuts

    class PASupportLinks(Base, IndraDBTable):
        __tablename__ = 'pa_support_links'
        _always_disp = ['supporting_mk_hash', 'supported_mk_hash']
        id = Column(Integer, primary_key=True)
        supporting_mk_hash = Column(BigInteger,
                                    ForeignKey('pa_statements.mk_hash'),
                                    nullable=False)
        supported_mk_hash = Column(BigInteger,
                                   ForeignKey('pa_statements.mk_hash'),
                                   nullable=False)
    table_dict[PASupportLinks.__tablename__] = PASupportLinks

    class Curation(Base, IndraDBTable):
        __tablename__ = 'curation'
        _always_disp = ['pa_hash', 'source_hash', 'tag', 'curator', 'date']
        id = Column(Integer, primary_key=True)
        pa_hash = Column(BigInteger, ForeignKey('pa_statements.mk_hash'))
        pa_statements = relationship(PAStatements)
        source_hash = Column(BigInteger)
        tag = Column(String)
        text = Column(String)
        curator = Column(String, nullable=False)
        auth_id = Column(Integer)
        source = Column(String)
        ip = Column(INET)
        date = Column(DateTime, default=func.now())

        def to_json(self):
            return {attr: getattr(self, attr)
                    for attr in ['id', 'pa_hash', 'source_hash', 'tag', 'text',
                                 'date']}

    table_dict[Curation.__tablename__] = Curation

    return table_dict
