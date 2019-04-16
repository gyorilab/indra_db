from sqlalchemy.exc import NoSuchTableError

__all__ = ['sqltypes', 'texttypes', 'formats', 'DatabaseManager',
           'IndraDbException', 'sql_expressions', 'readers', 'reader_versions']

import re
import random
import logging
from uuid import uuid4
from io import BytesIO
from numbers import Number
from datetime import datetime

from sqlalchemy.sql import expression as sql_expressions
from sqlalchemy.schema import DropTable
from sqlalchemy.sql.expression import Delete, Update
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy import Column, Integer, String, UniqueConstraint, ForeignKey, \
    create_engine, inspect, LargeBinary, Boolean, DateTime, func, BigInteger
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.dialects.postgresql import BYTEA, INET

from indra_db.exceptions import IndraDbException
from indra.util import batch_iter

try:
    import networkx as nx
    WITH_NX = True
except ImportError:
    WITH_NX = False


logger = logging.getLogger(__name__)


# Solution to fix postgres drop tables
# See: https://stackoverflow.com/questions/38678336/sqlalchemy-how-to-implement-drop-table-cascade
@compiles(DropTable, "postgresql")
def _compile_drop_table(element, compiler, **kwargs):
    return compiler.visit_drop_table(element) + " CASCADE"


# Solution to fix deletes with constraints from multiple tables.
# See: https://groups.google.com/forum/#!topic/sqlalchemy/cIvgH2y01_o
@compiles(Delete)
def compile_delete(element, compiler, **kw):
    text = compiler.visit_delete(element, **kw)
    extra_froms = Update._extra_froms.__get__(element)
    if extra_froms:
        text = re.sub(
                    r"(FROM \S+)",
                    lambda m: "%s USING %s" % (
                        m.group(1),
                        ", ".join(
                            compiler.process(fr, asfrom=True, **kw)
                            for fr in extra_froms
                        )
                    ),
                    text
                )
    return text


try:
    from pgcopy import CopyManager
    CAN_COPY = True
except ImportError:
    print("WARNING: pgcopy unavailable. Bulk copies will be slow.")

    class CopyManager(object):
        def __init__(self, conn, table, cols):
            raise NotImplementedError("CopyManager could not be imported from"
                                      "pgcopy.")

    CAN_COPY = False


def _isiterable(obj):
    "Bool determines if an object is an iterable (not a string)"
    return hasattr(obj, '__iter__') and not isinstance(obj, str)


class _map_class(object):
    @classmethod
    def _getattrs(self):
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith('_')
            }

    @classmethod
    def items(self):
        return self._getattrs().items()

    @classmethod
    def values(self):
        return self._getattrs().values()

    @classmethod
    def keys(self):
        return self._getattrs().keys()


class sqltypes(_map_class):
    POSTGRESQL = 'postgresql'
    SQLITE = 'sqlite'


class texttypes(_map_class):
    FULLTEXT = 'fulltext'
    ABSTRACT = 'abstract'
    TITLE = 'title'


class formats(_map_class):
    XML = 'xml'
    TEXT = 'text'
    JSON = 'json'
    EKB = 'ekb'


readers = {'REACH': 1, 'SPARSER': 2, 'TRIPS': 3}


# Specify versions of readers, and preference. Later in the list is better.
reader_versions = {
    'sparser': ['sept14-linux\n', 'sept14-linux', 'June2018-linux',
                'October2018-linux'],
    'reach': ['61059a-biores-e9ee36', '1.3.3-61059a-biores-'],
    'trips': ['STATIC']
}


class IndraTableError(IndraDbException):
    def __init__(self, table, issue):
        msg = 'Error in table %s: %s' % (table, issue)
        super(IndraTableError, self).__init__(self, msg)


class Displayable(object):
    _skip_disp = []

    def _make_str(self):
        s = self.__tablename__ + ':\n'
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if k in self._skip_disp:
                    s += '\t%s: [not shown]\n' % k
                else:
                    s += '\t%s: %s\n' % (k, v)
        return s

    def display(self):
        """Display the values of this entry."""
        print(self._make_str())

    def __str__(self):
        return self._make_str()


class MaterializedView(Displayable):
    __definition__ = NotImplemented
    _indices = []

    @classmethod
    def create(cls, db, with_data=True, commit=True):
        sql = "CREATE MATERIALIZED VIEW public.%s AS %s WITH %s DATA;" \
              % (cls.__tablename__, cls.get_definition(),
                 '' if with_data else "NO")
        if commit:
            cls.execute(db, sql)
        return sql

    @classmethod
    def update(cls, db, with_data=True, commit=True):
        sql = "REFRESH MATERIALIZED VIEW %s WITH %s DATA;" \
              % (cls.__tablename__, '' if with_data else 'NO')
        if commit:
            cls.execute(db, sql)
        return sql

    @classmethod
    def get_definition(cls):
        return cls.__definition__

    @staticmethod
    def execute(db, sql):
        conn = db.engine.raw_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        return

    @classmethod
    def create_index(cls, db, index, commit=True):
        inp_data = {'idx_name': index.name,
                    'table_name': cls.__tablename__,
                    'idx_def': index.definition}
        sql = ("CREATE INDEX {idx_name} ON public.{table_name} "
               "USING {idx_def} TABLESPACE pg_default;".format(**inp_data))
        if commit:
            cls.execute(db, sql)
        return sql

    @classmethod
    def build_indices(cls, db):
        for index in cls._indices:
            print("Building index: %s" % index.name)
            cls.create_index(db, index)


class BtreeIndex(object):
    def __init__(self, name, colname, opts=None):
        self.name = name
        self.colname = colname
        contents = colname
        if opts is not None:
            contents += ' ' + opts
        self.definition = ('btree (%s)' % contents)


class StringIndex(BtreeIndex):
    def __init__(self, name, colname):
        opts = 'COLLATE pg_catalog."en_US.utf8" varchar_ops ASC NULLS LAST'
        super().__init__(name, colname, opts)


class NamespaceLookup(MaterializedView):
    __dbname__ = NotImplemented

    @classmethod
    def get_definition(cls):
        return ("SELECT db_id, ag_id, role, ag_num, type, "
                "mk_hash, ev_count FROM pa_meta "
                "WHERE db_name = '%s'" % cls.__dbname__)


class DatabaseManager(object):
    """An object used to access INDRA's database.

    This object can be used to access and manage indra's database. It includes
    both basic methods and some useful, more high-level methods. It is designed
    to be used with postgresql, or sqlite.

    This object is primarily built around sqlalchemy, which is a required
    package for its use. It also optionally makes use of the pgcopy package for
    large data transfers.

    If you wish to access the primary database, you can simply use the
    `get_primary_db` to get an instance of this object using the default
    settings.

    Parameters
    ----------
    host : str
        The database to which you want to interface.
    sqltype : OPTIONAL[str]
        The type of sql library used. Use one of the sql types provided by
        `sqltypes`. Default is `sqltypes.POSTGRESQL`
    label : OPTIONAL[str]
        A short string to indicate the purpose of the db instance. Set as
        primary when initialized be `get_primary_db`.

    Example
    -------
    If you wish to acces the primary database and find the the metadata for a
    particular pmid, 1234567:

    >> from indra.db import get_primary_db()
    >> db = get_primary_db()
    >> res = db.select_all(db.TextRef, db.TextRef.pmid == '1234567')

    You will get a list of objects whose attributes give the metadata contained
    in the columns of the table.

    For more sophisticated examples, several use cases can be found in
    `indra.tests.test_db`.
    """
    def __init__(self, host, sqltype=sqltypes.POSTGRESQL, label=None):
        self.host = host
        self.session = None
        self.Base = declarative_base()
        self.sqltype = sqltype
        self.label = label
        self.tables = {}

        if sqltype is sqltypes.POSTGRESQL:
            Bytea = BYTEA
        else:
            Bytea = LargeBinary

        # Normal Tables -------------------------------------------------------
        class TextRef(self.Base, Displayable):
            __tablename__ = 'text_ref'
            id = Column(Integer, primary_key=True)
            pmid = Column(String(20))
            pmcid = Column(String(20))
            doi = Column(String(100))
            pii = Column(String(250))
            url = Column(String(250), unique=True)  # Maybe longer?
            manuscript_id = Column(String(100), unique=True)
            create_date = Column(DateTime, default=func.now())
            last_updated = Column(DateTime, onupdate=func.now())
            __table_args__ = (
                UniqueConstraint('pmid', 'doi', name='pmid-doi'),
                UniqueConstraint('pmcid', 'doi', name='pmcid-doi')
                )

        self.TextRef = TextRef
        self.tables[TextRef.__tablename__] = TextRef

        class SourceFile(self.Base, Displayable):
            __tablename__ = 'source_file'
            id = Column(Integer, primary_key=True)
            source = Column(String(250), nullable=False)
            name = Column(String(250), nullable=False)
            load_date = Column(DateTime, default=func.now())
            __table_args__ = (
                UniqueConstraint('source', 'name', name='source-name'),
                )
        self.SourceFile = SourceFile
        self.tables[SourceFile.__tablename__] = SourceFile

        class Updates(self.Base, Displayable):
            __tablename__ = 'updates'
            _skip_disp = ['unresolved_conflicts_file']
            id = Column(Integer, primary_key=True)
            init_upload = Column(Boolean, nullable=False)
            source = Column(String(250), nullable=False)
            unresolved_conflicts_file = Column(Bytea)
            datetime = Column(DateTime, default=func.now())
        self.Updates = Updates
        self.tables[Updates.__tablename__] = Updates

        class TextContent(self.Base, Displayable):
            __tablename__ = 'text_content'
            _skip_disp = ['content']
            id = Column(Integer, primary_key=True)
            text_ref_id = Column(Integer,
                                 ForeignKey('text_ref.id'),
                                 nullable=False)
            text_ref = relationship(TextRef)
            source = Column(String(250), nullable=False)
            format = Column(String(250), nullable=False)
            text_type = Column(String(250), nullable=False)
            content = Column(Bytea, nullable=False)
            insert_date = Column(DateTime, default=func.now())
            last_updated = Column(DateTime, onupdate=func.now())
            __table_args__ = (
                UniqueConstraint('text_ref_id', 'source', 'format',
                                 'text_type', name='content-uniqueness'),
                )
        self.TextContent = TextContent
        self.tables[TextContent.__tablename__] = TextContent

        class Reading(self.Base, Displayable):
            __tablename__ = 'reading'
            _skip_disp = ['bytes']
            id = Column(BigInteger, primary_key=True, default=None)
            text_content_id = Column(Integer,
                                     ForeignKey('text_content.id'),
                                     nullable=False)
            batch_id = Column(Integer, nullable=False)
            text_content = relationship(TextContent)
            reader = Column(String(20), nullable=False)
            reader_version = Column(String(20), nullable=False)
            format = Column(String(20), nullable=False)  # xml, json, etc.
            bytes = Column(Bytea)
            create_date = Column(DateTime, default=func.now())
            last_updated = Column(DateTime, onupdate=func.now())
            __table_args__ = (
                UniqueConstraint('text_content_id', 'reader', 'reader_version',
                                 name='reading-uniqueness'),
                )
        self.Reading = Reading
        self.tables[Reading.__tablename__] = Reading

        class ReadingUpdates(self.Base, Displayable):
            __tablename__ = 'reading_updates'
            id = Column(Integer, primary_key=True)
            complete_read = Column(Boolean, nullable=False)
            reader = Column(String(250), nullable=False)
            reader_version = Column(String(250), nullable=False)
            run_datetime = Column(DateTime, default=func.now())
            earliest_datetime = Column(DateTime)
            latest_datetime = Column(DateTime, nullable=False)
        self.ReadingUpdates = ReadingUpdates
        self.tables[ReadingUpdates.__tablename__] = ReadingUpdates

        class DBInfo(self.Base, Displayable):
            __tablename__ = 'db_info'
            id = Column(Integer, primary_key=True)
            db_name = Column(String(20), nullable=False)
            source_api = Column(String, nullable=False)
            create_date = Column(DateTime, default=func.now())
            last_updated = Column(DateTime, onupdate=func.now())
        self.DBInfo = DBInfo
        self.tables[DBInfo.__tablename__] = DBInfo

        class RawStatements(self.Base, Displayable):
            __tablename__ = 'raw_statements'
            _skip_disp = ['json']
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
            json = Column(Bytea, nullable=False)
            create_date = Column(DateTime, default=func.now())
            __table_args__ = (
                UniqueConstraint('mk_hash', 'text_hash', 'reading_id',
                                 name='reading_raw_statement_uniqueness'),
                UniqueConstraint('mk_hash', 'source_hash', 'db_info_id',
                                 name='db_info_raw_statement_uniqueness'),
                )
        self.RawStatements = RawStatements
        self.tables[RawStatements.__tablename__] = RawStatements

        class RejectedStatements(self.Base, Displayable):
            __tablename__ = 'rejected_statements'
            _skip_disp = ['json']
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
            json = Column(Bytea, nullable=False)
            create_date = Column(DateTime, default=func.now())
        self.RejectedStatements = RejectedStatements
        self.tables[RejectedStatements.__tablename__] = RejectedStatements

        class RawAgents(self.Base, Displayable):
            __tablename__ = 'raw_agents'
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
        self.RawAgents = RawAgents
        self.tables[RawAgents.__tablename__] = RawAgents

        class RawMods(self.Base, Displayable):
            __tablename__ = 'raw_mods'
            id = Column(Integer, primary_key=True)
            stmt_id = Column(Integer, ForeignKey('raw_statements.id'),
                             nullable=False)
            statements = relationship(RawStatements)
            type = Column(String, nullable=False)
            position = Column(String(10))
            residue = Column(String(5))
            modified = Column(Boolean)
            ag_num = Column(Integer, nullable=False)
        self.RawMods = RawMods
        self.tables[RawMods.__tablename__] = RawMods

        class RawMuts(self.Base, Displayable):
            __tablename__ = 'raw_muts'
            id = Column(Integer, primary_key=True)
            stmt_id = Column(Integer, ForeignKey('raw_statements.id'),
                             nullable=False)
            statements = relationship(RawStatements)
            position = Column(String(10))
            residue_from = Column(String(5))
            residue_to = Column(String(5))
            ag_num = Column(Integer, nullable=False)
        self.RawMuts = RawMuts
        self.tables[RawMuts.__tablename__] = RawMuts

        class RawUniqueLinks(self.Base, Displayable):
            __tablename__ = 'raw_unique_links'
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
        self.RawUniqueLinks = RawUniqueLinks
        self.tables[RawUniqueLinks.__tablename__] = RawUniqueLinks

        class PreassemblyUpdates(self.Base, Displayable):
            __tablename__ = 'preassembly_updates'
            id = Column(Integer, primary_key=True)
            corpus_init = Column(Boolean, nullable=False)
            run_datetime = Column(DateTime, default=func.now())
        self.PreassemblyUpdates = PreassemblyUpdates
        self.tables[PreassemblyUpdates.__tablename__] = PreassemblyUpdates

        class PAStatements(self.Base, Displayable):
            __tablename__ = 'pa_statements'
            _skip_disp = ['json']
            mk_hash = Column(BigInteger, primary_key=True)
            matches_key = Column(String, unique=True, nullable=False)
            uuid = Column(String(40), unique=True, nullable=False)
            type = Column(String(100), nullable=False)
            indra_version = Column(String(100), nullable=False)
            json = Column(Bytea, nullable=False)
            create_date = Column(DateTime, default=func.now())
        self.PAStatements = PAStatements
        self.tables[PAStatements.__tablename__] = PAStatements

        class PAAgents(self.Base, Displayable):
            __tablename__ = 'pa_agents'
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
        self.PAAgents = PAAgents
        self.tables[PAAgents.__tablename__] = PAAgents

        class PAMods(self.Base, Displayable):
            __tablename__ = 'pa_mods'
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
        self.PAMods = PAMods
        self.tables[PAMods.__tablename__] = PAMods

        class PAMuts(self.Base, Displayable):
            __tablename__ = 'pa_muts'
            id = Column(Integer, primary_key=True)
            stmt_mk_hash = Column(BigInteger,
                                  ForeignKey('pa_statements.mk_hash'),
                                  nullable=False)
            statements = relationship(PAStatements)
            position = Column(String(10))
            residue_from = Column(String(5))
            residue_to = Column(String(5))
            ag_num = Column(Integer, nullable=False)
        self.PAMuts = PAMuts
        self.tables[PAMuts.__tablename__] = PAMuts

        class Curation(self.Base, Displayable):
            __tablename__ = 'curation'
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
        self.Curation = Curation
        self.tables[Curation.__tablename__] = Curation

        class PASupportLinks(self.Base, Displayable):
            __tablename__ = 'pa_support_links'
            id = Column(Integer, primary_key=True)
            supporting_mk_hash = Column(BigInteger,
                                        ForeignKey('pa_statements.mk_hash'),
                                        nullable=False)
            supported_mk_hash = Column(BigInteger,
                                       ForeignKey('pa_statements.mk_hash'),
                                       nullable=False)
        self.PASupportLinks = PASupportLinks
        self.tables[PASupportLinks.__tablename__] = PASupportLinks

        class Auth(self.Base, Displayable):
            __tablename__ = 'auth'
            id = Column(Integer, primary_key=True)
            name = Column(String, unique=True)
            api_key = Column(String, unique=True)
            elsevier_access = Column(Boolean, default=False)
        self.__Auth = Auth

        # Materialized Views
        # ---------------------------------------------------------------------
        # We use materialized views to allow fast and efficient load of data,
        # and to add a layer of separation between the processes of updating
        # the content of the database and accessing the content of the
        # database. However, it is not practical to have the views created
        # through sqlalchemy: instead they are generated and updated manually
        # (or by other non-sqlalchemy scripts).
        #
        # The following views must be built in this specific order:
        #   1. fast_raw_pa_link
        #   2. evidence_counts
        #   3. pa_meta
        #   4. raw_stmt_src
        #   5. pa_stmt_src
        # The following can be built at any time and in any order:
        #   - reading_ref_link
        # Note that the order of views below is determined not by the above
        # order but by constraints imposed by use-case.

        self.m_views = {}

        class EvidenceCounts(self.Base, MaterializedView):
            __tablename__ = 'evidence_counts'
            __definition__ = ('SELECT count(id) AS ev_count, mk_hash '
                              'FROM fast_raw_pa_link '
                              'GROUP BY mk_hash')
            mk_hash = Column(BigInteger, primary_key=True)
            ev_count = Column(Integer)
        self.EvidenceCounts = EvidenceCounts
        self.m_views[EvidenceCounts.__tablename__] = EvidenceCounts

        class ReadingRefLink(self.Base, MaterializedView):
            __tablename__ = 'reading_ref_link'
            __definition__ = ('SELECT pmid, pmcid, tr.id AS trid, doi, '
                              'pii, url, manuscript_id, tc.id AS tcid, '
                              'source, r.id AS rid, reader '
                              'FROM text_ref AS tr JOIN text_content AS tc '
                              'ON tr.id = tc.text_ref_id JOIN reading AS r '
                              'ON tc.id = r.text_content_id')
            _indices = [BtreeIndex('rid_idx', 'rid')]
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
        self.ReadingRefLink = ReadingRefLink
        self.m_views[ReadingRefLink.__tablename__] = ReadingRefLink

        class FastRawPaLink(self.Base, MaterializedView):
            __tablename__ = 'fast_raw_pa_link'
            __definition__ = ('SELECT raw.id AS id, raw.json AS raw_json, '
                              'raw.reading_id, raw.db_info_id, '
                              'pa.mk_hash, pa.json AS pa_json, pa.type '
                              'FROM raw_statements AS raw, '
                              'pa_statements AS pa, '
                              'raw_unique_links AS link '
                              'WHERE link.raw_stmt_id = raw.id '
                              'AND link.pa_stmt_mk_hash = pa.mk_hash')
            _skip_disp = ['raw_json', 'pa_json']
            _indices = [BtreeIndex('hash_index', 'mk_hash')]
            id = Column(Integer, primary_key=True)
            raw_json = Column(BYTEA)
            reading_id = Column(BigInteger, ForeignKey('reading_ref_link.rid'))
            reading_ref = relationship(ReadingRefLink)
            db_info_id = Column(Integer)
            mk_hash = Column(BigInteger, ForeignKey('evidence_counts.mk_hash'))
            ev_counts = relationship(EvidenceCounts)
            pa_json = Column(BYTEA)
            type = Column(String)
        self.FastRawPaLink = FastRawPaLink
        self.m_views[FastRawPaLink.__tablename__] = FastRawPaLink

        class PaMeta(self.Base, MaterializedView):
            __tablename__ = 'pa_meta'
            __definition__ = ('SELECT pa_agents.db_name, pa_agents.db_id, '
                              'pa_agents.id AS ag_id, pa_agents.role, '
                              'pa_agents.ag_num, pa_statements.type, '
                              'pa_statements.mk_hash, evidence_counts.ev_count '
                              'FROM pa_agents, pa_statements, evidence_counts '
                              'WHERE pa_agents.stmt_mk_hash = pa_statements.mk_hash '
                              'AND pa_statements.mk_hash = evidence_counts.mk_hash')
            _indices = [StringIndex('pa_meta_db_name_idx', 'db_name'),
                        StringIndex('pa_meta_db_id_idx', 'db_id')]
            ag_id = Column(Integer, primary_key=True)
            ag_num = Column(Integer)
            db_name = Column(String)
            db_id = Column(String)
            role = Column(String(20))
            type = Column(String(100))
            mk_hash = Column(BigInteger, ForeignKey('fast_raw_pa_link.mk_hash'))
            raw_pa_link = relationship(FastRawPaLink)
            ev_count = Column(Integer)
        self.PaMeta = PaMeta
        self.m_views[PaMeta.__tablename__] = PaMeta

        class TextMeta(self.Base, NamespaceLookup):
            __tablename__ = 'text_meta'
            __dbname__ = 'TEXT'
            _indices = [StringIndex('text_meta_db_id_idx', 'db_id'),
                        StringIndex('text_meta_type_idx', 'type')]
            ag_id = Column(Integer, primary_key=True)
            ag_num = Column(Integer)
            db_id = Column(String)
            role = Column(String(20))
            type = Column(String(100))
            mk_hash = Column(BigInteger, ForeignKey('fast_raw_pa_link.mk_hash'))
            raw_pa_link = relationship(FastRawPaLink)
            ev_count = Column(Integer)
        self.TextMeta = TextMeta
        self.m_views[TextMeta.__tablename__] = TextMeta

        class NameMeta(self.Base, NamespaceLookup):
            __tablename__ = 'name_meta'
            __dbname__ = 'NAME'
            _indices = [StringIndex('name_meta_db_id_idx', 'db_id'),
                        StringIndex('name_meta_type_idx', 'type')]
            ag_id = Column(Integer, primary_key=True)
            ag_num = Column(Integer)
            db_id = Column(String)
            role = Column(String(20))
            type = Column(String(100))
            mk_hash = Column(BigInteger, ForeignKey('fast_raw_pa_link.mk_hash'))
            raw_pa_link = relationship(FastRawPaLink)
            ev_count = Column(Integer)
        self.NameMeta = NameMeta
        self.m_views[NameMeta.__tablename__] = NameMeta

        class OtherMeta(self.Base, MaterializedView):
            __tablename__ = 'other_meta'
            __definition__ = ("SELECT db_name, db_id, ag_id, role, ag_num, "
                              "type, mk_hash, ev_count FROM pa_meta "
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
            mk_hash = Column(BigInteger, ForeignKey('fast_raw_pa_link.mk_hash'))
            raw_pa_link = relationship(FastRawPaLink)
            ev_count = Column(Integer)
        self.OtherMeta = OtherMeta
        self.m_views[OtherMeta.__tablename__] = OtherMeta

        class RawStmtSrc(self.Base, MaterializedView):
            __tablename__ = 'raw_stmt_src'
            __definition__ = ('SELECT raw_statements.id AS sid, '
                              'lower(reading.reader) AS src '
                              'FROM raw_statements, reading '
                              'WHERE reading.id = raw_statements.reading_id '
                              'UNION '
                              'SELECT raw_statements.id AS sid, '
                              'lower(db_info.db_name) AS src '
                              'FROM raw_statements, db_info '
                              'WHERE db_info.id = raw_statements.db_info_id')
            sid = Column(Integer, primary_key=True)
            src = Column(String)
        self.RawStmtSrc = RawStmtSrc
        self.m_views[RawStmtSrc.__tablename__] = RawStmtSrc

        class PaStmtSrc(self.Base, MaterializedView):
            __tablename__ = 'pa_stmt_src'
            __definition_fmt__ = ("SELECT * FROM crosstab("
                                  "'SELECT mk_hash, src, count(sid) "
                                  "  FROM raw_stmt_src "
                                  "   JOIN fast_raw_pa_link ON sid = id "
                                  "  GROUP BY (mk_hash, src)', "
                                  "$$SELECT unnest('{%s}'::text[])$$"
                                  " ) final_result(mk_hash bigint, %s)")
            loaded = False

            @classmethod
            def definition(cls, db):
                db.grab_session()
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

            @classmethod
            def create(cls, db, with_data=True, commit=True):
                # Make sure the necessary extension is installed.
                with db.engine.connect() as conn:
                    conn.execute('CREATE EXTENSION IF NOT EXISTS tablefunc;')

                # Create the materialized view.
                cls.__definition__ = cls.definition(db)
                sql = super(cls, cls).create(db, with_data, commit)
                cls.loaded = True
                return sql

            @classmethod
            def update(cls, db, with_data=True, commit=True):
                # Make sure the necessary extension is installed.
                with db.engine.connect() as conn:
                    conn.execute('CREATE EXTENSION IF NOT EXISTS tablefunc;')

                # Drop the table entirely and replace it because the columns
                # may have changed.
                sql_fmt = 'DROP MATERIALIZED VIEW IF EXISTS %s; %s'
                create_sql = cls.create(db, with_data, commit=False)
                sql = sql_fmt % (cls.__tablename__, create_sql)
                cls.execute(db, sql)
                cls.load_cols(db.engine)
                return sql

            @classmethod
            def load_cols(cls, engine):
                if cls.loaded:
                    return

                try:
                    cols = inspect(engine).get_columns('pa_stmt_src')
                except NoSuchTableError:
                    return

                for col in cols:
                    if col['name'] == 'mk_hash':
                        continue

                    setattr(cls, col['name'], Column(BigInteger))

                cls.loaded = True
                return

            mk_hash = Column(BigInteger, primary_key=True)
        self.__PaStmtSrc = PaStmtSrc
        self.m_views[PaStmtSrc.__tablename__] = PaStmtSrc

        self.engine = create_engine(host)

        # There are some useful shortcuts that can be used if
        # networkx is available, specifically the DatabaseManager.link
        if WITH_NX:
            G = nx.Graph()
            G.add_edges_from([
                ('pa_agents', 'pa_statements'),
                ('raw_unique_links', 'pa_statements'),
                ('raw_unique_links', 'raw_statements'),
                ('raw_statements', 'reading'),
                ('raw_agents', 'raw_statements'),
                ('raw_statements', 'db_info'),
                ('reading', 'text_content'),
                ('text_content', 'text_ref')
                ])
            self.__foreign_key_graph = G
        else:
            self.__foreign_key_graph = None

        self._conn = None

        return

    def __del__(self, *args, **kwargs):
        try:
            self.grab_session()
            self.session.rollback()
        except:
            print("Failed to execute rollback of database upon deletion.")

    def __getattribute__(self, item):
        if item == 'PaStmtSrc':
            self.__PaStmtSrc.load_cols(self.engine)
            return self.__PaStmtSrc
        return super(DatabaseManager, self).__getattribute__(item)

    def _init_auth(self):
        """Create the auth table."""
        self.__Auth.__table__.create(bind=self.engine)

    def _get_auth(self, api_key):
        res = self.select_all(self.__Auth, self.__Auth.api_key == api_key)
        if not res:
            return None
        return res[0]

    def _get_auth_info(self, api_key):
        """Check if an api key is valid."""
        if api_key is None:
            return None
        auth = self._get_auth(api_key)
        if auth is None:
            return None
        else:
            return auth.id, auth.name

    def _get_api_key(self, name):
        """Get an API key from the username."""
        if name is None:
            return None
        api_key = self.select_one(self.__Auth, self.__Auth.name == name)
        if not api_key:
            return None
        return api_key

    def _add_auth(self, name, elsevier_access=False):
        """Add a new api key to the database."""
        new_uuid = str(uuid4())
        dbid = self.insert(self.__Auth, api_key=new_uuid, name=name,
                           elsevier_access=elsevier_access)
        return dbid, new_uuid

    def create_tables(self, tbl_list=None):
        "Create the tables for INDRA database."
        ordered_tables = ['text_ref', 'text_content', 'reading', 'db_info',
                          'raw_statements', 'raw_agents', 'raw_mods',
                          'raw_muts', 'pa_statements', 'pa_agents', 'pa_mods',
                          'pa_muts', 'raw_unique_links', 'support_links']
        if tbl_list is None:
            tbl_list = list(self.tables.keys())

        tbl_name_list = []
        for tbl in tbl_list:
            if isinstance(tbl, str):
                tbl_name_list.append(tbl)
            else:
                tbl_name_list.append(tbl.__tablename__)
        # These tables must be created in this order.
        for tbl_name in ordered_tables:
            if tbl_name in tbl_name_list:
                tbl_name_list.remove(tbl_name)
                logger.debug("Creating %s..." % tbl_name)
                if not self.tables[tbl_name].__table__.exists(self.engine):
                    self.tables[tbl_name].__table__.create(bind=self.engine)
                    logger.debug("Table created.")
                else:
                    logger.debug("Table already existed.")
        # The rest can be started any time.
        for tbl_name in tbl_name_list:
            logger.debug("Creating %s..." % tbl_name)
            self.tables[tbl_name].__table__.create(bind=self.engine)
            logger.debug("Table created.")
        return

    def manage_views(self, mode, view_list=None, with_data=True):
        """Manage the materialized views.

        Parameters
        ----------
        mode : 'create' or 'update'
            Select which management task you wish to perform.
        view_list : list or None
            Default None. A list of materialized view names or None. If None,
            all available views will be build.
        with_data : bool
            Default True. If True, the views are updated "with data", meaning
            they are more like instantiated tables, otherwise they are only a
            pre-computation.
        """
        ordered_views = ['fast_raw_pa_link', 'evidence_counts', 'pa_meta',
                         'name_meta', 'text_meta', 'other_meta',
                         'raw_stmt_src', 'pa_stmt_src']
        other_views = {'reading_ref_link'}
        active_views = self.get_active_views()

        def iter_views():
            for i, view in enumerate(ordered_views):
                yield str(i), view
            for view in other_views:
                yield '-', view

        for i, view_name in iter_views():
            if view_list is not None and view_name not in view_list:
                continue

            view = self.m_views[view_name]

            if mode == 'create':
                if view_name in active_views:
                    logger.info('[%s] View %s already exists. Skipping.'
                                % (i, view_name))
                    continue
                logger.info('[%s] Creating %s view...' % (i, view_name))
                view.create(self, with_data)
            elif mode == 'update':
                logger.info('[%s] Updating %s view...' % (i, view_name))
                view.update(self, with_data)
            else:
                raise ValueError("Invalid mode: %s." % mode)
        return

    def drop_tables(self, tbl_list=None, force=False):
        """Drop the tables for INDRA database given in tbl_list.

        If tbl_list is None, all tables will be dropped. Note that if `force`
        is False, a warning prompt will be raised to asking for confirmation,
        as this action will remove all data from that table.
        """
        if tbl_list is not None:
            tbl_objs = []
            for tbl in tbl_list:
                if isinstance(tbl, str):
                    tbl_objs.append(self.tables[tbl])
                else:
                    tbl_objs.append(tbl)
        if not force:
            # Build the message
            if tbl_list is None:
                msg = ("Do you really want to clear the %s database? [y/N]: "
                       % self.label)
            else:
                msg = "You are going to clear the following tables:\n"
                msg += str([tbl.__tablename__ for tbl in tbl_objs]) + '\n'
                msg += ("Do you really want to clear these tables from %s? "
                        "[y/N]: " % self.label)
            # Check to make sure.
            resp = input(msg)
            if resp != 'y' and resp != 'yes':
                logger.info('Aborting clear.')
                return False
        if tbl_list is None:
            logger.info("Removing all materialized views...")
            with self.engine.connect() as conn:
                conn.execute('DROP MATERIALIZED VIEW IF EXISTS %s CASCADE;'
                             % (', '.join(self.m_views.keys())))
            logger.info("Removing all tables...")
            self.Base.metadata.drop_all(self.engine)
            logger.debug("All tables removed.")
        else:
            for tbl in tbl_list:
                logger.info("Removing %s..." % tbl.__tablename__)
                if tbl in self.m_views.values():
                    with self.engine.connect() as conn:
                        conn.execute('DROP MATERIALIZED VIEW IF EXISTS %s '
                                     'CASCADE;' % tbl.__tablename__)
                elif tbl.__table__.exists(self.engine):
                    tbl.__table__.drop(self.engine)
                    logger.debug("Table removed.")
                else:
                    logger.debug("Table doesn't exist.")
        return True

    def _clear(self, tbl_list=None, force=False):
        "Brutal clearing of all tables in tbl_list, or all tables."
        # This is intended for testing purposes, not general use.
        # Use with care.
        self.grab_session()
        logger.debug("Rolling back before clear...")
        self.session.rollback()
        logger.debug("Rolled back.")
        if self.drop_tables(tbl_list, force=force):
            self.create_tables(tbl_list)
            return True
        else:
            return False

    def grab_session(self):
        "Get an active session with the database."
        if self.session is None or not self.session.is_active:
            logger.debug('Attempting to get session...')
            DBSession = sessionmaker(bind=self.engine)
            logger.debug('Got session.')
            self.session = DBSession()
            if self.session is None:
                raise IndraDbException("Failed to grab session.")

    def get_tables(self):
        "Get a list of available tables."
        return [tbl_name for tbl_name in self.tables.keys()]

    def show_tables(self):
        "Print a list of all the available tables."
        print(self.get_tables())

    def get_active_tables(self):
        "Get the tables currently active in the database."
        return inspect(self.engine).get_table_names()

    def get_active_views(self):
        return inspect(self.engine).get_view_names()

    def get_column_names(self, tbl_name):
        "Get a list of the column labels for a table."
        return self.get_column_objects(tbl_name).keys()

    def get_column_objects(self, table):
        'Get a list of the column object for the given table.'
        if isinstance(table, type(self.Base)):
            table = table.__tablename__
        return self.Base.metadata.tables[table].columns

    def commit(self, err_msg):
        "Commit, and give useful info if there is an exception."
        try:
            logger.debug('Attempting to commit...')
            self.session.commit()
            logger.debug('Message committed.')
        except Exception as e:
            if self.session is not None:
                logger.error('Got exception in commit, rolling back...')
                self.session.rollback()
                logger.debug('Rolled back.')
            logger.exception(e)
            logger.error(err_msg)
            raise

    def commit_copy(self, err_msg):
        if self._conn is not None:
            try:
                logger.debug('Attempting to commit...')
                self._conn.commit()
                self._conn = None
                logger.debug('Message committed.')
            except Exception as e:
                self._conn = None
                logger.exception(e)
                logger.error(err_msg)
                raise

    def _get_foreign_key_constraint(self, table_name_1, table_name_2):
        cols = self.get_column_objects(self.tables[table_name_1])
        ret = None
        for col in cols:
            for fk in col.foreign_keys:
                target_table_name, target_col = fk.target_fullname.split('.')
                if table_name_2 == target_table_name:
                    ret = (col == getattr(self.tables[table_name_2],
                                          target_col))
                    break
        return ret

    def link(self, table_1, table_2):
        """Get the joining clause between two tables, if one exists.

        If no link exists, an exception will be raised. Note that this only
        works for directly links.
        """
        table_name_1 = table_1.__tablename__
        table_name_2 = table_2.__tablename__
        if WITH_NX:
            fk_path = nx.shortest_path(self.__foreign_key_graph, table_name_1,
                                       table_name_2)
        else:
            fk_path = [table_name_1, table_name_2]

        links = []
        for i in range(len(fk_path) - 1):
            link = self._get_foreign_key_constraint(fk_path[i], fk_path[i+1])
            if link is None:
                link = self._get_foreign_key_constraint(fk_path[i+1],
                                                        fk_path[i])
            if link is None:
                raise IndraDbException("There is no foreign key in %s "
                                         "pointing to %s."
                                       % (table_name_1, table_name_2))
            links.append(link)
        return links

    def get_values(self, entry_list, col_names=None, keyed=False):
        "Get the column values from the entries in entry_list"
        if col_names is None and len(entry_list) > 0:  # Get everything.
            col_names = self.get_column_names(entry_list[0].__tablename__)
        ret = []
        for entry in entry_list:
            if _isiterable(col_names):
                if not keyed:
                    ret.append([getattr(entry, col) for col in col_names])
                else:
                    ret.append({col: getattr(entry, col) for col in col_names})
            else:
                ret.append(getattr(entry, col_names))
        return ret

    def insert(self, table, ret_info=None, **input_dict):
        "Insert a an entry into specified table, and return id."
        self.grab_session()
        # Resolve the table instance
        if isinstance(table, str):
            inputs = dict.fromkeys(self.get_column_names(table))
            table = self.tables[table]
        else:
            inputs = dict.fromkeys(self.get_column_names(table.__tablename__))

        # Get the default return info
        if ret_info is None:
            ret_info = inspect(table).primary_key[0].name

        # Do the insert
        inputs.update(input_dict)
        new_entry = table(**inputs)
        self.session.add(new_entry)
        self.commit("Excepted while trying to insert %s into %s" %
                    (inputs, table.__tablename__))
        return self.get_values([new_entry], ret_info)[0]

    def insert_many(self, table, input_data_list, ret_info=None, cols=None):
        "Insert many records into the table given by table_name."
        self.grab_session()

        # Resolve the table instance
        if isinstance(table, str):
            inputs = dict.fromkeys(self.get_column_names(table))
            table = self.tables[table]
        else:
            inputs = dict.fromkeys(self.get_column_names(table.__tablename__))

        # Set the default return info
        if ret_info is None:
            ret_info = inspect(table).primary_key[0].name

        # Prepare and insert the data
        entry_list = []
        for input_data in input_data_list:
            if cols:
                input_dict = zip(cols, input_data)
            else:
                input_dict = input_data
            inputs.update(input_dict)
            entry_list.append(table(**inputs))
            inputs = inputs.fromkeys(inputs)  # Clear the values of the dict.
        self.session.add_all(entry_list)
        self.commit("Excepted while trying to insert:\n%s,\ninto %s" %
                    (input_data_list, table.__tablename__))
        return self.get_values(entry_list, ret_info)

    def delete_all(self, entry_list):
        "Remove the given records from the given table."
        self.grab_session()
        for entry in entry_list:
            self.session.delete(entry)
        self.commit("Could not remove %d records from the database." %
                    len(entry_list))
        return

    def make_copy_batch_id(self):
        """Generate a random batch id for copying into the database.

        This allows for easy retrieval of the assigned ids immediately after
        copying in. At this time, only Reading and RawStatements use the
        feature.
        """
        return random.randint(-2**30, 2**30)

    def copy(self, tbl_name, data, cols=None, lazy=False, push_conflict=False,
             constraint=None, commit=True):
        "Use pg_copy to copy over a large amount of data."
        logger.info("Received request to copy %d entries into %s." %
                    (len(data), tbl_name))
        if len(data) is 0:
            return  # Nothing to do....

        # If cols is not specified, use all the cols in the table, else check
        # to make sure the names are valid.
        if cols is None:
            cols = self.get_column_names(tbl_name)
        else:
            db_cols = self.get_column_names(tbl_name)
            assert all([col in db_cols for col in cols]),\
                "Do not recognize one of the columns in %s for table %s." % \
                (cols, tbl_name)

        # Do the copy. Use pgcopy if available.
        if self.sqltype == sqltypes.POSTGRESQL and CAN_COPY:
            # Check for automatic timestamps which won't be applied by the
            # database when using copy, and manually insert them.
            auto_timestamp_type = type(func.now())
            for col in self.get_column_objects(tbl_name):
                if col.default is not None:
                    if isinstance(col.default.arg, auto_timestamp_type) \
                       and col.name not in cols:
                        logger.info("Applying timestamps to %s." % col.name)
                        now = datetime.utcnow()
                        cols += (col.name,)
                        data = [datum + (now,) for datum in data]

            # Format the data for the copy.
            data_bts = []
            for entry in data:
                new_entry = []
                for element in entry:
                    if isinstance(element, str):
                        new_entry.append(element.encode('utf8'))
                    elif (isinstance(element, bytes)
                          or element is None
                          or isinstance(element, Number)
                          or isinstance(element, datetime)):
                        new_entry.append(element)
                    else:
                        raise IndraDbException(
                            "Don't know what to do with element of type %s."
                            "Should be str, bytes, datetime, None, or a "
                            "number." % type(element)
                            )
                data_bts.append(tuple(new_entry))

            # Actually do the copy.
            if self._conn is None:
                self._conn = self.engine.raw_connection()
                self._conn.rollback()

            if lazy:
                # We need a constraint for if we are going to update on-
                # conflict, so if we didn't get a constraint, we can try to
                # guess it.
                if push_conflict and constraint is None:
                    tbl = self.tables[tbl_name]

                    # Look for table arguments that are constraints, and
                    # moreover that involve a subset of the columns being
                    # copied. If the column isn't in the input data, it can't
                    # possibly violate a constraint. It is also because of this
                    # line of code that constraints MUST be named. This process
                    # will not catch foreign key constraints, which may not
                    # even apply.
                    constraints = [c.name for c in tbl.__table_args__
                                   if isinstance(c, UniqueConstraint)
                                   and set(c.columns.keys()) < set(cols)]

                    # Include the primary key in the list, if applicable.
                    if inspect(tbl).primary_key[0].name in cols:
                        constraints.append(tbl_name + '_pkey')

                    # Hopefully at this point there is
                    if len(constraints) > 1:
                        raise ValueError("Cannot infer constraint. Only "
                                         "one constraint is allowed, and "
                                         "there are multiple "
                                         "possibilities. Please specify a "
                                         "single constraint.")
                    elif len(constraints) == 1:
                        constraint = constraints[0]
                    else:
                        raise ValueError("Could not infer a relevant "
                                         "constraint. If no columns have "
                                         "constraints on them, the lazy "
                                         "option is unnecessary. Note that I "
                                         "cannot guess a foreign key "
                                         "constraint.")

                mngr = LazyCopyManager(self._conn, tbl_name, cols,
                                       push_conflict=push_conflict,
                                       constraint=constraint)
                mngr.copy(data_bts, BytesIO)
            else:
                mngr = CopyManager(self._conn, tbl_name, cols)
                mngr.copy(data_bts, BytesIO)
            if commit:
                self._conn.commit()
                self._conn = None
        else:
            # TODO: use bulk insert mappings?
            logger.warning("You are not using postgresql or do not have "
                           "pgcopy, so this will likely be very slow.")
            self.insert_many(tbl_name, [dict(zip(cols, ro)) for ro in data])
        return

    def generate_materialized_view(self, mode, view, with_data=True):
        """Create or refresh the given materialized view."""

    def filter_query(self, tbls, *args):
        "Query a table and filter results."
        self.grab_session()
        ok_classes = [type(self.Base), InstrumentedAttribute]
        if _isiterable(tbls) and not isinstance(tbls, dict):
            if all([any([isinstance(tbl, ok_class) for ok_class in ok_classes])
                    for tbl in tbls]):
                query_args = tbls
            elif isinstance(tbls[0], str):
                query_args = [self.tables[tbl] for tbl in tbls]
            else:
                raise IndraDbException(
                    'Unrecognized table specification type: %s.' %
                    type(tbls[0])
                    )
        else:
            if any([isinstance(tbls, ok_class) for ok_class in ok_classes]):
                query_args = [tbls]
            elif isinstance(tbls, str):
                query_args = [self.tables[tbls]]
            else:
                raise IndraDbException(
                    'Unrecognized table specification type: %s.' %
                    type(tbls)
                    )

        return self.session.query(*query_args).filter(*args)

    def count(self, tbl, *args):
        """Get a count of the results to a query."""
        if isinstance(tbl, list):
            assert len(tbl) == 1, "Only one table can be counted at a time."
            tbl = tbl[0]
        if isinstance(tbl, DeclarativeMeta):
            tbl = self.get_primary_key(tbl)
        q = self.session.query(func.count(tbl)).filter(*args)
        res = q.all()
        assert len(res) == 1
        assert len(res[0]) == 1
        return res[0][0]

    def get_primary_key(self, tbl):
        """Get an instance for the primary key column of a given table."""
        return inspect(tbl).primary_key[0]

    def select_one(self, tbls, *args):
        """Select the first value that matches requirements.

        Requirements are given in kwargs from table indicated by tbl_name. See
        `select_all`.

        Note that if your specification yields multiple results, this method
        will just return the first result without exception.
        """
        return self.filter_query(tbls, *args).first()

    def select_all(self, tbls, *args, **kwargs):
        """Select any and all entries from table given by tbl_name.

        The results will be filtered by your keyword arguments. For example if
        you want to get a text ref with pmid '10532205', you would call:

        .. code-block:: python

            db.select_all('text_ref', db.TextRef.pmid == '10532205')

        Note that double equals are required, not a single equal. Equivalently
        you could call:

        .. code-block:: python

            db.select_all(db.TextRef, db.TextRef.pmid == '10532205')

        For a more complicated example, suppose you want to get all text refs
        that have full text from pmc oa, you could select:

        .. code-block:: python

           db.select_all(
               [db.TextRef, db.TextContent],
               db.TextContent.text_ref_id == db.TextRef.id,
               db.TextContent.source == 'pmc_oa',
               db.TextContent.text_type == 'fulltext'
               )

        Parameters
        ----------
        tbls, *args
            See above for usage.
        **kwargs
            yield_per: int or None
            If the result to your query is expected to be large, you can choose
            to only load `yield_per` items at a time, using the eponymous
            feature of sqlalchemy queries. Default is None, meaning all results
            will be loaded simultaneously.

        Returns
        -------

        """
        yield_per = kwargs.get('yield_per')
        if yield_per is not None:
            return self.filter_query(tbls, *args).yield_per(yield_per)
        return self.filter_query(tbls, *args).all()

    def select_all_batched(self, batch_size, tbls, *args, skip_idx=None,
                           order_by=None):
        """Load the results of a query in batches of size batch_size.

        Note that this differs from using yeild_per in that the results are not
        returned as a single iterable, but as an iterator of iterables.

        Note also that the order of results, and thus the contents of offsets,
        may vary for large queries unless an explicit order_by clause is added
        to the query.
        """
        q = self.filter_query(tbls, *args)
        if order_by:
            q = q.order_by(order_by)
        res_iter = q.yield_per(batch_size)
        for i, batch in enumerate(batch_iter(res_iter, batch_size)):
            if i != skip_idx:
                yield i, batch

    def select_sample_from_table(self, number, table, *args, **kwargs):
        """Select a number of random samples from the given table.

        Parameters
        ----------
        number : int
            The number of samples to return
        table : str, table class, or column attribute of table class
            The table or table column to be sampled.
        *args, **kwargs :
            All other arguments are passed to `select_all`, including any and
            all filtering clauses.

        Returns
        -------
        A list of sqlalchemy orm objects
        """
        # Get the base set of tables needed.
        if isinstance(table, str):
            # This should be the string name for a table or view.
            if table in self.tables.keys() or table in self.m_views.keys():
                true_table = getattr(self, table)
            else:
                raise IndraDbException("Invalid table name: %s." % table)
        elif hasattr(table, 'class_'):
            # This is technically an attribute of a table.
            true_table = table.class_
        elif table in self.tables.values() or table in self.m_views.values():
            # This is an actual table object
            true_table = table
        else:
            raise IndraDbException("Unrecognized table: %s of type %s"
                                   % (table, type(table)))

        # Get all ids for this table given query filters
        logger.info("Getting all relevant ids.")
        pk = self.get_primary_key(true_table)
        id_tuples = self.select_all(getattr(true_table, pk.name), *args,
                                    **kwargs)
        id_list = list({entry_id for entry_id, in id_tuples})

        # Sample from the list of ids
        logger.info("Getting sample of %d from %d members."
                    % (number, len(id_list)))
        id_sample = random.sample(id_list, number)
        if hasattr(table, 'key') and table.key == 'id':
            return [(entry_id,) for entry_id in id_sample]

        return self.select_all(table, getattr(table, pk.name).in_(id_sample))

    def has_entry(self, tbls, *args):
        "Check whether an entry/entries matching given specs live in the db."
        q = self.filter_query(tbls, *args)
        return self.session.query(q.exists()).first()[0]


class LazyCopyManager(CopyManager):
    """A copy manager that ignores entries which violate constraints."""
    def __init__(self, conn, table, cols, push_conflict=False,
                 constraint=None):
        super(LazyCopyManager, self).__init__(conn, table, cols)
        if push_conflict and constraint is None:
            raise ValueError("A constraint is required if you are updating "
                             "on-conflict.")
        self.push_conflict = push_conflict
        self.constraint = constraint
        return

    def copystream(self, datastream):
        cmd_fmt = ('CREATE TEMP TABLE "tmp_{table}" '
                   'ON COMMIT DROP '
                   'AS SELECT "{cols}" FROM "{schema}"."{table}" '
                   'WITH NO DATA; '
                   '\n'
                   'COPY "tmp_{table}" ("{cols}") '
                   'FROM STDIN WITH BINARY; '
                   '\n'
                   'INSERT INTO "{schema}"."{table}" ("{cols}") '
                   'SELECT "{cols}" '
                   'FROM "tmp_{table}" ')
        cmd_fmt += 'ON CONFLICT '
        if self.push_conflict:
            update = ', '.join('{0} = EXCLUDED.{0}'.format(c)
                               for c in self.cols)
            cmd_fmt += 'ON CONSTRAINT "%s" DO UPDATE SET %s;' \
                       % (self.constraint, update)
        else:
            if self.constraint:
                cmd_fmt += 'ON CONSTRAINT "%s" ' % self.constraint
            cmd_fmt += 'DO NOTHING;'
        columns = '", "'.join(self.cols)
        sql = cmd_fmt.format(schema=self.schema,
                             table=self.table,
                             cols=columns)
        print(sql)
        cursor = self.conn.cursor()
        try:
            cursor.copy_expert(sql, datastream)
        except Exception as e:
            templ = "error doing lazy binary copy into {0}.{1}:\n{2}"
            e.message = templ.format(self.schema, self.table, e)
            raise e
