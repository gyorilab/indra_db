import logging
from termcolor import colored
from psycopg2.errors import DuplicateTable
from sqlalchemy import inspect, Column, BigInteger, tuple_, and_
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm.attributes import InstrumentedAttribute

logger = logging.getLogger(__name__)


class DbIndexError(Exception):
    pass


class IndraDBTableMetaClass(type):
    """This serves as a meta class for all tables, allowing `str` to be useful.

    In particular, this makes it so that the string gives a representation of
    the SQL table, including columns.
    """
    def __init__(cls, *args, **kwargs):
        assert hasattr(cls, 'full_name'), \
            ("Class using metaclass IndraDBTableMetaClass missing critical "
             "class method: `full_name`")
        super(IndraDBTableMetaClass, cls).__init__(*args, **kwargs)

    def __str__(cls):
        col_names = [colored(attr_name, 'magenta')
                     for attr_name, attr_val in cls.__dict__.items()
                     if isinstance(attr_val, Column)
                     or isinstance(attr_val, InstrumentedAttribute)]
        cols = '\n  '
        cols += ', '.join(col_names)
        cols += '\n'
        full_name = colored(cls.full_name(force_schema=True), attrs=['bold'])
        return f"{full_name}({cols})"


class IndraDBTable(metaclass=IndraDBTableMetaClass):
    _indices = []
    _skip_disp = []
    _always_disp = ['id']
    _default_insert_order_by = 'id'

    @classmethod
    def create_index(cls, db, index, commit=True):
        full_name = cls.full_name(force_schema=True)
        sql = (f"CREATE INDEX {index.name} ON {full_name} "
               f"USING {index.definition} TABLESPACE pg_default;")
        if commit:
            try:
                cls.execute(db, sql)
            except DuplicateTable:
                logger.info("%s exists, skipping." % index.name)

            if index.cluster:
                cluster_sql = f"CLUSTER {full_name} USING {index.name};"
                cls.execute(db, cluster_sql)
        return sql

    @classmethod
    def build_indices(cls, db):
        found_a_clustered = False
        for index in cls._indices:
            if index.cluster:
                if not found_a_clustered:
                    found_a_clustered = True
                else:
                    raise DbIndexError("Only one index may be clustered "
                                       "at a time.")
            logger.info("Building index: %s" % index.name)
            cls.create_index(db, index)

    @staticmethod
    def execute(db, sql):
        conn = db.get_raw_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        return

    @classmethod
    def get_schema(cls, default=None):
        """Get the schema of this table."""
        if not hasattr(cls, '__table_args__'):
            return default

        if isinstance(cls.__table_args__, dict):
            return cls.__table_args__.get('schema')
        elif isinstance(cls.__table_args__, tuple):
            for arg in cls.__table_args__:
                if isinstance(arg, dict) and 'schema' in arg.keys():
                    return arg['schema']

        return default

    @classmethod
    def full_name(cls, force_schema=False):
        """Get the full name including the schema, if supplied."""
        name = cls.__tablename__

        # If we are definitely going to include the schema, default to public
        schema_name = cls.get_schema('public' if force_schema else None)

        # Prepend if we found something.
        if schema_name:
            name = schema_name + '.' + name

        return name

    def _make_str(self):
        s = self.full_name() + ':\n'
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

    def __repr__(self):
        ret = self.__class__.__name__ + '('

        entries = []
        for attr_name in self._always_disp:
            attr = getattr(self, attr_name)
            if isinstance(attr, str):
                fmt = '%s="%s"'
            elif isinstance(attr, bytes):
                fmt = '%s=b"%s"'
            else:
                fmt = '%s=%s'

            entries.append(fmt % (attr_name, attr))

        ret += ', '.join(entries)
        ret += ')'
        return ret


class ReadonlyTable(IndraDBTable):
    __definition__ = NotImplemented
    __table_args__ = NotImplemented
    __create_table_fmt__ = "CREATE TABLE IF NOT EXISTS %s AS %s;"

    # These tables are created all at once, so there isn't really an "order" to
    # which entries were inserted. They were inserted all at once.
    _default_insert_order_by = NotImplemented

    # Some tables may mark themselves as "temp", meaning they are not intended
    # to live beyond the readonly build process.
    _temp = False

    @classmethod
    def create(cls, db, commit=True):
        sql = cls.__create_table_fmt__ \
              % (cls.full_name(force_schema=True),
                 cls.get_definition())
        if commit:
            cls.execute(db, sql)
        return sql

    @classmethod
    def get_definition(cls):
        return cls.__definition__

    @classmethod
    def definition(cls, db):
        return cls.get_definition()


class SpecialColumnTable(ReadonlyTable):

    @classmethod
    def create(cls, db, commit=True):
        cls.__definition__ = cls.definition(db)
        sql = cls.__create_table_fmt__ \
            % (cls.full_name(force_schema=True),
               cls.__definition__)
        if commit:
            cls.execute(db, sql)
        cls.loaded = True
        return sql

    @classmethod
    def load_cols(cls, engine, cols=None):
        if cls.loaded:
            return

        if cols is None:
            try:
                schema = cls.__table_args__.get('schema', 'public')
                cols = inspect(engine).get_columns(cls.__tablename__,
                                                   schema=schema)
            except NoSuchTableError:
                return

        existing_cols = {col.name for col in cls.__table__.columns}
        for col in cols:
            if col['name'] in existing_cols:
                continue

            setattr(cls, col['name'], Column(BigInteger))

        cls.loaded = True
        return


class NamespaceLookup(ReadonlyTable):
    __dbname__ = NotImplemented

    @classmethod
    def get_definition(cls):
        return ("SELECT db_id, ag_id, role_num, ag_num, type_num,\n"
                "       mk_hash, ev_count, belief, activity, is_active,\n"
                "       agent_count, is_complex_dup\n"
                "FROM readonly.pa_meta\n"
                "WHERE db_name = '%s'" % cls.__dbname__)


class IndraDBRefTable:
    """Define an API and methods for a table of text references."""
    pmid = NotImplemented
    pmid_num = NotImplemented
    pmcid = NotImplemented
    pmcid_num = NotImplemented
    pmcid_version = NotImplemented
    doi = NotImplemented
    doi_ns = NotImplemented
    doi_id = NotImplemented

    @staticmethod
    def process_pmid(pmid):
        if not pmid:
            return None, None

        if not pmid.isdigit():
            return pmid, None

        return pmid, int(pmid)

    @staticmethod
    def process_pmcid(pmcid):
        if not pmcid:
            return None, None, None

        if not pmcid.startswith('PMC'):
            return pmcid, None, None

        if '.' in pmcid:
            pmcid, version_number_str = pmcid.split('.')
            if version_number_str.isdigit():
                version_number = int(version_number_str)
            else:
                version_number = None
        else:
            version_number = None

        if not pmcid[3:].isdigit():
            return pmcid, None, version_number

        return pmcid, int(pmcid[3:]), version_number

    @staticmethod
    def process_doi(doi):
        # Check for invalid DOIs
        if not doi:
            return None, None, None

        # Regularize case.
        doi = doi.upper()

        if not doi.startswith('10.'):
            return doi, None, None

        # Split up the parts of the DOI
        parts = doi[3:].split('/')
        if len(parts) < 2:
            return doi, None, None

        # Check the namespace number, make it an integer.
        namespace_str = parts[0]
        if not namespace_str.isdigit():
            return doi, None, None
        namespace = int(namespace_str)

        # Join the res of the parts together.
        group_id = '/'.join(parts[1:])

        return doi, namespace, group_id

    @classmethod
    def pmid_in(cls, pmid_list, filter_ids=False):
        """Get sqlalchemy clauses for a list of pmids."""
        # Process the ID list.
        pmid_num_set = set()
        for pmid in pmid_list:
            _, pmid_num = cls.process_pmid(pmid)
            if pmid_num is None:
                if filter_ids:
                    logger.warning('"%s" is not a valid pmid. Skipping.'
                                   % pmid)
                    continue
                else:
                    ValueError('"%s" is not a valid pmid.' % pmid)
            pmid_num_set.add(pmid_num)

        # Return the constraint
        if len(pmid_num_set) == 1:
            return cls.pmid_num == pmid_num_set.pop()
        else:
            return cls.pmid_num.in_(pmid_num_set)

    @classmethod
    def pmcid_in(cls, pmcid_list, filter_ids=False):
        """Get the sqlalchemy clauses for a list of pmcids."""
        # Process the ID list.
        pmcid_num_set = set()
        for pmcid in pmcid_list:
            _, pmcid_num, _ = cls.process_pmcid(pmcid)
            if not pmcid_num:
                if filter_ids:
                    logger.warning('"%s" does not look like a valid '
                                   'pmcid. Skipping.' % pmcid)
                    continue
                else:
                    raise ValueError('"%s" is not a valid pmcid.' % pmcid)
            else:
                pmcid_num_set.add(pmcid_num)

        # Return the constraint
        if len(pmcid_num_set) == 1:
            return cls.pmcid_num == pmcid_num_set.pop()
        else:
            return cls.pmcid_num.in_(pmcid_num_set)

    @classmethod
    def doi_in(cls, doi_list, filter_ids=False):
        """Get clause for looking up a list of dois."""
        # Parse the DOIs in the list.
        doi_tuple_set = set()
        for doi in doi_list:
            doi, doi_ns, doi_id = cls.process_doi(doi)
            if not doi_ns:
                if filter_ids:
                    logger.warning('"%s" does not look like a normal doi. '
                                   'Skipping.' % doi)
                    continue
                else:
                    raise ValueError('"%s" is not a valid doi.' % doi)
            else:
                doi_tuple_set.add((doi_ns, doi_id))

        # Return the constraint
        if len(doi_tuple_set) == 1:
            doi_ns, doi_id = doi_tuple_set.pop()
            return and_(cls.doi_ns == doi_ns, cls.doi_id == doi_id)
        else:
            return tuple_(cls.doi_ns, cls.doi_id).in_(doi_tuple_set)

    @classmethod
    def has_ref(cls, id_type, id_list, filter_ids=False):
        """Get the appropriate constraint for the given ID list."""
        if id_type == 'pmid':
            return cls.pmid_in(id_list, filter_ids)
        elif id_type == 'pmcid':
            return cls.pmcid_in(id_list, filter_ids)
        elif id_type == 'doi':
            return cls.doi_in(id_list, filter_ids)
        else:
            return getattr(cls, id_type).in_(id_list)

    def get_ref_dict(self):
        ref_dict = {}
        for ref in self._ref_cols:
            val = getattr(self, ref, None)
            if val:
                ref_dict[ref.upper()] = val
        ref_dict['TRID'] = self.id
        return ref_dict

