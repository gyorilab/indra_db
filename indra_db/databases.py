__all__ = ['texttypes', 'formats', 'DatabaseManager', 'IndraDbException',
           'sql_expressions', 'readers', 'reader_versions',
           'PrincipalDatabaseManager', 'ReadonlyDatabaseManager']

import re
import random
import logging
from io import BytesIO
from numbers import Number
from functools import wraps
from datetime import datetime

from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from sqlalchemy.sql import expression as sql_expressions
from sqlalchemy.schema import DropTable
from sqlalchemy.sql.expression import Delete, Update
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy import create_engine, inspect, UniqueConstraint, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.engine.url import make_url

from indra.util import batch_iter
from indra_db.util import S3Path
from indra_db.exceptions import IndraDbException
from indra_db.schemas import principal_schema, readonly_schema
from indra_db.schemas.readonly_schema import CREATE_ORDER, CREATE_UNORDERED


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
    from indra_db.copy import *
    CAN_COPY = True
except ImportError as e:
    logger.warning("Copy utilities unavailable: %s" % str(e))
    CAN_COPY = False


def _copy_method(get_null_return=lambda: None):
    def super_wrapper(meth):
        @wraps(meth)
        def wrapper(obj, tbl_name, data, cols=None, commit=True, *args, **kwargs):
            logger.info("Received request to %s %d entries into %s."
                        % (meth.__name__, len(data), tbl_name))
            if not CAN_COPY:
                raise RuntimeError("Cannot use copy methods. `pg_copy` is not "
                                   "available.")
            if len(data) is 0:
                return get_null_return()  # Nothing to do....

            res = meth(obj, tbl_name, data, cols, commit, *args, **kwargs)

            if commit:
                obj.commit_copy('Failed to commit %s.' % meth.__name__)

            return res
        return wrapper
    return super_wrapper


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


class texttypes(_map_class):
    FULLTEXT = 'fulltext'
    ABSTRACT = 'abstract'
    TITLE = 'title'


class formats(_map_class):
    XML = 'xml'
    TEXT = 'text'
    JSON = 'json'
    EKB = 'ekb'


readers = {'REACH': 1, 'SPARSER': 2, 'TRIPS': 3, 'ISI': 4, 'EIDOS': 5}


# Specify versions of readers, and preference. Later in the list is better.
reader_versions = {
    'sparser': ['sept14-linux\n', 'sept14-linux', 'June2018-linux',
                'October2018-linux', 'February2020-linux', 'April2020-linux'],
    'reach': ['61059a-biores-e9ee36', '1.3.3-61059a-biores-'],
    'trips': ['STATIC', '2019Nov14'],
    'isi': ['20180503'],
    'eidos': ['0.2.3-SNAPSHOT'],
}


class IndraTableError(IndraDbException):
    def __init__(self, table, issue):
        msg = 'Error in table %s: %s' % (table, issue)
        super(IndraTableError, self).__init__(self, msg)


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
    url : str
        The database to which you want to interface.
    label : OPTIONAL[str]
        A short string to indicate the purpose of the db instance. Set as
        primary when initialized be `get_primary_db` or `get_db`.

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
    def __init__(self, url, label=None):
        self.url = make_url(url)
        self.session = None
        self.Base = declarative_base()
        self.label = label
        self.engine = create_engine(self.url)
        self._conn = None
        return

    def _init_foreign_key_map(self, foreign_key_map):
        # There are some useful shortcuts that can be used if
        # networkx is available, specifically the DatabaseManager.link
        if WITH_NX and foreign_key_map:
            G = nx.Graph()
            G.add_edges_from(foreign_key_map)
            self.__foreign_key_graph = G
        else:
            self.__foreign_key_graph = None

    def __del__(self, *args, **kwargs):
        try:
            self.grab_session()
            self.session.rollback()
        except:
            print("Failed to execute rollback of database upon deletion.")

    def create_tables(self, tbl_list=None):
        "Create the tables for INDRA database."
        ordered_tables = ['text_ref', 'mesh_ref_annotations', 'text_content',
                          'reading', 'db_info', 'raw_statements', 'raw_agents',
                          'raw_mods', 'raw_muts', 'pa_statements', 'pa_agents',
                          'pa_mods', 'pa_muts', 'raw_unique_links',
                          'support_links']
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
            logger.info("Removing all tables...")
            self.Base.metadata.drop_all(self.engine)
            logger.debug("All tables removed.")
        else:
            for tbl in tbl_list:
                logger.info("Removing %s..." % tbl.__tablename__)
                if tbl.__table__.exists(self.engine):
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

    def get_schemas(self):
        """Return the list of schema names currently in the database."""
        res = []
        with self.engine.connect() as con:
            raw_res = con.execute('SELECT schema_name '
                                  'FROM information_schema.schemata;')
            for r, in raw_res:
                res.append(r)
        return res

    def create_schema(self, schema_name):
        """Create a schema with the given name."""
        with self.engine.connect() as con:
            con.execute('CREATE SCHEMA IF NOT EXISTS %s;' % schema_name)
        return

    def drop_schema(self, schema_name, cascade=True):
        """Drop a schema (rather forcefully by default)"""
        with self.engine.connect() as con:
            logger.info("Dropping schema %s." % schema_name)
            con.execute('DROP SCHEMA IF EXISTS %s %s;'
                        % (schema_name, 'CASCADE' if cascade else ''))
        return

    def get_column_names(self, table):
        """"Get a list of the column labels for a table.

        Note that if the table involves a schema, the schema name must be
        prepended to the table name.
        """
        return self.get_column_objects(table).keys()

    def get_column_objects(self, table):
        """Get a list of the column object for the given table.

        Note that if the table involves a schema, the schema name must be
        prepended to the table name.
        """
        if isinstance(table, type(self.Base)):
            table = table.full_name()
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

    def _prep_copy(self, tbl_name, data, cols):

        # If cols is not specified, use all the cols in the table, else check
        # to make sure the names are valid.
        if cols is None:
            cols = self.get_column_names(tbl_name)
        else:
            db_cols = self.get_column_names(tbl_name)
            assert all([col in db_cols for col in cols]), \
                "Do not recognize one of the columns in %s for table %s." % \
                (cols, tbl_name)

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
        n_cols = len(cols)
        for entry in data:
            # Make sure that the number of columns matches the number of columns in
            # the data.
            if n_cols != len(entry):
                raise ValueError("Number of columns does not match number of "
                                 "columns in data.")

            # Convert the entry to bytes
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

        # Prep the connection.
        if self._conn is None:
            self._conn = self.engine.raw_connection()
            self._conn.rollback()

        return cols, data_bts

    @_copy_method(list)
    def copy_report_lazy(self, tbl_name, data, cols=None, commit=True,
                         constraint=None, return_cols=None, order_by=None):
        """Copy lazily, and report what rows were skipped."""
        cols, data_bts = self._prep_copy(tbl_name, data, cols)

        if not order_by:
            order_by = getattr(self.tables[tbl_name],
                               '_default_insert_order_by')
            if not order_by:
                raise ValueError("%s does not have an `order_by` attribute, "
                                 "and no `order_by` was specified." % tbl_name)

        mngr = LazyCopyManager(self._conn, tbl_name, cols,
                               constraint=constraint)
        return mngr.report_copy(data_bts, order_by, return_cols, BytesIO)

    @_copy_method()
    def copy_lazy(self, tbl_name, data, cols=None, commit=True,
                  constraint=None):
        "Copy lazily, skip any rows that violate constraints."
        cols, data_bts = self._prep_copy(tbl_name, data, cols)

        mngr = LazyCopyManager(self._conn, tbl_name, cols,
                               constraint=constraint)
        mngr.copy(data_bts, BytesIO)
        return

    def _infer_constraint(self, tbl_name, cols):
        """Try to infer a single constrain for a given table and columns.

        Look for table arguments that are constraints, and moreover that
        involve a subset of the columns being copied. If the column isn't in
        the input data, it can't possibly violate a constraint. It is also
        because of this line of code that constraints MUST be named. This
        process will not catch foreign key constraints, which may not even
        apply.
        """
        tbl = self.tables[tbl_name]

        constraints = [c.name for c in tbl.__table_args__
                       if isinstance(c, UniqueConstraint)
                       and set(c.columns.keys()) < set(cols)]

        # Include the primary key in the list, if applicable.
        if inspect(tbl).primary_key[0].name in cols:
            constraints.append(tbl_name + '_pkey')

        # Hopefully at this point there is exactly one constraint.
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
        return constraint

    @_copy_method()
    def copy_push(self, tbl_name, data, cols=None, commit=True,
                  constraint=None):
        "Copy, pushing any changes to constraint violating rows."
        cols, data_bts = self._prep_copy(tbl_name, data, cols)

        if constraint is None:
            constraint = self._infer_constraint(tbl_name, cols)

        mngr = PushCopyManager(self._conn, tbl_name, cols,
                               constraint=constraint)
        mngr.copy(data_bts, BytesIO)
        return

    @_copy_method(list)
    def copy_report_push(self, tbl_name, data, cols=None, commit=True,
                         constraint=None, return_cols=None, order_by=None):
        """Report on the rows skipped when pushing and copying."""
        cols, data_bts = self._prep_copy(tbl_name, data, cols)

        if constraint is None:
            constraint = self._infer_constraint(tbl_name, cols)

        if not order_by:
            order_by = self.tables[tbl_name]._default_insert_order_by
            if not order_by:
                raise ValueError("Table %s have no `_default_insert_order_by` "
                                 "attribute, and no `order_by` was specified."
                                 % tbl_name)

        mngr = PushCopyManager(self._conn, tbl_name, cols,
                               constraint=constraint)
        return mngr.report_copy(data_bts, order_by, return_cols, BytesIO)

    @_copy_method()
    def copy(self, tbl_name, data, cols=None, commit=True):
        "Use pg_copy to copy over a large amount of data."
        cols, data_bts = self._prep_copy(tbl_name, data, cols)
        mngr = CopyManager(self._conn, tbl_name, cols)
        mngr.copy(data_bts, BytesIO)
        return

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
            if table in self.tables.keys() or table in self.views.keys():
                true_table = getattr(self, table)
            else:
                raise IndraDbException("Invalid table name: %s." % table)
        elif hasattr(table, 'class_'):
            # This is technically an attribute of a table.
            true_table = table.class_
        elif table in self.tables.values() or table in self.views.values():
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

    def _form_pg_args(self):
        """Arrange the url elements into a list of arguments for pg calls."""
        return ['-h', self.url.host,
                '-U', self.url.username,
                '-w',  # Don't prompt for a password, forces use of env.
                '-d', self.url.database]

    def vacuum(self, analyze=True):
        conn = self.engine.raw_connection()
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute('VACUUM' + (' ANALYZE;' if analyze else ''))
        return


class PrincipalDatabaseManager(DatabaseManager):
    """This class represents the methods special to the principal database."""
    def __init__(self, host, label=None):
        super(self.__class__, self).__init__(host, label)

        self.tables = principal_schema.get_schema(self.Base)
        self.readonly = readonly_schema.get_schema(self.Base)

        for tbl in (t for d in [self.tables, self.readonly]
                    for t in d.values()):
            if tbl.__name__ == 'PaStmtSrc':
                self.__PaStmtSrc = tbl
            elif tbl.__name__ == 'SourceMeta':
                self.__SourceMeta = tbl
            else:
                setattr(self, tbl.__name__, tbl)

        self._init_foreign_key_map(principal_schema.foreign_key_map)
        return

    def __getattribute__(self, item):
        if item == 'PaStmtSrc':
            self.__PaStmtSrc.load_cols(self.engine)
            return self.__PaStmtSrc
        elif item == 'SourceMeta':
            self.__SourceMeta.load_cols(self.engine)
            return self.__SourceMeta
        return super(DatabaseManager, self).__getattribute__(item)

    def generate_readonly(self, ro_list=None, allow_continue=True):
        """Manage the materialized views.

        Parameters
        ----------
        ro_list : list or None
            Default None. A list of readonly table names or None. If None,
            all defined readonly tables will be build.
        allow_continue : bool
            If True (default), continue to build the schema if it already
            exists. If False, give up if the schema already exists.
        """
        if 'readonly' in self.get_schemas():
            if allow_continue:
                logger.warning("Schema already exists. State could be "
                               "outdated. Will proceed (allow_continue=True),")
            else:
                logger.error("Schema already exists, will not proceed (allow_"
                             "continue=False).")
                return
        else:
            logger.info("Creating the schema.")
            self.create_schema('readonly')

        # Create each of the readonly view tables (in order, where necessary).
        def iter_names():
            for i, view in enumerate(CREATE_ORDER):
                yield str(i), view
            for view in CREATE_UNORDERED:
                yield '-', view

        for i, ro_name in iter_names():
            if ro_list is not None and ro_name not in ro_list:
                continue

            ro_tbl = self.readonly[ro_name]

            logger.info('[%s] Creating %s readonly table...' % (i, ro_name))
            ro_tbl.create(self)
            ro_tbl.build_indices(self)
        return

    def dump_readonly(self, dump_file=None):
        """Dump the readonly schema to s3."""
        if isinstance(dump_file, str):
            dump_file = S3Path.from_string(dump_file)
        elif dump_file is not None and not isinstance(dump_file, S3Path):
            raise ValueError("Argument `dump_file` must be appropriately "
                             "formatted string or S3Path object, not %s."
                             % type(dump_file))

        from subprocess import check_call
        from indra_db.config import get_s3_dump
        from os import environ

        # Make sure the session is fresh and any previous session are done.
        self.session.close()
        self.grab_session()

        # Add the password to the env
        my_env = environ.copy()
        my_env['PGPASSWORD'] = self.url.password

        # Form the name of the s3 file, if not given.
        if dump_file is None:
            now_str = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
            dump_loc = get_s3_dump()
            dump_file = dump_loc.get_element_path('readonly-%s.dump' % now_str)

        # Dump the database onto s3, piping through this machine (errors if
        # anything went wrong).
        cmd = ' '.join(["pg_dump", *self._form_pg_args(),
                        '-n', 'readonly', '-Fc',
                        '|', 'aws', 's3', 'cp', '-', dump_file.to_string()])
        check_call(cmd, shell=True, env=my_env)

        return dump_file

    @staticmethod
    def get_latest_dump_file():
        import boto3
        from indra.util.aws import iter_s3_keys
        from indra_db.config import get_s3_dump

        s3 = boto3.client('s3')
        s3_path = get_s3_dump()

        logger.debug("Looking for the latest dump file on s3 to %s." % s3_path)

        # Get the most recent file from s3.
        max_date_str = None
        max_lm_date = None
        latest_key = None
        for key, lm_date in iter_s3_keys(s3, with_dt=True, **s3_path.kw()):

            # Get the date string from the name, ignoring non-standard files.
            suffix = key.split('/')[-1]
            m = re.match('readonly-(\S+).dump', suffix)
            if m is None:
                logger.debug("{key} is not a standard key, will not be "
                             "considered.".format(key=key))
                continue
            date_str, = m.groups()

            # Compare the the current maxes. If the date_str and the last
            # -modified date don't agree, raise an error.
            if not max_lm_date \
                    or date_str > max_date_str and lm_date > max_lm_date:
                max_date_str = date_str
                max_lm_date = lm_date
                latest_key = key
            elif max_lm_date \
                    and (date_str > max_date_str or lm_date > max_lm_date):
                raise S3DumpTimeAmbiguityError(key, date_str > max_date_str,
                                               lm_date > max_lm_date)
        logger.debug("Latest dump file from %s was found to be %s."
                     % (s3_path, latest_key))

        return S3Path(s3_path.bucket, latest_key)


class S3DumpTimeAmbiguityError(Exception):
    def __init__(self, key, is_latest_str, is_last_modified):
        msg = ('%s is ' % key) + ('' if is_latest_str else 'not ') \
              + 'the largest date string but is ' \
              + ('' if is_last_modified else 'not ')\
              + 'the latest time stamp.'
        super().__init__(msg)
        return


class ReadonlyDatabaseManager(DatabaseManager):
    """This class represents the readonly database."""

    def __init__(self, host, label=None):
        super(self.__class__, self).__init__(host, label)

        self.tables = readonly_schema.get_schema(self.Base)
        for tbl in self.tables.values():
            if tbl.__name__ == 'PaStmtSrc':
                self.__PaStmtSrc = tbl
            elif tbl.__name__ == 'SourceMeta':
                self.__SourceMeta = tbl
            else:
                setattr(self, tbl.__name__, tbl)

    def __getattribute__(self, item):
        if item == 'PaStmtSrc':
            self.__PaStmtSrc.load_cols(self.engine)
            return self.__PaStmtSrc
        elif item == 'SourceMeta':
            self.__SourceMeta.load_cols(self.engine)
            return self.__SourceMeta
        return super(DatabaseManager, self).__getattribute__(item)

    def load_dump(self, dump_file, force_clear=True):
        """Load from a dump of the readonly schema on s3."""
        if isinstance(dump_file, str):
            dump_file = S3Path.from_string(dump_file)
        elif dump_file is not None and not isinstance(dump_file, S3Path):
            raise ValueError("Argument `dump_file` must be appropriately "
                             "formatted string or S3Path object, not %s."
                             % type(dump_file))

        from subprocess import run
        from os import environ

        self.session.close()
        self.grab_session()

        # Add the password to the env
        my_env = environ.copy()
        my_env['PGPASSWORD'] = self.url.password

        # Make sure the database is clear.
        if 'readonly' in self.get_schemas():
            if force_clear:
                # For some reason, dropping tables does not work.
                self.drop_schema('readonly')
            else:
                raise IndraDbException("Tables already exist and force_clear "
                                       "is False.")

        # Pipe the database dump from s3 through this machine into the database
        logger.info("Dumping into the database.")
        run(' '.join(['aws', 's3', 'cp', dump_file.to_string(), '-', '|',
                      'pg_restore', *self._form_pg_args(), '--no-owner']),
            env=my_env, shell=True, check=True)
        self.session.close()
        self.grab_session()

        # Run Vacuuming
        logger.info("Running vacuuming.")
        self.vacuum()

        return

