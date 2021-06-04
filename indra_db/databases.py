__all__ = ['texttypes', 'formats', 'DatabaseManager', 'IndraDbException',
           'sql_expressions', 'readers', 'reader_versions',
           'PrincipalDatabaseManager', 'ReadonlyDatabaseManager']

import re
import json
import random
import logging
import string
from io import BytesIO
from numbers import Number
from datetime import datetime
from time import sleep

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
from indra_db.config import CONFIG, build_db_url, is_db_testing
from indra_db.schemas.mixins import IndraDBTableMetaClass
from indra_db.util import S3Path
from indra_db.exceptions import IndraDbException
from indra_db.schemas import PrincipalSchema, foreign_key_map, ReadonlySchema
from indra_db.schemas.readonly_schema import CREATE_ORDER


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


readers = {'REACH': 1, 'SPARSER': 2, 'TRIPS': 3, 'ISI': 4, 'EIDOS': 5, 'MTI': 6}
"""A dict mapping each reader a unique integer ID.

These ID's are used in creating the reading primary ID hashes. Thus, for a new
reader to be fully integrated, it must be added to the above dictionary.
"""


# Specify versions of readers, and preference. Later in the list is better.
reader_versions = {
    'sparser': ['sept14-linux\n', 'sept14-linux', 'June2018-linux',
                'October2018-linux', 'February2020-linux', 'April2020-linux'],
    'reach': ['61059a-biores-e9ee36', '1.3.3-61059a-biores-', '1.6.1'],
    'trips': ['STATIC', '2019Nov14', '2021Jan26'],
    'isi': ['20180503'],
    'eidos': ['0.2.3-SNAPSHOT'],
    'mti': ['1.0'],
}
"""A dict of list values keyed by reader name, tracking reader versions.

The oldest versions are to the left, and the newest to the right. We keep track
of all past versions as it is often not practical nor necessary to re-run a
reading on all content. Even in cases where it is, it is often useful to be
able to compare results.

As with the :data:`readers` variable above, this is used in the creation of
the unique hash for a reading entry. For a new reader version to work, it must
be added to the appropriate list.
"""


class IndraTableError(IndraDbException):
    def __init__(self, table, issue):
        msg = 'Error in table %s: %s' % (table, issue)
        super(IndraTableError, self).__init__(self, msg)


class RdsInstanceNotFoundError(IndraDbException):
    def __init__(self, instance_identifier):
        msg = f"No instance with name \"{instance_identifier}\" found on RDS."
        super(RdsInstanceNotFoundError, self).__init__(msg)


def get_instance_attribute(attribute, instance_identifier):
    """Get the current status of a database."""
    # Get descriptions for all instances (apparently you can't get just one).
    import boto3
    rds = boto3.client('rds')
    resp = rds.describe_db_instances()

    # If we find the one they're looking for, return the status.
    for desc in resp['DBInstances']:
        if desc['DBInstanceIdentifier'] == instance_identifier:

            # Try to match some common patterns for attribute labels.
            if attribute in desc:
                return desc[attribute]

            if attribute.capitalize() in desc:
                return desc[attribute.capitalize()]

            inst_attr = f'DBInstance{attribute.capitalize()}'
            if inst_attr in desc:
                return desc[inst_attr]

            # Give explosively up if the above fail.
            raise ValueError(f"Invalid attribute: {attribute}. Did you mean "
                             f"one of these: {list(desc.keys())}?")

    # Otherwise, fail.
    raise RdsInstanceNotFoundError(instance_identifier)


class DatabaseManager(object):
    """An object used to access INDRA's database.

    This object can be used to access and manage indra's database. It includes
    both basic methods and some useful, more high-level methods. It is designed
    to be used with postgresql, or sqlite.

    This object is primarily built around sqlalchemy, which is a required
    package for its use. It also optionally makes use of the pgcopy package for
    large data transfers.

    If you wish to access the primary database, you can simply use the
    `get_db` function to get an instance of this object using the default
    settings.

    Parameters
    ----------
    url : str
        The database to which you want to interface.
    label : OPTIONAL[str]
        A short string to indicate the purpose of the db instance. Set as
        ``db_label`` when initialized with ``get_db(db_label)``.

    Example
    -------
    If you wish to access the primary database and find the the metadata for a
    particular pmid, 1234567:

    .. code-block:: python

        from indra.db import get_db
        db = get_db('primary')
        res = db.select_all(db.TextRef, db.TextRef.pmid == '1234567')

    You will get a list of objects whose attributes give the metadata contained
    in the columns of the table.

    For more sophisticated examples, several use cases can be found in
    ``indra.tests.test_db``.
    """
    _instance_type = NotImplemented
    _instance_name_fmt = NotImplemented
    _db_name = NotImplemented

    def __init__(self, url, label=None, protected=False):
        self.url = make_url(url)
        self.session = None
        self.label = label
        self.__protected = protected
        self._conn = None

        # To stringify table classes, we must merge the two meta classes.
        class BaseMeta(DeclarativeMeta, IndraDBTableMetaClass):
            pass
        self.Base = declarative_base(metaclass=BaseMeta)

        # Check to see if the database if available.
        self.available = True
        try:
            create_engine(
                self.url,
                connect_args={'connect_timeout': 1}
            ).execute('SELECT 1 AS ping;')
        except Exception as err:
            logger.warning(f"Database {repr(self.url)} is not available: {err}")
            self.available = False
            return

        # Create the engine (connection manager).
        self.__engine = create_engine(self.url)
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

    def is_protected(self):
        return self.__protected

    def get_raw_connection(self):
        if self.__protected:
            logger.error("Cannot get a raw connection if protected mode is on.")
            return
        return self.__engine.raw_connection()

    def get_conn(self):
        if self.__protected:
            logger.error("Cannot get a direct connection in protected mode.")
            return
        return self.__engine.connect()

    def __del__(self, *args, **kwargs):
        if not hasattr(self, 'available') or self.available:
            return
        try:
            self.grab_session()
            self.session.rollback()
        except:
            print("Failed to execute rollback of database upon deletion.")

    @classmethod
    def create_instance(cls, instance_name, size, tag_dict=None):
        """Allocate the resources on RDS for a database, and return handle."""
        # Load boto3 locally to avoid unnecessary dependencies.
        import boto3
        rds = boto3.client('rds')

        # Convert tags to boto3's goofy format.
        tags = ([{'Key': k, 'Value': v} for k, v in tag_dict.items()]
                if tag_dict else [])

        # Create a new password.
        pw_chars = random.choices(string.ascii_letters + string.digits, k=24)
        password = ''.join(pw_chars)

        # Load the rds general config settings.
        rds_config = CONFIG['rds-settings']

        # Create the database.
        inp_identifier = cls._instance_name_fmt.format(
            name=instance_name.lower()
        )
        resp = rds.create_db_instance(
            DBInstanceIdentifier=inp_identifier,
            DBName=cls._db_name,
            AllocatedStorage=size,
            DBInstanceClass=cls._instance_type,
            Engine='postgres',
            MasterUsername=rds_config['master_user'],
            MasterUserPassword=password,
            VpcSecurityGroupIds=[rds_config['security_group']],
            AvailabilityZone=rds_config['availability_zone'],
            DBSubnetGroupName='default',
            Tags=tags,
            DeletionProtection=True
        )

        # Perform a basic sanity check.
        assert resp['DBInstance']['DBInstanceIdentifier'] == inp_identifier, \
            f"Bad response from creating RDS instance {inp_identifier}:\n{resp}"

        # Wait for the database to be created.
        logger.info("Waiting for database to be created...")
        while get_instance_attribute('status', inp_identifier) == 'creating':
            sleep(5)

        # Use the given info to return a handle to the new database.
        endpoint = get_instance_attribute('endpoint', inp_identifier)
        url_str = build_db_url(dialect='postgres', host=endpoint['Address'],
                               port=endpoint['Port'], password=password,
                               name=cls._db_name,
                               username=rds_config['master_user'])
        return cls(url_str)

    def get_config_string(self):
        """Print a config entry for this handle.

        This is useful after using `create_instance`.
        """
        data = {
            'dialect': self.url.drivername,
            'driver': None,
            'username': self.url.username,
            'password': self.url.password_original,
            'host': self.url.host,
            'port': self.url.port,
            'name': self.url.database
        }
        return '\n'.join(f'{key} = {value}' if value else f'{key} ='
                         for key, value in data.items())

    def get_env_string(self):
        """Generate the string for an environment variable.

        This is useful after using `create_instance`.
        """
        return str(self.url)

    def grab_session(self):
        """Get an active session with the database."""
        if not self.available:
            return
        if self.session is None or not self.session.is_active:
            logger.debug('Attempting to get session...')
            if not self.__protected:
                DBSession = sessionmaker(bind=self.__engine)
            else:
                DBSession = sessionmaker(bind=self.__engine, autoflush=False,
                                         autocommit=False)
            logger.debug('Got session.')
            self.session = DBSession()
            if self.session is None:
                raise IndraDbException("Failed to grab session.")
            if self.__protected:
                def no_flush(*a, **k):
                    #  If further errors occur as a result, please first think
                    #  carefully whether you really want to write, and if you
                    #  do, instantiate your database handle with
                    #  "protected=False". Note that you should NOT be writing to
                    #  readonly unless you are doing the initial load, or are
                    #  testing something on a dev database. Do NOT write to a
                    #  stable deployment.
                    logger.info("Session flush attempted. Write not allowed in "
                                "protected mode.")
                self.session.flush = no_flush

    def get_tables(self):
        """Get a list of available tables."""
        return [tbl_name for tbl_name in self.tables.keys()]

    def show_tables(self, active_only=False, schema=None):
        """Print a list of all the available tables."""
        def print_table(table_name):
            if tbl_name in self.tables:
                print(self.tables[table_name])

        if not active_only:
            for tbl_name in self.get_tables():
                tbl = self.tables[tbl_name]
                if schema is None \
                   or tbl.get_schema(default='public') == schema:
                    print_table(tbl_name)
        else:
            if schema is None:
                for active_schema in self.get_schemas():
                    for tbl_name in self.get_active_tables(active_schema):
                        print_table(tbl_name)
            else:
                for tbl_name in self.get_active_tables(schema):
                    print_table(tbl_name)

    def get_active_tables(self, schema=None):
        """Get the tables currently active in the database.

        Parameters
        ----------
        schema : None or st
            The name of the schema whose tables you wish to see. The default is
            public.
        """
        return inspect(self.__engine).get_table_names(schema=schema)

    def get_schemas(self):
        """Return the list of schema names currently in the database."""
        res = []
        with self.__engine.connect() as con:
            raw_res = con.execute('SELECT schema_name '
                                  'FROM information_schema.schemata;')
            for r, in raw_res:
                res.append(r)
        return res

    def create_schema(self, schema_name):
        """Create a schema with the given name."""
        if self.__protected:
            logger.error("Running in protected mode, writes not allowed!")
            return
        with self.__engine.connect() as con:
            con.execute('CREATE SCHEMA IF NOT EXISTS %s;' % schema_name)
        return

    def drop_schema(self, schema_name, cascade=True):
        """Drop a schema (rather forcefully by default)"""
        if self.__protected:
            logger.error("Running in protected mode, writes not allowed!")
            return
        with self.__engine.connect() as con:
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

    def get_copy_cursor(self):
        """Execute SQL queries in the context of a copy operation."""
        # Prep the connection.
        if self._conn is None:
            self._conn = self.__engine.raw_connection()
            self._conn.rollback()
        return self._conn.cursor()

    def make_copy_batch_id(self):
        """Generate a random batch id for copying into the database.

        This allows for easy retrieval of the assigned ids immediately after
        copying in. At this time, only Reading and RawStatements use the
        feature.
        """
        return random.randint(-2**30, 2**30)

    def _precheck_copy(self, tbl_name, data, meth_name):
        logger.info(f"Received request to '{meth_name}' {len(data)} entries "
                    f"into table '{tbl_name}'.")
        if not CAN_COPY:
            raise RuntimeError("Cannot use copy methods. `pg_copy` is not "
                               "available.")
        if self.is_protected():
            raise RuntimeError("Attempt to copy while in protected mode!")
        if len(data) == 0:
            return False
        return True

    def _prep_copy(self, tbl_name, data, cols):
        assert not self.__protected,\
            "This should not be called if db in protected mode."

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
            # Make sure that the number of columns matches the number of columns
            # in the data.
            if n_cols != len(entry):
                raise ValueError("Number of columns does not match number of "
                                 "columns in data.")

            # Convert the entry to bytes
            new_entry = []
            for element in entry:
                if isinstance(element, str):
                    new_entry.append(element.encode('utf8'))
                elif isinstance(element, dict):
                    new_entry.append(json.dumps(element).encode('utf-8'))
                elif (isinstance(element, bytes)
                      or element is None
                      or isinstance(element, Number)
                      or isinstance(element, datetime)):
                    new_entry.append(element)
                else:
                    raise IndraDbException(
                        "Don't know what to do with element of type %s. "
                        "Should be str, bytes, datetime, None, or a "
                        "number." % type(element)
                    )
            data_bts.append(tuple(new_entry))

        # Prep the connection.
        if self._conn is None:
            self._conn = self.__engine.raw_connection()
            self._conn.rollback()

        return cols, data_bts

    def _infer_copy_order_by(self, order_by, tbl_name):
        if not order_by:
            order_by = getattr(self.tables[tbl_name],
                               '_default_insert_order_by')
            if not order_by:
                raise ValueError("%s does not have an `order_by` attribute, "
                                 "and no `order_by` was specified." % tbl_name)
        return order_by

    def _infer_copy_constraint(self, constraint, tbl_name, cols,
                               failure_ok=False):
        """Try to infer a single constrain for a given table and columns.

        Look for table arguments that are constraints, and moreover that
        involve a subset of the columns being copied. If the column isn't in
        the input data, it can't possibly violate a constraint. It is also
        because of this line of code that constraints MUST be named. This
        process will not catch foreign key constraints, which may not even
        apply.
        """
        # Get the table object.
        tbl = self.tables[tbl_name]

        # If a constraint was given, just return it, ensuring it is the object
        # and not just the name.
        if constraint:
            if not isinstance(constraint, str):
                if not isinstance(constraint, UniqueConstraint):
                    raise ValueError("`constraint` should by type "
                                     "UniqueConstraint or the (str) name of a "
                                     "UniqueConstraint.")
                return constraint.name
            return constraint

        constraints = [c.name for c in tbl.iter_constraints(cols)]

        # Include the primary key in the list, if applicable.
        if inspect(tbl).primary_key[0].name in cols:
            constraints.append(tbl_name + '_pkey')

        # Hopefully at this point there is exactly one constraint.
        if len(constraints) > 1 and not failure_ok:
            raise ValueError(f"Cannot infer constraint. Only one constraint is "
                             f"allowed, and there are multiple possibilities: "
                             f"{constraints}. Please specify a single "
                             f"constraint.")
        elif len(constraints) == 1:
            constraint = constraints[0]
        elif  not failure_ok:
            raise ValueError("Could not infer a relevant constraint. If no "
                             "columns have constraints on them, the lazy "
                             "option is unnecessary. Note that I cannot guess "
                             "a foreign key constraint.")
        else:
            constraint = None
        return constraint

    def _get_constraint_cols(self, constraint, tbl_name, cols):
        """Get the column pairs in cols involved in unique constraints."""
        tbl = self.tables[tbl_name]
        if constraint is None:
            constraint_cols = [tuple(c.columns.keys())
                               for c in tbl.iter_constraints(cols)]
        else:
            if isinstance(constraint, str):
                constraint = tbl.get_constraint(constraint)
            constraint_cols = [constraint.columns.keys()]
        return constraint_cols

    def copy_report_lazy(self, tbl_name, data, cols=None, commit=True,
                         constraint=None, return_cols=None, order_by=None):
        """Copy lazily, and report what rows were skipped."""
        # General overhead.
        if not self._precheck_copy(tbl_name, data, 'copy_report_lazy'):
            return []
        cols, data_bts = self._prep_copy(tbl_name, data, cols)

        # Guess parameters.
        order_by = self._infer_copy_order_by(order_by, tbl_name)

        # Do the copy.
        mngr = LazyCopyManager(self._conn, tbl_name, cols,
                               constraint=constraint)
        ret = mngr.report_copy(data_bts, order_by, return_cols, BytesIO)

        # Commit the copy.
        if commit:
            self.commit_copy(f'Failed to commit copy_report_lazy to '
                             f'{tbl_name}.')

        return ret

    def copy_detailed_report_lazy(self, tbl_name, data, inp_cols=None,
                                  ret_cols=None, commit=True, constraint=None,
                                  skipped_cols=None, order_by=None):
        """Copy lazily, returning data from some of the columns such as IDs."""
        # General overhead.
        if not self._precheck_copy(tbl_name, data,
                                   'copy_report_and_return_lazy'):
            return [], [], []
        inp_cols, data_bts = self._prep_copy(tbl_name, data, inp_cols)

        # Handle guessed-parameters
        order_by = self._infer_copy_order_by(order_by, tbl_name)
        constraint_cols = self._get_constraint_cols(constraint, tbl_name,
                                                    inp_cols)
        if ret_cols is None:
            ret_cols = (self.get_primary_key(tbl_name).name,)

        # Do the copy.
        mngr = ReturningCopyManager(self._conn, tbl_name, inp_cols, ret_cols,
                                    constraint=constraint)
        ret = mngr.detailed_report_copy(data_bts, constraint_cols, skipped_cols,
                                        order_by)

        # Commit
        if commit:
            self.commit_copy(f'Failed to commit copy_report_and_return_lazy to '
                             f'{tbl_name}.')

        return ret

    def copy_lazy(self, tbl_name, data, cols=None, commit=True,
                  constraint=None):
        """Copy lazily, skip any rows that violate constraints."""
        # General overhead.
        if not self._precheck_copy(tbl_name, data, 'copy_lazy'):
            return
        cols, data_bts = self._prep_copy(tbl_name, data, cols)

        # Handle guessed-parameters
        # NOTE: It is OK for the constraint to be None in this case. We are not
        # trying to return any values, so an anonymous "do for any constraint"
        # would be fine.
        constraint = self._infer_copy_constraint(constraint, tbl_name, cols,
                                                 failure_ok=True)

        # Do the copy.
        mngr = LazyCopyManager(self._conn, tbl_name, cols,
                               constraint=constraint)
        mngr.copy(data_bts, BytesIO)

        # Commit
        if commit:
            self.commit_copy(f'Failed to commit copy_lazy to {tbl_name}.')

        return

    def copy_push(self, tbl_name, data, cols=None, commit=True,
                  constraint=None):
        """Copy, pushing any changes to constraint violating rows."""
        # General overhead.
        if not self._precheck_copy(tbl_name, data, 'copy_push'):
            return
        cols, data_bts = self._prep_copy(tbl_name, data, cols)

        # Handle guessed-parameters
        constraint = self._infer_copy_constraint(constraint, tbl_name, cols)

        # Do the copy.
        mngr = PushCopyManager(self._conn, tbl_name, cols,
                               constraint=constraint)
        mngr.copy(data_bts, BytesIO)

        # Commit
        if commit:
            self.commit_copy(f'Failed to commit copy_push to {tbl_name}.')
        return

    def copy_report_push(self, tbl_name, data, cols=None, commit=True,
                         constraint=None, return_cols=None, order_by=None):
        """Report on the rows skipped when pushing and copying."""
        if not self._precheck_copy(tbl_name, data, 'copy_report_push'):
            return
        cols, data_bts = self._prep_copy(tbl_name, data, cols)

        constraint = self._infer_copy_constraint(constraint, tbl_name, cols)
        order_by = self._infer_copy_order_by(order_by, tbl_name)

        mngr = PushCopyManager(self._conn, tbl_name, cols,
                               constraint=constraint)
        ret = mngr.report_copy(data_bts, order_by, return_cols, BytesIO)

        if commit:
            self.commit_copy(f'Failed to commit copy_report_push to '
                             f'{tbl_name}.')
        return ret

    def copy(self, tbl_name, data, cols=None, commit=True):
        """Use pg_copy to copy over a large amount of data."""
        if not self._precheck_copy(tbl_name, data, 'copy'):
            return
        cols, data_bts = self._prep_copy(tbl_name, data, cols)
        mngr = CopyManager(self._conn, tbl_name, cols)
        mngr.copy(data_bts, BytesIO)
        if commit:
            self.commit_copy(f'Failed to commit copy to {tbl_name}.')
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
        if isinstance(tbl, str):
            tbl = self.tables[tbl]
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

    def pg_dump(self, dump_file, **options):
        """Use the pg_dump command to dump part of the database onto s3.

        The `pg_dump` tool must be installed, and must be a compatible version
        with the database(s) being used.

        All keyword arguments are converted into flags/arguments of pg_dump. For
        documentation run `pg_dump --help`. This will also confirm you have
        `pg_dump` installed.

        By default, the "General" and "Connection" options are already set. The
        most likely specification you will want to use is `--table` or
        `--schema`, specifying either a particular table or schema to dump.

        Parameters
        ----------
        dump_file : S3Path or str
            The location on s3 where the content should be dumped.
        """
        if self.__protected:
            logger.error("Cannot execute pg_dump in protected mode.")
            return

        if isinstance(dump_file, str):
            dump_file = S3Path.from_string(dump_file)
        elif dump_file is not None and not isinstance(dump_file, S3Path):
            raise ValueError("Argument `dump_file` must be appropriately "
                             "formatted string or S3Path object, not %s."
                             % type(dump_file))

        from os import environ
        from subprocess import run, PIPE

        # Make sure the session is fresh and any previous session are done.
        self.session.close()
        self.grab_session()

        # Add the password to the env
        my_env = environ.copy()
        my_env['PGPASSWORD'] = self.url.password

        # Dump the database onto s3, piping through this machine (errors if
        # anything went wrong).
        option_list = [f'--{opt}' if isinstance(val, bool) and val
                       else f'--{opt}={val}' for opt, val in options.items()]
        cmd = ["pg_dump", *self._form_pg_args(), *option_list, '-Fc']

        # If we are testing the database, we
        if not is_db_testing():
            cmd += ['|', 'aws', 's3', 'cp', '-', dump_file.to_string()]
            run(' '.join(cmd), shell=True, env=my_env, check=True)
        else:
            import boto3
            res = run(' '.join(cmd), shell=True, env=my_env, stdout=PIPE,
                      check=True)
            dump_file.upload(boto3.client('s3'), res.stdout)
        return dump_file

    def vacuum(self, analyze=True):
        if self.__protected:
            logger.error("Vacuuming not allowed in protected mode.")
            return
        conn = self.__engine.raw_connection()
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute('VACUUM' + (' ANALYZE;' if analyze else ''))
        return

    def pg_restore(self, dump_file, **options):
        """Load content into the database from a dump file on s3."""
        if self.__protected:
            logger.error("Cannot execute pg_restore in protected mode.")
            return

        if isinstance(dump_file, str):
            dump_file = S3Path.from_string(dump_file)
        elif dump_file is not None and not isinstance(dump_file, S3Path):
            raise ValueError("Argument `dump_file` must be appropriately "
                             "formatted string or S3Path object, not %s."
                             % type(dump_file))

        from subprocess import run, PIPE
        from os import environ

        self.session.close()
        self.grab_session()

        # Add the password to the env
        my_env = environ.copy()
        my_env['PGPASSWORD'] = self.url.password

        # Pipe the database dump from s3 through this machine into the database
        logger.info("Dumping into the database.")
        option_list = [f'--{opt}' if isinstance(val, bool) and val
                       else f'--{opt}={val}' for opt, val in options.items()]
        cmd = ['pg_restore', *self._form_pg_args(), *option_list, '--no-owner']
        if not is_db_testing():
            cmd = ['aws', 's3', 'cp', dump_file.to_string(), '-', '|'] + cmd
            run(' '.join(cmd), shell=True, env=my_env, check=True)
        else:
            import boto3
            res = dump_file.get(boto3.client('s3'))
            run(' '.join(cmd), shell=True, env=my_env, input=res['Body'].read(),
                check=True)
        self.session.close()
        self.grab_session()
        return dump_file


class PrincipalDatabaseManager(DatabaseManager):
    """This class represents the methods special to the principal database."""

    # Note that these are NOT guaranteed to apply to older deployed instances.
    _instance_type = 'db.m5.large'
    _instance_name_fmt = 'indradb-{name}'
    _db_name = 'indradb_principal'

    def __init__(self, host, label=None, protected=False):
        super(self.__class__, self).__init__(host, label, protected)
        if not self.available:
            return
        self.__protected = self._DatabaseManager__protected
        self.__engine = self._DatabaseManager__engine

        self.public = PrincipalSchema(self.Base).build_table_dict()
        self.readonly = ReadonlySchema(self.Base).build_table_dict()
        self.tables = {k: v for d in [self.public, self.readonly]
                       for k, v in d.items()}

        for tbl in self.tables.values():
            if tbl.__name__ == '_PaStmtSrc':
                self.__PaStmtSrc = tbl
            elif tbl.__name__ == 'SourceMeta':
                self.__SourceMeta = tbl
            else:
                setattr(self, tbl.__name__, tbl)

        self._init_foreign_key_map(foreign_key_map)
        return

    def __getattribute__(self, item):
        if item == '_PaStmtSrc':
            self.load_pa_stmt_src_cols()
            return self.__PaStmtSrc
        elif item == 'SourceMeta':
            self.load_source_meta_cols()
            return self.__SourceMeta
        return super(DatabaseManager, self).__getattribute__(item)

    def load_pa_stmt_src_cols(self, cols=None):
        self.__PaStmtSrc.load_cols(self.__engine, cols)

    def load_source_meta_cols(self, cols=None):
        self.__SourceMeta.load_cols(self.__engine, cols)

    def generate_readonly(self, belief_dict, allow_continue=True):
        """Manage the materialized views.

        Parameters
        ----------
        belief_dict : dict
            The dictionary, keyed by hash, of belief calculated for Statements.
        allow_continue : bool
            If True (default), continue to build the schema if it already
            exists. If False, give up if the schema already exists.
        """
        if self.__protected:
            logger.error("Cannot generate readonly in protected mode.")
            return

        # Optionally create the schema.
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

        # Create function to quickly check if a table will be used further down
        # the line.
        def table_is_used(tbl, other_tables):
            if not tbl._temp:
                return True
            for other_idx, tbl_name in enumerate(other_tables):
                if tbl_name in self.get_active_tables(schema='readonly'):
                    continue
                other_tbl = self.readonly[tbl_name]
                if not table_is_used(other_tbl, other_tables[other_idx+1:]):
                    continue
                if tbl.full_name() in other_tbl.definition(self):
                    return True
            return False

        # Perform some sanity checks (this would fail only due to developer
        # errors.)
        assert len(set(CREATE_ORDER)) == len(CREATE_ORDER),\
            "Elements in CREATE_ORDERED are NOT unique."
        to_create = set(CREATE_ORDER)
        in_ro = set(self.readonly.keys()) - {'belief'}  # belief is pre-loaded
        assert to_create == in_ro,\
            f"Not all readonly tables included in CREATE_ORDER:\n" \
            f"extra in create_order={to_create-in_ro}\n" \
            f"extra in tables={in_ro-to_create}."

        # Dump the belief dict into the database.
        self.Belief.__table__.create(bind=self.__engine)
        self.copy(self.Belief.full_name(),
                  [(int(h), n) for h, n in belief_dict.items()],
                  ('mk_hash', 'belief'))

        # Build the tables.
        for i, ro_name in enumerate(CREATE_ORDER):
            # Check to see if the table has already been build (skip if so).
            if ro_name in self.get_active_tables(schema='readonly'):
                logger.info(f"[{i}] Build of {ro_name} done, continuing...")
                continue

            # Get table object, and check to see that if it is temp it is used.
            ro_tbl = self.readonly[ro_name]
            if not table_is_used(ro_tbl, CREATE_ORDER[i+1:]):
                logger.info(f"[{i}] {ro_name} is marked as a temp table "
                            f"but is not used in future tables. Skipping.")
                continue

            # Build the table and its indices.
            logger.info(f"[{i}] Creating {ro_name} readonly table...")
            ro_tbl.create(self)
            ro_tbl.build_indices(self)

            # Drop any temp tables that will not be used further down the line.
            to_drop = []
            for existing_tbl in self.get_active_tables(schema='readonly'):
                if not table_is_used(self.readonly[existing_tbl],
                                     CREATE_ORDER[i+1:]):
                    to_drop.append(existing_tbl)
            self.drop_tables(to_drop, force=True)

        return

    def dump_readonly(self, dump_file=None):
        """Dump the readonly schema to s3."""

        # Form the name of the s3 file, if not given.
        if dump_file is None:
            from indra_db.config import get_s3_dump
            now_str = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
            dump_loc = get_s3_dump()
            dump_file = dump_loc.get_element_path('readonly-%s.dump' % now_str)
        return self.pg_dump(dump_file, schema='readonly')

    def create_table(self, table_obj):
        table_obj.__table__.create(self.__engine)

    def create_tables(self, tbl_list=None):
        """Create the public tables for INDRA database."""
        if self.__protected:
            logger.error("Cannot create tables in protected mode.")
            return
        ordered_tables = ['text_ref', 'mesh_ref_annotations', 'text_content',
                          'reading', 'db_info', 'raw_statements', 'raw_agents',
                          'raw_mods', 'raw_muts', 'pa_statements', 'pa_agents',
                          'pa_mods', 'pa_muts', 'raw_unique_links',
                          'support_links']
        if tbl_list is None:
            tbl_list = list(self.public.keys())

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
                if not self.public[tbl_name].__table__.exists(self.__engine):
                    self.public[tbl_name].__table__.create(bind=self.__engine)
                    logger.debug("Table created.")
                else:
                    logger.debug("Table already existed.")

        # The rest can be started any time.
        for tbl_name in tbl_name_list:
            logger.debug("Creating %s..." % tbl_name)
            self.public[tbl_name].__table__.create(bind=self.__engine)
            logger.debug("Table created.")
        return

    def drop_tables(self, tbl_list=None, force=False):
        """Drop the tables for INDRA database given in tbl_list.

        If tbl_list is None, all tables will be dropped. Note that if `force`
        is False, a warning prompt will be raised to asking for confirmation,
        as this action will remove all data from that table.
        """
        if self.__protected:
            logger.error("Cannot drop tables in protected mode.")
            return False

        if tbl_list is not None:
            for i, tbl in enumerate(tbl_list[:]):
                if isinstance(tbl, str):
                    if tbl in self.tables:
                        tbl_list[i] = self.tables[tbl]
                    else:
                        raise ValueError(f"Did not recognize table name: {tbl}")
        if not force:
            # Build the message
            if tbl_list is None:
                msg = ("Do you really want to clear the %s database? [y/N]: "
                       % self.label)
            else:
                msg = "You are going to clear the following tables:\n"
                msg += str([tbl.__tablename__ for tbl in tbl_list]) + '\n'
                msg += ("Do you really want to clear these tables from %s? "
                        "[y/N]: " % self.label)
            # Check to make sure.
            resp = input(msg)
            if resp != 'y' and resp != 'yes':
                logger.info('Aborting clear.')
                return False
        if tbl_list is None:
            logger.info("Removing all tables...")
            self.Base.metadata.drop_all(self.__engine)
            logger.debug("All tables removed.")
        else:
            for tbl in tbl_list:
                logger.info("Removing %s..." % tbl.__tablename__)
                if tbl.__table__.exists(self.__engine):
                    tbl.__table__.drop(self.__engine)
                    logger.debug("Table removed.")
                else:
                    logger.debug("Table doesn't exist.")
        return True

    def _clear(self, tbl_list=None, force=False):
        """Brutal clearing of all tables in tbl_list, or all public."""
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


class ReadonlyDatabaseManager(DatabaseManager):
    """This class represents the readonly database."""
    _instance_type = 'db.m5.xlarge'
    _instance_name_fmt = 'indradb-readonly-{name}'
    _db_name = 'indradb_readonly'

    def __init__(self, host, label=None, protected=True):
        super(self.__class__, self).__init__(host, label, protected)
        if not self.available:
            return
        self.__protected = self._DatabaseManager__protected
        self.__engine = self._DatabaseManager__engine

        self.tables = ReadonlySchema(self.Base).build_table_dict()
        for tbl in self.tables.values():
            if tbl.__name__ == '_PaStmtSrc':
                self.__PaStmtSrc = tbl
            elif tbl.__name__ == 'SourceMeta':
                self.__SourceMeta = tbl
            else:
                setattr(self, tbl.__name__, tbl)
        self.__non_source_cols = None

    def get_config_string(self):
        res = super(ReadonlyDatabaseManager, self).get_config_string()
        res = 'role = readonly\n' + res
        return res

    def get_source_names(self) -> set:
        """Get a list of the source names as they appear in SourceMeta cols."""
        all_cols = set(self.get_column_names(self.SourceMeta))
        return all_cols - self.__non_source_cols

    def __getattribute__(self, item):
        if item == '_PaStmtSrc':
            self.__PaStmtSrc.load_cols(self.__engine)
            return self.__PaStmtSrc
        elif item == 'SourceMeta':
            if self.__non_source_cols is None:
                self.__non_source_cols = \
                    set(self.get_column_names(self.__SourceMeta))
            self.__SourceMeta.load_cols(self.__engine)
            return self.__SourceMeta
        return super(DatabaseManager, self).__getattribute__(item)

    def get_active_tables(self, schema='readonly'):
        """Get the tables currently active in the database.

        Parameters
        ----------
        schema : None or st
            The name of the schema whose tables you wish to see. The default is
            readonly.
        """
        return super(ReadonlyDatabaseManager, self).get_active_tables(schema)

    def load_dump(self, dump_file, force_clear=True):
        """Load from a dump of the readonly schema on s3."""
        if self.__protected:
            logger.error("Cannot load a dump while in protected mode.")
            return

        # Make sure the database is clear.
        if 'readonly' in self.get_schemas():
            if force_clear:
                # For some reason, dropping tables does not work.
                self.drop_schema('readonly')
            else:
                raise IndraDbException("Tables already exist and force_clear "
                                       "is False.")

        # Do the restore
        self.pg_restore(dump_file)

        # Run Vacuuming
        logger.info("Running vacuuming.")
        self.vacuum()

        return

