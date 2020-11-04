__all__ = ['get_primary_db', 'get_db', 'get_ro', 'get_ro_host']

import logging

from indra_db.databases import PrincipalDatabaseManager, \
    ReadonlyDatabaseManager
from indra_db.exceptions import IndraDbException
from indra_db.config import get_databases, get_readonly_databases, is_db_testing

logger = logging.getLogger('util-constructors')


__PRIMARY_DB = None


def get_primary_db(force_new=False):
    """Get a DatabaseManager instance for the primary database host.

    The primary database host is defined in the defaults.txt file, or in a file
    given by the environment variable DEFAULTS_FILE. Alternatively, it may be
    defined by the INDRADBPRIMARY environment variable. If none of the above
    are specified, this function will raise an exception.

    Note: by default, calling this function twice will return the same
    `DatabaseManager` instance. In other words::

        db1 = get_primary_db()
        db2 = get_primary_db()
        db1 is db2

    This means also that, for example `db1.select_one(db2.TextRef)` will work,
    in the above context.

    It is still recommended that when creating a script or function, or other
    general application, you should not rely on this feature to get your access
    to the database, as it can make substituting a different database host both
    complicated and messy. Rather, a database instance should be explicitly
    passed between different users as is done in `get_statements_by_gene_role_type`
    function's call to `get_statements` in `indra.db.query_db_stmts`.

    Parameters
    ----------
    force_new : bool
        If true, a new instance will be created and returned, regardless of
        whether there is an existing instance or not. Default is False, so that
        if this function has been called before within the global scope, a the
        instance that was first created will be returned.

    Returns
    -------
    primary_db : :py:class:`DatabaseManager`
        An instance of the database manager that is attached to the primary
        database.
    """
    logger.warning("DEPRECATION WARNING: This function is being deprecated.")
    defaults = get_databases()
    if 'primary' in defaults.keys():
        primary_host = defaults['primary']
    else:
        raise IndraDbException("No primary host available in defaults file.")

    global __PRIMARY_DB
    if __PRIMARY_DB is None or force_new:
        __PRIMARY_DB = PrincipalDatabaseManager(primary_host, label='primary')
        __PRIMARY_DB.grab_session()
    return __PRIMARY_DB


class WontDoIt(Exception):
    pass


def get_db(db_label):
    """Get a db instance base on it's name in the config or env.

    If the label does not exist or the database labeled can't be reached, None
    is returned.
    """
    # If we are running certain tests, we want to make sure the real database
    # is not used for any reason.
    if is_db_testing():
        raise WontDoIt(f"Cannot instantiate {db_label} database during test.")

    # Instantiate a database handle
    defaults = get_databases()
    if db_label not in defaults:
        logger.error(f"No such database available: {db_label}. Check config "
                     f"file or environment variables.")
        return
    db_url = defaults[db_label]
    db = PrincipalDatabaseManager(db_url, label=db_label)
    if not db.available:
        return
    db.grab_session()
    return db


def get_ro(ro_label):
    """Get a readonly database instance, based on its name.

    If the label does not exist or the database labeled can't be reached, None
    is returned.
    """
    # If we are running certain tests, we want to make sure the real database
    # is not used for any reason.
    if is_db_testing():
        raise WontDoIt(f"Cannot instantiate {ro_label} readonly database "
                       f"during test.")

    # Instantiate a readonly database.
    defaults = get_readonly_databases()
    if ro_label == 'primary' and 'override' in defaults:
        logger.info("Found an override database: using in place of primary.")
        ro_label = 'override'
    if ro_label not in defaults:
        logger.error(f"No such readonly database available: {ro_label}. Check "
                     f"config file or environment variables.")
        return
    db_url = defaults[ro_label]
    ro = ReadonlyDatabaseManager(db_url, label=ro_label)
    if not ro.available:
        return
    ro.grab_session()
    return ro


def get_ro_host(ro_label):
    """Get the host of the current readonly database."""
    ro = get_ro(ro_label)
    if not ro:
        return None
    return ro.url.host
