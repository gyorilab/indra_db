import sys
import logging
from functools import wraps
from os import path, mkdir, environ
from shutil import copyfile

from indra_db.util.s3_path import S3Path

if sys.version_info[0] == 3:
    from configparser import ConfigParser
else:
    from ConfigParser import ConfigParser

FILE_PATH = path.dirname(path.abspath(__file__))
DEFAULT_DB_CONFIG_PATH = path.join(FILE_PATH, 'resources/default_db_config.ini')

DB_CONFIG_DIR = path.expanduser('~/.config/indra')
DB_CONFIG_PATH = path.join(DB_CONFIG_DIR, 'db_config.ini')

PRINCIPAL_ENV_PREFIX = 'INDRADB'
READONLY_ENV_PREFIX = 'INDRARO'
S3_DUMP_ENV_VAR = 'INDRA_DB_S3_PREFIX'
LAMBDA_NAME_ENV_VAR = 'DB_SERVICE_LAMBDA_NAME'
TESTING_ENV_VAR = 'INDRA_DB_TESTING'


logger = logging.getLogger('db_config')


CONFIG_EXISTS = True
if not path.exists(DB_CONFIG_DIR):
    try:
        mkdir(DB_CONFIG_DIR)
    except Exception as e:
        CONFIG_EXISTS = False
        logger.warning("Unable to copy config dir: %s" % e)


if not path.exists(DB_CONFIG_PATH) and CONFIG_EXISTS:
    try:
        copyfile(DEFAULT_DB_CONFIG_PATH, DB_CONFIG_PATH)
    except Exception as e:
        CONFIG_EXISTS = False
        logger.warning("Unable to copy config file into config dir: %s" % e)


# In hindsight I profoundly wish I had done all of this with objects rather
# than globals, however this could lead to some profound backwards
# compatibility/signature change problems.
#
# Actually, for that matter, I wish I had made the config a simple JSON
# file rather than this cumbersome config format, but that presents even more
# widespread compatibility problems.
CONFIG = None


def build_db_url(**kwargs):
    fmt = "{prefix}://{username}{password}{host}{port}/{name}"

    # Extract all the database connection data
    if kwargs['host']:
        kwargs['host'] = '@' + kwargs['host']
    kwargs['prefix'] = kwargs.get('dialect', kwargs.get('prefix'))
    if kwargs.get('driver') and kwargs.get('prefix'):
        kwargs['prefix'] += kwargs['driver']
    if kwargs.get('port'):
        kwargs['port'] = ':' + str(kwargs['port'])
    if kwargs.get('password'):
        kwargs['password'] = ':' + kwargs['password']

    # Get the role of the database
    return fmt.format(**kwargs)


def _get_urls_from_env(prefix):
    return {k[len(prefix):].lower(): v
            for k, v in environ.items()
            if k.startswith(prefix)}


def _load_config():
    global CONFIG
    assert isinstance(CONFIG, dict)
    parser = ConfigParser()
    parser.read(DB_CONFIG_PATH)
    for section in parser.sections():
        def_dict = {k: parser.get(section, k)
                    for k in parser.options(section)}

        # Handle general parameters
        if section == 'general':
            if def_dict.pop('testing', None) == 'true':
                CONFIG['testing'] = True
            else:
                CONFIG['testing'] = False
            CONFIG.update(def_dict)
            continue

        # Handle the case for the s3 bucket spec.
        if section.startswith('aws-'):
            CONFIG[section[4:]] = def_dict
            continue

        url = build_db_url(**def_dict)
        if def_dict.get('role') == 'readonly':
            # Include the entry both with and without the -ro. This is only
            # needed when sometimes a readonly database has the same name
            # as a principal database (e.g. primary).
            if section.endswith('-ro'):
                CONFIG['readonly'][section[:-3]] = url
            CONFIG['readonly'][section] = url
        else:
            CONFIG['databases'][section] = url


def _load_env_config():
    assert CONFIG, "CONFIG must be defined BEFORE calling this function."

    CONFIG['databases'].update(_get_urls_from_env(PRINCIPAL_ENV_PREFIX))
    CONFIG['readonly'].update(_get_urls_from_env(READONLY_ENV_PREFIX))

    if S3_DUMP_ENV_VAR in environ:
        bucket, prefix = environ[S3_DUMP_ENV_VAR].split(':')
        CONFIG['s3_dump'] = {'bucket': bucket, 'prefix': prefix}

    if LAMBDA_NAME_ENV_VAR in environ:
        role, function = environ[LAMBDA_NAME_ENV_VAR].split(':')
        CONFIG['lambda'] = {'role': role, 'function': function}

    if TESTING_ENV_VAR in environ:
        env_is_testing = environ[TESTING_ENV_VAR].lower()
        if env_is_testing == 'true':
            CONFIG['testing'] = True
        else:
            CONFIG['testing'] = False

    return


def _load(include_config=True):
    global CONFIG
    CONFIG = {'databases': {}, 'readonly': {}, 's3_dump': {}, 'lambda': {},
              'testing': False}
    if CONFIG_EXISTS and include_config:
        _load_config()
    _load_env_config()
    return


def get_databases(force_update=False, include_config=True):
    if not CONFIG or force_update:
        _load(include_config)

    return CONFIG['databases']


def get_readonly_databases(force_update=False, include_config=True):
    if not CONFIG or force_update:
        _load(include_config)

    return CONFIG['readonly']


def get_s3_dump(force_update=False, include_config=True):
    if not CONFIG or force_update:
        _load(include_config)

    if 's3_dump' not in CONFIG:
        return None

    return S3Path(CONFIG['s3_dump']['bucket'], CONFIG['s3_dump'].get('prefix'))


_load(True)


# ====================
# Testing config tools
# ====================


def run_in_test_mode(func):
    """Run the wrapped function in testing mode."""
    @wraps(func)
    def wrap_func(*args, **kwargs):
        global CONFIG
        assert isinstance(CONFIG, dict)
        global TEST_RECORDS
        TEST_RECORDS = []
        orig_value = CONFIG['testing']
        CONFIG['testing'] = True
        try:
            ret = func(*args, **kwargs)
        finally:
            CONFIG['testing'] = orig_value
        return ret
    return wrap_func


def is_db_testing():
    """Check whether we are in testing mode."""
    assert isinstance(CONFIG, dict)
    return CONFIG.get('testing', False)


class WontDoIt(Exception):
    """Raised in testing mode when an off-limits function is called."""
    pass


def nope_in_test(func):
    """Raise an error if the wrapped function is used in "test mode".

    You can enter "test mode" by wrapping a function or method (usually a test)
    with `run_in_test_mode`.
    """
    @wraps(func)
    def wrap_func(*args, **kwargs):
        if is_db_testing():
            raise WontDoIt(f"Cannot run {func.__name__} during test.")
        return func(*args, **kwargs)
    return wrap_func


TEST_RECORDS = None


class TestFuncCallRecord:
    """Info on a func wrapped with `record_in_test` and called during a test."""
    def __init__(self, function, args, kwargs):
        import inspect
        from datetime import datetime

        self.func_name = function.__name__
        self._function = function
        if inspect.ismethod(function) or hasattr(function, '__self__'):
            self.meth_class = function.__self__.__class__.__name__
        else:
            self.meth_class = None

        self.args = args
        self.kwargs = kwargs
        self.called_at = datetime.utcnow()

    def __fullname(self):
        if self.meth_class is not None:
            return f"{self.meth_class}.{self.func_name}"
        else:
            return self.func_name

    def __str__(self):
        return f"{self.__fullname()}({', '.join(self.args)}, " \
               f"{',  '.join(f'{k}={v}' for k, v in self.kwargs.items())})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__fullname()} " \
               f"at {self.called_at})"


def record_in_test(func):
    """Record the function call and args but do nothing during a test.

    The resulting calls can be retrieved using `get_test_call_records`.
    """
    @wraps(func)
    def wrap_func(*args, **kwargs):
        if is_db_testing():
            assert isinstance(TEST_RECORDS, list)
            TEST_RECORDS.append(TestFuncCallRecord(func, args, kwargs))
        else:
            return func(*args, **kwargs)
    return wrap_func


def get_test_call_records():
    """Get the records of functions avoided during a test."""
    if TEST_RECORDS is None:
        return
    return TEST_RECORDS[:]

