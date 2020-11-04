import sys
import logging
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
    parser = ConfigParser()
    parser.read(DB_CONFIG_PATH)
    for section in parser.sections():
        def_dict = {k: parser.get(section, k)
                    for k in parser.options(section)}

        # Handle general parameters
        if section == 'general':
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
        elif env_is_testing == 'false':
            CONFIG['testing'] = False
        else:
            raise ValueError(f"Unknown option: {environ[TESTING_ENV_VAR]}, "
                             f"should be \"true\" or \"false\"")

    return


def run_in_db_test_mode(func):
    _load()

    def wrap_func(*args, **kwargs):
        global CONFIG
        orig_value = CONFIG['testing']
        CONFIG['testing'] = True
        ret = func(*args, **kwargs)
        CONFIG['testing'] = orig_value
        return ret
    return wrap_func


def is_db_testing():
    return CONFIG['testing']


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
