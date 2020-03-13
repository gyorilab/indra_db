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

DB_STR_FMT = "{prefix}://{username}{password}{host}{port}/{name}"
PRINCIPAL_ENV_PREFIX = 'INDRADB'
READONLY_ENV_PREFIX = 'INDRARO'
S3_DUMP_ENV_VAR = 'INDRA_DB_S3_PREFIX'
LAMBDA_NAME_ENV_VAR = 'DB_SERVICE_LAMBDA_NAME'


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

        # Handle the case for the s3 bucket spec.
        if section.startswith('aws-'):
            CONFIG[section[4:]] = def_dict
            continue

        # Extract all the database connection data
        if def_dict['host']:
            def_dict['host'] = '@' + def_dict['host']
        def_dict['prefix'] = def_dict['dialect']
        if def_dict['driver']:
            def_dict['prefix'] += def_dict['driver']
        if def_dict['port']:
            def_dict['port'] = ':' + def_dict['port']
        if def_dict['password']:
            def_dict['password'] = ':' + def_dict['password']

        # Get the role of the database
        url = DB_STR_FMT.format(**def_dict)
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

    return


def _load(include_config=True):
    global CONFIG
    CONFIG = {'databases': {}, 'readonly': {}, 's3_dump': {}}
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

    return S3Path(**CONFIG['s3_dump'])


_load(True)
