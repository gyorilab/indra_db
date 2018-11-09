import logging

logger = logging.getLogger("db_curation_client")

from indra_db.util import get_primary_db
from indra_db.exceptions import NoAuthError


def submit_curation(hash_val, tag, curator, ip, api_key, text=None,
                    ev_hash=None, source='direct_client', db=None):
    """Submit a curation for a given preassembled or raw extraction.

    Parameters
    ----------
    level : 'pa' or 'raw'
        This indicates the level of curation, whether at the single extraction/
        sentence level ('raw'), or a the de-duplicated, logical level ('pa').
    hash_val : int
        The hash corresponding to the statement.
    tag : str
        A very short phrase categorizing the error or type of curation.
    curator : str
        The name or identifier for the curator.
    ip : str
        The ip address of user's computer.
    api_key : str
        If you have one, this can help identify you as a curator, and may lend
        extra weight to your curation(s).
    text : str
        A brief description of the problem.
    ev_hash : int
        A hash of the sentence and other evidence information. Elsewhere
        referred to as `source_hash`.
    source : str
        The name of the access point through which the curation was performed.
        The default is 'direct_client', meaning this function was used
        directly. Any higher-level application should identify itself here.
    db : DatabaseManager
        A database manager object used to access the database.
    """
    if db is None:
        db = get_primary_db()

    inp = {'tag': tag, 'text': text, 'curator': curator, 'ip': ip,
           'source': source, 'pa_hash': hash_val, 'source_hash': ev_hash}

    auth = db._get_auth_info(api_key)
    if auth is None:
        raise NoAuthError(api_key, 'curation')
    inp['auth_id'] = auth[0]

    logger.info("Adding curation: %s" % str(inp))

    dbid = db.insert(db.Curation, **inp)
    return dbid


def get_curations(db=None, **params):
    """Get all curations for a certain level given certain criteria."""
    if db is None:
        db = get_primary_db()
    cur = db.Curation

    constraints = []
    for key, val in params.items():
        if key == 'hash_val':
            key = 'pa_hash'
        if key == 'ev_hash':
            key = 'source_hash'
        if isinstance(val, list) or isinstance(val, set) \
           or isinstance(val, tuple):
            constraints.append(getattr(cur, key).in_(val))
        else:
            constraints.append(getattr(cur, key) == val)

    return db.select_all(cur, *constraints)
