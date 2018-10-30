from indra_db.util import get_primary_db
from indra_db.exceptions import NoAuthError


def submit_curation(level, hash_val, tag, text, curator, ip, api_key,
                    source='direct_client', db=None):
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
    text : str
        A brief description of the problem.
    curator : str
        The name or identifier for the curator.
    ip : str
        The ip address of user's computer.
    source : str
        The name of the access point through which the curation was performed.
        The default is 'direct_client', meaning this function was used
        directly. Any higher-level application should identify itself here.
    api_key : str
        If you have one, this can help identify you as a curator, and may lend
        extra weight to your curation(s).
    db : DatabaseManager
        A database manager object used to access the database.
    """
    if db is None:
        db = get_primary_db()

    if level not in ['pa', 'raw']:
        raise ValueError('Level must be either "pa" or "raw", but got "%s".'
                         % level)

    inp = {'tag': tag, 'text': text, 'curator': curator, 'ip': ip,
           'source': source}

    auth = db._get_auth_info(api_key)
    if auth is None:
        raise NoAuthError(api_key, 'curation')
    inp['auth_id'] = auth[0]

    if level == 'pa':
        cur = db.PACuration
        inp['pa_hash'] = hash_val
    else:
        cur = db.RawCuration
        inp['raw_hash'] = hash_val

    db.insert(cur, **inp)
    return


def get_curations(level, db=None, **params):
    """Get all curations for a certain level given certain criteria."""
    if db is None:
        db = get_primary_db()

    if level not in ['pa', 'raw']:
        raise ValueError('Level must be either "pa" or "raw", but got "%s".'
                         % level)
    if level == 'pa':
        cur = db.PACuration
        hash_key = 'pa_hash'
    else:
        cur = db.RawCuration
        hash_key = 'raw_hash'

    constraints = []
    for key, val in params.items():
        if key == 'hash_val':
            key = hash_key
        if isinstance(val, list) or isinstance(val, set) \
           or isinstance(val, tuple):
            constraints.append(getattr(cur, key).in_(val))
        else:
            constraints.append(getattr(cur, key) == val)

    return db.select_all(cur, *constraints)
