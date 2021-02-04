__all__ = ['submit_curation', 'get_curations', 'get_grounding_curations']

import json
import re
import logging
import datetime
from collections import Counter

from sqlalchemy.exc import IntegrityError

from indra_db import get_db
from indra_db.exceptions import BadHashError

logger = logging.getLogger(__name__)


def submit_curation(hash_val, tag, curator, ip, text=None, ev_hash=None,
                    source='direct_client', pa_json=None, ev_json=None,
                    db=None):
    """Submit a curation for a given preassembled or raw extraction.

    Parameters
    ----------
    hash_val : int
        The hash corresponding to the statement.
    tag : str
        A very short phrase categorizing the error or type of curation.
    curator : str
        The name or identifier for the curator.
    ip : str
        The ip address of user's computer.
    text : str
        A brief description of the problem.
    ev_hash : int
        A hash of the sentence and other evidence information. Elsewhere
        referred to as `source_hash`.
    source : str
        The name of the access point through which the curation was performed.
        The default is 'direct_client', meaning this function was used
        directly. Any higher-level application should identify itself here.
    pa_json : Optional[dict]
        The JSON of a preassembled or raw statement that was curated. If None,
        we will try to get the pa_json from the database.
    ev_json : Optional[dict]
        The JSON of the evidence that was curated. This cannot be retrieved from
        the database if not given.
    db : DatabaseManager
        A database manager object used to access the database.
    """
    if db is None:
        db = get_db('primary')

    if pa_json is None:
        pa_json_strs = db.select_one(db.PAStatements.json,
                                     db.PAStatements.mk_hash == int(hash_val))
        if pa_json_strs is not None:
            pa_json = json.loads(pa_json_strs[0])

    inp = {'tag': tag, 'text': text, 'curator': curator, 'ip': ip,
           'source': source, 'pa_hash': hash_val, 'source_hash': ev_hash,
           'pa_json': pa_json, 'ev_json': ev_json}

    logger.info("Adding curation: %s" % str(inp))

    try:
        dbid = db.insert(db.Curation, **inp)
    except IntegrityError as e:
        logger.error("Got a bad entry.")
        msg = e.args[0]
        detail_line = msg.splitlines()[1]
        m = re.match("DETAIL: .*?\(pa_hash\)=\((\d+)\).*?not present.*?pa.*?",
                     detail_line)
        if m is None:
            raise e
        else:
            h = m.groups()[0]
            assert int(h) == int(hash_val), \
                "Erred hash %s does not match input hash %s." % (h, hash_val)
            logger.error("Bad hash: %s" % h)
            raise BadHashError(h)
    return dbid


def get_curations(db=None, **params):
    """Get all curations for a certain level given certain criteria."""
    if db is None:
        db = get_db('primary')
    cur = db.Curation

    constraints = []
    for key, val in params.items():
        if key == 'hash_val':
            key = 'pa_hash'
        elif key == 'ev_hash':
            key = 'source_hash'

        if isinstance(val, list) or isinstance(val, set) \
           or isinstance(val, tuple):
            constraints.append(getattr(cur, key).in_(val))
        else:
            constraints.append(getattr(cur, key) == val)

    return [c.to_json() for c in db.select_all(cur, *constraints)]


def get_grounding_curations(db=None):
    """Return a dict of curated groundings from a given database.

    Parameters
    ----------
    db : Optional[DatabaseManager]
        A database manager object used to access the database. If not given,
        the database configured as primary is used.

    Returns
    -------
    dict
        A dict whose keys are raw text strings and whose values are dicts of DB
        name space to DB ID mappings corresponding to the curated grounding.
    """
    # Get all the grounding curations
    curs = get_curations(db=db, tag='grounding')
    groundings = {}
    for cur in curs:
        # If there is no curation given, we skip it
        if not cur['text']:
            continue
        # We now try to match the standard pattern for grounding curation
        cur_text = cur['text'].strip()
        match = re.match('^\[(.*)\] -> ([^ ]+)$', cur_text)
        # We log any instances of curations that don't match the pattern
        if not match:
            logger.info('"%s" by %s does not match the grounding curation '
                        'pattern.' % (cur_text, cur['curator']))
            continue
        txt, dbid_str = match.groups()
        # We now get a dict of curated mappings to return
        try:
            dbid_entries = [entry.split(':', maxsplit=1)
                            for entry in dbid_str.split('|')]
            dbids = {k: v for k, v in dbid_entries}
        except Exception as e:
            logger.info('Could not interpret DB IDs: %s for %s' %
                        (dbid_str, txt))
            continue
        if txt in groundings and groundings[txt] != dbids:
            logger.info('There is already a curation for %s: %s, '
                        'overwriting with %s' % (txt, str(groundings[txt]),
                                                 str(dbids)))
        groundings[txt] = dbids
    return groundings


def get_curator_counts(db=None):
    """Return a Counter of the number of curations submitted by each user.

    Parameters
    ----------
    db : Optional[DatabaseManager]
        A database manager object used to access the database. If not given,
        the database configured as primary is used.

    Returns
    -------
    collections.Counter
        A Counter of curator users by the number of curations they have
        submitted.
    """
    if db is None:
        db = get_db('primary')
    res = db.select_all(db.Curation)
    curators = [r.curator for r in res]
    counter = Counter(curators)
    return counter


def plot_curators(curator_counter, topk=10, fname=None):
    """Plot curation statistics based on curation counts per user.

    Parameters
    ----------
    curator_counter : collections.Counter
        A Counter of curator users by the number of curations they have
        submitted.
    topk : Optional[int]
        Only plot the top k curators, Default: 10
    fname : Optional[str]
        If provided, an image of the plot with the given file name is saved.
        Otherwise the plot is just displayed.
    """
    import matplotlib.pyplot as plt
    # Get today's date
    today = datetime.datetime.today()
    today_str = today.strftime('%Y-%m-%d')

    # Just get the top k
    sorted_curators = curator_counter.most_common(topk)
    curator_names = [c[0].replace('@', '@\n') if '@' else c[0]
                     for c in sorted_curators]
    ticks = range(len(sorted_curators)-1, -1, -1)
    plt.barh(ticks, [c[1] for c in sorted_curators], color='red')
    plt.yticks(ticks, curator_names)

    plt.title('Curation statistics as of %s' % today_str)
    plt.xlabel('Number of curations')
    plt.subplots_adjust(left=0.21, right=0.97, top=0.91, bottom=0.11)
    if fname is not None:
        plt.savefig(fname)
        return fname
    else:
        plt.show()
    return
