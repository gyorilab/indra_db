__all__ = ['get_stmts_with_agent_text_like', 'get_text_content_from_stmt_ids']

import json
from sqlalchemy import text
from collections import defaultdict
from cachetools.keys import hashkey
from contextlib import contextmanager
from cachetools import cached, LRUCache


from .constructors import get_db
from .helpers import unpack, _get_trids


@contextmanager
def managed_db(db_label='primary', protected=False):
    db = get_db(db_label, protected)
    try:
        yield db
    finally:
        db.session.rollback()
        db.session.close()


@cached(cache=LRUCache(maxsize=1024))
def get_pa_statements_for_pair(curie1, curie2):
    """Return dict with info for preassembled statements connecting two agents

    Parameters
    ----------
    curie1 : str
       String of the form f'{namespace}:{identifier}' such as
       'HGNC:6091' or 'FPLX:PI3K'.

    curie2: str
        See above
    
    Returns
    -------
    dict
        Dictionary mapping stmt_mk_hashes for preassembled statements to
        statement types.
    """
    query = """--
    SELECT
        pa1.stmt_mk_hash, pa1.db_name, pa1.db_id,
        pa2.db_name, pa2.db_id, ps.type, ps.json
    FROM
        pa_agents pa1
    INNER JOIN
        pa_agents pa2
    ON
        pa1.stmt_mk_hash = pa2.stmt_mk_hash AND
        MD5(pa1.db_name || pa1.db_id) = MD5(:db_ns1 || :db_id1) AND
        MD5(pa2.db_name || pa2.db_id) = MD5(:db_ns2 || :db_id2) AND
        pa1.role != 'OBJECT' AND pa2.role != 'SUBJECT'
    INNER JOIN
        pa_statements ps
    ON
        pa2.stmt_mk_hash = ps.mk_hash
    """
    db_ns1, db_id1 = curie1.split(':', maxsplit=1)
    db_ns2, db_id2 = curie2.split(':', maxsplit=1)
    with managed_db() as db:
        res = db.session.execute(text(query),
                                 {'db_ns1': db_ns1, 'db_id1': db_id1,
                                  'db_ns2': db_ns2, 'db_id2': db_id2})
    # Although absurdly unlikely, we filter MD5 hash collisions just
    # on principle. Also filter complexes with more than two members
    return {stmt_mk_hash: stmt_type for
            stmt_mk_hash, db_name1, id1, db_name2, id2,
            stmt_type, stmt_json in res
            if (stmt_type != 'Complex' or
                len(json.loads(stmt_json.tobytes())['members']) == 2) and
            db_name1 == db_ns1 and id1 == db_id1 and
            db_name2 == db_ns2 and id2 == db_id2}


def get_reach_support_for_pa_statements(stmt_mk_hashes):
    """Return reading_ids and raw_stmt_ids of reach support for input

    Parameters
    ----------
    stmt_mk_hashes : list of int
        List of stmt_mk_hashes for preassembled statements

    Returns
    -------
    generator of tuple
        yields tuples of the form 
        (stmt_mk_hash, raw_stmt_id, reading_id)
        where stmt_mk_hash is the mk_hash of a preassembled statement in the
        input, raw_stmt_id is a raw statement id for a raw statement from REACH
        which supports the preassembled statement, and reading_id is that
        associated reading id in the reading table.
    """
    query = """--
    SELECT rl.pa_stmt_mk_hash, rs.id, rs.reading_id
    FROM
        raw_unique_links rl
    INNER JOIN
        raw_statements rs
    ON 
        rl.pa_stmt_mk_hash IN :stmt_mk_hashes AND
        rl.raw_stmt_id = rs.id
    INNER JOIN
        reading rd
    ON
        rs.reading_id = rd.id AND
        rd.reader = 'REACH'
    """
    with managed_db() as db:
        res = db.session.execute(text(query),
                                 {'stmt_mk_hashes': tuple(set(stmt_mk_hashes))})
    return ((stmt_mk_hash, raw_stmt_id,
             reading_id) for stmt_mk_hash, raw_stmt_id, reading_id
            in res)


def get_readings_for_reading_ids(reading_ids):
    """Get json output associated to reading ids

    Parameters
    ----------
    reading_ids : list of ints
        reading ids for rows in readings table

    Returns
    -------
    dict
        dict mapping reading ids to jsons of reading output
    """
    query = 'SELECT id, bytes FROM reading WHERE id IN :reading_ids'
    with managed_db() as db:
        res = db.session.execute(text(query),
                                 {'reading_ids': tuple(set(reading_ids))})
    return {reading_id: json.loads(unpack(bytes_))
            for reading_id, bytes_ in res}


def get_raw_statement_jsons(stmt_ids):
    """Get statement jsons associated to each in a list of raw statement ids

    Parameters
    ----------
    stmt_ids : list of int
        list of raw statement ids

    Returns
    --------
    dict
        dict mapping raw statement ids to statement jsons.
    """
    query = 'SELECT id, json FROM raw_statements WHERE id in :stmt_ids'
    with managed_db() as db:
        res = db.session.execute(text(query),
                                 {'stmt_ids': tuple(set(stmt_ids))})
    return {stmt_id: json.loads(json_.tobytes()) for stmt_id, json_ in res}


def get_stmts_with_agent_text_like(pattern, filter_genes=False,
                                   db=None):

    """Get statement ids with agent with rawtext matching pattern


    Parameters
    ----------
    pattern : str
        a pattern understood by sqlalchemy's like operator.
        For example '__' for two letter agents

    filter_genes : Optional[bool]
       if True, only returns map for agent texts for which there is at least
       one HGNC grounding in the database. Default: False

    db : Optional[:py:class:`DatabaseManager`]
        User has the option to pass in a database manager. If None
        the primary database is used. Default: None

    Returns
    -------
    dict
        dict mapping agent texts to statement ids. agent text are those
        matching the input pattern. Each agent text maps to the list of
        statement ids for statements containing an agent with that TEXT
        in its db_refs
    """
    if db is None:
        db = get_db('primary')

    # Query Raw agents table for agents with TEXT db_ref matching pattern
    # Selects agent texts, statement ids and agent numbers. The agent number
    # corresponds to the agents index into the agent list
    agents = db.select_all([db.RawAgents.db_id,
                            db.RawAgents.stmt_id,
                            db.RawAgents.ag_num],
                           db.RawAgents.db_name.like('TEXT'),
                           db.RawAgents.db_id.like(pattern),
                           db.RawAgents.stmt_id.isnot(None))
    if filter_genes:
        # If filtering to only genes, get statement ids and agent numbers
        # for all agents grounded to HGNC. Check if agent text has been
        # grounded to HGNC at least once
        hgnc_agents = db.select_all([db.RawAgents.stmt_id,
                                     db.RawAgents.ag_num],
                                    db.RawAgents.db_name.like('HGNC'),
                                    db.RawAgents.stmt_id.isnot(None))
        hgnc_agents = set(hgnc_agents)
        agents = [(agent_text, stmt_id, ag_num)
                  for agent_text, stmt_id, ag_num in agents
                  if (stmt_id, ag_num) in hgnc_agents]
    output = defaultdict(list)
    for agent_text, stmt_id, ag_num in agents:
        if stmt_id not in output[agent_text]:
            output[agent_text].append(stmt_id)
    return dict(output)


def get_stmts_with_agent_text_in(agent_texts, filter_genes=False, db=None):
    """Get statement ids with agent with rawtext in list


    Parameters
    ----------
    agent_texts : list of str
        a list of agent texts

    filter_genes : Optional[bool]
        if True, only returns map for agent texts for which there is at least
        one HGNC grounding in the database. Default: False

    db : Optional[:py:class:`DatabaseManager`]
        User has the option to pass in a database manager. If None
        the primary database is used. Default: None

    Returns
    -------
    dict
        dict mapping agent texts to lists of statement ids for statements
        containing an agent with that TEXT in its db_refs.
    """
    if db is None:
        db = get_db('primary')

    # Query Raw agents table for agents with TEXT db_ref matching pattern
    # Selects agent texts, statement ids and agent numbers. The agent number
    # corresponds to the agents index into the agent list
    agents = db.select_all([db.RawAgents.db_id,
                            db.RawAgents.stmt_id,
                            db.RawAgents.ag_num],
                           db.RawAgents.db_name.like('TEXT'),
                           db.RawAgents.stmt_id.isnot(None))
    agents = [(agent_text, stmt_id, ag_num)
              for agent_text, stmt_id, ag_num in agents
              if agent_text in agent_texts]
    if filter_genes:
        # If filtering to only genes, get statement ids and agent numbers
        # for all agents grounded to HGNC. Check if agent text has been
        # grounded to HGNC at least once
        hgnc_agents = db.select_all([db.RawAgents.stmt_id,
                                     db.RawAgents.ag_num],
                                    db.RawAgents.db_name.like('HGNC'),
                                    db.RawAgents.stmt_id.isnot(None))
        hgnc_agents = set(hgnc_agents)
        agents = [(agent_text, stmt_id, ag_num)
                  for agent_text, stmt_id, ag_num in agents
                  if (stmt_id, ag_num) in hgnc_agents]
    output = defaultdict(list)
    for agent_text, stmt_id, ag_num in agents:
        if stmt_id not in output[agent_text]:
            output[agent_text].append(stmt_id)
    return dict(output)


def get_text_content_from_pmids(pmids, db=None):
    """Get best available text content for list of pmids

    For each pmid, gets the best piece of text content with the priority
    fulltext > abstract > title.

    Parameters
    ----------
    pmids : list of str

    db : Optional[:py:class:`DatabaseManager`]
        User has the option to pass in a database manager. If None
        the primary database is used. Default: None

    Returns
    -------
    identifiers : dict
        dict mapping pmids to identifiers for pieces of content.
        These identifiers are tuples of the form
        (text_ref_id, source, text_type). Each tuple uniquely specifies
        a piece of content in the database
        No entries exist for statements with no associated text content
        (these typically come from databases)

    content : dict
        dict mapping content identifiers used as values in the ref_dict
        to the best available text content.
    """
    if db is None:
        db = get_db('primary')
    identifiers = get_content_identifiers_from_pmids(pmids)
    content = _get_text_content(identifiers.values())
    return identifiers, content


def get_content_identifiers_from_pmids(pmids, db=None):
    """Get content identifiers from list of pmids

    An identifier is a triple containing a text_ref_id, source, and text_type
    Gets the identifier for best piece of text content with priority
    fulltext > abstract > title

    Parameters
    ----------
    pmids : list of str

    db : Optional[:py:class:`DatabaseManager`]
        User has the option to pass in a database manager. If None
        the primary is used. Default: None

    Returns
    -------
    ref_dict: dict
        dict mapping statement ids to identifiers for pieces of content.
        These identifiers take the form `<text_ref_id>/<source>/<text_type>'.
        No entries exist for statements with no associated text content
        (these typically come from databases)


    text_dict: dict
        dict mapping content identifiers used as values in the ref_dict
        to best available text content. The order of preference is
        fulltext xml > plaintext abstract > title
    """
    if db is None:
        db = get_db('primary')
    pmids = tuple(set(pmids))
    query = """SELECT
                   tr.pmid, tr.id, tc.source, tc.format, tc.text_type
               FROM
                   text_content AS tc
               JOIN
                   text_ref as tr
               ON
                   tr.id = tc.text_ref_id
               WHERE
                   tr.pmid IN :pmids
            """
    res = db.session.execute(text(query), {'pmids': pmids})
    return _collect_content_identifiers(res)


def _collect_content_identifiers(res):
    priority = {'fulltext': 2, 'abstract': 1, 'title': 0}
    seen_text_refs = {}
    ref_dict = {}
    for id_, text_ref_id, source, format_, text_type in res.fetchall():
        new_identifier = (text_ref_id, source, format_, text_type)
        if (id_, text_ref_id) not in seen_text_refs:
            seen_text_refs[(id_, text_ref_id)] = new_identifier
            ref_dict[id_] = new_identifier
        else:
            # update if we find text_type with higher priority for
            # a given text_ref
            old_identifier = seen_text_refs[(id_, text_ref_id)]
            old_text_type = old_identifier[3]
            if priority[text_type] > priority[old_text_type]:
                seen_text_refs[(id_, text_ref_id)] = new_identifier
                ref_dict[id_] = new_identifier
    return ref_dict


def _get_text_content(content_identifiers, db=None):
    """Return text_content associated to a list of content identifiers

    Parameters
    ----------
    content_identifiers : iterable of tuple
        A content identifier is a triple with three elements, text_ref_id,
        source, and text_type. These three pieces of information uniquely
        specify a piece of content in the database. content_identifiers
        is a list of these triples

     db : Optional[:py:class:`DatabaseManager`]
        User has the option to pass in a database manager. If None
        the primary database is used. Default: None

    Returns
    -------
    dict
        A dictionary mapping content identifiers to pieces of
        text content. content identifiers for which no content
        exists in the database are excluded as keys.
    """
    if db is None:
        db = get_db('primary')
    # Remove duplicate identifiers
    content_identifiers = set(content_identifiers)
    # Query finds content associated to each identifier by joining
    # the text_content table with a virtual table containing the
    # input identifiers. The query string is generated programmatically
    id_str = ', '.join('(:trid%d, :source%d, :format%d, :text_type%d)'
                       % (i, i, i, i)
                       for i in range(len(content_identifiers)))
    params = {}
    for i, (trid, source,
            format_, text_type) in enumerate(content_identifiers):
        params.update({'trid%s' % i: trid,
                       'source%i' % i: source,
                       'format%i' % i: format_,
                       'text_type%i' % i: text_type})
    query = """SELECT
                   tc.text_ref_id, tc.source, tc.format, tc.text_type, content
               FROM
                   text_content AS tc
               JOIN (VALUES %s)
               AS
                  ids (text_ref_id, source, format, text_type)
               ON
                   tc.text_ref_id = ids.text_ref_id
                   AND tc.source = ids.source
                   AND tc.format = ids.format
                   AND tc.text_type = ids.text_type
            """ % id_str

    res = db.session.execute(text(query), params)
    return {(trid, source, format, text_type): unpack(content)
            for trid, source, format, text_type, content in res}


class TextContentSessionHandler(object):
    """Allows querying of text content from text_refs

    Doesn't directly expose the db.

    Parameters
    ----------
    db : Optional[:py:class:`DatabaseManager`]
        User has the option to pass in a database manager. If None
        the primary database is used. Default: None
    """
    def __init__(self, db=None):
        default = False
        if db is None:
            db = get_db('primary')
            default = True
        self.__db = db
        self.default = default

    def close(self):
        self.__db.session.rollback()
        self.__db.session.close()

    def get_text_content_from_text_refs(self, text_refs, use_cache=True):
        """Get text_content from an evidence object's text_refs attribute


        Parameters
        ----------
        text_refs : dict of str: str
            text_refs dictionary as contained in an evidence object
            The dictionary should be keyed on id_types. The valid keys
            are 'PMID', 'PMCID', 'DOI', 'PII', 'URL', 'MANUSCRIPT_ID'.

        use_cache : Optional[bool]
            Whether or not to use cached results. Only relevant when
            querying the primary database. Will not work if primary
            database is passed in with keyword argument. Only if
            keyword db argument is absent or set to None.
            Default: True

        Returns
        -------
        text : str
            fulltext corresponding to the text_refs if it exists in the
            database, otherwise the abstract. Returns None if no content
            exists for the text_refs in the database
        """
        if self.default and use_cache:
            frozen_text_refs = frozenset(text_refs.items())
            result = self.\
                _get_text_content_from_text_refs_cached(frozen_text_refs)
        else:
            text_ref_id = self._get_text_ref_id_from_text_refs(text_refs)
            if text_ref_id is None:
                result = None
            else:
                result = self._get_text_content_from_trid(text_ref_id)
        return result

    @cached(cache=LRUCache(maxsize=10000),
            key=lambda self, frozen_text_refs: hashkey(frozen_text_refs))
    def _get_text_content_from_text_refs_cached(self, frozen_text_refs):
        text_refs = dict(frozen_text_refs)
        text_ref_id = self._get_text_ref_id_from_text_refs(text_refs)
        if text_ref_id is None:
            result = None
        else:
            result = self._get_text_content_from_trid(text_ref_id)
        return result

    def _get_text_ref_id_from_text_refs(self, text_refs):
        # In some cases the TRID is already there so we can just
        # return it
        if 'TRID' in text_refs:
            return text_refs['TRID']
        text_ref_id = None
        for id_type in ['pmid', 'pmcid', 'doi',
                        'pii', 'url', 'manuscript_id']:
            try:
                id_val = text_refs[id_type.upper()]
                trids = _get_trids(self.__db, id_val, id_type)
                if trids:
                    text_ref_id = trids[0]
                    break
            except KeyError:
                pass
        return text_ref_id

    def _get_text_content_from_trid(self, text_ref_id):
        texts = self.__db.select_all([self.__db.TextContent.content,
                                      self.__db.TextContent.text_type],
                                     self.__db.TextContent.text_ref_id ==
                                     text_ref_id)
        contents = defaultdict(list)
        for content, text_type in texts:
            contents[text_type].append(content)
        # Look at text types in order of priority
        for text_type in ('fulltext', 'abstract', 'title'):
            # There are cases when we get a list of results for the same
            # content type with some that are None and some actual content,
            # so we iterate to find a non-empty content to return
            for content in contents.get(text_type, []):
                if content:
                    return unpack(content)
        return None


def _extract_db_refs(stmt_json):
    agent_types = ['sub', 'subj', 'obj', 'enz', 'agent', 'gef;', 'ras',
                   'gap', 'obj_from', 'obj_to']
    db_ref_list = []

    for agent_type in agent_types:
        try:
            agent = stmt_json[agent_type]
        except KeyError:
            continue
        try:
            db_refs = agent['db_refs']
        except KeyError:
            continue
        db_ref_list.append(db_refs)

    members = stmt_json.get('members')
    if members is not None:
        for member in members:
            try:
                db_refs = member['db_refs']
            except KeyError:
                continue
            db_ref_list.append(db_refs)
    return db_ref_list
