__all__ = ['get_stmts_with_agent_text_like', 'get_text_content_from_stmt_ids']


import json
from collections import defaultdict


from .constructors import get_primary_db
from .helpers import unpack, _get_trids


def get_stmts_with_agent_text_like(pattern, filter_genes=False):
    """Get statement ids with agent with rawtext matching pattern


    Parameters
    ----------
    pattern : str
        a pattern understood by sqlalchemy's like operator.
        For example '__' for two letter agents

    filter_genes : Optional[bool]
       if True, only considers agents matching the pattern for which there
       is an HGNC grounding

    Returns
    -------
    dict
        dict mapping agent texts matching the input pattern to lists of
        ids for statements with at least one agent with raw text matching
        the pattern.
    """
    db = get_primary_db()

    agents = db.select_all([db.RawAgents.db_id,
                            db.RawAgents.stmt_id],
                           db.RawAgents.db_name == 'TEXT',
                           db.RawAgents.db_id.like(pattern),
                           db.RawAgents.stmt_id.isnot(None))
    stmt_dict = defaultdict(list)
    for agent_text, stmt_id in agents:
        stmt_dict[agent_text].append(stmt_id)
    if not filter_genes:
        return dict(stmt_dict)

    output = defaultdict(list)
    unique_stmts = set(stmt_id for stmts in stmt_dict.values()
                       for stmt_id in stmts)
    stmt_jsons = db.select_all([db.RawStatements.id,
                                db.RawStatements.json],
                               db.RawStatements.id.in_(unique_stmts))
    json_dict = {stmt_id: json.loads(jsn) for stmt_id, jsn in stmt_jsons}
    for agent_text, stmts in stmt_dict.items():
        for stmt_id in stmts:
            stmt_json = json_dict[stmt_id]
            db_refs = _extract_db_refs(stmt_json)
            for ref in db_refs:
                if ('TEXT' in ref and 'HGNC' in ref and
                    ref['TEXT'] == agent_text and
                        stmt_id not in output[agent_text]):
                    output[agent_text].append(stmt_id)
    return dict(output)


def get_stmts_with_agent_text_in(texts, filter_genes=False,
                                 batch_size=100):
    """Get statement ids with agent with rawtext in list


    Parameters
    ----------
    tests : list of str
        a list of agent texts

    filter_genes : Optional[bool]
       if True, only considers agents matching the pattern for which there
       is an HGNC grounding

    Returns
    -------
    dict
        dict mapping agent texts to lists of stmt_ids for statements
        containing an agent with the given text
    """
    db = get_primary_db()
    agents = db.select_all([db.RawAgents.db_id,
                            db.RawAgents.stmt_id],
                           db.RawAgents.db_name == 'TEXT',
                           db.RawAgents.stmt_id.isnot(None))
    stmt_dict = defaultdict(list)
    for agent_text, stmt_id in agents:
        if agent_text in texts:
            stmt_dict[agent_text].append(stmt_id)
    if not filter_genes:
        return dict(stmt_dict)

    output = defaultdict(list)
    unique_stmts = set(stmt_id for stmts in stmt_dict.values()
                       for stmt_id in stmts)
    stmt_jsons = db.select_all([db.RawStatements.id,
                                db.RawStatements.json],
                               db.RawStatements.id.in_(unique_stmts))
    json_dict = {stmt_id: json.loads(jsn) for stmt_id, jsn in stmt_jsons}
    for agent_text, stmts in stmt_dict.items():
        for stmt_id in stmts:
            stmt_json = json_dict[stmt_id]
            db_refs = _extract_db_refs(stmt_json)
            for ref in db_refs:
                if ('TEXT' in ref and 'HGNC' in ref and
                    ref['TEXT'] == agent_text and
                        stmt_id not in output[agent_text]):
                    output[agent_text].append(stmt_id)
    return dict(output)


def get_text_content_from_stmt_ids(stmt_ids):
    """Get text content for statements from a list of ids

    Gets the fulltext if it is available, even if the statement came from an
    abstract.

    Parameters
    ----------
    stmt_ids : list of str

    Returns
    -------
    dict of str: str
        dictionary mapping statement ids to text content. Uses fulltext
        if one is available, falls back upon using the abstract.
        A statement id will map to None if no text content is available.
    """
    db = get_primary_db()
    text_refs = db.select_all([db.RawStatements.id, db.TextRef.id],
                              db.RawStatements.id.in_(stmt_ids),
                              *db.link(db.RawStatements, db.TextRef))
    text_refs = dict(text_refs)
    texts = db.select_all([db.TextContent.text_ref_id,
                           db.TextContent.content,
                           db.TextContent.text_type],
                          db.TextContent.text_ref_id.in_(text_refs.values()))
    fulltexts = {text_id: unpack(text)
                 for text_id, text, text_type in texts
                 if text_type == 'fulltext'}
    abstracts = {text_id: unpack(text)
                 for text_id, text, text_type in texts
                 if text_type == 'abstract'}
    ref_dict = {}
    text_dict = {}
    for stmt_id in stmt_ids:
        # first check if we have text content for this statement
        try:
            text_ref = text_refs[stmt_id]
        except KeyError:
            # if not, set fulltext to None
            ref_dict[stmt_id] = None
            continue
        ref_dict[stmt_id] = text_ref
        fulltext = fulltexts.get(text_ref)
        abstract = abstracts.get(text_ref)
        # use the fulltext if we have one
        if fulltext is not None:
            # if so, the text content is xml and will need to be processed
            text_dict[text_ref] = fulltext
        # otherwise use the abstract
        elif abstract is not None:
            text_dict[text_ref] = abstract
        # if we have neither, set result to None
        else:
            text_dict[text_ref] = None
    return ref_dict, text_dict


def get_text_content_from_text_refs(text_refs):
    """Get text_content from an evidence object's text_refs attribute


    Parameters
    ----------
    text_refs : dict of str: str
        text_refs dictionary as contained in an evidence object
        The dictionary should be keyed on id_types. The valid keys
        are 'PMID', 'PMCID', 'DOI', 'PII', 'URL', 'MANUSCRIPT_ID'.

    Returns
    -------
    text : str
        fulltext corresponding to the text_refs if it exists in the
        database, otherwise the abstract. Returns None if no content
        exists for the text_refs in the database
    """
    db = get_primary_db()
    text_ref_id = None
    for id_type in ['pmid', 'pmcid', 'doi',
                    'pii', 'url', 'manuscript_id']:
        try:
            id_val = text_refs[id_type.upper()]
            trids = _get_trids(db, id_val, id_type)
            if trids:
                text_ref_id = trids[0]
                break
        except KeyError:
            pass
    if text_ref_id is None:
        return None
    texts = db.select_all([db.TextContent.content,
                           db.TextContent.text_type],
                          db.TextContent.text_ref_id == text_ref_id)
    fulltext = [unpack(content) for content, text_type in texts
                if text_type == 'fulltext']
    if fulltext:
        return fulltext[0]
    abstract = [unpack(content) for content, text_type in texts
                if text_type == 'abstract']
    if abstract:
        return abstract[0]
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
