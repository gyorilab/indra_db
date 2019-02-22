__all__ = ['get_stmts_with_agent_text_like', 'get_text_content_from_stmt_ids']

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

    text_dict = db.select_all([db.RawAgents.stmt_id,
                               db.RawAgents.db_id],
                              db.RawAgents.db_name == 'TEXT',
                              db.RawAgents.db_id.like(pattern),
                              db.RawAgents.stmt_id.isnot(None))

    if filter_genes:
        hgnc_stmts = db.select_all(db.RawAgents.stmt_id,
                                   db.RawAgents.db_name == 'HGNC',
                                   db.RawAgents.stmt_id.isnot(None))
        hgnc_stmts = set(stmt_id[0] for stmt_id in hgnc_stmts)
        hgnc_rawtexts = set()
        for stmt_id, db_id in text_dict:
            if stmt_id not in hgnc_stmts:
                continue
        hgnc_rawtexts.add(db_id)

    result_dict = defaultdict(list)
    for stmt_id, db_id in text_dict:
        if not filter_genes or db_id in hgnc_rawtexts:
            result_dict[db_id].append(stmt_id)
    return dict(result_dict)


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
    result = {}
    for stmt_id in stmt_ids:
        # first check if we have text content for this statement
        try:
            text_ref = text_refs[stmt_id]
        except KeyError:
            # if not, set fulltext to None
            result[stmt_id] = None
            continue
        fulltext = fulltexts.get(text_ref)
        abstract = abstracts.get(text_ref)
        # use the fulltext if we have one
        if fulltext is not None:
            # if so, the text content is xml and will need to be processed
            result[stmt_id] = fulltext
        # otherwise use the abstract
        elif abstract is not None:
            result[stmt_id] = abstract
        # if we have neither, set result to None
        else:
            result[stmt_id] = None
    return result


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
    for id_type in ['pmid', 'pmcid', 'doi',
                    'pii', 'url', 'manuscript_id']:
        try:
            id_val = text_refs[id_type.upper()]
            text_ref_id = _get_trids(db, id_val, id_type)[0]
            break
        except KeyError:
            pass
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
