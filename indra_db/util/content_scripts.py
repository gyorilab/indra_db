from collections import defaultdict

import indra_db.util as dbu


def get_statements_with_agent_text_like(pattern):
    """Get statement ids with agent with rawtext matching pattern


    Parameters
    ----------
    pattern : str
        a pattern understood by sqlalchemy's like operator.
        For example '__' for two letter agents

    Returns
    -------
    dict
        dict mapping agent texts matching the input pattern to lists of
        ids for statements with at least one agent with raw text matching
        the pattern.
    """
    db = dbu.get_primary_db()
    # get all stmts with at least one hgnc grounded agent
    hgnc_stmts = db.select_all(db.RawAgents.stmt_id,
                               db.RawAgents.db_name == 'HGNC',
                               db.RawAgents.stmt_id.isnot(None))
    hgnc_stmts = set(stmt_id[0] for stmt_id in hgnc_stmts)
    text_dict = db.select_all([db.RawAgents.stmt_id,
                               db.RawAgents.db_id],
                              db.RawAgents.db_name == 'TEXT',
                              db.RawAgents.db_id.like(pattern),
                              db.RawAgents.stmt_id.isnot(None))
    hgnc_rawtexts = set()
    for stmt_id, db_id in text_dict:
        if stmt_id not in hgnc_stmts:
            continue
        hgnc_rawtexts.add(db_id)

    result_dict = defaultdict(list)
    for stmt_id, db_id in text_dict:
        if db_id in hgnc_rawtexts:
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
    db = dbu.get_primary_db()
    text_refs = db.select_all([db.RawStatements.id, db.TextRef.id],
                              db.RawStatements.id.in_(stmt_ids),
                              *db.link(db.RawStatements, db.TextRef))
    text_refs = dict(text_refs)
    texts = db.select_all([db.TextContent.text_ref_id,
                           db.TextContent.content,
                           db.TextContent.text_type],
                          db.TextContent.text_ref_id.in_(text_refs.values()))
    fulltexts = {text_id: dbu.unpack(text)
                 for text_id, text, text_type in texts
                 if text_type == 'fulltext'}
    abstracts = {text_id: dbu.unpack(text)
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
