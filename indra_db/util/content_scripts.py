__all__ = ['get_stmts_with_agent_text_like', 'get_text_content_from_stmt_ids']


from collections import defaultdict
from sqlalchemy import func
from .constructors import get_primary_db
from .helpers import unpack, _get_trids


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
        db = get_primary_db()

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
        db = get_primary_db()

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


def get_text_content_from_stmt_ids(stmt_ids, db=None):
    """Get text content for statements from a list of ids

    Gets the fulltext if it is available, even if the statement came from an
    abstract.

    Parameters
    ----------
    stmt_ids : list of str

    db : Optional[:py:class:`DatabaseManager`]
        User has the option to pass in a database manager. If None
        the primary database is used. Default: None

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
        db = get_primary_db()

    # Query for all text content for which we have statements for its text ref.
    trid_q = (db.session.query(func.array_agg(db.RawStatements.id)
                               .label('stmt_ids'),
                               db.TextRef.id.label('text_ref_id'))
              .filter(db.RawStatements.id.in_(stmt_ids),
                      *db.link(db.RawStatements, db.TextRef))
              .group_by(db.TextRef.id))
    trid_q = trid_q.subquery('text_ref_ids')
    texts_q = (db.session.query(db.TextContent.content,
                                db.TextContent.source,
                                db.TextContent.text_type,
                                trid_q)
                 .filter(trid_q.c.text_ref_id ==
                         db.TextContent.text_ref_id))

    # Process the results.
    priority = {'fulltext': 2, 'abstract': 1, 'title': 0}
    seen_text_refs = {}
    ref_dict = {}
    text_dict = {}
    for content, source, text_type, stmts, text_ref_id in texts_q.all():
        # key uniquely identifies each piece of content
        new_key = '/'.join([str(text_ref_id), source, text_type])
        if text_ref_id not in seen_text_refs:
            text_dict[new_key] = content
            seen_text_refs[text_ref_id] = new_key
            ref_dict.update({stmt_id: new_key for stmt_id in stmts})
        else:
            # update if we find text_type with higher priority for
            # a given text_ref
            old_key = seen_text_refs[text_ref_id]
            old_text_type = old_key.split('/')[2]
            if priority[text_type] > priority[old_text_type]:
                seen_text_refs[text_ref_id] = new_key
                text_dict[new_key] = content
                text_dict.pop(old_key)
                ref_dict.update({stmt_id: new_key for stmt_id in stmts})

    text_dict = {key: unpack(content) for key, content in text_dict.items()}
    return ref_dict, text_dict


def get_text_content_from_text_refs(text_refs, db=None):
    """Get text_content from an evidence object's text_refs attribute


    Parameters
    ----------
    text_refs : dict of str: str
        text_refs dictionary as contained in an evidence object
        The dictionary should be keyed on id_types. The valid keys
        are 'PMID', 'PMCID', 'DOI', 'PII', 'URL', 'MANUSCRIPT_ID'.

    db : Optional[:py:class:`DatabaseManager`]
        User has the option to pass in a database manager. If None
        the primary database is used. Default: None

    Returns
    -------
    text : str
        fulltext corresponding to the text_refs if it exists in the
        database, otherwise the abstract. Returns None if no content
        exists for the text_refs in the database
    """
    if db is None:
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
