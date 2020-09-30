__all__ = ['get_statement_essentials', 'get_relation_dict',
           'export_relation_dict_to_tsv']

import logging
from itertools import permutations
from sqlalchemy import or_

from indra.databases import hgnc_client
from indra_db.util import get_db, get_statement_object

logger = logging.getLogger(__name__)


def get_statement_essentials(clauses, count=1000, db=None, preassembled=True):
    """Get the type, agents, and id data for the specified statements.

    This function is useful for light-weight searches of basic mechanistic
    information, without the need to follow as many links in the database to
    populate the Statement objects.

    To get full statements, use `get_statements`.

    Parameters
    ----------
    clauses : list
        list of sqlalchemy WHERE clauses to pass to the filter query.
    count : int
        Number of statements to retrieve and process in each batch.
    do_stmt_count : bool
        Whether or not to perform an initial statement counting step to give
        more meaningful progress messages.
    db : :py:class:`DatabaseManager`
        Optionally specify a database manager that attaches to something
        besides the primary database, for example a local database instance.
    preassembled : bool
        If true, statements will be selected from the table of pre-assembled
        statements. Otherwise, they will be selected from the raw statements.
        Default is True.

    Returns
    -------
    A list of tuples containing:
        `(uuid, sid, hash, type, (agent_1, agent_2, ...))`.
    """
    if db is None:
        db = get_db('primary')

    stmts_tblname = 'pa_statements' if preassembled else 'raw_statements'

    stmt_data = []
    db_stmts = db.select_all(stmts_tblname, *clauses, yield_per=count)
    for db_stmt in db_stmts:
        stmt = get_statement_object(db_stmt)
        sid = db_stmt.id if hasattr(db_stmt, 'id') else None
        stmt_data.append((db_stmt.uuid, sid, stmt.get_hash(shallow=True),
                          db_stmt.type, stmt.agent_list()))

    return stmt_data


def get_relation_dict(db, groundings=None, with_evidence_count=False,
                      with_support_count=False):
    """Get a dictionary of entity interactions from the database.

    Use only metadata from the database to rapidly get simple interaction data.
    This is much faster than handling the full Statement jsons, while providing
    some basic valuable functionality.

    Parameters
    ----------
    db : DatabaseManager instance
        An instance of a database manager.
    groundings : list[str] or None
        Select which types of grounding namespaces to include, e.g. HGNC, or
        FPLX, or both. Only agent refs with these groundings will be selected.
        If None, only HGNC is used.
    with_evidence_count : bool
        Default is False. If True, an additional query will be made for each
        statement to get the count of supporting evidence, which is a useful
        proxy for belief.
    with_support_count : bool
        Default is False. Like `with_evidence_count`, except the number of
        supporting statements is counted.
    """
    other_params = []
    if groundings is None:
        other_params.append(db.PAAgents.db_name.like('HGNC'))
    elif len(groundings) == 1:
        other_params.append(db.PAAgents.db_name.like(groundings[0]))
    else:
        ors = []
        for gdng in groundings:
            ors.append(db.PAAgents.db_name.like(gdng))
        other_params.append(or_(*ors))

    vals = [db.PAAgents.id, db.PAAgents.db_id, db.PAAgents.role,
            db.PAAgents.db_name, db.PAStatements.type, db.PAStatements.mk_hash]

    if with_evidence_count:
        other_params.append(
            db.EvidenceCounts.mk_hash == db.PAStatements.mk_hash
            )
        vals.append(db.EvidenceCounts.ev_count)

    # Query the database
    results = db.select_all(
        vals,
        db.PAStatements.mk_hash == db.PAAgents.stmt_mk_hash,
        *other_params, **{'yield_per': 10000}
        )

    # Sort into a dict.
    stmt_dict = {}
    for res in results:
        if with_evidence_count:
            ag_id, ag_dbid, ag_role, ag_dbname, st_type, stmt_hash, n_ev = res
        else:
            ag_id, ag_dbid, ag_role, ag_dbname, st_type, stmt_hash = res

        # Handle the case that this is or isn't HGNC
        if ag_dbname == 'HGNC':
            ag_tpl = (ag_id, ag_role, ag_dbname, ag_dbid,
                      hgnc_client.get_hgnc_name(ag_dbid))
        else:
            ag_tpl = (ag_id, ag_role, ag_dbname, ag_dbid, ag_dbid)

        # Add the tuple to the dict in the appropriate manner.
        if stmt_hash not in stmt_dict.keys():
            stmt_dict[stmt_hash] = {'type': st_type, 'agents': [ag_tpl]}
            if with_evidence_count:
                stmt_dict[stmt_hash]['n_ev'] = n_ev
            if with_support_count:
                logger.info('Getting a count of support for %d' % stmt_hash)
                n_sup = db.count(
                    db.PASupportLinks,
                    db.PASupportLinks.supported_mk_hash == stmt_hash
                    )
                stmt_dict[stmt_hash]['n_sup'] = n_sup
        else:
            assert stmt_dict[stmt_hash]['type'] == st_type
            stmt_dict[stmt_hash]['agents'].append(ag_tpl)

    # Only return the entries with at least 2 agents.
    return {k: d for k, d in stmt_dict.items() if len(d['agents']) >= 2}


def export_relation_dict_to_tsv(relation_dict, out_base, out_types=None):
    """Export a relation dict (from get_relation_dict) to a tsv.

    Available output types are:
    - "full_tsv" : get a tsv with directed pairs of entities (e.g. HGNC
        symbols), the type of relation (e.g. Phosphorylation) and the hash
        of the preassembled statement. Columns are agent_1, agent_2 (where
        agent_1 affects agent_2), type, hash.
    - "short_tsv" : like the above, but without the hashes, so only one
        instance of each pair and type trio occurs. However, the information
        cannot be traced. Columns are agent_1, agent_2, type, where agent_1
        affects agent_2.
    - "pairs_tsv" : like the above, but without the relation type. Similarly,
        each row is unique. In addition, the agents are undirected. Thus this
        is purely a list of pairs of related entities. The columns are just
        agent_1 and agent_2, where nothing is implied by the ordering.

    Parameters
    ----------
    relation_dict : dict
        This should be the output from `get_relation_dict`, or something
        equivalently constructed.
    out_base : str
        The base-name for the output files.
    out_types : list[str]
        A list of the types of tsv to output. See above for details.
    """
    # Check to make sure the output types are valid.
    ok_types = ['full_tsv', 'short_tsv', 'pairs_tsv']
    if out_types is None:
        out_types = ok_types[:]

    if any(ot not in ok_types for ot in out_types):
        raise ValueError('Invalid output_types: %s. Allowed types are: %s'
                         % (out_types, ok_types))

    # Now write any tsv's.
    def write_tsv_line(f, row_tpl):
        f.write('\t'.join(list(row_tpl)) + '\n')

    # Open the tsv files.
    tsv_files = {}
    for output_type in out_types:
        tsv_files[output_type] = open('%s_%s.tsv' % (out_base, output_type),
                                      'w')

    # Write the tsv files.
    short_set = set()
    very_short_set = set()
    for h, d in relation_dict.items():
        # Do some pre-processing
        roles = sorted([ag_tpl[1] for ag_tpl in d['agents']])
        ag_by_roles = dict.fromkeys(roles)
        for role in roles:
            ag_by_roles[role] = [ag_tpl[-1] for ag_tpl in d['agents']
                                 if ag_tpl[1] == role]
        if roles == ['OBJECT', 'SUBJECT']:
            data_tpls = [(ag_by_roles['SUBJECT'][0], ag_by_roles['OBJECT'][0],
                          d['type'], str(h))]
        elif set(roles) == {'OTHER'}:
            data_tpls = [(a, b, d['type'], str(h))
                         for a, b in permutations(ag_by_roles['OTHER'], 2)]
        elif d['type'] == 'Conversion':
            continue  # TODO: Handle conversions.
        else:
            print("This is weird...", h, d)
            continue

        # Handle writing the various files.
        if 'full_tsv' in out_types:
            for data_tpl in data_tpls:
                write_tsv_line(tsv_files['full_tsv'], data_tpl)

        if 'short_tsv' in out_types:
            short_tpls = [t[:-1] for t in data_tpls]
            for t in short_tpls:
                if t not in short_set:
                    short_set.add(t)
                    write_tsv_line(tsv_files['short_tsv'], t)

        if 'pairs_tsv' in out_types:
            vs_tpls ={tuple(sorted(t[:-2])) for t in data_tpls}
            for t in vs_tpls:
                if t not in very_short_set:
                    very_short_set.add(t)
                    write_tsv_line(tsv_files['pairs_tsv'], t)

    # Close the tsv files.
    for file_handle in tsv_files.values():
        file_handle.close()

    return relation_dict
