__all__ = ['insert_agents', 'insert_pa_agents_directly', 'insert_pa_stmts',
           'insert_db_stmts', 'regularize_agent_id']

import json
import logging
from sqlalchemy import exists

from indra.util import clockit
from indra.util.get_version import get_version
from indra.statements import Complex, SelfModification, ActiveForm,\
    Conversion, Translocation

from indra_db.exceptions import IndraDbException

from .helpers import get_statement_object


logger = logging.getLogger('util-insert')


@clockit
def get_statements_without_agents(db, prefix, *other_stmt_clauses, **kwargs):
    """Get a generator for orm statement objects which do not have agents."""
    num_per_yield = kwargs.pop('num_per_yield', 100)
    verbose = kwargs.pop('verbose', False)

    # Get the objects for either raw or pa statements.
    stmt_tbl_obj = db.tables[prefix + '_statements']
    agent_tbl_obj = db.tables[prefix + '_agents']

    # Build a dict mapping stmt UUIDs to statement IDs
    logger.info("Getting %s that lack %s in the database."
                % (stmt_tbl_obj.__tablename__, agent_tbl_obj.__tablename__))
    if prefix == 'pa':
        agents_link = (stmt_tbl_obj.mk_hash == agent_tbl_obj.stmt_mk_hash)
    elif prefix == 'raw':
        agents_link = (stmt_tbl_obj.id == agent_tbl_obj.stmt_id)
    else:
        raise IndraDbException("Unrecognized prefix: %s." % prefix)
    stmts_wo_agents_q = (db.session
                         .query(stmt_tbl_obj)
                         .filter(*other_stmt_clauses)
                         .filter(~exists().where(agents_link)))

    # Start printing some data
    if verbose:
        num_stmts = stmts_wo_agents_q.count()
        print("Adding agents for %d statements." % num_stmts)
    else:
        num_stmts = None

    # Get the iterator
    return stmts_wo_agents_q.yield_per(num_per_yield), num_stmts


@clockit
def insert_agents(db, prefix, stmts_wo_agents=None, **kwargs):
    """Insert agents for statements that don't have any agents.

    Note: This method currently works for both Statements and PAStatements and
    their corresponding agents (Agents and PAAgents). However, if you already
    have preassembled INDRA Statement objects that you know don't have agents
    in the database, you can use `insert_pa_agents_directly` to insert the
    agents much faster.

    Parameters
    ----------
    db : :py:class:`DatabaseManager`
        The manager for the database into which you are adding agents.
    prefix : str
        Select which stage of statements for which you wish to insert agents.
        The choices are 'pa' for preassembled statements or 'raw' for raw
        statements.
    verbose : bool
        If True, print extra information and a status bar while compiling
        agents for insert from statements. Default False.
    num_per_yield : int
        To conserve memory, statements are loaded in batches of `num_per_yeild`
        using the `yeild_per` feature of sqlalchemy queries.
    """
    verbose = kwargs.get('verbose', False)

    agent_tbl_obj = db.tables[prefix + '_agents']

    if stmts_wo_agents is None:
        stmts_wo_agents, num_stmts = \
            get_statements_without_agents(db, prefix, **kwargs)
    else:
        num_stmts = None

    if verbose:
        if num_stmts is None:
            try:
                num_stmts = len(stmts_wo_agents)
            except TypeError:
                logger.info("Could not get length from type: %s. Turning off "
                            "verbose messaging." % type(stmts_wo_agents))
                verbose = False

    # Construct the agent records
    logger.info("Building agent data for insert...")
    if verbose:
        print("Loading:", end='', flush=True)
    agent_data = []
    for i, db_stmt in enumerate(stmts_wo_agents):
        # Convert the database statement entry object into an indra statement.
        stmt = get_statement_object(db_stmt)

        if prefix == 'pa':
            stmt_id = db_stmt.mk_hash
        else:  # prefix == 'raw'
            stmt_id = db_stmt.id

        agent_data.extend(_get_agent_tuples(stmt, stmt_id))

        # Optionally print another tick on the progress bar.
        if verbose and num_stmts > 25 and i % (num_stmts//25) == 0:
            print('|', end='', flush=True)

    if verbose and num_stmts > 25:
        print()

    if prefix == 'pa':
        cols = ('stmt_mk_hash', 'db_name', 'db_id', 'role')
    else:  # prefix == 'raw'
        cols = ('stmt_id', 'db_name', 'db_id', 'role')
    for row in agent_data:
        if None in row:
            logger.warning("Found None in agent input:\n\t%s\n\t%s"
                           % (cols, row))
    db.copy(agent_tbl_obj.__tablename__, agent_data, cols)
    return


@clockit
def insert_pa_agents_directly(db, stmts, verbose=False):
    """Insert agents for preasembled statements.

    Unlike raw statements, preassembled statements are indexed by a hash,
    allowing for bulk import without a lookup beforehand, and allowing for a
    much simpler API.

    Parameters
    ----------
    db : :py:class:`DatabaseManager`
        The manager for the database into which you are adding agents.
    stmts : list[:py:class:`Statement`]
        A list of statements for which statements should be inserted.
    verbose : bool
        If True, print extra information and a status bar while compiling
        agents for insert from statements. Default False.
    """
    if verbose:
        num_stmts = len(stmts)

    # Construct the agent records
    logger.info("Building agent data for insert...")
    if verbose:
        print("Loading:", end='', flush=True)
    agent_data = []
    for i, stmt in enumerate(stmts):
        agent_data.extend(_get_agent_tuples(stmt, stmt.get_hash(shallow=True)))

        # Optionally print another tick on the progress bar.
        if verbose and num_stmts > 25 and i % (num_stmts//25) == 0:
            print('|', end='', flush=True)

    if verbose and num_stmts > 25:
        print()

    cols = ('stmt_mk_hash', 'db_name', 'db_id', 'role')
    db.copy('pa_agents', agent_data, cols)
    return


def regularize_agent_id(id_val, id_ns):
    """Change agent ids for better search-ability and index-ability."""
    new_id_val = id_val
    if id_ns.upper() == 'CHEBI':
        if id_val.startswith('CHEBI'):
            new_id_val = id_val[6:]
            logger.info("Fixed agent id: %s -> %s" % (id_val, new_id_val))
    return new_id_val


def _get_agent_tuples(stmt, stmt_id):
    """Create the tuples for copying agents into the database."""
    # Figure out how the agents are structured and assign roles.
    ag_list = stmt.agent_list()
    nary_stmt_types = [Complex, SelfModification, ActiveForm, Conversion,
                       Translocation]
    if any([isinstance(stmt, tp) for tp in nary_stmt_types]):
        agents = {('OTHER', ag) for ag in ag_list}
    elif len(ag_list) == 2:
        agents = {('SUBJECT', ag_list[0]), ('OBJECT', ag_list[1])}
    else:
        raise IndraDbException("Unhandled agent structure for stmt %s "
                               "with agents: %s."
                               % (str(stmt), str(stmt.agent_list())))

    def all_agent_refs(agents):
        """Smooth out the iteration over agents and their refs."""
        for role, ag in agents:
            # If no agent, or no db_refs for the agent, skip the insert
            # that follows.
            if ag is None or ag.db_refs is None:
                continue
            for ns, ag_id in ag.db_refs.items():
                if isinstance(ag_id, list):
                    for sub_id in ag_id:
                        yield ns, sub_id, role
                else:
                    yield ns, ag_id, role

    # Prep the agents for copy into the database.
    agent_data = []
    for ns, ag_id, role in all_agent_refs(agents):
        if ag_id is not None:
            agent_data.append((stmt_id, ns, regularize_agent_id(ag_id, ns),
                               role))
        else:
            logger.warning("Found agent for %s with None value." % ns)
    return agent_data


def insert_db_stmts(db, stmts, db_ref_id, verbose=False):
    """Insert statement, their database, and any affiliated agents.

    Note that this method is for uploading statements that came from a
    database to our databse, not for inserting any statements to the database.

    Parameters
    ----------
    db : :py:class:`DatabaseManager`
        The manager for the database into which you are loading statements.
    stmts : list [:py:class:`indra.statements.Statement`]
        A list of un-assembled indra statements to be uploaded to the datbase.
    db_ref_id : int
        The id to the db_ref entry corresponding to these statements.
    verbose : bool
        If True, print extra information and a status bar while compiling
        statements for insert. Default False.
    """
    # Preparing the statements for copying
    stmt_data = []
    cols = ('uuid', 'mk_hash', 'source_hash', 'db_info_id', 'type', 'json',
            'indra_version')
    if verbose:
        print("Loading:", end='', flush=True)
    for i, stmt in enumerate(stmts):
        # Only one evidence is allowed for each statement.
        for ev in stmt.evidence:
            new_stmt = stmt.make_generic_copy()
            new_stmt.evidence.append(ev)
            stmt_rec = (
                new_stmt.uuid,
                new_stmt.get_hash(shallow=False),
                new_stmt.evidence[0].get_source_hash(),
                db_ref_id,
                new_stmt.__class__.__name__,
                json.dumps(new_stmt.to_json()).encode('utf8'),
                get_version()
            )
            stmt_data.append(stmt_rec)
        if verbose and i % (len(stmts)//25) == 0:
            print('|', end='', flush=True)
    if verbose:
        print(" Done loading %d statements." % len(stmts))
    db.copy('raw_statements', stmt_data, cols)
    stmts_to_add_agents, num_stmts = \
        get_statements_without_agents(db, 'raw',
                                      db.RawStatements.db_info_id == db_ref_id)
    insert_agents(db, 'raw', stmts_to_add_agents)
    return


def insert_pa_stmts(db, stmts, verbose=False, do_copy=True,
                    direct_agent_load=True):
    """Insert pre-assembled statements, and any affiliated agents.

    Parameters
    ----------
    db : :py:class:`DatabaseManager`
        The manager for the database into which you are loading pre-assembled
        statements.
    stmts : iterable [:py:class:`indra.statements.Statement`]
        A list of pre-assembled indra statements to be uploaded to the datbase.
    verbose : bool
        If True, print extra information and a status bar while compiling
        statements for insert. Default False.
    do_copy : bool
        If True (default), use pgcopy to quickly insert the agents.
    direct_agent_load : bool
        If True (default), use the Statement get_hash method to get the id's of
        the Statements for insert, instead of looking up the ids of Statements
        from the database.
    """
    logger.info("Beginning to insert pre-assembled statements.")
    stmt_data = []
    indra_version = get_version()
    cols = ('uuid', 'matches_key', 'mk_hash', 'type', 'json', 'indra_version')
    if verbose:
        print("Loading:", end='', flush=True)
    for i, stmt in enumerate(stmts):
        stmt_rec = (
            stmt.uuid,
            stmt.matches_key(),
            stmt.get_hash(shallow=True),
            stmt.__class__.__name__,
            json.dumps(stmt.to_json()).encode('utf8'),
            indra_version
        )
        stmt_data.append(stmt_rec)
        if verbose and i % (len(stmts)//25) == 0:
            print('|', end='', flush=True)
    if verbose:
        print(" Done loading %d statements." % len(stmts))
    if do_copy:
        db.copy('pa_statements', stmt_data, cols)
    else:
        db.insert_many('pa_statements', stmt_data, cols=cols)
    if direct_agent_load:
        insert_pa_agents_directly(db, stmts, verbose=verbose)
    else:
        insert_agents(db, 'pa', verbose=verbose)
    return
