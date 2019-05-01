__all__ = ['insert_raw_agents', 'insert_pa_agents', 'insert_pa_stmts',
           'insert_db_stmts', 'regularize_agent_id', 'extract_agent_data']

import json
import pickle
import logging

from indra.util import clockit
from indra.util.get_version import get_version
from indra.statements import Complex, SelfModification, ActiveForm, \
    Conversion, Translocation

from indra_db.exceptions import IndraDbException

from .helpers import get_statement_object


logger = logging.getLogger('util-insert')


@clockit
def insert_raw_agents(db, batch_id, stmts=None, verbose=False,
                      num_per_yield=100):
    """Insert agents for statements that don't have any agents.

    Parameters
    ----------
    db : :py:class:`DatabaseManager`
        The manager for the database into which you are adding agents.
    batch_id : int
        Every set of new raw statements must be given an id unique to that copy
        That id is used to get the set of statements that need agents added.
    stmts : list[indra.statements.Statement]
        The list of statements that include those whose agents are being
        uploaded.
    verbose : bool
        If True, print extra information and a status bar while compiling
        agents for insert from statements. Default False.
    num_per_yield : int
        To conserve memory, statements are loaded in batches of `num_per_yeild`
        using the `yeild_per` feature of sqlalchemy queries.
    """
    ref_tuples = []
    mod_tuples = []
    mut_tuples = []
    if not stmts:
        tbls = [db.RawStatements.id, db.RawStatements]
        stmt_dict = None
    else:
        tbls = [db.RawStatements.id, db.RawStatements.uuid]
        stmt_dict = {s.uuid: s for s in stmts}

    q = db.filter_query(tbls, db.RawStatements.batch_id == batch_id)
    if verbose:
        num_stmts = q.count()
        print("Loading:", end='', flush=True)

    db_stmts = q.yield_per(num_per_yield)

    for i, (stmt_id, db_stmt) in enumerate(db_stmts):
        if stmts is None:
            stmt = get_statement_object(db_stmt)
        else:
            stmt = stmt_dict[db_stmt]

        ref_data, mod_data, mut_data = extract_agent_data(stmt, stmt_id)
        ref_tuples.extend(ref_data)
        mod_tuples.extend(mod_data)
        mut_tuples.extend(mut_data)

        # Optionally print another tick on the progress bar.
        if verbose and num_stmts > 25 and i % (num_stmts//25) == 0:
            print('|', end='', flush=True)

    if verbose and num_stmts > 25:
        print()

    db.copy('raw_agents', ref_tuples,
            ('stmt_id', 'ag_num', 'db_name', 'db_id', 'role'), commit=False)
    db.copy('raw_mods', mod_tuples,
            ('stmt_id', 'ag_num', 'type', 'position', 'residue', 'modified'),
            commit=False)
    db.copy('raw_muts', mut_tuples,
            ('stmt_id', 'ag_num', 'position', 'residue_from', 'residue_to'),
            commit=False)
    db.commit_copy('Error copying raw agents, mods, and muts.')
    return


def insert_pa_agents(db, stmts, verbose=False, skip=None):
    if skip is None:
        skip = []

    if verbose:
        num_stmts = len(stmts)

    # Construct the agent records
    logger.info("Building data from agents for insert into pa_agents, "
                "pa_mods, and pa_muts...")

    if verbose:
        print("Loading:", end='', flush=True)

    ref_data = []
    mod_data = []
    mut_data = []
    for i, stmt in enumerate(stmts):
        refs, mods, muts = extract_agent_data(stmt, stmt.get_hash())
        ref_data.extend(refs)
        mod_data.extend(mods)
        mut_data.extend(muts)

        # Optionally print another tick on the progress bar.
        if verbose and num_stmts > 25 and i % (num_stmts//25) == 0:
            print('|', end='', flush=True)

    if verbose and num_stmts > 25:
        print()

    if 'agents' not in skip:
        db.copy('pa_agents', ref_data,
                ('stmt_mk_hash', 'ag_num', 'db_name', 'db_id', 'role'),
                lazy=True, commit=False)
    if 'mods' not in skip:
        db.copy('pa_mods', mod_data,
                ('stmt_mk_hash', 'ag_num', 'type', 'position', 'residue',
                 'modified'), commit=False)
    if 'muts' not in skip:
        db.copy('pa_muts', mut_data,
                ('stmt_mk_hash', 'ag_num', 'position', 'residue_from',
                 'residue_to'), commit=False)
    db.commit_copy('Error copying pa agents, mods, and muts, excluding: %s.'
                   % (', '.join(skip)))
    return


def regularize_agent_id(id_val, id_ns):
    """Change agent ids for better search-ability and index-ability."""
    ns_abbrevs = [('CHEBI', ':'), ('GO', ':'), ('HMDB', ''), ('PF', ''),
                  ('IP', '')]
    for ns, div in ns_abbrevs:
        if id_ns.upper() == ns and id_val.startswith(ns):
            new_id_val = id_val[len(ns) + len(div)]
            break
    else:
        return id_val

    # logger.info("Fixed agent id: %s -> %s" % (id_val, new_id_val))
    return new_id_val


def extract_agent_data(stmt, stmt_id):
    """Create the tuples for copying agents into the database."""
    # Figure out how the agents are structured and assign roles.
    ag_list = stmt.agent_list(deep_sorted=True)
    nary_stmt_types = [Complex, SelfModification, ActiveForm, Conversion,
                       Translocation]
    if any([isinstance(stmt, tp) for tp in nary_stmt_types]):
        agents = {('OTHER', ag, i) for i, ag in enumerate(ag_list)}
    elif len(ag_list) == 2:
        agents = {(role, ag_list[i], i)
                  for i, role in enumerate(['SUBJECT', 'OBJECT'])}
    else:
        raise IndraDbException("Unhandled agent structure for stmt %s "
                               "with agents: %s."
                               % (str(stmt), str(stmt.agent_list())))

    def all_agent_refs(ag):
        """Smooth out the iteration over agents and their refs."""
        for ns, ag_id in ag.db_refs.items():
            if isinstance(ag_id, list):
                for sub_id in ag_id:
                    yield ns, sub_id
            else:
                yield ns, ag_id
        yield 'NAME', ag.name

    # Prep the agents for copy into the database.
    ref_data = []
    mod_data = []
    mut_data = []
    warnings = set()
    for role, ag, idx in agents:
        # If no agent, or no db_refs for the agent, skip the insert
        # that follows.
        if ag is None or ag.db_refs is None:
            continue

        # Get the db refs data.
        for ns, ag_id in all_agent_refs(ag):
            if ag_id is not None:
                ref_data.append((stmt_id, idx, ns,
                                 regularize_agent_id(ag_id, ns), role))
            else:
                if ns not in warnings:
                    warnings.add(ns)
                    logger.warning("Found agent for %s with None value." % ns)

        # Get the modification data
        for mod in ag.mods:
            mod_data.append((stmt_id, idx, mod.mod_type, mod.position,
                             mod.residue, mod.is_modified))

        # Get the mutation data
        for mut in ag.mutations:
            mut_data.append((stmt_id, idx, mut.position, mut.residue_from,
                             mut.residue_to))

    return ref_data, mod_data, mut_data


def insert_db_stmts(db, stmts, db_ref_id, verbose=False, batch_id=None,
                    lazy=False):
    """Insert statement, their database, and any affiliated agents.

    Note that this method is for uploading statements that came from a
    database to our databse, not for inserting any statements to the database.

    Parameters
    ----------
    db : :py:class:`DatabaseManager`
        The manager for the database into which you are loading statements.
    stmts : list [:py:class:`indra.statements.Statement`]
        (Cannot be a generator) A list of un-assembled indra statements, each
        with EXACTLY one evidence and no exact duplicates, to be uploaded to
        the database.
    db_ref_id : int
        The id to the db_ref entry corresponding to these statements.
    verbose : bool
        If True, print extra information and a status bar while compiling
        statements for insert. Default False.
    batch_id : int or None
        Select a batch id to use for this upload. It can be used to trace what
        content has been added.
    """
    # Preparing the statements for copying
    if batch_id is None:
        batch_id = db.make_copy_batch_id()

    stmt_data = []

    cols = ('uuid', 'mk_hash', 'source_hash', 'db_info_id', 'type', 'json',
            'indra_version', 'batch_id')
    if verbose:
        print("Loading:", end='', flush=True)

    for i, stmt in enumerate(stmts):
        assert len(stmt.evidence) == 1, \
            'Statement with %s evidence.' % len(stmt.evidence)

        stmt_rec = (stmt.uuid, stmt.get_hash(refresh=True),
                    stmt.evidence[0].get_source_hash(refresh=True), db_ref_id,
                    stmt.__class__.__name__,
                    json.dumps(stmt.to_json()).encode('utf8'),
                    get_version(), batch_id)

        stmt_data.append(stmt_rec)

        if verbose and i % (len(stmts)//25) == 0:
            print('|', end='', flush=True)

    if verbose:
        print(" Done preparing %d statements." % len(stmts))

    try:
        # TODO: Make it possible to not commit this immediately. That would
        # require developing a more sophisticated copy procedure for raw
        # statements and agents.
        db.copy('raw_statements', stmt_data, cols, lazy=lazy,
                push_conflict=lazy)
    except Exception as e:
        with open('stmt_data_dump.pkl', 'wb') as f:
            pickle.dump(stmt_data, f)
        raise e
    insert_raw_agents(db, batch_id, stmts)
    return


def insert_pa_stmts(db, stmts, verbose=False, do_copy=True,
                    ignore_agents=False):
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
    if not ignore_agents:
        insert_pa_agents(db, stmts, verbose=verbose)
    return
