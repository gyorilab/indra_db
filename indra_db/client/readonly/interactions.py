__all__ = ['get_interaction_jsons_from_agents',
           'get_interaction_statements_from_agents',
           'stmt_from_interaction']

from collections import defaultdict

from indra.statements import get_statement_by_name, Agent, ActiveForm
from indra.util import clockit

from indra_db.util import get_ro
from indra_db.client.tools import _apply_limits
from indra_db.client.readonly.pa_statements import _make_mk_hashes_query


@clockit
def get_interaction_jsons_from_agents(agents=None, stmt_type=None, ro=None,
                                      best_first=True, max_relations=None,
                                      offset=None, detail_level='hashes'):
    """Get simple reaction information available from Statement metadata.

    There are three levels of detail:
        hashes -> Each entry in the result corresponds to a single preassembled
                statement, distinguished by its hash.
        relations -> Each entry in the result corresponds to a relation, meaning
                an interaction type, and the names of the agents involved.
        agents -> Each entry is simply a pair (or more) of Agents involved in
                an interaction.
    """
    if detail_level not in {'hashes', 'relations', 'agents'}:
        raise ValueError("Invalid detail_level: %s" % detail_level)

    if ro is None:
        ro = get_ro('primary')

    # Make a really fancy query to the database.
    mk_hashes_q, mk_hashes_al = _make_mk_hashes_query(ro, agents, stmt_type)

    mk_hashes_q = _apply_limits(ro, mk_hashes_q, best_first, max_relations,
                                offset, mk_hashes_alias=mk_hashes_al)

    mk_hashes_sq = mk_hashes_q.subquery('mk_hashes')
    q = (ro.session.query(ro.NameMeta.mk_hash, ro.NameMeta.db_id,
                          ro.NameMeta.ag_num, ro.NameMeta.type,
                          ro.PaStmtSrc)
         .filter(ro.NameMeta.mk_hash == mk_hashes_sq.c.mk_hash,
                 ro.PaStmtSrc.mk_hash == mk_hashes_sq.c.mk_hash))
    names = q.all()

    # Group the agents together.
    meta_dict = {}
    for h, ag_name, ag_num, stmt_type, srcs in names:
        if h not in meta_dict.keys():
            meta_dict[h] = {'type': stmt_type, 'agents': {},
                            'source_counts': srcs.get_sources()}
        meta_dict[h]['agents'][ag_num] = ag_name

    # Condense the results, as indicated.
    result = []
    if detail_level == 'hashes':
        for h, data in meta_dict.items():
            data['hash'] = h
            data['id'] = str(h)
            data['total_count'] = sum(data['source_counts'].values())
            result.append(data)
    else:

        # Re-aggregate the statements.
        condensed = {}
        for h, data in meta_dict.items():
            print(h)

            # Make the agent key
            ag_dict = data['agents']

            num_agents = max(ag_dict.keys()) + 1  # Could be trailing Nones...
            ordered_agents = [ag_dict.get(n) for n in range(num_agents)]
            agent_key = '(' + ', '.join(str(ag) for ag in ordered_agents) + ')'

            # Make the overall key
            if detail_level == 'relations':
                key = data['type'] + agent_key
            else:
                key = 'Interaction' + agent_key

            # Handle new entries
            if key not in condensed:
                condensed[key] = {'hashes': {}, 'id': key,
                                  'source_counts': defaultdict(lambda: 0),
                                  'total_count': 0, 'agents': data['agents']}
                if detail_level == 'relations':
                    condensed[key]['type'] = data['type']
                else:
                    condensed[key]['types'] = defaultdict(lambda: 0)

            # Update existing entries.
            entry = condensed[key]
            if detail_level == 'agents':
                entry['types'][data['type']] += sum(data['source_counts'].values())

            for src, cnt in data['source_counts'].items():
                entry['source_counts'][src] += cnt
                entry['total_count'] += cnt
            entry['hashes'][h] = sum(data['source_counts'].values())

        # Convert defaultdict to normal dict and add to list.
        for entry in condensed.values():
            entry['source_counts'] = dict(entry['source_counts'])
            if detail_level == 'agents':
                entry['types'] = dict(entry['types'])
            result.append(entry)

    return sorted(result, key=lambda data: data['total_count'], reverse=True)


def stmt_from_interaction(interaction):
    """Get a shell statement from an interaction."""
    StmtClass = get_statement_by_name(interaction['type'])
    if issubclass(StmtClass, ActiveForm):
        return None
    if interaction['type'] == 'Complex':
        agents = [Agent(name) for name in interaction['agents'].values()]
        stmt = StmtClass(agents)
    else:
        agents = [Agent(interaction['agents'][i])
                  if interaction['agents'].get(i)
                  else None
                  for i in range(len(StmtClass._agent_order))]
        stmt = StmtClass(*agents)
    return stmt
    

def get_interaction_statements_from_agents(*args, **kwargs):
    """Get high-level statements for interactions apparent in db metadata.

    This function is a fairly thin wrapper around
    `get_interaction_jsons_from_agents`
    """
    meta_dicts = get_interaction_jsons_from_agents(*args, **kwargs)
    stmts = []
    for meta in meta_dicts:
        stmt = stmt_from_interactions(meta)
        if stmt is None:
            continue
        stmts.append(stmt)
    return stmts
