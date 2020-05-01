__all__ = ['stmt_from_interaction']

from indra.statements import get_statement_by_name, Agent, ActiveForm


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
