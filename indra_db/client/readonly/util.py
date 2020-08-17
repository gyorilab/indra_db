__all__ = ['stmt_from_interaction']

import logging

from indra.statements import get_statement_by_name, Agent, ActiveForm

logger = logging.getLogger(__name__)


snowflakes = ['Complex', 'Translocation', 'ActiveForm', 'Conversion',
              'Autophosphorylation']


def stmt_from_interaction(interaction):
    """Get a shell statement from an interaction."""
    StmtClass = get_statement_by_name(interaction['type'])
    if interaction['type'] == 'Complex':
        agents = [Agent(name) for name in interaction['agents'].values()]
        stmt = StmtClass(agents)
    elif interaction['type'] == 'ActiveForm':
        name = interaction['agents'][0]
        agent = Agent(name)
        stmt = StmtClass(agent, interaction['activity'],
                         interaction['is_active'])
    else:
        agents = [Agent(interaction['agents'][i])
                  if interaction['agents'].get(i)
                  else None
                  for i in range(len(StmtClass._agent_order))]
        stmt = StmtClass(*agents)
    return stmt


def _iter_agents(stmt_json, agent_order):
    for i, ag_key in enumerate(agent_order):
        ag = stmt_json.get(ag_key)
        if ag is None:
            continue
        if isinstance(ag, list):
            # Like a complex
            for ag_obj in ag:
                if stmt_json['type'] in snowflakes:
                    yield None, ag_obj
                else:
                    yield ['subject', 'object'][i], ag_obj
        else:
            if stmt_json['type'] in snowflakes:
                yield None, ag
            else:
                yield ['subject', 'object'][i], ag
