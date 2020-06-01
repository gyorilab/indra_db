import logging

logger = logging.getLogger(__name__)

# =============================================================================
# The API
# =============================================================================


snowflakes = ['Complex', 'Translocation', 'ActiveForm', 'Conversion',
              'Autophosphorylation']


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
