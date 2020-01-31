import re
import logging
from io import StringIO
from datetime import datetime

from flask import Response, abort


from indra.statements import make_statement_camel
from indra_db.client import get_statement_jsons_from_agents

logger = logging.getLogger('db rest api - util')


class DbAPIError(Exception):
    pass

# ==============================================
# Define some utilities used to resolve queries.
# ==============================================


def __process_agent(agent_param):
    """Get the agent id and namespace from an input param."""

    if not agent_param.endswith('@TEXT'):
        param_parts = agent_param.split('@')
        if len(param_parts) == 2:
            ag, ns = param_parts
        elif len(param_parts) == 1:
            ns = 'NAME'
            ag = param_parts[0]
        else:
            raise DbAPIError('Unrecognized agent spec: \"%s\"' % agent_param)
    else:
        ag = agent_param[:-5]
        ns = 'TEXT'

    if ns == 'HGNC-SYMBOL':
        ns = 'NAME'

    logger.info("Resolved %s to ag=%s, ns=%s" % (agent_param, ag, ns))
    return ag, ns


def get_source(ev_json):
    notes = ev_json.get('annotations')
    if notes is None:
        return
    src = notes.get('content_source')
    if src is None:
        return
    return src.lower()


def sec_since(t):
    return (datetime.now() - t).total_seconds()


class LogTracker(object):
    log_path = '.rest_api_tracker.log'

    def __init__(self):
        root_logger = logging.getLogger()
        self.stream = StringIO()
        sh = logging.StreamHandler(self.stream)
        formatter = logging.Formatter('%(levelname)s: %(name)s %(message)s')
        sh.setFormatter(formatter)
        sh.setLevel(logging.WARNING)
        root_logger.addHandler(sh)
        self.root_logger = root_logger
        return

    def get_messages(self):
        conts = self.stream.getvalue()
        print(conts)
        ret = conts.splitlines()
        return ret

    def get_level_stats(self):
        msg_list = self.get_messages()
        ret = {}
        for msg in msg_list:
            level = msg.split(':')[0]
            if level not in ret.keys():
                ret[level] = 0
            ret[level] += 1
        return ret


# ==========================================
# Define wrappers for common API call types.
# ==========================================


def _answer_binary_query(act_raw, roled_agents, free_agents, offs, max_stmts,
                         ev_limit, best_first, censured_sources):
    # Fix the case, if we got a statement type.
    act = None if act_raw is None else make_statement_camel(act_raw)

    # Make sure we got SOME agents. We will not simply return all
    # phosphorylations, or all activations.
    if not any(roled_agents.values()) and not free_agents:
        logger.error("No agents.")
        abort(Response(("No agents. Must have 'subject', 'object', or "
                        "'other'!\n"), 400))

    # Check to make sure none of the agents are None.
    assert None not in roled_agents.values() and None not in free_agents, \
        "None agents found. No agents should be None."

    # Now find the statements.
    logger.info("Getting statements...")
    agent_iter = [(role, ag_dbid, ns)
                  for role, (ag_dbid, ns) in roled_agents.items()]
    agent_iter += [(None, ag_dbid, ns) for ag_dbid, ns in free_agents]

    result = \
        get_statement_jsons_from_agents(agent_iter, stmt_type=act, offset=offs,
                                        max_stmts=max_stmts, ev_limit=ev_limit,
                                        best_first=best_first,
                                        censured_sources=censured_sources)
    return result


RELS = {'=': 'eq', '<': 'lt', '>': 'gt', 'is': 'is', 'is not': 'is not',
        '>=': 'gte', '<=': 'lte'}


def _build_source_filter_patt():
    padded_rels = set()
    for pair in RELS.items():
        for rel in pair:
            if rel.isalpha():
                pad = r'\s+'
            else:
                pad = r'\s*'
            padded_rels.add(pad + rel + pad)

    patt_str = r'^(\w+)(' + r'|'.join(padded_rels) + r')(\w+)$'

    return re.compile(patt_str)


source_patt = _build_source_filter_patt()


def _parse_source_str(source_relation):
    # Match to the source relation.
    m = source_patt.match(source_relation)
    if m is None:
        raise ValueError("Could not parse source relation: %s"
                         % source_relation)

    # Extract the groups.
    source, rel, value = m.groups()

    # Verify that rel is valid, and normalize.
    rel = rel.strip()
    if rel not in RELS.values():
        if rel not in RELS.keys():
            raise ValueError("Unrecognized relation: %s. Options are: %s"
                             % (rel, {s for pair in RELS.items()
                                      for s in pair}))
        else:
            # replace with a standardized representation.
            rel = RELS[rel]

    # Convert/verify the type of the value.
    if value.lower() in {'null', 'none'}:
        value = None

        # Assume the obvious intention behind =
        if rel == 'eq':
            rel = 'is'

        # Make sure that relation is "is" or "is not". Size comparisons don't
        # make sense.
        if rel not in {'is', 'is not'}:
            raise ValueError("Cannot us comparator \"%s\" with None." % rel)

    elif value.isdigit():
        value = int(value)
    else:
        raise ValueError("Can only match value to null or int, not: %s."
                         % value)

    return source.lower(), rel, value

