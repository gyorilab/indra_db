import json
import logging
from io import StringIO
from datetime import datetime

from indra_db.client.readonly.query import gilda_ground

logger = logging.getLogger('db rest api - util')


class DbAPIError(Exception):
    pass


class NoGroundingFound(DbAPIError):
    pass


def get_s3_client():
    import boto3
    from botocore import config
    return boto3.client('s3', boto3.session.Session().region_name,
                        config=config.Config(s3={'addressing_style': 'path'}))

# ==============================================
# Define some utilities used to resolve queries.
# ==============================================


def process_agent(agent_param):
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


def process_mesh_term(mesh_term):
    """Use gilda to translate a mesh term into a MESH ID if possible."""
    if mesh_term is None:
        return mesh_term

    # Check to see if this is a mesh ID.
    if any(mesh_term.startswith(c) for c in ['D', 'C']) \
            and mesh_term[1:].isdigit():
        return mesh_term

    # Try to ground the term.
    results = gilda_ground(mesh_term)
    for res in results:
        if res['term']['db'] == 'MESH':
            logger.info(f"Auto-mapped {mesh_term} to {res['term']['id']} "
                        f"({res['term']['entry_name']}) using Gilda.")
            return res['term']['id']
    raise NoGroundingFound(f"Could not find MESH id for {mesh_term} among "
                           f"gilda results:\n{json.dumps(results, indent=2)}")


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
