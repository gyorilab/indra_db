import json
import logging
import re
import sys
from datetime import datetime
from functools import wraps
from io import StringIO
from os import path

from flask import Flask, request, abort, Response, redirect, url_for, jsonify
from flask_compress import Compress
from flask_cors import CORS
from flask_jwt_extended import get_jwt_identity, jwt_optional
from jinja2 import Environment, ChoiceLoader

from indra.assemblers.html import HtmlAssembler
from indra.assemblers.html.assembler import loader as indra_loader
from indra.statements import make_statement_camel, stmts_from_json
from indra.util import batch_iter

from indralab_auth_tools.auth import auth, resolve_auth, config_auth

from indra_db.exceptions import BadHashError
from indra_db.client import get_statement_jsons_from_agents, \
    get_statement_jsons_from_hashes, get_statement_jsons_from_papers, \
    submit_curation, get_interaction_jsons_from_agents

logger = logging.getLogger("db-api")
logger.setLevel(logging.INFO)

app = Flask(__name__)
app.register_blueprint(auth)

app.config['DEBUG'] = True
SC, jwt = config_auth(app)

Compress(app)
CORS(app)

print("Loading file")
logger.info("INFO working.")
logger.warning("WARNING working.")
logger.error("ERROR working.")

MAX_STATEMENTS = int(1e3)
TITLE = "The INDRA Database"
HERE = path.abspath(path.dirname(__file__))

# Instantiate a jinja2 env.
env = Environment(loader=ChoiceLoader([app.jinja_loader, auth.jinja_loader,
                                       indra_loader]))

# Here we can add functions to the jinja2 env.
env.globals.update(url_for=url_for)


def render_my_template(template, title, **kwargs):
    kwargs['title'] = TITLE + ': ' + title
    kwargs['identity'] = get_jwt_identity()
    return env.get_template(template).render(**kwargs)


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


REDACT_MESSAGE = '[MISSING/INVALID API KEY: limited to 200 char for Elsevier]'


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


def _query_wrapper(f):
    logger.info("Calling outer wrapper.")

    @wraps(f)
    @jwt_optional
    def decorator(*args, **kwargs):

        tracker = LogTracker()
        start_time = datetime.now()
        logger.info("Got query for %s at %s!" % (f.__name__, start_time))

        query = request.args.copy()
        offs = query.pop('offset', None)
        ev_lim = query.pop('ev_limit', None)
        best_first_str = query.pop('best_first', 'true')
        best_first = True if best_first_str.lower() == 'true' \
                             or best_first_str else False
        max_stmts = min(int(query.pop('max_stmts', MAX_STATEMENTS)),
                        MAX_STATEMENTS)
        fmt = query.pop('format', 'json')

        # Figure out authorization.
        has = dict.fromkeys(['elsevier', 'medscan'], False)
        user, roles = resolve_auth(query)
        for role in roles:
            for resource in has.keys():
                has[resource] |= role.permissions.get(resource, False)
        logger.info('Auths: %s' % str(has))

        # Avoid loading medscan:
        censured_sources = set()
        if not has['medscan']:
            censured_sources.add('medscan')

        # Actually run the function.
        logger.info("Running function %s after %s seconds."
                    % (f.__name__, sec_since(start_time)))
        result = f(query, offs, max_stmts, ev_lim, best_first,
                   censured_sources, *args, **kwargs)
        if isinstance(result, Response):
            return result

        logger.info("Finished function %s after %s seconds."
                    % (f.__name__, sec_since(start_time)))

        # Handle any necessary redactions
        stmts_json = result.pop('statements')
        elsevier_redactions = 0
        source_counts = result['source_counts']
        if not has['elsevier']:
            for h, stmt_json in stmts_json.copy().items():
                for ev_json in stmt_json['evidence'][:]:

                    # Check for elsevier and redact if necessary
                    if get_source(ev_json) == 'elsevier':
                        text = ev_json['text']
                        if len(text) > 200:
                            ev_json['text'] = text[:200] + REDACT_MESSAGE
                            elsevier_redactions += 1

        logger.info(f"Redacted {elsevier_redactions} pieces of elsevier "
                    f"evidence.")

        logger.info("Finished redacting evidence for %s after %s seconds."
                    % (f.__name__, sec_since(start_time)))

        # Add derived values to the result.
        result['offset'] = offs
        result['evidence_limit'] = ev_lim
        result['statement_limit'] = MAX_STATEMENTS
        result['statements_returned'] = len(stmts_json)
        result['end_of_statements'] = (len(stmts_json) < MAX_STATEMENTS)
        result['statements_removed'] = 0

        if fmt == 'html':
            title = TITLE + ': ' + 'Results'
            ev_totals = result.pop('evidence_totals')
            stmts = stmts_from_json(stmts_json.values())
            html_assembler = HtmlAssembler(stmts, result, ev_totals,
                                           source_counts, title=title,
                                           db_rest_url=request.url_root[:-1])
            idbr_template = env.get_template('idbr_statements_view.html')
            content = html_assembler.make_model(idbr_template,
                                    identity=user.identity() if user else None)
            if tracker.get_messages():
                level_stats = ['%d %ss' % (n, lvl.lower())
                               for lvl, n in tracker.get_level_stats().items()]
                msg = ' '.join(level_stats)
                content = html_assembler.append_warning(msg)
            mimetype = 'text/html'
        else:  # Return JSON for all other values of the format argument
            result.update(tracker.get_level_stats())
            result['statements'] = stmts_json
            result['source_counts'] = source_counts
            content = json.dumps(result)
            mimetype = 'application/json'

        resp = Response(content, mimetype=mimetype)
        logger.info("Exiting with %d statements with %d/%d evidence of size "
                    "%f MB after %s seconds."
                    % (result['statements_returned'],
                       result['evidence_returned'], result['total_evidence'],
                       sys.getsizeof(resp.data) / 1e6, sec_since(start_time)))
        return resp

    return decorator


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


# ==========================
# Here begins the API proper
# ==========================


@app.route('/', methods=['GET'])
def iamalive():
    return redirect('statements', code=302)


@app.route('/statements', methods=['GET'])
@jwt_optional
def get_statements_query_format():
    # Create a template object from the template file, load once
    return render_my_template('search_statements.html', 'Search',
                              message="Welcome! Try asking a question.",
                              endpoint=request.url_root)


@app.route('/statements/from_agents', methods=['GET'])
@_query_wrapper
def get_statements(query_dict, offs, max_stmts, ev_limit, best_first,
                   censured_sources):
    """Get some statements constrained by query."""
    logger.info("Getting query details.")
    if ev_limit is None:
        ev_limit = 10
    try:
        # Get the agents without specified locations (subject or object).
        free_agents = [__process_agent(ag)
                       for ag in query_dict.poplist('agent')]
        ofaks = {k for k in query_dict.keys() if k.startswith('agent')}
        free_agents += [__process_agent(query_dict.pop(k)) for k in ofaks]

        # Get the agents with specified roles.
        roled_agents = {role: __process_agent(query_dict.pop(role))
                        for role in ['subject', 'object']
                        if query_dict.get(role) is not None}
    except DbAPIError as e:
        logger.exception(e)
        abort(Response('Failed to make agents from names: %s\n' % str(e), 400))
        return

    # Get the raw name of the statement type (we allow for variation in case).
    act_raw = query_dict.pop('type', None)

    # If there was something else in the query, there shouldn't be, so
    # someone's probably confused.
    if query_dict:
        abort(Response("Unrecognized query options; %s."
                       % list(query_dict.keys()), 400))
        return

    return _answer_binary_query(act_raw, roled_agents, free_agents, offs,
                                max_stmts, ev_limit, best_first,
                                censured_sources)


@app.route('/statements/from_hashes', methods=['POST'])
@_query_wrapper
def get_statements_by_hashes(query_dict, offs, max_stmts, ev_lim, best_first,
                             censured_sources):
    if ev_lim is None:
        ev_lim = 20
    hashes = request.json.get('hashes')
    if not hashes:
        logger.error("No hashes provided!")
        abort(Response("No hashes given!", 400))
    if len(hashes) > max_stmts:
        logger.error("Too many hashes given!")
        abort(Response("Too many hashes given, %d allowed." % max_stmts,
                       400))

    result = get_statement_jsons_from_hashes(hashes, max_stmts=max_stmts,
                                             offset=offs, ev_limit=ev_lim,
                                             best_first=best_first,
                                             censured_sources=censured_sources)
    return result


@app.route('/statements/from_hash/<hash_val>', methods=['GET'])
@_query_wrapper
def get_statement_by_hash(query_dict, offs, max_stmts, ev_limit, best_first,
                          censured_sources, hash_val):
    if ev_limit is None:
        ev_limit = 10000
    return get_statement_jsons_from_hashes([hash_val], max_stmts=max_stmts,
                                           offset=offs, ev_limit=ev_limit,
                                           best_first=best_first,
                                           censured_sources=censured_sources)


@app.route('/statements/from_papers', methods=['POST'])
@_query_wrapper
def get_paper_statements(query_dict, offs, max_stmts, ev_limit, best_first,
                         censured_sources):
    """Get Statements from a papers with the given ids."""
    if ev_limit is None:
        ev_limit = 10

    # Get the paper id.
    ids = request.json.get('ids')
    if not ids:
        logger.error("No ids provided!")
        abort(Response("No ids in request!", 400))

    # Format the ids.
    id_tpls = set()
    for id_dict in ids:
        val = id_dict['id']
        typ = id_dict['type']

        # Turn tcids and trids into integers.
        id_val = int(val) if typ in ['tcid', 'trid'] else val

        id_tpls.add((typ, id_val))

    # Now get the statements.
    logger.info('Getting statements for %d papers.' % len(id_tpls))
    result = get_statement_jsons_from_papers(id_tpls, max_stmts=max_stmts,
                                             offset=offs, ev_limit=ev_limit,
                                             best_first=best_first,
                                             censured_sources=censured_sources)
    return result


@app.route('/curation', methods=['GET'])
def describe_curation():
    return redirect('/statements', code=302)


@app.route('/curation/submit/<hash_val>', methods=['POST'])
@jwt_optional
def submit_curation_endpoint(hash_val, **kwargs):
    user, roles = resolve_auth(dict(request.args))
    if not roles and not user:
        res_dict = {"result": "failure", "reason": "Invalid Credentials"}
        return jsonify(res_dict), 401

    if user:
        email = user.email
    else:
        email = request.json.get('email')
        if not email:
            res_dict = {"result": "failure",
                        "reason": "POST with API key requires a user email."}
            return jsonify(res_dict), 400

    logger.info("Adding curation for statement %s." % hash_val)
    ev_hash = request.json.get('ev_hash')
    source_api = request.json.pop('source', 'DB REST API')
    tag = request.json.get('tag')
    ip = request.remote_addr
    text = request.json.get('text')
    is_test = 'test' in request.args
    if not is_test:
        assert tag is not 'test'
        try:
            dbid = submit_curation(hash_val, tag, email, ip, text, ev_hash,
                                   source_api)
        except BadHashError as e:
            abort(Response("Invalid hash: %s." % e.mk_hash, 400))
        res = {'result': 'success', 'ref': {'id': dbid}}
    else:
        res = {'result': 'test passed', 'ref': None}
    logger.info("Got result: %s" % str(res))
    return jsonify(res)


@app.route('/metadata/<level>/from_agents', methods=['GET'])
@jwt_optional
def get_metadata(level):
    start = datetime.utcnow()
    query = request.args.copy()

    # Figure out authorization.
    has = dict.fromkeys(['elsevier', 'medscan'], False)
    user, roles = resolve_auth(query)
    for role in roles:
        for resource in has.keys():
            has[resource] |= role.permissions.get(resource, False)
    logger.info('Auths: %s' % str(has))

    logger.info("Getting query details.")
    try:
        # Get the agents without specified locations (subject or object).
        agents = [(None,) + __process_agent(ag)
                  for ag in query.poplist('agent')]
        ofaks = {k for k in query.keys() if k.startswith('agent')}
        agents += [(None,) + __process_agent(query.pop(k)) for k in ofaks]

        # Get the agents with specified roles.
        agents += [(role,) +__process_agent(query.pop(role))
                   for role in ['subject', 'object']
                   if query.get(role) is not None]
    except DbAPIError as e:
        logger.exception(e)
        abort(Response('Failed to make agents from names: %s\n' % str(e), 400))
        return

    def pop(k, default=None):
        if isinstance(default, bool):
            return query.pop(k, str(default).lower()) == 'true'
        return query.pop(k, default)

    stmt_type = pop('type')
    stmt_type = None if stmt_type is None else make_statement_camel(stmt_type)
    res = get_interaction_jsons_from_agents(agents=agents, detail_level=level,
                                            stmt_type=stmt_type,
                                            max_relations=pop('limit'),
                                            offset=pop('offset'),
                                            best_first=pop('best_first', True))

    dt = (datetime.utcnow() - start).total_seconds()
    logger.info("Got %s results after %.2f." % (len(res), dt))

    # Currently, a hash could get through (in one of the less detailed results)
    # that is entirely dependent on medscan.
    if not has['medscan']:
        censored_res = []
        for entry in res:
            entry['total_count'] -= entry['source_counts'].pop('medscan', 0)
            if entry['source_counts']:
                censored_res.append(entry)
        res = censored_res

    dt = (datetime.utcnow() - start).total_seconds()
    logger.info("Returning with %s results after %.2f seconds."
                % (len(res), dt))

    resp = Response(json.dumps(sorted(res, key=lambda e: e['total_count'],
                                      reverse=True)),
                    mimetype='application/json')

    dt = (datetime.utcnow() - start).total_seconds()
    logger.info("Result prepared after %.2f seconds." % dt)
    return resp


if __name__ == '__main__':
    app.run()
