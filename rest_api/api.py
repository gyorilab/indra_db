import json
import logging
import sys
from datetime import datetime
from functools import wraps
from http.cookies import SimpleCookie
from io import StringIO
from os import path, environ

from flask import Flask, request, abort, Response, redirect, url_for, \
    render_template_string, jsonify
from flask_compress import Compress
from flask_cors import CORS
from flask_jwt_extended import create_access_token, jwt_required, \
    get_jwt_identity, JWTManager, set_access_cookies
from flask_restful import Resource, Api, reqparse
from jinja2 import Environment

from indra.assemblers.html import HtmlAssembler, IndraHTMLLoader
from indra.statements import make_statement_camel, stmts_from_json
from indra.util import batch_iter
from indra_db.client import get_statement_jsons_from_agents, \
    get_statement_jsons_from_hashes, get_statement_jsons_from_papers, \
    submit_curation, _has_auth, BadHashError
from rest_api.models import User

logger = logging.getLogger("db-api")
logger.setLevel(logging.INFO)

app = Flask(__name__)

app.config['DEBUG'] = True
app.config['JWT_SECRET_KEY'] = environ['INDRADB_JWT_SECRET']
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 2592000  # 30 days
app.config['JWT_TOKEN_LOCATION'] = ['cookies']
app.config['JWT_COOKIE_SECURE'] = True
app.config['JWT_SESSION_COOKIE'] = False

Compress(app)
CORS(app)
SC = SimpleCookie()
api = Api(app)
jwt = JWTManager(app)

print("Loading file")
logger.info("INFO working.")
logger.warning("WARNING working.")
logger.error("ERROR working.")

MAX_STATEMENTS = int(1e3)
TITLE = "The INDRA Database"
HERE = path.abspath(path.dirname(__file__))

# Instantiate a jinja2 env.
env = Environment(
    loader=IndraHTMLLoader({None: HERE,
                            'indra': IndraHTMLLoader.native_path})
)

# Here we can add functions to the jinja2 env.
env.globals.update(url_for=url_for)


def render_my_template(template, title, **kwargs):
    kwargs['title'] = TITLE + ': ' + title
    return env.get_template(template).render(**kwargs)


class DbAPIError(Exception):
    pass


parser = reqparse.RequestParser()
parser.add_argument('email', help='Enter your email.', required=True)
parser.add_argument('password', help='Enter your password.', required=True)


class UserRegistration(Resource):
    def post(self):
        data = parser.parse_args()

        new_user = User.new_user(
            email=data['email'],
            password=data['password']
        )

        try:
            new_user.save()
            return {'message': 'User {} was created'.format(data['email'])}
        except Exception as e:
            return {'message': 'Something went wrong: %s' % str(e)}, 500


class UserLogin(Resource):
    def post(self):
        data = parser.parse_args()
        current_user = User.get_user_by_email(data['email'], data['password'])

        if not current_user:
            return {'message': 'Username or password was incorrect.'}

        access_token = create_access_token(identity=data['email'])
        resp = {'login': True}
        set_access_cookies(resp, access_token)
        return resp


api.add_resource(UserRegistration, '/register')
api.add_resource(UserLogin, '/login')


def __process_agent(agent_param):
    """Get the agent id and namespace from an input param."""
    if not agent_param.endswith('@TEXT'):
        param_parts = agent_param.split('@')
        if len(param_parts) == 2:
            ag, ns = param_parts
        elif len(param_parts) == 1:
            ag = agent_param
            ns = 'NAME'
        else:
            raise DbAPIError('Unrecognized agent spec: \"%s\"' % agent_param)
    else:
        ag = agent_param[:-5]
        ns = 'TEXT'

    if ns == 'HGNC-SYMBOL':
        ns = 'NAME'

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


def _query_wrapper(f):
    logger.info("Calling outer wrapper.")

    @wraps(f)
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
        do_stream_str = query.pop('stream', 'false')
        do_stream = True if do_stream_str == 'true' else False
        max_stmts = min(int(query.pop('max_stmts', MAX_STATEMENTS)),
                        MAX_STATEMENTS)
        format = query.pop('format', 'json')

        api_key = query.pop('api_key', kwargs.pop('api_key', None))
        logger.info("Running function %s after %s seconds."
                    % (f.__name__, sec_since(start_time)))
        result = f(query, offs, max_stmts, ev_lim, best_first, *args, **kwargs)
        if isinstance(result, Response):
            return result

        logger.info("Finished function %s after %s seconds."
                    % (f.__name__, sec_since(start_time)))

        # Handle any necessary redactions
        stmts_json = result.pop('statements')
        has = {src: _has_auth(src, api_key) for src in ['elsevier', 'medscan']}
        logger.info('Auths: %s' % str(has))
        medscan_redactions = 0
        medscan_removals = 0
        elsevier_redactions = 0
        source_counts = result['source_counts']
        if not all(has.values()):
            for h, stmt_json in stmts_json.copy().items():
                for ev_json in stmt_json['evidence'][:]:

                    # Check for elsevier and redact if necessary
                    if get_source(ev_json) == 'elsevier' \
                            and not has['elsevier']:
                        text = ev_json['text']
                        if len(text) > 200:
                            ev_json['text'] = text[:200] + REDACT_MESSAGE
                            elsevier_redactions += 1

                    # Check for medscan and redact if necessary
                    elif ev_json['source_api'] == 'medscan' \
                            and not has['medscan']:
                        medscan_redactions += 1
                        stmt_json['evidence'].remove(ev_json)
                        if len(stmt_json['evidence']) == 0:
                            medscan_removals += 1
                            stmts_json.pop(h)
                            result['source_counts'].pop(h)

            # Take medscan counts out of the source counts.
            if not has['medscan']:
                source_counts = {h: {src: count
                                     for src, count in src_count.items()
                                     if src is not 'medscan'}
                                 for h, src_count in source_counts.items()}

        logger.info(f"Redacted {medscan_redactions} medscan evidence, "
                    f"resulting in the complete removal of {medscan_removals} "
                    f"statements.")
        logger.info(f"Redacted {elsevier_redactions} pieces of elsevier "
                    f"evidence.")

        logger.info("Finished redacting evidence for %s after %s seconds."
                    % (f.__name__, sec_since(start_time)))

        # Add derived values to the result.
        result['offset'] = offs
        result['evidence_limit'] = ev_lim
        result['statement_limit'] = MAX_STATEMENTS
        result['statements_returned'] = len(stmts_json)

        if format == 'html':
            title = TITLE + ': ' + 'Results'
            ev_totals = result.pop('evidence_totals')
            stmts = stmts_from_json(stmts_json.values())
            html_assembler = HtmlAssembler(stmts, result, ev_totals,
                                           source_counts, title=title,
                                           db_rest_url=request.url_root[:-1])
            idbr_template = env.get_template('idbr_statements_view.html')
            content = html_assembler.make_model(idbr_template)
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

        if do_stream:
            # Returning a generator should stream the data.
            resp_json_bts = content
            gen = batch_iter(resp_json_bts, 10000)
            resp = Response(gen, mimetype=mimetype)
        else:
            resp = Response(content, mimetype=mimetype)
        logger.info("Exiting with %d statements with %d/%d evidence of size "
                    "%f MB after %s seconds."
                    % (result['statements_returned'],
                       result['evidence_returned'], result['total_evidence'],
                       sys.getsizeof(resp.data) / 1e6, sec_since(start_time)))
        return resp

    return decorator


@app.route('/test_security')
@jwt_required
def home():
    return render_template_string('Hello authorized person!')


@app.route('/', methods=['GET'])
def iamalive():
    return redirect('welcome', code=302)


@app.route('/welcome', methods=['GET'])
def welcome():
    logger.info("Browser welcome page.")
    onclick = ("window.location = window.location.href.replace('welcome', "
               "'statements')")
    return render_my_template('welcome.html', 'Welcome',
                              onclick_action=onclick)


with open(path.join(HERE, 'static', 'curationFunctions.js'), 'r') as f:
    CURATION_JS = f.read()


@app.route('/code/curationFunctions.js')
def serve_js():
    return Response(CURATION_JS, mimetype='text/javascript')


with open(path.join(HERE, 'static', 'favicon.ico'), 'rb') as f:
    ICON = f.read()


@app.route('/favicon.ico')
def serve_icon():
    return Response(ICON, mimetype='image/x-icon')


@app.route('/statements', methods=['GET'])
def get_statements_query_format():
    # Create a template object from the template file, load once
    return render_my_template('search_statements.html', 'Search',
                              message="Welcome! Try asking a question.",
                              endpoint=request.url_root)


def _answer_binary_query(act_raw, roled_agents, free_agents, offs, max_stmts,
                         ev_limit, best_first):
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
                                        best_first=best_first)
    return result


@app.route('/statements/from_agents', methods=['GET'])
@_query_wrapper
def get_statements(query_dict, offs, max_stmts, ev_limit, best_first):
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
                                max_stmts, ev_limit, best_first)


@app.route('/statements/from_hashes', methods=['POST'])
@_query_wrapper
def get_statements_by_hashes(query_dict, offs, max_stmts, ev_lim, best_first):
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
                                             best_first=best_first)
    return result


@app.route('/statements/from_hash/<hash_val>', methods=['GET'])
@_query_wrapper
def get_statement_by_hash(query_dict, offs, max_stmts, ev_limit, best_first,
                          hash_val):
    if ev_limit is None:
        ev_limit = 10000
    return get_statement_jsons_from_hashes([hash_val], max_stmts=max_stmts,
                                           offset=offs, ev_limit=ev_limit,
                                           best_first=best_first)


@app.route('/statements/from_papers', methods=['POST'])
@_query_wrapper
def get_paper_statements(query_dict, offs, max_stmts, ev_limit, best_first):
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
                                             best_first=best_first)
    return result


@app.route('/curation', methods=['GET'])
def describe_curation():
    return redirect('/statements', code=302)


@app.route('/curation/submit/<hash_val>', methods=['POST'])
@jwt_required
def submit_curation_endpoint(hash_val, **kwargs):
    logger.info("Adding curation for statement %s." % hash_val)
    current_user = get_jwt_identity()
    logger.info(current_user)
    if not isinstance(current_user, str):
        assert False, "Current user not str."
    ev_hash = request.json.get('ev_hash')
    source_api = request.json.pop('source', 'DB REST API')
    tag = request.json.get('tag')
    ip = request.remote_addr
    text = request.json.get('text')
    is_test = 'test' in request.args
    if not is_test:
        assert tag is not 'test'
        try:
            dbid = submit_curation(hash_val, tag, current_user, ip, '', text,
                                   ev_hash, source_api)
        except BadHashError as e:
            abort(Response("Invalid hash: %s." % e.mk_hash, 400))
        res = {'result': 'success', 'ref': {'id': dbid}}
    else:
        res = {'result': 'test passed', 'ref': None}
    logger.info("Got result: %s" % str(res))
    return Response(json.dumps(res), mimetype='application/json')


if __name__ == '__main__':
    app.run()
