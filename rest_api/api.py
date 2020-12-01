import sys
import json
import logging
from os import path
from datetime import datetime
from argparse import ArgumentParser
from collections import defaultdict

from flask_cors import CORS
from flask_compress import Compress
from flask import url_for as base_url_for
from jinja2 import Environment, ChoiceLoader
from flask_jwt_extended import get_jwt_identity
from flask import Flask, request, abort, Response, redirect, jsonify

from indra.statements import get_all_descendants, Statement
from indra.assemblers.html.assembler import loader as indra_loader

from indra_db.exceptions import BadHashError
from indra_db.client.principal.curation import *
from indra_db.client.readonly import AgentJsonExpander
from indra_db.util.constructors import get_ro_host

from indralab_auth_tools.auth import auth, resolve_auth, config_auth
from indralab_auth_tools.log import note_in_log, set_log_service_name, \
    user_log_endpoint

from rest_api.config import *
from rest_api.call_handlers import *
from rest_api.util import sec_since, get_s3_client, gilda_ground, \
    _make_english_from_meta, get_html_source_info


# =========================
# A lot of config and setup
# =========================

# Get a logger, and assert the logging level.
logger = logging.getLogger("db rest api")
logger.setLevel(logging.INFO)

# Set the name of this service for the usage logs.
set_log_service_name(f"db-rest-api-{DEPLOYMENT if DEPLOYMENT else 'stable'}")


# Define a custom flask class to handle the deployment name prefix.
class MyFlask(Flask):
    def route(self, url, *args, **kwargs):
        if DEPLOYMENT is not None:
            url = f'/{DEPLOYMENT}{url}'
        flask_dec = super(MyFlask, self).route(url, **kwargs)
        return flask_dec


# Propagate the deployment name to the static path and auth URLs.
static_url_path = None
if DEPLOYMENT is not None:
    static_url_path = f'/{DEPLOYMENT}/static'
    auth.url_prefix = f'/{DEPLOYMENT}'


# Initialize the flask application (with modified static path).
app = MyFlask(__name__, static_url_path=static_url_path)


# Register the auth application, and config it if we are not testing.
app.register_blueprint(auth)
app.config['DEBUG'] = True
if not TESTING['status']:
    SC, jwt = config_auth(app)
else:
    logger.warning("TESTING: No auth will be enabled.")

# Apply wrappers to the app that will compress responses and enable CORS.
Compress(app)
CORS(app)

# The directory path to this location (works in any file system).
HERE = path.abspath(path.dirname(__file__))

# Instantiate a jinja2 env.
env = Environment(loader=ChoiceLoader([app.jinja_loader, auth.jinja_loader,
                                       indra_loader]))


# Overwrite url_for function in jinja to handle DEPLOYMENT prefix gracefully.
def url_for(*args, **kwargs):
    """Generate a url for a given endpoint, applying the DEPLOYMENT prefix."""
    res = base_url_for(*args, **kwargs)
    if DEPLOYMENT is not None:
        if not res.startswith(f'/{DEPLOYMENT}'):
            res = f'/{DEPLOYMENT}' + res
    return res


env.globals.update(url_for=url_for)


# Define a useful helper function.
def render_my_template(template, title, **kwargs):
    """Render a Jinja2 template wrapping in identity and other details."""
    kwargs['title'] = TITLE + ': ' + title
    if not TESTING['status']:
        kwargs['identity'] = get_jwt_identity()

    # Set nav elements as inactive by default.
    for nav_element in ['search', 'old_search']:
        key = f'{nav_element}_active'
        kwargs[key] = kwargs.pop(key, False)

    kwargs['simple'] = False
    return env.get_template(template).render(**kwargs)


# ==========================
# Here begins the API proper
# ==========================


@app.route('/', methods=['GET'])
def root():
    return redirect(url_for('search'), code=302)


@app.route('/healthcheck', methods=['GET'])
def i_am_alive():
    return jsonify({'status': 'testing' if TESTING['status'] else 'healthy'})


@app.route('/ground', methods=['GET'])
def ground():
    ag = request.args['agent']
    res_json = gilda_ground(ag)
    return jsonify(res_json)


@app.route('/search', methods=['GET'])
@jwt_nontest_optional
@user_log_endpoint
def search():
    stmt_types = {c.__name__ for c in get_all_descendants(Statement)}
    stmt_types -= {'Influence', 'Event', 'Unresolved'}
    stmt_types_json = json.dumps(sorted(list(stmt_types)))
    source_info, source_colors = get_html_source_info()
    if TESTING['status']:
        vue_src = url_for("serve_indralab_vue", file='IndralabVue.umd.js')
        vue_style = url_for("serve_indralab_vue", file='IndralabVue.css')
    else:
        vue_src = f'{VUE_ROOT}/IndralabVue.umd.js'
        vue_style = f'{VUE_ROOT}/IndralabVue.css'
    return render_my_template(
        'search.html', 'Search', source_colors=source_colors,
        source_info=source_info, search_active=True, vue_src=vue_src,
        vue_style=vue_style, stmt_types_json=stmt_types_json)


@app.route('/data-vis/<path:file_path>')
def serve_data_vis(file_path):
    full_path = path.join(HERE, 'data-vis/dist', file_path)
    logger.info('data-vis: ' + full_path)
    if not path.exists(full_path):
        return abort(404)
    ext = full_path.split('.')[-1]
    if ext == 'js':
        ct = 'application/javascript'
    elif ext == 'css':
        ct = 'text/css'
    else:
        ct = None
    with open(full_path, 'rb') as f:
        return Response(f.read(),
                        content_type=ct)


if TESTING['status']:
    assert path.exists(VUE_ROOT), "Cannot test API with Vue packages."

    @app.route('/ilv/<path:file>')
    def serve_indralab_vue(file):
        full_path = path.join(HERE, VUE_ROOT, file)
        if not path.exists(full_path):
            return abort(404)
        ext = full_path.split('.')[-1]
        if ext == 'js':
            ct = 'application/javascript'
        elif ext == 'css':
            ct = 'text/css'
        else:
            ct = None
        with open(full_path, 'rb') as f:
            return Response(f.read(),
                            content_type=ct)


@app.route('/monitor')
def get_data_explorer():
    return render_my_template('daily_data.html', 'Monitor')


@app.route('/monitor/data/runtime')
def serve_runtime():
    from indra_db.util.data_gatherer import S3_DATA_LOC

    s3 = get_s3_client()
    res = s3.get_object(Bucket=S3_DATA_LOC['bucket'],
                        Key=S3_DATA_LOC['prefix']+'runtimes.json')
    return jsonify(json.loads(res['Body'].read()))


@app.route('/monitor/data/liststages')
def list_stages():
    from indra_db.util.data_gatherer import S3_DATA_LOC

    s3 = get_s3_client()
    res = s3.list_objects_v2(Bucket=S3_DATA_LOC['bucket'],
                             Prefix=S3_DATA_LOC['prefix'],
                             Delimiter='/')

    ret = [k[:-len('.json')] for k in (e['Key'][len(S3_DATA_LOC['prefix']):]
                                       for e in res['Contents'])
           if k.endswith('.json') and not k.startswith('runtimes')]
    print(ret)
    return jsonify(ret)


@app.route('/monitor/data/<stage>')
def serve_stages(stage):
    from indra_db.util.data_gatherer import S3_DATA_LOC

    s3 = get_s3_client()
    res = s3.get_object(Bucket=S3_DATA_LOC['bucket'],
                        Key=S3_DATA_LOC['prefix'] + stage + '.json')

    return jsonify(json.loads(res['Body'].read()))


@app.route('/statements', methods=['GET'])
@jwt_nontest_optional
@user_log_endpoint
def old_search():
    # Create a template object from the template file, load once
    url_base = request.url_root
    if DEPLOYMENT is not None:
        url_base = f'{url_base}{DEPLOYMENT}/'
    source_info, source_colors = get_html_source_info()
    return render_my_template('search_statements.html', 'Search',
                              message="Welcome! Try asking a question.",
                              old_search_active=True, source_info=source_info,
                              source_colors=source_colors, endpoint=url_base)


@app.route('/<result_type>/<path:method>', methods=['GET', 'POST'])
@app.route('/metadata/<result_type>/<path:method>', methods=['GET', 'POST'])
@user_log_endpoint
def get_statements(result_type, method):
    """Get some statements constrained by query."""
    if result_type not in ApiCall.valid_result_types:
        return Response('Page not found.', 404)

    note_in_log(method=method, result_type=result_type)
    note_in_log(db_host=get_ro_host('primary'))

    if method == 'from_agents' and request.method == 'GET':
        call = FromAgentsApiCall(env)
    elif method == 'from_hashes' and request.method == 'POST':
        call = FromHashesApiCall(env)
    elif method.startswith('from_hash/') and request.method == 'GET':
        call = FromHashApiCall(env)
        call.web_query['hash'] = method[len('from_hash/'):]
    elif method == 'from_papers' and request.method == 'POST':
        call = FromPapersApiCall(env)
    elif method == 'from_agent_json' and request.method == 'POST':
        call = FromAgentJsonApiCall(env)
    elif method == 'from_query_json' and request.method == 'POST':
        call = FromQueryJsonApiCall(env)
    else:
        return abort(Response('Page not found.', 404))

    return call.run(result_type=result_type)


@app.route('/expand', methods=['POST'])
def expand_meta_row():
    start_time = datetime.now()

    # Get the agent_json and hashes
    agent_json = request.json.get('agent_json')
    if not agent_json:
        logger.error("No agent_json provided!")
        return abort(Response("No agent_json in request!", 400))
    stmt_type = request.json.get('stmt_type')
    hashes = request.json.get('hashes')
    logger.info(f"Expanding on agent_json={agent_json}, stmt_type={stmt_type}, "
                f"hashes={hashes}")

    w_cur_counts = \
        request.args.get('with_cur_counts', 'False').lower() == 'true'

    # Figure out authorization.
    has_medscan = False
    if not TESTING['status']:
        user, roles = resolve_auth(request.args.copy())
        for role in roles:
            has_medscan |= role.permissions.get('medscan', False)
    else:
        api_key = request.args.get('api_key', None)
        has_medscan = api_key is not None
    logger.info(f'Auths for medscan: {has_medscan}')

    # Get the sorting parameter.
    sort_by = request.args.get('sort_by', 'ev_count')

    # Get the more detailed results.
    q = AgentJsonExpander(agent_json, stmt_type=stmt_type, hashes=hashes)
    result = q.expand(sort_by=sort_by)

    # Filter out any medscan content, and construct english.
    entry_hash_lookup = defaultdict(list)
    for key, entry in result.results.copy().items():
        # Filter medscan...
        if not has_medscan:
            result.evidence_totals[key] -= \
                entry['source_counts'].pop('medscan', 0)
            entry['total_count'] = result.evidence_totals[key]
            if not entry['source_counts']:
                logger.warning("Censored content present. Removing it.")
                result.results.pop(key)
                result.evidence_totals.pop(key)
                continue

        # Add english...
        eng = _make_english_from_meta(entry['agents'],
                                      entry.get('type'))
        if not eng:
            logger.warning(f"English not formed for {key}:\n"
                           f"{entry}")
        entry['english'] = eng

        # Prep curation counts
        if w_cur_counts:
            if 'hashes' in entry:
                for h in entry['hashes']:
                    entry['cur_count'] = 0
                    entry_hash_lookup[h].append(entry)
            elif 'hash' in entry:
                entry['cur_count'] = 0
                entry_hash_lookup[entry['hash']].append(entry)
            else:
                assert False, "Entry has no hash info."

        # Stringify hashes for JavaScript
        if 'hashes' in entry:
            entry['hashes'] = [str(h) for h in entry['hashes']]
        else:
            entry['hash'] = str(entry['hash'])

    if w_cur_counts:
        curations = get_curations(pa_hash=set(entry_hash_lookup.keys()))
        for cur in curations:
            for entry in entry_hash_lookup[cur['pa_hash']]:
                entry['cur_count'] += 1

    res_json = result.json()
    res_json['relations'] = list(res_json['results'].values())
    res_json.pop('results')
    resp = Response(json.dumps(res_json), mimetype='application/json')
    logger.info(f"Returning expansion with {len(result.results)} meta results "
                f"that represent {result.total_evidence} total evidence. Size "
                f"is {sys.getsizeof(resp.data) / 1e6} MB after "
                f"{sec_since(start_time)} seconds.")
    return resp


@app.route('/query/<result_type>', methods=['GET', 'POST'])
@user_log_endpoint
def get_statements_by_query_json(result_type):
    note_in_log(result_type=result_type)
    return FallbackQueryApiCall(env).run(result_type)


@app.route('/curation', methods=['GET'])
def describe_curation():
    return redirect('/statements', code=302)


@app.route('/curation/submit/<hash_val>', methods=['POST'])
@jwt_nontest_optional
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


@app.route('/curation/list/<stmt_hash>/<src_hash>', methods=['GET'])
def list_curations(stmt_hash, src_hash):
    curations = get_curations(pa_hash=stmt_hash, source_hash=src_hash)
    return jsonify(curations)


def main():
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000)
    args = parser.parse_args()
    app.run(port=args.port)


if __name__ == '__main__':
    main()
