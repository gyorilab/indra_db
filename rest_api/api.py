import sys
import json
import logging
from os import path, environ
from functools import wraps
from datetime import datetime

from flask import Flask, request, abort, Response, redirect, jsonify
from flask import url_for as base_url_for
from flask_compress import Compress
from flask_cors import CORS
from flask_jwt_extended import get_jwt_identity, jwt_optional
from jinja2 import Environment, ChoiceLoader

from indra.assemblers.html.assembler import loader as indra_loader, \
    stmts_from_json, HtmlAssembler, SOURCE_COLORS, _format_evidence_text, \
    _format_stmt_text
from indra.assemblers.english import EnglishAssembler
from indra.statements import make_statement_camel, get_all_descendants
from indra_db.client.readonly.query import HasAgent, HasType, HasNumAgents, \
    HasOnlySource, StatementQueryResult, HasHash, QueryCore

from indralab_auth_tools.auth import auth, resolve_auth, config_auth

from indra_db.exceptions import BadHashError
from indra_db.client import get_statement_jsons_from_hashes, \
    get_statement_jsons_from_papers, submit_curation, \
    get_interaction_jsons_from_agents, stmt_from_interaction, get_curations
from .util import process_agent, _answer_binary_query, DbAPIError, LogTracker, \
    sec_since, get_source, get_s3_client, gilda_ground

logger = logging.getLogger("db rest api")
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

TITLE = "The INDRA Database"
HERE = path.abspath(path.dirname(__file__))
DEPLOYMENT = environ.get('INDRA_DB_API_DEPLOYMENT')

# Instantiate a jinja2 env.
env = Environment(loader=ChoiceLoader([app.jinja_loader, auth.jinja_loader,
                                       indra_loader]))


def url_for(*args, **kwargs):
    res = base_url_for(*args, **kwargs)
    if DEPLOYMENT is not None:
        pass
    return res


# Here we can add functions to the jinja2 env.
env.globals.update(url_for=url_for)


MAX_STATEMENTS = int(1e3)
REDACT_MESSAGE = '[MISSING/INVALID API KEY: limited to 200 char for Elsevier]'


def render_my_template(template, title, **kwargs):
    kwargs['title'] = TITLE + ': ' + title
    kwargs['identity'] = get_jwt_identity()
    return env.get_template(template).render(**kwargs)


def _query_wrapper(get_db_query):
    logger.info("Calling outer wrapper.")

    @wraps(get_db_query)
    @jwt_optional
    def decorator(*args, **kwargs):
        tracker = LogTracker()
        start_time = datetime.now()
        logger.info("Got query for %s at %s!"
                    % (get_db_query.__name__, start_time))

        web_query = request.args.copy()
        offs = web_query.pop('offset', None)
        ev_lim = web_query.pop('ev_limit', None)
        best_first = web_query.pop('best_first', 'true').lower() == 'true'
        max_stmts = min(int(web_query.pop('max_stmts', MAX_STATEMENTS)),
                        MAX_STATEMENTS)
        fmt = web_query.pop('format', 'json')
        w_english = web_query.pop('with_english', 'false').lower() == 'true'
        w_cur_counts = \
            web_query.pop('with_cur_counts', 'false').lower() == 'true'

        # Figure out authorization.
        has = dict.fromkeys(['elsevier', 'medscan'], False)
        user, roles = resolve_auth(web_query)
        for role in roles:
            for resource in has.keys():
                has[resource] |= role.permissions.get(resource, False)
        logger.info('Auths: %s' % str(has))

        # Actually run the function.
        logger.info("Running function %s after %s seconds."
                    % (get_db_query.__name__, sec_since(start_time)))
        db_query = get_db_query(web_query, *args, **kwargs)
        if isinstance(db_query, Response):
            return db_query
        elif not isinstance(db_query, QueryCore):
            raise RuntimeError("Result should be a child of QueryCore.")

        if ev_lim is None:
            if get_db_query is get_statement_by_hash:
                ev_lim = 10000
            else:
                ev_lim = 10

        if not has['medscan']:
            minus_q = ~HasOnlySource('medscan')
            db_query &= minus_q
            ev_filter = minus_q.ev_filter()
        else:
            ev_filter = None

        result = db_query.get_statements(offset=offs, limit=max_stmts,
                                         ev_limit=ev_lim, best_first=best_first,
                                         evidence_filter=ev_filter)

        logger.info("Finished function %s after %s seconds."
                    % (get_db_query.__name__, sec_since(start_time)))

        # Handle any necessary redactions
        res_json = result.json()
        stmts_json = res_json.pop('results')
        elsevier_redactions = 0
        source_counts = result.source_counts
        if not all(has.values()) or fmt == 'json-js' or w_english:
            for h, stmt_json in stmts_json.copy().items():
                if w_english:
                    stmt = stmts_from_json([stmt_json])[0]
                    stmt_json['english'] = _format_stmt_text(stmt)
                    stmt_json['evidence'] = _format_evidence_text(stmt)

                if has['elsevier'] and fmt != 'json-js' and not w_english:
                    continue

                if not has['medscan']:
                    source_counts[h].pop('medscan', 0)

                for ev_json in stmt_json['evidence'][:]:
                    if fmt == 'json-js':
                        ev_json['source_hash'] = str(ev_json['source_hash'])

                    # Check for elsevier and redact if necessary
                    if not has['elsevier'] and \
                            get_source(ev_json) == 'elsevier':
                        text = ev_json['text']
                        if len(text) > 200:
                            ev_json['text'] = text[:200] + REDACT_MESSAGE
                            elsevier_redactions += 1

        logger.info(f"Redacted {elsevier_redactions} pieces of elsevier "
                    f"evidence.")

        logger.info("Finished redacting evidence for %s after %s seconds."
                    % (get_db_query.__name__, sec_since(start_time)))

        # Get counts of the curations for the resulting statements.
        if w_cur_counts:
            curations = get_curations(pa_hash=set(stmts_json.keys()))
            logger.info("Found %d curations" % len(curations))
            cur_counts = {}
            for curation in curations:
                # Update the overall counts.
                if curation.pa_hash not in cur_counts:
                    cur_counts[curation.pa_hash] = 0
                cur_counts[curation.pa_hash] += 1

                # Work these counts into the evidence dict structure.
                for ev_json in stmts_json[curation.pa_hash]['evidence']:
                    if str(ev_json['source_hash']) == str(curation.source_hash):
                        ev_json['num_curations'] = \
                            ev_json.get('num_curations', 0) + 1
                        break
            result['num_curations'] = cur_counts

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
            identity = user.identity() if user else None
            content = html_assembler.make_model(idbr_template,
                                                identity=identity)
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

# ==========================
# Here begins the API proper
# ==========================


def dep_route(url, **kwargs):
    if DEPLOYMENT is not None:
        url = f'/{DEPLOYMENT}{url}'
    flask_dec = app.route(url, **kwargs)
    return flask_dec


@dep_route('/', methods=['GET'])
def iamalive():
    return redirect(url_for('search'), code=302)


@dep_route('/ground', methods=['GET'])
def ground():
    ag = request.args['agent']
    res_json = gilda_ground(ag)
    return jsonify(res_json)


@dep_route('/search', methods=['GET'])
def search():
    return render_my_template('search.html', 'Search',
                              source_colors=SOURCE_COLORS)


@dep_route('/data-vis/<path:file_path>')
def serve_data_vis(file_path):
    full_path = path.join(HERE, 'data-vis/dist', file_path)
    logger.info('data-vis: ' + full_path)
    if not path.exists(full_path):
        abort(404)
        return
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


@dep_route('/monitor')
def get_data_explorer():
    return render_my_template('daily_data.html', 'Monitor')


@dep_route('/monitor/data/runtime')
def serve_runtime():
    from indra_db.util.data_gatherer import S3_DATA_LOC

    s3 = get_s3_client()
    res = s3.get_object(Bucket=S3_DATA_LOC['bucket'],
                        Key=S3_DATA_LOC['prefix']+'runtimes.json')
    return jsonify(json.loads(res['Body'].read()))


@dep_route('/monitor/data/liststages')
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


@dep_route('/monitor/data/<stage>')
def serve_stages(stage):
    from indra_db.util.data_gatherer import S3_DATA_LOC

    s3 = get_s3_client()
    res = s3.get_object(Bucket=S3_DATA_LOC['bucket'],
                        Key=S3_DATA_LOC['prefix'] + stage + '.json')

    return jsonify(json.loads(res['Body'].read()))


@dep_route('/statements', methods=['GET'])
@jwt_optional
def get_statements_query_format():
    # Create a template object from the template file, load once
    return render_my_template('search_statements.html', 'Search',
                              message="Welcome! Try asking a question.",
                              endpoint=request.url_root)


@dep_route('/statements/from_agents', methods=['GET'])
@_query_wrapper
def get_statements(query_dict):
    """Get some statements constrained by query."""
    logger.info("Getting query details.")
    db_query = None
    num_agents = 0
    try:
        # Get the agents without specified locations (subject or object).
        free_agents = (ag for ag_gen in [query_dict.poplist('agent'),
                                         (query_dict.pop(k)
                                          for k in {k for k in query_dict.keys()
                                                    if k.startswith('agent')})]
                       for ag in ag_gen)
        for raw_ag in free_agents:
            num_agents += 1
            ag, ns = process_agent(raw_ag)
            new_q = HasAgent(ag, namespace=ns)
            if db_query is None:
                db_query = new_q
            else:
                db_query &= new_q

        # Get the agents with specified roles.
        for role in ['subject', 'object']:
            num_agents += 1
            raw_ag = query_dict.pop(role)
            if raw_ag is None:
                continue
            ag, ns = process_agent(raw_ag)
            new_q = HasAgent(ag, namespace=ns, role=role.upper())
            if db_query is None:
                db_query = new_q
            else:
                db_query &= new_q
    except DbAPIError as e:
        logger.exception(e)
        abort(Response('Failed to make agents from names: %s\n' % str(e), 400))
        return

    if db_query is None:
        abort(Response('No agents found in request.', 400))

    # Get the raw name of the statement type (we allow for variation in case).
    act_raw = query_dict.pop('type', None)
    if act_raw is not None:
        act = make_statement_camel(act_raw)
        db_query &= HasType([act])

    # Get whether the user wants a strict match
    if query_dict.pop('strict', 'false').lower() == 'true':
        db_query &= HasNumAgents((num_agents,))

    # If there was something else in the query, there shouldn't be, so
    # someone's probably confused.
    if query_dict:
        abort(Response("Unrecognized query options; %s."
                       % list(query_dict.keys()), 400))
        return

    return db_query


@dep_route('/statements/from_hashes', methods=['POST'])
@_query_wrapper
def get_statements_by_hashes(query_dict):
    hashes = request.json.get('hashes')
    if not hashes:
        logger.error("No hashes provided!")
        abort(Response("No hashes given!", 400))
    if len(hashes) > MAX_STATEMENTS:
        logger.error("Too many hashes given!")
        abort(Response("Too many hashes given, %d allowed." % MAX_STATEMENTS,
                       400))

    return HasHash(hashes)


@dep_route('/statements/from_hash/<hash_val>', methods=['GET'])
@_query_wrapper
def get_statement_by_hash(query_dict, hash_val):
    return HasHash([hash_val])


@dep_route('/statements/from_papers', methods=['POST'])
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


@dep_route('/curation', methods=['GET'])
def describe_curation():
    return redirect('/statements', code=302)


@dep_route('/curation/submit/<hash_val>', methods=['POST'])
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


@dep_route('/curation/list/<stmt_hash>/<src_hash>', methods=['GET'])
def list_curations(stmt_hash, src_hash):
    curations = get_curations(pa_hash=stmt_hash, source_hash=src_hash)
    curation_json = [cur.to_json() for cur in curations]
    return jsonify(curation_json)


@dep_route('/metadata/<level>/from_agents', methods=['GET'])
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
        agents = [(None,) + process_agent(ag)
                  for ag in query.poplist('agent')]
        ofaks = {k for k in query.keys() if k.startswith('agent')}
        agents += [(None,) + process_agent(query.pop(k)) for k in ofaks]

        # Get the agents with specified roles.
        agents += [(role,) + process_agent(query.pop(role))
                   for role in ['subject', 'object']
                   if query.get(role) is not None]
    except DbAPIError as e:
        logger.exception(e)
        abort(Response('Failed to make agents from names: %s\n' % str(e), 400))
        return

    def pop(k, default=None, type_cast=None):
        if isinstance(default, bool):
            val = query.pop(k, str(default).lower()) == 'true'
        else:
            val = query.pop(k, default)

        if type_cast is not None and val is not None:
            return type_cast(val)
        return val

    w_curations = pop('with_cur_counts', False)
    stmt_type = pop('type', type_cast=make_statement_camel)
    max_relations = pop('limit', type_cast=int)
    offset = pop('offset', type_cast=int)
    best_first = pop('best_first', True)
    res = get_interaction_jsons_from_agents(agents=agents, detail_level=level,
                                            stmt_type=stmt_type,
                                            max_relations=max_relations,
                                            offset=offset,
                                            best_first=best_first)

    dt = (datetime.utcnow() - start).total_seconds()
    logger.info("Got %s results after %.2f." % (len(res), dt))

    # Currently, a hash could get through (in one of the less detailed results)
    # that is entirely dependent on medscan.
    if not has['medscan']:
        censored_res = []
        for entry in res['relations']:
            entry['total_count'] -= entry['source_counts'].pop('medscan', 0)
            if entry['source_counts']:
                censored_res.append(entry)
        res['relations'] = censored_res

    for entry in res['relations']:
        if entry['type'] == 'ActiveForm':
            entry['english'] = entry['agents'][0] + ' has active form.'
        else:
            entry['english'] = \
                EnglishAssembler([stmt_from_interaction(entry)]).make_model()

    # Look up curations, if result with_curations was set.
    if w_curations:
        rel_hash_lookup = {}
        if level == 'hashes':
            for rel in res['relations']:
                rel['cur_count'] = 0
                rel_hash_lookup[rel['hash']] = rel
        else:
            for rel in res['relations']:
                for h in rel['hashes']:
                    rel['cur_count'] = 0
                    rel_hash_lookup[h] = rel
        curations = get_curations(pa_hash=set(rel_hash_lookup.keys()))
        for cur in curations:
            rel_hash_lookup[cur.pa_hash]['cur_count'] += 1

    # Finish up the query.
    dt = (datetime.utcnow() - start).total_seconds()
    logger.info("Returning with %s results after %.2f seconds."
                % (len(res), dt))

    resp = Response(json.dumps(res),
                    mimetype='application/json')

    dt = (datetime.utcnow() - start).total_seconds()
    logger.info("Result prepared after %.2f seconds." % dt)
    return resp


if __name__ == '__main__':
    app.run()
