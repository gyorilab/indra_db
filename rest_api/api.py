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
from indra.statements import make_statement_camel, get_all_descendants, \
    Statement
from indra_db.client.readonly.query import HasAgent, HasType, HasNumAgents, \
    HasOnlySource, HasHash, Query, FromPapers, FromMeshIds, EvidenceFilter, \
    EmptyQuery

from indralab_auth_tools.auth import auth, resolve_auth, config_auth

from indra_db.exceptions import BadHashError
from indra_db.client import submit_curation, stmt_from_interaction,\
    get_curations
from .util import process_agent, DbAPIError, LogTracker, sec_since, get_source,\
    get_s3_client, gilda_ground

logger = logging.getLogger("db rest api")
logger.setLevel(logging.INFO)

app = Flask(__name__)
if environ.get('TESTING_DB_APP') == '1':
    logger.warning("TESTING: No auth will be enabled.")
    TESTING = True
else:
    TESTING = False

app.register_blueprint(auth)
if not TESTING:
    app.config['DEBUG'] = True
    SC, jwt = config_auth(app)

Compress(app)
CORS(app)

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
    if not TESTING:
        kwargs['identity'] = get_jwt_identity()
    return env.get_template(template).render(**kwargs)


def jwt_nontest_optional(func):
    if TESTING:
        return func
    else:
        return jwt_optional(func)


class ApiCall:
    default_ev_lim = 10

    @jwt_nontest_optional
    def __init__(self):
        self.tracker = LogTracker()
        self.start_time = datetime.now()

        self.web_query = request.args.copy()
        self.offs = self._pop('offset', type_cast=int)
        self.best_first = self._pop('best_first', True, bool)
        self.max_stmts = min(self._pop('max_stmts', MAX_STATEMENTS, int),
                             MAX_STATEMENTS)
        self.fmt = self._pop('format', 'json')
        self.w_english = self._pop('with_english', False, bool)
        self.w_cur_counts = self._pop('with_cur_counts', False, bool)

        # Figure out authorization.
        self.has = dict.fromkeys(['elsevier', 'medscan'], False)
        if not TESTING:
            self.user, roles = resolve_auth(self.web_query)
            for role in roles:
                for resource in self.has.keys():
                    self.has[resource] |= role.permissions.get(resource, False)
            logger.info('Auths: %s' % str(self.has))
        else:
            self.web_query.pop('api_key', None)
            self.has['elsevier'] = False
            self.has['medscan'] = False
        return

    def run(self, result_type):
        # Get the db query object.
        logger.info("Running function %s after %s seconds."
                    % (self.__class__.__name__, sec_since(self.start_time)))
        db_query = self.get_db_query()
        if isinstance(db_query, Response):
            return db_query
        elif not isinstance(db_query, Query):
            raise RuntimeError("Result should be a child of Query.")

        if not self.has['medscan']:
            minus_q = ~HasOnlySource('medscan')
            db_query &= minus_q
        else:
            minus_q = None

        # Actually run the function
        params = dict(offset=self.offs, limit=self.max_stmts,
                      best_first=self.best_first)
        if result_type == 'statements':
            if minus_q:
                ev_filter = minus_q.ev_filter()
            else:
                ev_filter = None
            ev_lim = self._pop('ev_limit', self.default_ev_lim, int)
            res = db_query.get_statements(ev_lim=ev_lim,
                                          evidence_filter=ev_filter, **params)
            self.filter_evidence(res)
        elif result_type == 'interactions':
            res = db_query.get_statements(**params)
        elif result_type == 'relations':
            with_hashes = self._pop('with_hashes', type_cast=bool)
            res = db_query.get_relations(with_hashes=with_hashes, **params)
        elif result_type == 'agents':
            with_hashes = self._pop('with_hashes', type_cast=bool)
            res = db_query.get_agents(with_hashes=with_hashes, **params)
        elif result_type == 'hashes':
            res = db_query.get_hashes(**params)
        else:
            raise ValueError(f"Invalid result type: {result_type}")
        return self.process_result(res)

    def _pop(self, key, default=None, type_cast=None):
        if isinstance(default, bool):
            val = self.web_query.pop(key, str(default).lower()) == 'true'
        else:
            val = self.web_query.pop(key, default)

        if type_cast is not None and val is not None:
            return type_cast(val)
        return val

    def get_db_query(self):
        raise NotImplementedError()

    def process_result(self, result):
        res_json = result.json()
        content = json.dumps(res_json)

        resp = Response(content, mimetype='application/json')
        logger.info("Exiting with %d results that have %d total evidence, "
                    "with size %f MB after %s seconds."
                    % (len(res_json['results']),
                       res_json['total_evidence'],
                       sys.getsizeof(resp.data) / 1e6,
                       sec_since(self.start_time)))
        return resp

    def filter_evidence(self, result):
        elsevier_redactions = 0
        source_counts = result.source_counts
        if not all(self.has.values()) or self.fmt == 'json-js' \
                or self.w_english:
            for h, stmt_json in result.results.copy().items():
                if self.w_english:
                    stmt = stmts_from_json([stmt_json])[0]
                    stmt_json['english'] = _format_stmt_text(stmt)
                    stmt_json['evidence'] = _format_evidence_text(stmt)

                if self.has['elsevier'] and self.fmt != 'json-js' \
                        and not self.w_english:
                    continue

                if not self.has['medscan']:
                    source_counts[h].pop('medscan', 0)

                for ev_json in stmt_json['evidence'][:]:
                    if self.fmt == 'json-js':
                        ev_json['source_hash'] = str(ev_json['source_hash'])

                    # Check for elsevier and redact if necessary
                    if not self.has['elsevier'] and \
                            get_source(ev_json) == 'elsevier':
                        text = ev_json['text']
                        if len(text) > 200:
                            ev_json['text'] = text[:200] + REDACT_MESSAGE
                            elsevier_redactions += 1
        logger.info(f"Redacted {elsevier_redactions} pieces of elsevier "
                    f"evidence.")

        logger.info("Finished redacting evidence for %s after %s seconds."
                    % (self.__class__.__name__, sec_since(self.start_time)))
        return

    @staticmethod
    def get_curation_counts(result):
        # Get counts of the curations for the resulting statements.
        curations = get_curations(pa_hash=set(result.results.keys()))
        logger.info("Found %d curations" % len(curations))
        cur_counts = {}
        for curation in curations:
            # Update the overall counts.
            if curation.pa_hash not in cur_counts:
                cur_counts[curation.pa_hash] = 0
            cur_counts[curation.pa_hash] += 1

            # Work these counts into the evidence dict structure.
            for ev_json in result.results[curation.pa_hash]['evidence']:
                if str(ev_json['source_hash']) == str(curation.source_hash):
                    ev_json['num_curations'] = \
                        ev_json.get('num_curations', 0) + 1
                    break
        return cur_counts


class StatementApiCall(ApiCall):

    def get_db_query(self):
        raise NotImplementedError()

    def process_result(self, result):
        res_json = result.json()

        # Add derived values to the res_json.
        if self.w_cur_counts:
            res_json['curation_counts'] = self.get_curation_counts(result)
        res_json['statement_limit'] = MAX_STATEMENTS
        res_json['statements_returned'] = len(result.results)
        res_json['end_of_statements'] = (len(result.results) < MAX_STATEMENTS)
        res_json['statements_removed'] = 0
        res_json['evidence_returned'] = result.returned_evidence

        stmts_json = result.results
        if self.fmt == 'html':
            title = TITLE + ': ' + 'Results'
            ev_totals = res_json.pop('evidence_totals')
            stmts = stmts_from_json(stmts_json.values())
            html_assembler = HtmlAssembler(stmts, res_json, ev_totals,
                                           result.source_counts, title=title,
                                           db_rest_url=request.url_root[:-1])
            idbr_template = env.get_template('idbr_statements_view.html')
            identity = self.user.identity() if self.user else None
            content = html_assembler.make_model(idbr_template,
                                                identity=identity)
            if self.tracker.get_messages():
                level_stats = ['%d %ss' % (n, lvl.lower())
                               for lvl, n in self.tracker.get_level_stats().items()]
                msg = ' '.join(level_stats)
                content = html_assembler.append_warning(msg)
            mimetype = 'text/html'
        else:  # Return JSON for all other values of the format argument
            res_json.update(self.tracker.get_level_stats())
            res_json['statements'] = stmts_json
            content = json.dumps(res_json)
            mimetype = 'application/json'

        resp = Response(content, mimetype=mimetype)
        logger.info("Exiting with %d statements with %d/%d evidence of size "
                    "%f MB after %s seconds."
                    % (res_json['statements_returned'],
                       res_json['evidence_returned'],
                       res_json['total_evidence'],
                       sys.getsizeof(resp.data) / 1e6,
                       sec_since(self.start_time)))
        return resp


def iter_free_agents(query_dict):
    agent_keys = {k for k in query_dict.keys() if k.startswith('agent')}
    for k in agent_keys:
        entry = query_dict.pop(k)
        if isinstance(entry, list):
            for agent in entry:
                yield agent
        else:
            yield entry


def _db_query_from_web_query(query_dict, require_any=None, require_all=None,
                             empty_web_query=False):
    db_query = EmptyQuery()
    num_agents = 0

    logger.info(f"Making DB query from:\n{query_dict}")
    filter_ev = query_dict.pop('filter_ev', 'false').lower() == 'true'
    ev_filter = EvidenceFilter()

    # Get the agents without specified locations (subject or object).
    for raw_ag in iter_free_agents(query_dict):
        ag, ns = process_agent(raw_ag)
        db_query &= HasAgent(ag, namespace=ns)

    # Get the agents with specified roles.
    for role in ['subject', 'object']:
        raw_ag = query_dict.pop(role, None)
        if raw_ag is None:
            continue
        if isinstance(raw_ag, list):
            assert len(raw_ag) == 1, f'Malformed agent for {role}: {raw_ag}'
            raw_ag = raw_ag[0]
        num_agents += 1
        ag, ns = process_agent(raw_ag)
        db_query &= HasAgent(ag, namespace=ns, role=role.upper())

    # Get the raw name of the statement type (we allow for variation in case).
    act_raw = query_dict.pop('type', None)
    if act_raw is not None:
        if isinstance(act_raw, list):
            assert len(act_raw) == 1, \
                f"Got multiple entries for statement type: {act_raw}."
            act_raw = act_raw[0]
        act = make_statement_camel(act_raw)
        db_query &= HasType([act])

    # Get whether the user wants a strict match
    if _pop(query_dict, 'strict', False, bool):
        db_query &= HasNumAgents((num_agents,))

    # Unpack hashes, if present.
    hashes = query_dict.pop('hashes', None)
    if hashes:
        db_query &= HasHash(hashes)

    # Unpack paper ids, if present:
    id_tpls = set()
    for id_str in query_dict.pop('paper_ids', []):
        val, typ = id_str.split('@')

        # Turn tcids and trids into integers.
        id_val = int(val) if typ in ['tcid', 'trid'] else val
        id_tpls.add((typ, id_val))

    if id_tpls:
        paper_q = FromPapers(id_tpls)
        db_query &= paper_q
        if filter_ev:
            ev_filter &= paper_q.ev_filter()

    # Unpack mesh ids.
    mesh_ids = query_dict.pop('mesh_ids', [])
    if mesh_ids:
        mesh_q = FromMeshIds(mesh_ids)
        db_query &= mesh_q
        if filter_ev:
            ev_filter &= mesh_q.ev_filter()

    # Check for health of the resulting query, and some other things.
    if isinstance(db_query, EmptyQuery):
        raise DbAPIError(f"No arguments from web query {query_dict} mapped to "
                         f"db query.")
    assert isinstance(db_query, Query), "Somehow db_query is not Query."

    component_queries = set(db_query.get_component_queries())
    if require_any and not set(require_any) & component_queries:
        raise DbAPIError(f'None of the required query elements '
                         f'found: {require_any}')
    if require_all and set(require_all) > component_queries:
        raise DbAPIError(f"Required query elements not found: {require_all}")

    if query_dict and empty_web_query:
        raise DbAPIError(f"Invalid query options: {query_dict.keys()}.")

    return db_query


class FromAgentsApiCall(StatementApiCall):
    def get_db_query(self):
        logger.info("Getting query details.")
        try:
            inp_dict = {f'agent{i}': ag
                        for i, ag in enumerate(self.web_query.poplist('agent'))}
            inp_dict['mesh_ids'] = \
                {m for m in self._pop('mesh_ids', '').split(',') if m}
            inp_dict['paper_ids'] = \
                {i for i in self._pop('paper_ids', '').split(',') if i}
            inp_dict.update(self.web_query)
            db_query = _db_query_from_web_query(
                inp_dict,
                require_any={'HasAgent', 'FromPapers'},
                empty_web_query=True
            )
        except Exception as e:
            logger.exception(e)
            abort(Response(f'Problem forming query: {e}', 400))
            return

        return db_query


class FromHashesApiCall(StatementApiCall):
    def get_db_query(self):
        hashes = request.json.get('hashes')
        if not hashes:
            logger.error("No hashes provided!")
            abort(Response("No hashes given!", 400))
        if len(hashes) > MAX_STATEMENTS:
            logger.error("Too many hashes given!")
            abort(
                Response("Too many hashes given, %d allowed." % MAX_STATEMENTS,
                         400))

        return _db_query_from_web_query({'hashes': hashes})


class FromHashApiCall(StatementApiCall):
    default_ev_lim = 1000

    def get_db_query(self):
        hash = self._pop('hash')
        return _db_query_from_web_query({'hashes': [hash]})


class FromPapersApiCall(StatementApiCall):
    def get_db_query(self):
        # Get the paper id.
        ids = request.json.get('ids')
        if not ids:
            logger.error("No ids provided!")
            abort(Response("No ids in request!", 400))
        mesh_ids = request.json.get('mesh_ids', [])
        return _db_query_from_web_query({'paper_ids': ids,
                                         'mesh_ids': mesh_ids})


class QueryApiCall(ApiCall):
    def get_db_query(self):
        query_json = self._pop('json', '{}')
        return Query.from_json(query_json)


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
    stmt_types = {c.__name__ for c in get_all_descendants(Statement)}
    stmt_types -= {'Influence', 'Event', 'Unresolved'}
    return render_my_template('search.html', 'Search',
                              source_colors=SOURCE_COLORS,
                              stmt_types_json=json.dumps(sorted(list(stmt_types))))


@dep_route('/data-vis/<path:file_path>')
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
@jwt_nontest_optional
def get_statements_query_format():
    # Create a template object from the template file, load once
    return render_my_template('search_statements.html', 'Search',
                              message="Welcome! Try asking a question.",
                              endpoint=request.url_root)


@dep_route('/statements/<path:method>', methods=['GET', 'POST'])
def get_statements(method):
    """Get some statements constrained by query."""

    if method == 'from_agents' and request.method == 'GET':
        call = FromAgentsApiCall()
    elif method == 'from_hashes' and request.method == 'POST':
        call = FromHashesApiCall()
    elif method.startswith('from_hash') and request.method == 'GET':
        call = FromHashApiCall()
        call.web_query['hash'] = method[len('from_hash'):]
    elif method == 'from_papers' and request.method == 'POST':
        call = FromPapersApiCall()
    else:
        return abort(Response('Page not found.', 404))

    return call.run(result_type='statements')


@dep_route('/query/<result_type>', methods=['GET', 'POST'])
def get_statements_by_query_json(result_type):
    return QueryApiCall().run(result_type)


@dep_route('/curation', methods=['GET'])
def describe_curation():
    return redirect('/statements', code=302)


@dep_route('/curation/submit/<hash_val>', methods=['POST'])
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


@dep_route('/curation/list/<stmt_hash>/<src_hash>', methods=['GET'])
def list_curations(stmt_hash, src_hash):
    curations = get_curations(pa_hash=stmt_hash, source_hash=src_hash)
    curation_json = [cur.to_json() for cur in curations]
    return jsonify(curation_json)


def _pop(query, k, default=None, type_cast=None):
    if isinstance(default, bool):
        val = query.pop(k, str(default).lower()) == 'true'
    else:
        val = query.pop(k, default)

    if type_cast is not None and val is not None:
        return type_cast(val)
    return val


@dep_route('/metadata/<level>/from_agents', methods=['GET'])
@jwt_nontest_optional
def get_metadata(level):
    start = datetime.utcnow()
    query = request.args.copy()

    # Figure out authorization.
    has = dict.fromkeys(['elsevier', 'medscan'], False)
    if not TESTING:
        user, roles = resolve_auth(query)
        for role in roles:
            for resource in has.keys():
                has[resource] |= role.permissions.get(resource, False)
    logger.info('Auths: %s' % str(has))

    w_curations = _pop(query, 'with_cur_counts', False)

    kwargs = dict(limit=_pop(query, 'limit', type_cast=int),
                  offset=_pop(query, 'offset', type_cast=int),
                  best_first=_pop(query, 'best_first', True))
    try:
        inp_dict = {f'agent{i}': ag
                    for i, ag in enumerate(query.poplist('agent'))}
        inp_dict['mesh_ids'] = \
            {m for m in query.pop('mesh_ids', '').split(',') if m}
        inp_dict['paper_ids'] = \
            {i for i in query.pop('paper_ids', '').split(',') if i}
        inp_dict.update(query)
        db_query = _db_query_from_web_query(
            inp_dict,
            require_any={'HasAgent', 'FromPapers'},
            empty_web_query=True
        )
    except Exception as e:
        abort(Response(f'Problem forming query: {e}', 400))
        return

    if not has['medscan']:
        db_query -= HasOnlySource('medscan')

    if level == 'hashes':
        res = db_query.get_interactions(**kwargs)
    elif level == 'relations':
        res = db_query.get_relations(with_hashes=w_curations, **kwargs)
    elif level == 'agents':
        res = db_query.get_agents(with_hashes=w_curations, **kwargs)
    else:
        abort(Response(f'Invalid level: {level}'))
        return

    dt = (datetime.utcnow() - start).total_seconds()
    logger.info("Got %s results after %.2f." % (len(res.results), dt))

    ret = res.json()
    res_list = []
    for key, entry in ret.pop('results').items():
        # Filter medscan from source counts.
        if not has['medscan']:
            res.evidence_totals[key] -= entry['source_counts'].pop('medscan', 0)
            entry['total_count'] = res.evidence_totals[key]
            if not entry['source_counts']:
                logger.warning("Censored content present.")
                continue

        # Create english
        if level == 'agents':
            ag_dict = entry['agents']
            if len(ag_dict) == 0:
                eng = ''
            else:
                ag_list = list(ag_dict.values())
                eng = ag_list[0]
                if len(ag_dict) > 1:
                    eng += ' interacts with ' + ag_list[1]
                    if len(ag_dict) > 3:
                        eng += ', ' + ', '.join(ag_list[2:-1])
                    if len(ag_dict) > 2:
                        eng += ', and ' + ag_list[-1]
        else:
            eng = EnglishAssembler([stmt_from_interaction(entry)]).make_model()
        entry['english'] = eng

        res_list.append(entry)

    # Look up curations, if result with_curations was set.
    if w_curations:
        rel_hash_lookup = {}
        if level == 'hashes':
            for rel in res_list:
                rel['cur_count'] = 0
                rel_hash_lookup[rel['hash']] = rel
        else:
            for rel in res_list:
                for h in rel['hashes']:
                    rel['cur_count'] = 0
                    rel_hash_lookup[h] = rel
        curations = get_curations(pa_hash=set(rel_hash_lookup.keys()))
        for cur in curations:
            rel_hash_lookup[cur.pa_hash]['cur_count'] += 1

    # Finish up the query.
    dt = (datetime.utcnow() - start).total_seconds()
    logger.info("Returning with %s results after %.2f seconds."
                % (len(res_list), dt))

    ret['relations'] = res_list
    ret['query_str'] = str(db_query)
    resp = Response(json.dumps(ret), mimetype='application/json')

    dt = (datetime.utcnow() - start).total_seconds()
    logger.info("Result prepared after %.2f seconds." % dt)
    return resp


if __name__ == '__main__':
    app.run()
