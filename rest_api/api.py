import sys
import json
import logging
from os import path, environ
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
from indra.ontology.bio import bio_ontology
from indra.statements import make_statement_camel, get_all_descendants, \
    Statement, Complex
from indra_db.client.readonly.query import HasAgent, HasType, HasNumAgents, \
    HasOnlySource, HasHash, Query, FromPapers, FromMeshIds, EvidenceFilter, \
    EmptyQuery

from indralab_auth_tools.auth import auth, resolve_auth, config_auth

from indra_db.exceptions import BadHashError
from indra_db.client import submit_curation, stmt_from_interaction,\
    get_curations
from .util import process_agent, DbAPIError, LogTracker, sec_since, get_source, \
    get_s3_client, gilda_ground, process_mesh_term

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


MAX_STATEMENTS = int(0.5e3)
REDACT_MESSAGE = '[MISSING/INVALID CREDENTIALS: limited to 200 char for Elsevier]'


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


def iter_free_agents(query_dict):
    agent_keys = {k for k in query_dict.keys() if k.startswith('agent')}
    for k in agent_keys:
        entry = query_dict.pop(k)
        if isinstance(entry, list):
            for agent in entry:
                yield agent
        else:
            yield entry


def dep_route(url, **kwargs):
    if DEPLOYMENT is not None:
        url = f'/{DEPLOYMENT}{url}'
    flask_dec = app.route(url, **kwargs)
    return flask_dec


# ==========================
# API call class definitions
# ==========================

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
        self.strict = self._pop('strict', False, bool)
        self.agent_dict = None
        self.agent_set = None

        # Figure out authorization.
        self.has = dict.fromkeys(['elsevier', 'medscan'], False)
        if not TESTING:
            self.user, roles = resolve_auth(self.web_query)
            for role in roles:
                for resource in self.has.keys():
                    self.has[resource] |= role.permissions.get(resource, False)
            logger.info('Auths: %s' % str(self.has))
        else:
            api_key = self.web_query.pop('api_key', None)
            if api_key is None:  # any key will do for testing.
                self.has['elsevier'] = False
                self.has['medscan'] = False
            else:
                self.has['elsevier'] = True
                self.has['medscan'] = True

        self.db_query = None
        self.ev_filter = None
        self.special = {}
        return

    def run(self, result_type):

        # Get the db query object.
        logger.info("Running function %s after %s seconds."
                    % (self.__class__.__name__, sec_since(self.start_time)))

        # Actually run the function
        params = dict(offset=self.offs, limit=self.max_stmts,
                      best_first=self.best_first)
        if result_type == 'statements':
            self.special['ev_limit'] = \
                self._pop('ev_limit', self.default_ev_lim, int)
            res = self.get_db_query().get_statements(
                ev_limit=self.special['ev_limit'],
                evidence_filter=self.ev_filter,
                **params
            )
        elif result_type == 'interactions':
            res = self.get_db_query().get_interactions(**params)
        elif result_type == 'relations':
            self.special['with_hashes'] = self._pop('with_hashes', False, bool)
            res = self.get_db_query().get_relations(
                with_hashes=self.special['with_hashes'] or self.w_cur_counts,
                **params
            )
        elif result_type == 'agents':
            self.special['with_hashes'] = self._pop('with_hashes', False, bool)
            res = self.get_db_query().get_agents(
                with_hashes=self.special['with_hashes'] or self.w_cur_counts,
                **params
            )
        elif result_type == 'hashes':
            res = self.get_db_query().get_hashes(**params)
        else:
            raise ValueError(f"Invalid result type: {result_type}")
        self.process_entries(res)
        return self.produce_response(res)

    def _pop(self, key, default=None, type_cast=None):
        if isinstance(default, bool):
            val = self.web_query.pop(key, str(default).lower()) == 'true'
        else:
            val = self.web_query.pop(key, default)

        if type_cast is not None and val is not None:
            return type_cast(val)
        return val

    def get_db_query(self):
        if self.db_query is None:
            self.db_query = self._build_db_query()

            if not self.has['medscan']:
                minus_q = ~HasOnlySource('medscan')
                self.db_query &= minus_q
                if not self.ev_filter:
                    self.ev_filter = minus_q.ev_filter()
                else:
                    self.ev_filter &= minus_q.ev_filter()

            if self.strict:
                num_agents = (self.db_query.get_component_queries()
                              .count(HasAgent.__name__))
                self.db_query &= HasNumAgents((num_agents,))

        return self.db_query

    def _build_db_query(self):
        raise NotImplementedError()

    def produce_response(self, result):
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

    def process_entries(self, result):
        if result.result_type == 'hashes':
            # There is really nothing to do for hashes.
            return

        elsevier_redactions = 0
        if not all(self.has.values()) or self.fmt == 'json-js' \
                or self.w_english:
            for key, entry in result.results.copy().items():
                # Build english reps of each result (unless their just hashes)
                if self.w_english and result.result_type != 'hashes':
                    stmt = None
                    # Fix the agent order
                    if self.strict:
                        if result.result_type == 'statements':
                            stmt = stmts_from_json([entry])[0]
                            if type(stmt) == Complex:
                                id_lookup = {v: int(k)
                                             for k, v in self.agent_dict.items()}
                                stmt.members.sort(
                                    key=lambda ag: id_lookup.get(ag.name, 10)
                                )
                            agent_set = {ag.name
                                         for ag in stmt.agent_list()
                                         if ag is not None}
                        else:
                            agent_set = set(entry['agents'].values())
                            if result.result_type == 'relations' \
                                    and entry['type'] == 'Complex':
                                entry['agents'] = self.agent_dict
                        if agent_set < self.agent_set:
                            result.results.pop(key, None)
                            continue

                    # Construct the english.
                    if result.result_type == 'statements':
                        if stmt is None:
                            stmt = stmts_from_json([entry])[0]
                        eng = _format_stmt_text(stmt)
                        entry['evidence'] = _format_evidence_text(stmt)
                    elif result.result_type == 'agents':
                        ag_dict = entry['agents']
                        if len(ag_dict) == 0:
                            eng = ''
                        else:
                            ag_list = list(ag_dict.values())
                            eng = f'<b>{ag_list[0]}</b>'
                            if len(ag_dict) > 1:
                                eng += ' affects ' + f'<b>{ag_list[1]}</b>'
                                if len(ag_dict) > 3:
                                    eng += ', ' \
                                           + ', '.join(f'<b>{ag}</b>'
                                                       for ag in ag_list[2:-1])
                                if len(ag_dict) > 2:
                                    eng += ', and ' + f'<b>{ag_list[-1]}</b>'
                            else:
                                eng += ' is modified'
                    else:
                        eng = _format_stmt_text(stmt_from_interaction(entry))
                    if not eng:
                        logger.warning(f"English not formed for {key}:\n"
                                       f"{entry}")
                    entry['english'] = eng

                # Filter out medscan if user does not have medscan privileges.
                if not self.has['medscan']:
                    if result.result_type == 'statements':
                        result.source_counts[key].pop('medscan', 0)
                    else:
                        result.evidence_totals[key] -= \
                            entry['source_counts'].pop('medscan', 0)
                        entry['total_count'] = result.evidence_totals[key]
                        if not entry['source_counts']:
                            logger.warning("Censored content present.")
                            continue

                # In most cases we can stop here
                if self.has['elsevier'] and self.fmt != 'json-js' \
                        and not self.w_english \
                        or result.result_type != 'statements':
                    continue

                # If there is evidence, loop through it if necessary.
                for ev_json in entry['evidence'][:]:
                    if self.fmt == 'json-js':
                        ev_json['source_hash'] = str(ev_json['source_hash'])

                    # Check for elsevier and redact if necessary
                    if not self.has['elsevier'] and \
                            get_source(ev_json) == 'elsevier':
                        text = ev_json['text']
                        if len(text) > 200:
                            ev_json['text'] = text[:200] + REDACT_MESSAGE
                            elsevier_redactions += 1
        if result.result_type == 'statements':
            logger.info(f"Redacted {elsevier_redactions} pieces of elsevier "
                        f"evidence.")

        logger.info("Finished  for %s after %s seconds."
                    % (self.__class__.__name__, sec_since(self.start_time)))
        return


class StatementApiCall(ApiCall):
    def __init__(self):
        super(StatementApiCall, self).__init__()
        self.web_query['mesh_ids'] = \
            {m for m in self._pop('mesh_ids', '').split(',') if m}
        self.web_query['paper_ids'] = \
            {i for i in self._pop('paper_ids', '').split(',') if i}
        self.agent_dict = {}
        self.agent_set = set()
        return

    def _build_db_query(self):
        raise NotImplementedError()

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

    def produce_response(self, result):
        res_json = result.json()

        # Add derived values to the res_json.
        if self.w_cur_counts:
            res_json['num_curations'] = self.get_curation_counts(result)
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
            html_assembler = HtmlAssembler(stmts, summary_metadata=res_json,
                                           ev_counts=ev_totals, title=title,
                                           source_counts=result.source_counts,
                                           db_rest_url=request.url_root[:-1])
            idbr_template = env.get_template('idbr_statements_view.html')
            if not TESTING:
                identity = self.user.identity() if self.user else None
            else:
                identity = None
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

    def _require_agent(self, ag, ns, num=None):
        if not self.strict:
            return

        if ns in ['NAME', 'FPLX', ]:
            name = ag
        elif ns != 'TEXT':
            name = bio_ontology.get_name(ns, ag)
        else:
            # If the namespace is TEXT, what do we do?
            return

        self.agent_set.add(name)
        if num is not None:
            self.agent_dict[num] = name
        return

    def _agent_query_from_web_query(self, db_query):
        # Get the agents without specified locations (subject or object).
        for raw_ag in iter_free_agents(self.web_query):
            ag, ns = process_agent(raw_ag)
            db_query &= HasAgent(ag, namespace=ns)
            self._require_agent(ag, ns)

        # Get the agents with specified roles.
        for ag_num, role in enumerate(['subject', 'object']):
            raw_ag = self._pop(role, None)
            if raw_ag is None:
                continue
            if isinstance(raw_ag, list):
                assert len(raw_ag) == 1, f'Malformed agent for {role}: {raw_ag}'
                raw_ag = raw_ag[0]
            ag, ns = process_agent(raw_ag)
            db_query &= HasAgent(ag, namespace=ns, role=role.upper())
            self._require_agent(ag, ns, ag_num)

        # Get agents with specific agent numbers.
        for key in self.web_query.copy().keys():
            if not key.startswith('ag_num'):
                continue
            ag_num_str = key[len('ag_num_'):]
            if not ag_num_str.isdigit():
                return abort(Response(f"Invalid agent number: {ag_num_str}",
                                      400))

            raw_ag = self._pop(key)
            ag_num = int(ag_num_str)
            ag, ns = process_agent(raw_ag)
            db_query &= HasAgent(ag, namespace=ns, agent_num=ag_num)
            self._require_agent(ag, ns, ag_num)

        return db_query

    def _type_query_from_web_query(self, db_query):
        # Get the raw name of the statement type (we fix some variations).
        act_raw = self._pop('type', None)
        if act_raw is not None:
            if isinstance(act_raw, list):
                assert len(act_raw) == 1, \
                    f"Got multiple entries for statement type: {act_raw}."
                act_raw = act_raw[0]
            act = make_statement_camel(act_raw)
            db_query &= HasType([act])
        return db_query

    def _hashes_query_from_web_query(self, db_query):
        # Unpack hashes, if present.
        hashes = self._pop('hashes', None)
        if hashes:
            db_query &= HasHash(hashes)
        return db_query

    def _evidence_query_from_web_query(self, db_query):
        filter_ev = self._pop('filter_ev', False, bool)
        ev_filter = EvidenceFilter()

        # Unpack paper ids, if present:
        id_tpls = set()
        for paper_id in self._pop('paper_ids', []):
            if isinstance(paper_id, dict):
                val = paper_id['id']
                typ = paper_id['type']
            elif isinstance(paper_id, str):
                val, typ = paper_id.split('@')
            else:
                raise ValueError(f"Invalid paper_id type: {type(paper_id)}")

            # Turn tcids and trids into integers.
            id_val = int(val) if typ in ['tcid', 'trid'] else val
            id_tpls.add((typ, id_val))

        if id_tpls:
            paper_q = FromPapers(id_tpls)
            db_query &= paper_q
            if filter_ev:
                ev_filter &= paper_q.ev_filter()

        # Unpack mesh ids.
        mesh_ids = self._pop('mesh_ids', [])
        if mesh_ids:
            mesh_q = FromMeshIds([process_mesh_term(m) for m in mesh_ids])
            db_query &= mesh_q
            if filter_ev:
                ev_filter &= mesh_q.ev_filter()

        # Assign ev filter to class scope
        if ev_filter.filters:
            self.ev_filter = ev_filter
        return db_query

    def _check_db_query(self, db_query, require_any, require_all,
                        empty_web_query):
        # Check for health of the resulting query, and some other things.
        if isinstance(db_query, EmptyQuery):
            raise DbAPIError(f"No arguments from web query {self.web_query} "
                             f"mapped to db query.")
        assert isinstance(db_query, Query), "Somehow db_query is not Query."

        component_queries = set(db_query.get_component_queries())
        if require_any and not set(require_any) & component_queries:
            raise DbAPIError(f'None of the required query elements '
                             f'found: {require_any}')
        if require_all and set(require_all) > component_queries:
            raise DbAPIError(f"Required query elements not found: "
                             f"{require_all}")

        if self.web_query and empty_web_query:
            raise DbAPIError(f"Invalid query options: "
                             f"{list(self.web_query.keys())}.")
        return

    def _db_query_from_web_query(self, require_any=None, require_all=None,
                                 empty_web_query=False):
        db_query = EmptyQuery()

        logger.info(f"Making DB query from:\n{self.web_query}")

        db_query = self._agent_query_from_web_query(db_query)
        db_query = self._type_query_from_web_query(db_query)
        db_query = self._hashes_query_from_web_query(db_query)
        db_query = self._evidence_query_from_web_query(db_query)
        self._check_db_query(db_query, require_any, require_all,
                             empty_web_query)

        return db_query


class FromAgentsApiCall(StatementApiCall):
    def _build_db_query(self):
        logger.info("Getting query details.")
        try:
            self.web_query.update(
                {f'agent{i}': ag
                 for i, ag in enumerate(self.web_query.poplist('agent'))}
            )
            db_query = self._db_query_from_web_query(
                require_any={'HasAgent', 'FromPapers', 'FromMeshIds'},
                empty_web_query=True
            )
        except Exception as e:
            logger.exception(e)
            return abort(Response(f'Problem forming query: {e}', 400))

        return db_query


class FromHashesApiCall(StatementApiCall):
    def _build_db_query(self):
        hashes = request.json.get('hashes')
        if not hashes:
            logger.error("No hashes provided!")
            return abort(Response("No hashes given!", 400))
        if len(hashes) > MAX_STATEMENTS:
            logger.error("Too many hashes given!")
            return abort(
                Response(f"Too many hashes given, {MAX_STATEMENTS} allowed.",
                         400)
            )

        self.web_query['hashes'] = hashes
        return self._db_query_from_web_query()


class FromHashApiCall(StatementApiCall):
    default_ev_lim = 1000

    def _build_db_query(self):
        self.web_query['hashes'] = [self._pop('hash')]
        return self._db_query_from_web_query()


class FromPapersApiCall(StatementApiCall):
    def _build_db_query(self):
        # Get the paper id.
        ids = request.json.get('ids')
        if not ids:
            logger.error("No ids provided!")
            return abort(Response("No ids in request!", 400))
        mesh_ids = request.json.get('mesh_ids', [])
        self.web_query['paper_ids'] = ids
        self.web_query['mesh_ids'] = mesh_ids
        return self._db_query_from_web_query()


class MetadataApiCall(FromAgentsApiCall):
    def produce_response(self, result):
        # Look up curations, if result with_curations was set.
        if self.w_cur_counts:
            rel_hash_lookup = {}
            if result.result_type == 'hashes':
                for rel in result.results.values():
                    rel['cur_count'] = 0
                    rel_hash_lookup[rel['hash']] = rel
            else:
                for rel in result.results.values():
                    for h in rel['hashes']:
                        rel['cur_count'] = 0
                        rel_hash_lookup[h] = rel
                    if not self.special['with_hashes']:
                        rel['hashes'] = None
            curations = get_curations(pa_hash=set(rel_hash_lookup.keys()))
            for cur in curations:
                rel_hash_lookup[cur.pa_hash]['cur_count'] += 1

        logger.info("Returning with %s results after %.2f seconds."
                    % (len(result.results), sec_since(self.start_time)))

        res_json = result.json()
        res_json['relations'] = list(res_json['results'].values())
        res_json['query_str'] = str(self.db_query)
        resp = Response(json.dumps(res_json), mimetype='application/json')

        logger.info("Result prepared after %.2f seconds."
                    % sec_since(self.start_time))
        return resp


class QueryApiCall(ApiCall):
    def _build_db_query(self):
        query_json = json.loads(self._pop('json', '{}'))
        q = Query.from_json(query_json)
        required_queries = {'HasAgent', 'FromPapers', 'FromMeshIds'}
        if not required_queries & set(q.get_component_queries()):
            abort(Response(f"Query must contain at least one of "
                           f"{required_queries}."), 400)
        if q.full:
            abort(Response("Query would retrieve all statements. "
                           "Please constrain further.", 400))
        return q


# ==========================
# Here begins the API proper
# ==========================


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
    elif method.startswith('from_hash/') and request.method == 'GET':
        call = FromHashApiCall()
        call.web_query['hash'] = method[len('from_hash/'):]
    elif method == 'from_papers' and request.method == 'POST':
        call = FromPapersApiCall()
    else:
        return abort(Response('Page not found.', 404))

    return call.run(result_type='statements')


@dep_route('/query/<result_type>', methods=['GET', 'POST'])
def get_statements_by_query_json(result_type):
    return QueryApiCall().run(result_type)


@dep_route('/metadata/<result_type>/from_agents', methods=['GET'])
def get_metadata(result_type):
    return MetadataApiCall().run(result_type)


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


if __name__ == '__main__':
    app.run()
