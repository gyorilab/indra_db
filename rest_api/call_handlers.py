__all__ = ['ApiCall', 'FromAgentsApiCall', 'FromHashApiCall',
           'FromHashesApiCall', 'FromPapersApiCall', 'FromQueryJsonApiCall',
           'FromAgentJsonApiCall', 'FallbackQueryApiCall']

import sys
import json
import logging
from datetime import datetime
from collections import defaultdict

from flask import request, Response, abort

from indralab_auth_tools.auth import resolve_auth

from indra.ontology.bio import bio_ontology
from indra.statements import stmts_from_json, Complex, make_statement_camel
from indra.assemblers.html.assembler import HtmlAssembler, _format_stmt_text, \
    _format_evidence_text

from indra_db.client.readonly import *
from indra_db.client.principal.curation import *
from indralab_auth_tools.log import note_in_log, is_log_running
from indralab_auth_tools.src.models import UserDatabaseError

from rest_api.config import MAX_STMTS, REDACT_MESSAGE, TITLE, TESTING, \
    jwt_nontest_optional
from rest_api.util import LogTracker, sec_since, get_source, process_agent, \
    process_mesh_term, DbAPIError, iter_free_agents, _make_english_from_meta, \
    get_html_source_info

logger = logging.getLogger('call_handlers')


class ApiCall:
    default_ev_lim = 10

    @jwt_nontest_optional
    def __init__(self, env):
        self.tracker = LogTracker()
        self.start_time = datetime.now()
        self._env = env

        self.web_query = request.args.copy()

        # Get the offset and limit
        self.offs = self._pop('offset', type_cast=int)
        if 'limit' in self.web_query:
            self.limit = min(self._pop('limit', MAX_STMTS, int), MAX_STMTS)
        else:
            self.limit = min(self._pop('max_stmts', MAX_STMTS, int), MAX_STMTS)

        # Sort out the sorting.
        sort_by = self._pop('sort_by', None)
        best_first = self._pop('best_first', True, bool)
        if sort_by is not None:
            self.sort_by = sort_by
        elif best_first:
            self.sort_by = 'ev_count'
        else:
            self.sort_by = None

        # Gather other miscillaneous options
        self.fmt = self._pop('format', 'json')
        self.w_english = self._pop('with_english', False, bool)
        self.w_cur_counts = self._pop('with_cur_counts', False, bool)
        self.strict = self._pop('strict', False, bool)

        # Prime agent recorders.
        self.agent_dict = None
        self.agent_set = None

        # Figure out authorization.
        self.has = dict.fromkeys(['elsevier', 'medscan'], False)
        if not TESTING['status']:
            try:
                self.user, roles = resolve_auth(self.web_query)
            except UserDatabaseError:
                abort(Response("Invalid credentials.", 401))
                return
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

    valid_result_types = ['statements', 'interactions', 'agents', 'hashes']

    def run(self, result_type):

        # Get the db query object.
        logger.info("Running function %s after %s seconds."
                    % (self.__class__.__name__, sec_since(self.start_time)))

        # Actually run the function
        params = dict(offset=self.offs, limit=self.limit,
                      sort_by=self.sort_by)
        logger.info(f"Sending query with params: {params}")
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
            self.special['complexes_covered'] = \
                self._pop('complexes_covered', None)
            res = self.get_db_query().get_agents(
                with_hashes=self.special['with_hashes'] or self.w_cur_counts,
                complexes_covered=self.special['complexes_covered'],
                **params
            )
        elif result_type == 'hashes':
            res = self.get_db_query().get_hashes(**params)
        else:
            raise ValueError(f"Invalid result type: {result_type}")
        logger.info(f"Got results from query after "
                    f"{sec_since(self.start_time)} seconds.")
        self.process_entries(res)
        logger.info(f"Returning for query with params: {params}")
        return self.produce_response(res)

    def _pop(self, key, default=None, type_cast=None):
        if isinstance(default, bool):
            val = self.web_query.pop(key, str(default).lower()).lower() == 'true'
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
                num_agents = (self.db_query.list_component_queries()
                              .count(HasAgent.__name__))
                self.db_query &= HasNumAgents((num_agents,))

            # Note the query in the log, if one is running.
            if is_log_running():
                note_in_log(query=self.db_query.to_json())

        logger.info(f"Constructed query \"{self.db_query}\":\n"
                    f"{json.dumps(self.db_query.to_json(), indent=2)}")
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
                    else:
                        eng = _make_english_from_meta(entry['agents'],
                                                      entry.get('type'))
                    if not eng:
                        logger.warning(f"English not formed for {key}:\n"
                                       f"{entry}")
                    entry['english'] = eng

                # Filter out medscan if user does not have medscan privileges.
                if not self.has['medscan']:
                    if result.result_type == 'statements':
                        result.source_counts[key].pop('medscan', 0)
                    else:
                        result.evidence_counts[key] -= \
                            entry['source_counts'].pop('medscan', 0)
                        entry['total_count'] = result.evidence_counts[key]
                        if not entry['source_counts']:
                            logger.warning("Censored content present.")

                # In most cases we can stop here
                if self.has['elsevier'] and self.fmt != 'json-js' \
                        and not self.w_english:
                    continue

                if result.result_type == 'statements':
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
                elif result.result_type != 'hashes' and self.fmt == 'json-js':
                    # Stringify lists of hashes.
                    if 'hashes' in entry and entry['hashes'] is not None:
                        entry['hashes'] = [str(h) for h in entry['hashes']]
                    elif 'hash' in entry:
                        entry['hash'] = str(entry['hash'])

        if result.result_type == 'statements':
            logger.info(f"Redacted {elsevier_redactions} pieces of elsevier "
                        f"evidence.")

        logger.info(f"Process entries for {self.__class__.__name__} after "
                    f"{sec_since(self.start_time)} seconds.")
        return


class StatementApiCall(ApiCall):
    def __init__(self, env):
        super(StatementApiCall, self).__init__(env)
        self.web_query['mesh_ids'] = \
            {m for m in self._pop('mesh_ids', '').split(',') if m}
        self.web_query['paper_ids'] = \
            {i for i in self._pop('paper_ids', '').split(',') if i}
        self.filter_ev = self._pop('filter_ev', True, bool)
        logger.info(f"Evidence {'will' if self.filter_ev else 'will not'} be "
                    f"filtered.")
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
            if curation['pa_hash'] not in cur_counts:
                cur_counts[curation['pa_hash']] = 0
            cur_counts[curation['pa_hash']] += 1

            # Work these counts into the evidence dict structure.
            for ev_json in result.results[curation['pa_hash']]['evidence']:
                if str(ev_json['source_hash']) == str(curation['source_hash']):
                    ev_json['num_curations'] = \
                        ev_json.get('num_curations', 0) + 1
                    break
        return cur_counts

    def produce_response(self, result):
        if result.result_type == 'statements':
            res_json = result.json()

            # Add derived values to the res_json.
            if self.w_cur_counts:
                res_json['num_curations'] = self.get_curation_counts(result)
            res_json['statement_limit'] = MAX_STMTS
            res_json['statements_returned'] = len(result.results)
            res_json['end_of_statements'] = \
                (len(result.results) < MAX_STMTS)
            res_json['statements_removed'] = 0
            res_json['evidence_returned'] = result.returned_evidence

            # Build the HTML if HTML, else just tweak the JSON.
            stmts_json = result.results
            if self.fmt == 'html':
                title = TITLE
                ev_counts = res_json.pop('evidence_counts')
                beliefs = res_json.pop('belief_scores')
                stmts = stmts_from_json(stmts_json.values())
                db_rest_url = request.url_root[:-1] \
                    + self._env.globals['url_for']('root')[:-1]
                html_assembler = \
                    HtmlAssembler(stmts, summary_metadata=res_json,
                                  ev_counts=ev_counts, beliefs=beliefs,
                                  sort_by=self.sort_by, title=title,
                                  source_counts=result.source_counts,
                                  db_rest_url=db_rest_url)
                idbr_template = \
                    self._env.get_template('idbr_statements_view.html')
                if not TESTING['status']:
                    identity = self.user.identity() if self.user else None
                else:
                    identity = None
                source_info, source_colors = get_html_source_info()
                resp_content = html_assembler.make_model(
                    idbr_template, identity=identity, source_info=source_info,
                    source_colors=source_colors, simple=False
                )
                if self.tracker.get_messages():
                    level_stats = ['%d %ss' % (n, lvl.lower())
                                   for lvl, n
                                   in self.tracker.get_level_stats().items()]
                    msg = ' '.join(level_stats)
                    resp_content = html_assembler.append_warning(msg)
                mimetype = 'text/html'
            else:  # Return JSON for all other values of the format argument
                res_json.update(self.tracker.get_level_stats())
                res_json['statements'] = stmts_json
                resp_content = json.dumps(res_json)
                mimetype = 'application/json'

            resp = Response(resp_content, mimetype=mimetype)
            logger.info("Exiting with %d statements with %d/%d evidence of "
                        "size %f MB after %s seconds."
                        % (res_json['statements_returned'],
                           res_json['evidence_returned'],
                           res_json['total_evidence'],
                           sys.getsizeof(resp.data) / 1e6,
                           sec_since(self.start_time)))
        elif result.result_type != 'hashes':
            # Look up curations, if result with_curations was set.
            if self.w_cur_counts:
                rel_hash_lookup = defaultdict(list)
                if result.result_type == 'interactions':
                    for h, rel in result.results.items():
                        rel['cur_count'] = 0
                        rel_hash_lookup[int(h)].append(rel)
                else:
                    for rel in result.results.values():
                        for h in rel['hashes']:
                            rel['cur_count'] = 0
                            rel_hash_lookup[int(h)].append(rel)
                        if not self.special['with_hashes']:
                            rel['hashes'] = None
                curations = get_curations(pa_hash=set(rel_hash_lookup.keys()))
                for cur in curations:
                    for rel in rel_hash_lookup[cur['pa_hash']]:
                        rel['cur_count'] += 1

            logger.info("Returning with %s results after %.2f seconds."
                        % (len(result.results), sec_since(self.start_time)))

            res_json = result.json()
            res_json['relations'] = list(res_json['results'].values())
            if result.result_type == 'agents' and self.fmt == 'json-js':
                res_json['complexes_covered'] = \
                    [str(h) for h in res_json['complexes_covered']]
            res_json.pop('results')
            res_json['query_str'] = str(self.db_query)
            resp = Response(json.dumps(res_json), mimetype='application/json')

            logger.info("Result prepared after %.2f seconds."
                        % sec_since(self.start_time))
        else:
            return super(StatementApiCall, self).produce_response(result)
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

    def _evidence_query_from_web_query(self, db_query=None):
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
            if db_query is not None:
                db_query &= paper_q
            if self.filter_ev:
                ev_filter &= paper_q.ev_filter()

        # Unpack mesh ids.
        mesh_ids = self._pop('mesh_ids', [])
        if mesh_ids:
            mesh_q = FromMeshIds([process_mesh_term(m) for m in mesh_ids])
            if db_query is not None:
                db_query &= mesh_q
            if self.filter_ev:
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

        component_queries = set(db_query.list_component_queries())
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
        if len(hashes) > MAX_STMTS:
            logger.error("Too many hashes given!")
            return abort(
                Response(f"Too many hashes given, {MAX_STMTS} allowed.",
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


class FromAgentJsonApiCall(StatementApiCall):
    def _build_db_query(self):
        agent_json = request.json.get('agent_json')
        stmt_type = request.json.get('stmt_type')
        hashes = request.json.get('hashes')
        if hashes is not None:
            hashes = [int(h) for h in hashes]
        db_query = FromAgentJson(agent_json, stmt_type, hashes)
        self._evidence_query_from_web_query()
        return db_query


def _check_query(query):
    required_queries = {'HasAgent', 'FromPapers', 'FromMeshIds'}
    if not required_queries & set(query.list_component_queries()):
        abort(Response(f"Query must contain at least one of "
                       f"{required_queries}."), 400)
    if query.full:
        abort(Response("Query would retrieve all statements. "
                       "Please constrain further.", 400))
    return


class FromQueryJsonApiCall(StatementApiCall):
    def __init__(self, env):
        super(FromQueryJsonApiCall, self).__init__(env)
        self.web_query['complexes_covered'] = \
            request.json.get('complexes_covered')

    def _build_db_query(self):
        query_json = request.json['query']
        try:
            q = Query.from_simple_json(query_json)
            if self.filter_ev:
                self.ev_filter = q.ev_filter()
        except (KeyError, ValueError):
            abort(Response("Invalid JSON.", 400))
        _check_query(q)
        return q


class FallbackQueryApiCall(ApiCall):
    def _build_db_query(self):
        query_json = json.loads(self._pop('json', '{}'))
        q = Query.from_json(query_json)
        _check_query(q)
        return q