import sys
import json
import pickle
import logging
from os import path
from collections import defaultdict
from datetime import datetime
from itertools import combinations

import unittest
import requests
from unittest.case import SkipTest

from indra import get_config
from indra.databases import hgnc_client
from indra.statements import stmts_from_json
from indra_db import get_db

from indra_db.client.readonly.query import QueryResult
from indra_db.client import HasAgent, HasType, StatementQueryResult, FromMeshIds

from rest_api.util import get_source
from rest_api.config import MAX_STMTS, REDACT_MESSAGE, TESTING

logger = logging.getLogger('db_api_unit_tests')
try:
    from rest_api.api import app
except Exception as e:
    logger.warning(f"Could not load app: {e}")
    app = None
    TESTING = {'status': False}

HERE = path.dirname(path.abspath(__file__))

TIMEGOAL = 1
TIMELIMIT = 30
SIZELIMIT = 4e7
TESTING['status'] = True


def _check_stmt_agents(resp, agents):
    json_stmts = json.loads(resp.data)['statements']
    stmts = stmts_from_json(json_stmts)
    for stmt in stmts:
        for ag_ix, db_ns, db_id in agents:
            if ag_ix is not None:
                assert stmt.agent_list()[ag_ix].db_refs.get(db_ns) == db_id
            # If the ag_ix is None, we're just checking that the Statement
            # contains the agent we're looking for
            else:
                db_ids = [ag.db_refs.get(db_ns) for ag in stmt.agent_list()]
                assert db_id in db_ids


# Change this flag to choose to test a remote deployment.
TEST_DEPLOYMENT = False


class TestDbApi(unittest.TestCase):
    def setUp(self):
        if TEST_DEPLOYMENT:
            url = get_config('INDRA_DB_REST_URL', failure_ok=False)
            print("URL:", url)
            self.app = WebApp(url)
        else:
            app.testing = True
            self.app = app.test_client()

    @staticmethod
    def __check_time(dt, time_goal=TIMEGOAL):
        print(dt)
        assert dt <= TIMELIMIT, \
            ("Query took %f seconds. Must be less than %f seconds."
             % (dt, TIMELIMIT))
        if dt >= time_goal:
            logger.warning("Query took %f seconds, goal is less than %f seconds."
                           % (dt, time_goal))
        return

    def __time_get_query(self, end_point, query_str):
        return self.__time_query('get', end_point, query_str)

    @staticmethod
    def _get_api_key():
        if TEST_DEPLOYMENT:
            api_key = get_config('INDRA_DB_REST_API_KEY', failure_ok=True)
            if api_key is None:
                raise unittest.SkipTest(
                    "No API KEY available. Cannot test auth.")
            return api_key
        else:
            return 'TESTKEY'

    def _add_auth(self, url):
        api_key = self._get_api_key()
        if '?' not in url:
            url += '?'
        else:
            url += '&'
        url += f'api_key={api_key}'
        return url

    def __time_query(self, method, end_point, query_str=None, url_fmt='/%s?%s',
                     with_auth=False, **data):
        print(end_point)
        start_time = datetime.now()
        if query_str is not None:
            url = url_fmt % (end_point, query_str)
        else:
            url = end_point
        if with_auth:
            url = self._add_auth(url)
        meth_func = getattr(self.app, method)
        if data:
            resp = meth_func(url, data=json.dumps(data),
                             headers={'content-type': 'application/json'})
        else:
            resp = meth_func(url)
        dt = (datetime.now() - start_time).total_seconds()
        print(dt)
        size = int(resp.headers['Content-Length'])
        raw_size = sys.getsizeof(resp.data)
        print("Raw size: {raw:f}/{lim:f}, Compressed size: {comp:f}/{lim:f}."
              .format(raw=raw_size/1e6, lim=SIZELIMIT/1e6, comp=size/1e6))
        return resp, dt, size

    def __check_good_statement_query(self, *args, **kwargs):
        check_stmts = kwargs.pop('check_stmts', True)
        time_goal = max(kwargs.pop('time_goal', TIMEGOAL), TIMEGOAL)
        print("Checking json response.")
        query_str = '&'.join(['%s=%s' % (k, v) for k, v in kwargs.items()]
                             + list(args))
        resp, dt, size = self.__time_get_query('statements/from_agents',
                                               query_str)
        assert resp.status_code == 200, \
            ('Got error code %d: \"%s\".'
             % (resp.status_code, resp.data.decode()))
        resp_dict = json.loads(resp.data)
        assert size <= SIZELIMIT, \
            ("Query took up %f MB. Must be less than %f MB."
             % (size/1e6, SIZELIMIT/1e6))
        self.__check_stmts(resp_dict['statements'].values(),
                           check_stmts=check_stmts)

        self.__check_time(dt, time_goal)

        print("Checking html response.")
        query_str += '&format=html'
        _, dt, size = self.__time_get_query('statements/from_agents',
                                            query_str)
        self.__check_time(dt, time_goal)

        return resp

    def __check_stmts(self, json_stmts, check_support=False, check_stmts=False):
        assert len(json_stmts) is not 0, \
            'Did not get any statements.'
        stmts = stmts_from_json(json_stmts)
        for s in stmts:
            assert s.evidence, "Statement lacks evidence."
            for ev in s.evidence:
                if ev.source_api in {'reach', 'sparser', 'trips'} \
                        and ev.pmid is None:

                    # Check because occasionally there is genuinely no pmid.
                    from indra_db.util import get_db
                    db = get_db('primary')
                    tr = db.select_one(db.TextRef,
                                       db.TextRef.id == ev.text_refs['TRID'])
                    assert tr.pmid is None, \
                        ('Statement from reading missing pmid:\n%s\n%s.'
                         % (s, json.dumps(ev.to_json(), indent=2)))

        # To allow for faster response-times, we currently do not include
        # support links in the response.
        if check_support:
            assert any([s.supports + s.supported_by for s in stmts]),\
                ("Some statements lack support: %s."
                 % str([str(s) for s in stmts if not s.supports+s.supported_by]))
            if check_stmts:
                assert all([not s1.matches(s2)
                            for s1, s2 in combinations(stmts, 2)]),\
                    ("Some statements match: %s."
                     % str([(s1, s2) for s1, s2 in combinations(stmts, 2)
                            if s1.matches(s2)]))
        return

    def test_health_check(self):
        """Test that the health check works."""
        resp = self.app.get('healthcheck')
        assert resp.status_code == 200, resp.status_code
        assert resp.json == {'status': 'testing'}, resp.json

    def test_blank_response(self):
        """Test the response to an empty request."""
        resp, dt, size = self.__time_get_query('statements/from_agents', '')
        assert resp.status_code == 400, \
            ('Got unexpected response with code %d: %s.'
             % (resp.status_code, resp.data.decode()))
        assert dt <= TIMELIMIT, \
            ("Query took %f seconds. Must be less than %f seconds."
             % (dt, TIMELIMIT))
        assert size <= SIZELIMIT, \
            "Query took up %f MB. Must be less than %f MB." % (size/1e6,
                                                               SIZELIMIT/1e6)

    def test_specific_query(self):
        """Test whether we can get a "fully" specified statement."""
        resp = self.__check_good_statement_query(object='MAP2K1',
                                                 subject='MAPK1',
                                                 type='Phosphorylation')
        _check_stmt_agents(resp, agents=[
                (0, 'HGNC', hgnc_client.get_hgnc_id('MAPK1')),
                (1, 'HGNC', hgnc_client.get_hgnc_id('MAP2K1'))])

    def test_object_only_query(self):
        """Test whether we can get an object only statement."""
        resp = self.__check_good_statement_query(object='GLUL',
                                          type='IncreaseAmount')
        _check_stmt_agents(resp, agents=[
                (1, 'HGNC', hgnc_client.get_hgnc_id('GLUL'))])
        return

    def test_query_with_two_agents(self):
        """Test a query were the roles of the agents are not given."""
        resp = self.__check_good_statement_query(agent0='MAP2K1',
                                                 agent1='MAPK1',
                                                 type='Phosphorylation')
        _check_stmt_agents(resp, agents=[
                (None, 'HGNC', hgnc_client.get_hgnc_id('MAPK1')),
                (None, 'HGNC', hgnc_client.get_hgnc_id('MAP2K1'))])
        return

    def test_specific_buggy_query(self):
        resp = self.__check_good_statement_query(subject='PDGF@FPLX',
                                                 object='FOS',
                                                 type='IncreaseAmount')
        _check_stmt_agents(resp, agents=[
            (0, 'FPLX', 'PDGF'),
            (1, 'HGNC', '3796')
        ])
        return

    def test_query_with_other(self):
        """Test that we can get an ActiveForm."""
        resp = self.__check_good_statement_query(agent='MAPK1',
                                                 type='ActiveForm')
        _check_stmt_agents(resp, agents=[
                (0, 'HGNC', hgnc_client.get_hgnc_id('MAPK1'))])
        return

    def test_belief_sort_in_agent_search(self):
        """Test sorting by belief."""
        resp = self.__check_good_statement_query(agent='MAPK1',
                                                 sort_by='belief')
        assert len(resp.json['belief_scores']) \
               == len(resp.json['evidence_counts'])
        beliefs = list(resp.json['belief_scores'].values())
        assert all(b1 >= b2 for b1, b2 in zip(beliefs[:-1], beliefs[1:]))
        ev_counts = list(resp.json['evidence_counts'].values())
        assert not all(c1 >= c2 for c1, c2 in zip(ev_counts[:-1], ev_counts[1:]))

    def test_explicit_ev_count_sort_agent_search(self):
        """Test sorting by ev_count explicitly."""
        resp = self.__check_good_statement_query(agent='MAPK1',
                                                 sort_by='ev_count')
        assert len(resp.json['belief_scores']) \
               == len(resp.json['evidence_counts'])
        beliefs = list(resp.json['belief_scores'].values())
        assert not all(b1 >= b2 for b1, b2 in zip(beliefs[:-1], beliefs[1:]))
        ev_counts = list(resp.json['evidence_counts'].values())
        assert all(c1 >= c2 for c1, c2 in zip(ev_counts[:-1], ev_counts[1:]))

    def test_bad_camel(self):
        """Test that a type can be poorly formatted and resolve correctly."""
        resp = self.__check_good_statement_query(agent='MAPK1',
                                                 type='acTivefOrm')
        _check_stmt_agents(resp, agents=[
                (0, 'HGNC', hgnc_client.get_hgnc_id('MAPK1'))])
        return

    # Note that in these big_query tests do not check the quality of statements,
    # because there are likely to be so many statements that that would take
    # longer than needed, given that the quality is tested in other tests.
    def test_big_query_ATK1(self):
        self.__check_good_statement_query(agent='AKT1', check_stmts=False,
                                          time_goal=10)

    def test_big_query_MAPK1(self):
        self.__check_good_statement_query(agent='MAPK1', check_stmts=False,
                                          time_goal=20)

    def test_big_query_TP53(self):
        self.__check_good_statement_query(agent='TP53', check_stmts=False,
                                          time_goal=20)

    def test_big_query_NFkappaB(self):
        self.__check_good_statement_query(agent='NFkappaB@FPLX',
                                          check_stmts=False, time_goal=20)
        return

    def test_offset(self):
        resp1 = self.__check_good_statement_query(agent='NFkappaB@FPLX',
                                                  check_stmts=False,
                                                  time_goal=20)
        j1 = json.loads(resp1.data)
        hashes1 = set(j1['statements'].keys())
        ev_counts1 = j1['evidence_totals']
        resp2 = self.__check_good_statement_query(agent='NFkappaB@FPLX',
                                                  offset=MAX_STMTS,
                                                  check_stmts=False,
                                                  time_goal=20)
        j2 = json.loads(resp2.data)
        hashes2 = set(j2['statements'].keys())
        assert not hashes2 & hashes1

        ev_counts2 = j2['evidence_totals']
        assert max(ev_counts2.values()) <= min(ev_counts1.values())

        return

    def test_query_with_hgnc_ns(self):
        """Test specifying HGNC as a namespace."""
        resp = self.__check_good_statement_query(subject='6871@HGNC',
                                                 object='MAP2K1',
                                                 type='Phosphorylation')
        _check_stmt_agents(resp, agents=[
                (0, 'HGNC', '6871'),
                (1, 'HGNC', hgnc_client.get_hgnc_id('MAP2K1'))])
        return

    def test_query_with_text_ns(self):
        """Test specifying TEXT as a namespace."""
        resp = self.__check_good_statement_query(subject='ERK@TEXT',
                                                 type='Phosphorylation')
        _check_stmt_agents(resp, agents=[(0, 'TEXT', 'ERK')])
        return

    def test_query_with_hgnc_symbol_ns(self):
        """Test specifying HGNC-SYMBOL as a namespace."""
        resp = self.__check_good_statement_query(subject='MAPK1@HGNC-SYMBOL',
                                                 type='Phosphorylation')
        _check_stmt_agents(resp, agents=[
                (0, 'HGNC', hgnc_client.get_hgnc_id('MAPK1'))])
        return

    def test_query_with_chebi_ns(self):
        """Test specifying CHEBI as a namespace."""
        resp = self.__check_good_statement_query(subject='CHEBI:6801@CHEBI')
        _check_stmt_agents(resp, agents=[(0, 'CHEBI', 'CHEBI:6801')])
        return

    def test_query_with_chebi_ns_vemurafenib(self):
        """Test specifying CHEBI as a namespace."""
        resp = self.__check_good_statement_query(subject='CHEBI:63637@CHEBI')
        _check_stmt_agents(resp, agents=[(0, 'CHEBI', 'CHEBI:63637')])
        return

    def test_query_with_names(self):
        resp = self.__check_good_statement_query(subject='MEK', object='ERK',
                                                 type='Phosphorylation')
        _check_stmt_agents(resp, agents=[(0, 'FPLX', 'MEK'),
                                         (1, 'FPLX', 'ERK')])
        return

    def test_query_with_names_that_were_breaking(self):
        resp = self.__check_good_statement_query(subject='MAP2K1',
                                                 object='ERK')
        return

    def test_famplex_query(self):
        resp, dt, size = self.__time_get_query('statements/from_agents',
                                               ('object=PPP1C@FPLX'
                                                '&subject=CHEBI:44658@CHEBI'
                                                '&type=Inhibition'))
        resp_dict = json.loads(resp.data)
        stmts = stmts_from_json(resp_dict['statements'].values())
        assert len(stmts)
        _check_stmt_agents(resp, agents=[
                (0, 'CHEBI', 'CHEBI:44658'),
                (1, 'FPLX', 'PPP1C')])
        self.__check_time(dt)
        assert size <= SIZELIMIT, size

    def test_complex_query(self):
        resp = self.__check_good_statement_query(agent0='MEK@FPLX',
                                                 agent1='ERK@FPLX',
                                                 type='Complex')
        _check_stmt_agents(resp, agents=[(None, 'FPLX', 'MEK'),
                                         (None, 'FPLX', 'ERK')])
        resp_dict = json.loads(resp.data)
        print(len(resp_dict['statements']))
        for h, sj in resp_dict['statements'].items():
            fplx_set = {mem['db_refs'].get('FPLX') for mem in sj['members']}
            assert {'MEK', 'ERK'}.issubset(fplx_set), \
                ("Statement %s with hash %s does not have both members: %s."
                 % (stmts_from_json([sj])[0], h, fplx_set))

        return

    def test_max(self):
        resp = self.__check_good_statement_query(agent0='MEK@FPLX',
                                                 agent1='ERK@FPLX',
                                                 type='Phosphorylation',
                                                 max_stmts=2)
        resp_dict = json.loads(resp.data)
        assert len(resp_dict['statements']) == 2, len(resp_dict['statements'])

    def test_statements_by_hashes_query(self):
        hashes = [25011516823924690, -29396420431585282, 12592460208021981]
        resp, dt, size = self.__time_query('post', 'statements/from_hashes',
                                           hashes=hashes)
        assert resp.status_code == 200, \
            '%s: %s' % (resp.status_code, resp.data.decode())
        resp_dict = json.loads(resp.data)
        self.__check_stmts(resp_dict['statements'].values())
        self.__check_time(dt)
        return

    def test_statements_by_hashes_large_query(self):
        with open(path.join(HERE, 'sample_hashes.pkl'), 'rb') as f:
            hashes = pickle.load(f)
        hash_sample = hashes[:MAX_STMTS]

        # Run the test.
        resp, dt, size = self.__time_query('post', 'statements/from_hashes',
                                           hashes=hash_sample)
        assert resp.status_code == 200, \
            '%s: %s' % (resp.status_code, resp.data.decode())
        resp_dict = json.loads(resp.data)
        self.__check_stmts(resp_dict['statements'].values())
        self.__check_time(dt, time_goal=20)
        return

    def test_get_statement_by_single_hash_query(self):
        resp, dt, size = self.__time_query('get',
            'statements/from_hash/25011516823924690')
        assert resp.status_code == 200, \
            '%s: %s' % (resp.status_code, resp.data.decode())
        resp_dict = json.loads(resp.data)
        self.__check_stmts(resp_dict['statements'].values())
        assert len(resp_dict['statements']) == 1, len(resp_dict['statements'])
        self.__check_time(dt, time_goal=1)
        return

    def test_get_big_statement_by_single_hash_query(self):
        resp, dt, size = self.__time_query('get',
            'statements/from_hash/-29396420431585282')
        assert resp.status_code == 200, \
            '%s: %s' % (resp.status_code, resp.data.decode())
        resp_dict = json.loads(resp.data)
        self.__check_stmts(resp_dict['statements'].values())
        assert len(resp_dict['statements']) == 1, len(resp_dict['statements'])
        self.__check_time(dt, time_goal=1)
        return

    def __test_basic_paper_query(self, id_val, id_type, min_num_results=1):
        id_list = [{'id': id_val, 'type': id_type}]
        resp, dt, size = self.__time_query('post',
                                           'statements/from_papers',
                                           ids=id_list)
        self.__check_time(dt)
        assert size <= SIZELIMIT, size
        assert resp.status_code == 200, str(resp)
        json_dict = json.loads(resp.data)['statements']
        assert len(json_dict) >= min_num_results, (min_num_results,
                                                   len(json_dict))
        return json_dict

    def test_pmid_paper_query(self):
        pmid = '27014235'
        self.__test_basic_paper_query(pmid, 'pmid')

    def test_pmcid_paper_query(self):
        json_dict = self.__test_basic_paper_query('PMC5770457', 'pmcid')
        assert 40 < len(json_dict) < 60, \
            "Wrong number of results: %d." % len(json_dict)

    def test_trid_paper_query(self):
        self.__test_basic_paper_query('19649148', 'trid')

    def __test_redaction(self, method, endpoint, base_qstr, **data):
        resp, dt, size = self.__time_query(method, endpoint, base_qstr, **data)
        assert resp.status_code == 200, \
            '%s: %s' % (resp.status_code, resp.data.decode())
        resp_dict = json.loads(resp.data)
        stmt_dict_redact = resp_dict['statements']
        elsevier_found = 0
        elsevier_long_found = 0
        for s in stmt_dict_redact.values():
            for ev in s['evidence']:
                if get_source(ev) == 'elsevier':
                    elsevier_found += 1
                    if len(ev['text']) > 200:
                        elsevier_long_found += 1
                        assert ev['text'].endswith(REDACT_MESSAGE), \
                            'Found unredacted Elsevier text: %s.' % ev['text']
                else:
                    if 'text' in ev.keys():
                        assert not ev['text'].startswith('[Redacted'), \
                            'Found redacted non-elsevier text.'
        if elsevier_found == 0:
            raise SkipTest("No Elsevier content occurred.")
        if elsevier_long_found == 0:
            raise SkipTest("No redactable (>200 char) Elsevier content "
                           "occurred.")
        resp, dt, size = self.__time_query(method, endpoint, base_qstr,
                                           with_auth=True, **data)
        resp_dict = json.loads(resp.data)
        stmt_dict_intact = resp_dict['statements']
        assert stmt_dict_intact.keys() == stmt_dict_redact.keys(), \
            "Response content changed: different statements without redaction."
        elsevier_found = 0
        for s in stmt_dict_intact.values():
            for ev in s['evidence']:
                if get_source(ev) == 'elsevier':
                    elsevier_found += 1
                if 'text' in ev.keys() and len(ev['text']) > 200:
                    assert not ev['text'].endswith(REDACT_MESSAGE), \
                        'Found redacted text despite api key.'
        assert elsevier_found > 0, "Elsevier content references went missing."
        return

    def test_redaction_on_agents_query(self):
        return self.__test_redaction('get', 'statements/from_agents',
                                     'agent1=STAT5@FPLX&agent2=CRKL')

    def test_redaction_on_paper_query(self):
        ids = [{'id': '20914619', 'type': 'tcid'}]
        return self.__test_redaction('post', 'statements/from_papers', None,
                                     url_fmt='%s?%s', ids=ids)

    def test_redaction_on_hash_query(self):
        sample_hashes = [24340898017079193, -18002830651869995,
                         11108256246535015, 27972673344272623,
                         29058537924450063, -13534950859792956]
        return self.__test_redaction('post', 'statements/from_hashes', None,
                                     url_fmt='%s?%s', hashes=sample_hashes)

    def test_curation_submission(self):
        # This can only test the surface layer endpoint.
        self.__time_query('post', 'curation/submit/12345?test', tag='test',
                          curator='tester', text='This is text.')

    def test_interaction_query(self):
        self.__time_query('get', 'metadata/relations/from_agents',
                          'agent0=mek%40AUTO&limit=50&with_cur_counts=true')

    def test_simple_json_query(self):
        query = (HasAgent('MEK', namespace='NAME')
                 & HasAgent('ERK', namespace='NAME')
                 & HasType(['Phosphorylation']))
        limit = 50
        ev_limit = 5
        qr1 = query.get_statements(limit=50, ev_limit=5)
        qj = query.to_json()
        qjs = json.dumps(qj)
        resp, dt, size = \
            self.__time_query('get', 'query/statements',
                              f'json={qjs}&limit={limit}&ev_limit={ev_limit}')
        qr2 = QueryResult.from_json(json.loads(resp.data))
        assert isinstance(qr2, StatementQueryResult)

        # Check that we got the same statements.
        assert qr1.results.keys() == qr2.results.keys()

        # Make sure elsevier and medscan were filtered out
        for s1, s2 in zip(qr1.statements(), qr2.statements()):
            if any(ev.source_api == 'medscan' for ev in s1.evidence):
                assert len(s1.evidence) > len(s2.evidence),\
                    'Medscan result not filtered out.'
                continue  # TODO: Figure out how to test elsevier in this case.
            else:
                assert len(s1.evidence) == len(s2.evidence), \
                    "Evidence counts don't match."

            for ev1, ev2 in zip(s1.evidence, s2.evidence):
                if ev1.text_refs.get('SOURCE') == 'elsevier':
                    if len(ev1.text) > 200:
                        assert ev2.text.endswith(REDACT_MESSAGE),\
                            "Elsevier text not truncated."
                        assert len(ev2.text) == (200 + len(REDACT_MESSAGE)), \
                            "Elsevier text not truncated."
                    else:
                        assert len(ev1.text) == len(ev2.text),\
                            "Evidence text lengths don't match."

    def test_drill_down(self):
        def drill_down(relation, result_type):
            query_strs = ['with_cur_counts=true']
            query_data = {'agent_json': relation['agents'],
                          'hashes': relation['hashes']}
            if result_type == 'statements':
                query_data['type'] = relation['type']
                query_strs.extend(['ev_limit=10', 'format=json-js',
                                   'filter_ev=true', 'with_english=true'])
                endpoint = 'statements/from_agent_json'
            else:
                endpoint = 'expand'
            resp, dt, size = self.__time_query('post', endpoint,
                                               '&'.join(query_strs),
                                               **query_data)
            assert dt < 10
            res = json.loads(resp.data)
            if result_type == 'relations':
                rels = res['relations']
            elif result_type == 'statements':
                rels = res['statements'].values()
            assert all('english' in rel for rel in rels)
            if result_type == 'relations':
                assert all('cur_count' in rel for rel in rels)
                assert all(rel['hashes'] is not None for rel in rels)
                assert all(rel['agents'] == relation['agents']
                           for rel in res['relations'])
                num_complexes = sum(rel['type'] == 'Complex'
                                    for rel in res['relations'])
                assert num_complexes <= 1
            elif result_type == 'statements':
                assert 'num_curations' in res
                assert all('evidence' in rel for rel in rels)
            sum_src_cnt = defaultdict(lambda: 0)
            for rel in rels:
                if result_type == 'relations':
                    this_src_counts = rel['source_counts']
                else:
                    this_src_counts = res['source_counts'][rel['matches_hash']]
                for src, cnt in this_src_counts.items():
                    sum_src_cnt[src] += cnt
            sum_src_cnt = dict(sum_src_cnt)
            parent_src_cnt = relation['source_counts']
            parent_set = {s for s, c in parent_src_cnt.items() if c > 0}
            sum_set = {s for s, c in sum_src_cnt.items() if c > 0}
            assert parent_set == sum_set, \
                f'Set mismatch: {parent_set} vs. {sum_set}'
            all_match = all(sum_src_cnt[src] == parent_src_cnt[src]
                            for src in parent_set)
            assert all_match, \
                '\n'.join((f"{s}: parent={parent_src_cnt[s]}, "
                           f"child sum={sum_src_cnt[s]}")
                          if parent_src_cnt[s] != sum_src_cnt[s] else f'{s}: ok'
                          for s in parent_set)
            if result_type == 'relations':
                for rel in rels:
                    drill_down(rel, 'statements')

        url_base = "agents/from_query_json"
        query = HasAgent('MEK')
        query_str = ("limit=50&with_cur_counts=true&with_english=true"
                     "&with_hashes=true")
        complexes_covered = []
        resp, dt, size = self.__time_query('post', url_base, query_str,
                                           query=query.to_json(),
                                           complexes_covered=complexes_covered)
        res = json.loads(resp.data)
        complexes_covered = res['complexes_covered']
        assert len(res['relations']) == 50
        assert isinstance(res['relations'], list)
        assert dt < 15
        assert all('english' in rel for rel in res['relations'])
        assert all('cur_count' in rel for rel in res['relations'])
        assert all(rel['hashes'] is not None for rel in res['relations'])

        assert res['relations'][0]['id'] == 'Agents(None, MEK)'
        drill_down(res['relations'][0], 'relations')
        assert res['relations'][1]['id'] == 'Agents(MEK, ERK)'
        assert res['relations'][1]['agents'] == {'0': 'MEK', '1': 'ERK'}
        drill_down(res['relations'][1], 'relations')
        print(resp)

    def __simple_time_test(self, *args, **kwargs):
        resp, dt, size = \
            self.__time_query(*args, **kwargs)
        assert resp.status_code == 200, f"Query failed: {resp.data.decode()}"
        assert dt < 30, "Query would have timed out."
        if dt > 15:
            logger.warning(f"Query took a long time: {dt} seconds.")

    def test_IL6_html_with_creds(self):
        """Test the timing of a search for IL-6 with auth, "signed in"."""
        self.__simple_time_test('get', 'statements/from_agents',
                                'agent=IL-6@TEXT&format=html',
                                with_auth=True)

    def test_IL6_html_no_creds(self):
        """Test the timing of a search for IL-6 without auth, "signed out"."""
        self.__simple_time_test('get', 'statements/from_agents',
                                'agent=IL-6@TEXT&format=html',
                                with_auth=False)

    def test_IL6_agents_with_creds(self):
        """Test the timing of a query for agents with text=IL-6, "signed in"."""
        query = HasAgent('IL-6', 'TEXT')
        self.__simple_time_test('post', 'agents/from_query_json',
                                'format=json-js&with_english=true',
                                query=query.to_json(),
                                with_auth=True)

    def test_IL6_agents_no_creds(self):
        """Test timing of a query for agents with text=IL-6, "signed out"."""
        query = HasAgent('IL-6', 'TEXT')
        self.__simple_time_test('post', 'agents/from_query_json',
                                'format=json-js&with_english=true',
                                query=query.to_json(),
                                with_auth=False)

    def test_ROS_html_with_creds(self):
        """Test the timing of a search for ROS with auth, "signed in"."""
        self.__simple_time_test('get', 'statements/from_agents',
                                'agent=ROS@TEXT&format=html',
                                with_auth=True)

    def test_ROS_html_no_creds(self):
        """Test the timing of a search for ROS without auth, "signed out"."""
        self.__simple_time_test('get', 'statements/from_agents',
                                'agent=ROS@TEXT&format=html',
                                with_auth=False)

    def test_ROS_agents_with_creds(self):
        """Test the timing of a query for agents with text=ROS, "signed in"."""
        query = HasAgent('ROS', 'TEXT')
        self.__simple_time_test('post', 'agents/from_query_json',
                                'format=json-js&with_english=true',
                                query=query.to_json(),
                                with_auth=True)

    def test_ROS_agents_no_creds(self):
        """Test timing of a query for agents with text=ROS, "signed out"."""
        query = HasAgent('ROS', 'TEXT')
        self.__simple_time_test('post', 'agents/from_query_json',
                                'format=json-js&with_english=true',
                                query=query.to_json(),
                                with_auth=False)

    def test_mesh_concept_ev_limit(self):
        """Test a specific bug in which evidence was duplicated.

        When querying for mesh concepts, with an evidence limit, the evidence
        was repeated numerous times.
        """
        db = get_db('primary')
        q = HasAgent('ACE2') & FromMeshIds(['C000657245'])
        resp, dt, size = self.__time_query('post', 'statements/from_query_json',
                                           'limit=50&ev_limit=6',
                                           query=q.to_json(), with_auth=True)
        assert resp.status_code == 200, f"Query failed: {resp.data.decode()}"
        assert dt < 30, "Query would have timed out."
        if dt > 15:
            logger.warning(f"Query took a long time: {dt} seconds.")

        resp_json = json.loads(resp.data)
        pmids = set()
        for h, data in resp_json['results'].items():
            ev_list = data['evidence']
            assert len(ev_list) <= 6, "Evidence limit exceeded."
            ev_tuples = {(ev.get('text'), ev.get('source_hash'),
                          ev.get('source_api'), str(ev.get('text_refs')))
                         for ev in ev_list}
            assert len(ev_tuples) == len(ev_list), "Evidence is not unique."
            for ev in ev_list:
                found_pmid = False
                if 'pmid' in ev:
                    pmids.add(ev['pmid'])
                    found_pmid = True

                if 'text_refs' in ev:
                    tr_dict = ev['text_refs']
                    if 'TRID' in tr_dict:
                        tr = db.select_one(db.TextRef,
                                           db.TextRef.id == tr_dict['TRID'])
                        pmids.add(tr.pmid)
                        found_pmid = True
                    if 'PMID' in tr_dict:
                        pmids.add(tr_dict['PMID'])
                        found_pmid = True
                    if 'DOI' in tr_dict:
                        tr_list = db.select_all(
                            db.TextRef,
                            db.TextRef.doi_in([tr_dict['DOI']])
                        )
                        pmids |= {tr.pmid for tr in tr_list if tr.pmid}
                        found_pmid = True

                assert found_pmid,\
                    "How could this have been mapped to mesh?"
        pmids = {int(pmid) for pmid in pmids if pmid is not None}

        mesh_pmids = {n for n, in db.select_all(
            db.MeshRefAnnotations.pmid_num,
            db.MeshRefAnnotations.pmid_num.in_(pmids),
            db.MeshRefAnnotations.mesh_num == 657245,
            db.MeshRefAnnotations.is_concept.is_(True)
        )}
        mesh_pmids |= {n for n, in db.select_all(
            db.MtiRefAnnotationsTest.pmid_num,
            db.MtiRefAnnotationsTest.pmid_num.in_(pmids),
            db.MtiRefAnnotationsTest.mesh_num == 657245,
            db.MtiRefAnnotationsTest.is_concept.is_(True)
        )}

        assert pmids == mesh_pmids, "Not all pmids mapped ot mesh term."


class WebApp:
    """Mock the behavior of the "app" but on the real service."""
    def __init__(self, url):
        self.base_url = url
        if self.base_url.endswith('/'):
            self.base_url = self.base_url[:-1]

    def __process_url(self, url):
        if not url.startswith('/'):
            url = '/' + url
        return self.base_url + url

    def get(self, url):
        full_url = self.__process_url(url)
        raw_resp = requests.get(full_url)
        return WebResponse(raw_resp)

    def post(self, url, data, headers):
        full_url = self.__process_url(url)
        raw_resp = requests.post(full_url, data=data, headers=headers)
        return WebResponse(raw_resp)


class WebResponse:
    """Imitate the response from the "app", but from real request responses."""
    def __init__(self, resp):
        self._resp = resp
        self.data = resp.content

    def __getattribute__(self, item):
        """When in doubt, try to just get the item from the actual resp.

        This should work much of the time because the results from the "app" are
        intended to imitate and actual response.
        """
        try:
            return super(WebResponse, self).__getattribute__(item)
        except AttributeError:
            return getattr(self._resp, item)