from indra.util import zip_string

import indra_db.util as dbu
from indra_db.util.content_scripts import (get_stmts_with_agent_text_like,
                                           get_stmts_with_agent_text_in,
                                           get_text_content_from_stmt_ids,
                                           get_text_content_from_text_refs)


def test_get_stmts_with_agent_text_like():
    db = _get_prepped_db()
    agent_stmts0 = get_stmts_with_agent_text_like('__', filter_genes=True,
                                                  db=db)
    assert len(agent_stmts0) == 1
    assert 'ER' in agent_stmts0
    assert agent_stmts0['ER'] == [0]

    agent_stmts1 = get_stmts_with_agent_text_like('__', filter_genes=False,
                                                  db=db)
    assert len(agent_stmts1) == 2
    assert 'ER' in agent_stmts1
    assert 'DL' in agent_stmts1
    assert agent_stmts1['ER'] == [0]
    assert agent_stmts1['DL'] == [2]

    agent_stmts2 = get_stmts_with_agent_text_like('__s', filter_genes=True,
                                                  db=db)
    assert len(agent_stmts2) == 1
    assert 'NPs' in agent_stmts2
    assert agent_stmts2['NPs'] == [1]


def test_get_stmts_with_agent_text_in():
    db = _get_prepped_db()
    agent_stmts0 = get_stmts_with_agent_text_in(['damage', 'impact', 'health'],
                                                filter_genes=True, db=db)
    assert set(agent_stmts0.keys()) == set(['damage', 'impact'])
    assert agent_stmts0['damage'] == [0]
    assert agent_stmts0['impact'] == [1]
    agent_stmts1 = get_stmts_with_agent_text_in(['damage', 'impact', 'health'],
                                                filter_genes=False, db=db)
    assert set(agent_stmts1.keys()) == set(['damage', 'impact', 'health'])
    assert agent_stmts1['damage'] == [0]
    assert agent_stmts1['impact'] == [1]
    assert agent_stmts1['health'] == [2]


def test_get_text_content_from_stmt_ids():
    fulltext0 = ('We investigate properties of the estrogen receptor (ER).'
                 ' Our investigation made no new findings about ER, leading to'
                 ' damage in our groups abilty to secure funding.')
    fulltext1 = ('We describe an experiment about nanoparticles (NPs).'
                 ' The experiment was a complete failure. Our inability to'
                 ' produce sufficient quantities of NPs has made a troubling'
                 ' impact on the future of our lab. The following figure'
                 ' contains a schematic diagram of the apparatus of our'
                 ' experiment.')
    db = _get_prepped_db()
    ref_dict, text_dict = get_text_content_from_stmt_ids([0, 1], db=db)
    assert ref_dict == {0: 0, 1: 1}
    assert text_dict[0] == fulltext0
    assert text_dict[1] == fulltext1


def test_get_text_content_from_text_refs():
    fulltext0 = ('We investigate properties of the estrogen receptor (ER).'
                 ' Our investigation made no new findings about ER, leading to'
                 ' damage in our groups abilty to secure funding.')
    fulltext1 = ('We describe an experiment about nanoparticles (NPs).'
                 ' The experiment was a complete failure. Our inability to'
                 ' produce sufficient quantities of NPs has made a troubling'
                 ' impact on the future of our lab. The following figure'
                 ' contains a schematic diagram of the apparatus of our'
                 ' experiment.')
    db = _get_prepped_db()
    text = get_text_content_from_text_refs({'PMID': '000000'}, db=db)
    assert text == fulltext0
    text = get_text_content_from_text_refs({'PMID': '777777'}, db=db)
    assert text == fulltext1


def _get_prepped_db():
    dts = _DatabaseTestSetup()
    dts.load_tables()
    return dts.test_db


class _DatabaseTestSetup(object):
    """Sets up the test database"""
    def __init__(self):
        self.test_db = dbu.get_test_db()
        self.test_db._clear(force=True)
        self.test_data = _make_test_db_input()
        self.test_db._init_auth()
        _, api_key = self.test_db._add_auth('tester')
        self.tester_key = api_key

    def load_tables(self):
        """Load in all the background provenance metadata (e.g. text_ref).

        This must be done before you load any statements.
        """
        test_data = self.test_data
        tables = ['text_ref', 'text_content', 'reading', 'db_info',
                  'raw_statements', 'raw_agents']
        db_inputs = {table: set(test_data[table]['tuples'])
                     for table in tables}
        for table in tables:
            self.test_db.copy(table, db_inputs[table],
                              test_data[table]['cols'])
        return


def _make_test_db_input():
    test_database = {}
    raw_statement_cols, raw_statement_tuples, raw_statement_dict = \
        _make_raw_statements_input()
    test_database['raw_statements'] = {'cols': raw_statement_cols,
                                       'tuples': raw_statement_tuples,
                                       'dict': raw_statement_dict}

    raw_agent_cols, raw_agent_tuples, raw_agent_dict = \
        _make_raw_agents_input()
    test_database['raw_agents'] = {'cols': raw_agent_cols,
                                   'tuples': raw_agent_tuples,
                                   'dict': raw_agent_dict}

    reading_cols, reading_tuples, reading_dict = \
        _make_readings_input()
    test_database['reading'] = {'cols': reading_cols,
                                'tuples': reading_tuples,
                                'dict': reading_dict}

    text_content_cols, text_content_tuples, text_content_dict = \
        _make_text_content_input()
    test_database['text_content'] = {'cols': text_content_cols,
                                     'tuples': text_content_tuples,
                                     'dict': text_content_dict}

    text_ref_cols, text_ref_tuples, text_ref_dict = \
        _make_text_ref_input()
    test_database['text_ref'] = {'cols': text_ref_cols,
                                 'tuples': text_ref_tuples,
                                 'dict': text_ref_dict}
    test_database['db_info'] = {'cols': [],
                                'tuples': [],
                                'dict': {}}
    return test_database


def _make_raw_statements_input():
    raw_statement_cols = ('id', 'uuid', 'indra_version', 'mk_hash',
                          'reading_id', 'db_info_id', 'type', 'json',
                          'batch_id', 'source_hash')
    raw_statement_tuples = [(0, '0', '3.14', 0, 0, None, 'Activation',
                             b'{"type": "Activation", "sub": {"name": "ESR1",'
                             b'"db_refs": {"TEXT": "ER", "HGNC": "3467"}},'
                             b'"obj": {"name": "MAGEE1",'
                             b'"db_refs":'
                             b'{"TEXT": "damage", "HGNC": "24934"}},'
                             b'"evidence": [{"source_api": "rdr", "text":'
                             b'"Our investigation made no new findings about'
                             b' ER, leading to damage in our groups ability to'
                             b' secure funding."}]}', 1, 0),
                            (1, '1', '3.14', 1, 1, None, 'IncreaseAmount',
                             b'{"type": "IncreaseAmount", "sub":'
                             b' {"name": "NPS", "db_refs":'
                             b'{"TEXT": "NPs", "HGNC": "33940"}},'
                             b'"obj": {"name": "IMPACT",'
                             b'"db_refs":'
                             b'{"TEXT": "impact", "HGNC": "20387"}},'
                             b'"evidence": [{"source_api": "rdr", "text":'
                             b'"Our inability to produce sufficient quantities'
                             b' of NPs has made a troubling impact on the'
                             b' future of our lab ."}]}', 1, 1),
                            (2, '2', '3.14', 2, 2, None, 'IncreaseAmount',
                             b'{"type": "IncreaseAmount", "sub":'
                             b' {"name": "Deep Learning", "db_refs":'
                             b'{"TEXT": "DL", "MESH": "D000077321"}},'
                             b'"obj": {"name": "Health",'
                             b'"db_refs":'
                             b'{"TEXT": "health", "MESH": "D006262"}},'
                             b'"evidence": [{"source_api": "rdr", "text":'
                             b'"Research into DL is shown to increase'
                             b' the health of research programs."}]}', 1, 1)]

    raw_statement_dict = {x[0]: x for x in raw_statement_tuples}
    return raw_statement_cols, raw_statement_tuples, raw_statement_dict


def _make_raw_agents_input():
    raw_agent_cols = ('id', 'stmt_id', 'db_name', 'db_id', 'role', 'ag_num')
    raw_agent_tuples = [(0, 0, 'TEXT', 'ER', 'SUBJECT', 0),
                        (1, 0, 'HGNC', 'ESR1', 'SUBJECT', 0),
                        (2, 1, 'TEXT', 'NPs', 'SUBJECT', 0),
                        (3, 1, 'HGNC', '33940', 'SUBJECT', 0),
                        (4, 0, 'TEXT', 'damage', 'OBJECT', 0),
                        (5, 0, 'HGNC', '24934', 'OBJECT', 0),
                        (6, 1, 'TEXT', 'impact', 'OBJECT', 0),
                        (7, 1, 'HGNC', '20387', 'OBJECT', 0),
                        (8, 2, 'TEXT', 'DL', 'SUBJECT', 0),
                        (9, 2, 'MESH', 'D000077321', 'SUBJECT', 0),
                        (10, 2, 'TEXT', 'health', 'OBJECT', 0),
                        (11, 2, 'MESH', 'D006262', 'OBJECT', 0)]
    raw_agent_dict = {x[0]: x for x in raw_agent_tuples}
    return raw_agent_cols, raw_agent_tuples, raw_agent_dict


def _make_readings_input():
    reading_cols = ('id', 'text_content_id', 'reader', 'reader_version',
                    'bytes', 'format', 'batch_id')
    reading_tuples = [(0, 1, 'RDR', '2.78', b'\x1f', 'json', 1),
                      (1, 2, 'RDR', '2.78', b'\x82', 'json', 1),
                      (2, 4, 'RDR', '2.78', b'\x17', 'json', 1)]
    reading_dict = {x[0]: x for x in reading_tuples}
    return reading_cols, reading_tuples, reading_dict


def _make_text_content_input():
    text_content_cols = ('id', 'text_ref_id', 'source', 'text_type', 'format',
                         'content')
    abstract0 = 'We investigate properties of the estrogen receptor (ER).'
    fulltext0 = ('We investigate properties of the estrogen receptor (ER).'
                 ' Our investigation made no new findings about ER, leading to'
                 ' damage in our groups abilty to secure funding.')
    abstract1 = ('We describe an experiment about nanoparticles (NPs).'
                 ' The experiment was a complete failure. Our inability'
                 ' to produce sufficient quantities of NPs has made a'
                 ' troubling impact on the future of our lab.')
    fulltext1 = ('We describe an experiment about nanoparticles (NPs).'
                 ' The experiment was a complete failure. Our inability to'
                 ' produce sufficient quantities of NPs has made a troubling'
                 ' impact on the future of our lab. The following figure'
                 ' contains a schematic diagram of the apparatus of our'
                 ' experiment.')
    abstract2 = ('We describe applications of deep learning (DL) to'
                 ' grant procurement. We find that mentions of DL in'
                 ' grant proposals has a positive correlation with the'
                 ' likelihood an application is accepted. Research into'
                 ' DL is shown to increase the health of research'
                 ' programs')

    text_content_tuples = [(0, 0, 'pubmed', 'abstract', 'text',
                            zip_string(abstract0)),
                           (1, 0, 'pmc', 'fulltext', 'text',
                            zip_string(fulltext0)),
                           (2, 1, 'pubmed', 'abstract', 'text',
                            zip_string(abstract1)),
                           (3, 1, 'pmc', 'fulltext', 'text',
                            zip_string(fulltext1)),
                           (4, 2, 'pubmed', 'abstract', 'text',
                            zip_string(abstract2))]
    text_content_dict = {x[0]: x for x in text_content_tuples}
    return text_content_cols, text_content_tuples, text_content_dict


def _make_text_ref_input():
    text_ref_cols = ('id', 'pmid', 'pmcid', 'doi', 'manuscript_id', 'pii')
    text_ref_tuples = [(0, '000000', 'PMC777777', None, None, None),
                       (1, '777777', 'PMC000000', None, None, None),
                       (2, '000001', None, None, None, None)]
    text_ref_dict = {x[0]: x for x in text_ref_tuples}
    return text_ref_cols, text_ref_tuples, text_ref_dict
