

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
                          'reading_id', 'db_info_id', 'type', 'json')
    raw_statement_tuples = [(0, '0', '3.14', 0, 0, None, 'Activation',
                             b'{"type": "Activation", "sub": {"name": "ESR1",'
                             b'"db_refs": {"TEXT": "ER", "HGNC": "3467"}},'
                             b'"obj": {"name": "MAGEE1",'
                             b'"db_refs":'
                             b'{"TEXT": "damage", "HGNC": "24934"}},'
                             b'"evidence": [{"source_api": "rdr", "text":'
                             b'"Our investigation made no new findings about'
                             b' ER, leading to damage in our groups ability to'
                             b' secure funding."}]}'),
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
                             b' future of our lab."}]}')]

    raw_statement_dict = {x[0]: x for x in raw_statement_tuples}
    return raw_statement_cols, raw_statement_tuples, raw_statement_dict


def _make_raw_agents_input():
    raw_agent_cols = ('id', 'stmt_id', 'db_name', 'db_id', 'role')
    raw_agent_tuples = [(0, 0, 'TEXT', 'ER', 'SUBJECT'),
                        (1, 0, 'HGNC', 'ESR1', 'SUBJECT'),
                        (2, 1, 'TEXT', 'NPs', 'SUBJECT'),
                        (3, 1, 'HGNC', '33940', 'SUBJECT'),
                        (4, 0, 'TEXT', 'damage', 'OBJECT'),
                        (5, 0, 'HGNC', '24934', 'OBJECT'),
                        (6, 1, 'TEXT', 'impact', 'OBJECT'),
                        (7, 1, 'HGNC', '20387', 'OBJECT')]

    raw_agent_dict = {x[0]: x for x in raw_agent_tuples}
    return raw_agent_cols, raw_agent_tuples, raw_agent_dict


def _make_readings_input():
    reading_cols = ('id', 'text_content_id', 'reader', 'reader_version',
                    'bytes', 'format')
    reading_tuples = [(0, 1, 'RDR', '2.78', b'\x1f', 'json'),
                      (1, 2, 'RDR', '2.78', b'\x82', 'json')]
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

    text_content_tuples = [(0, 0, 'pubmed', 'abstract', 'text', abstract0),
                           (1, 0, 'pmc', 'fulltext', 'text', fulltext0),
                           (2, 1, 'pubmed', 'abstract', 'text', abstract1),
                           (3, 1, 'pmc', 'fulltext', 'text', fulltext1)]
    text_content_dict = {x[0]: x for x in text_content_tuples}
    return text_content_cols, text_content_tuples, text_content_dict


def _make_text_ref_input():
    text_ref_cols = ('id', 'pmid', 'pmcid', 'doi', 'manuscript_id', 'pii')
    text_ref_tuples = [(0, '000000', 'PMC777777', None, None, None),
                       (1, '777777', 'PMC000000', None, None, None)]
    text_ref_dict = {x[0]: x for x in text_ref_tuples}
    return text_ref_cols, text_ref_tuples, text_ref_dict
