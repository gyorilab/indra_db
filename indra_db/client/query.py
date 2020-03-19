from indra.statements import stmts_from_json


class StatementQueryResult(object):
    def __init__(self, results: dict, limit: int, offset: int,
                 query: object):
        self.results = results
        self.limit = limit
        self.offset = offset
        self.query = query

    def json(self):
        return {'results': self.results, 'limit': self.limit,
                'offset': self.offset, 'query': self.query.to_json()}

    def statements(self):
        return stmts_from_json(list(self.results.values()))


class StatementQuery(object):
    def __init__(self, limit=None, offset=0):
        self.limit = limit
        self.offset = offset

    def to_json(self) -> dict:
        return {'limit': self.limit, 'offset': self.offset,
                'constraint': self._get_constraint_json()}

    def _get_constraint_json(self) -> dict:
        raise NotImplementedError()

    def __and__(self, other):
        pass

    def __or__(self, other):
        pass


class ReadonlyQuery(StatementQuery):

    def run(self, ro, stmt_types=None, activity=None, is_active=None,
            agent_count=None, ev_count=None) -> StatementQueryResult:
        raise NotImplementedError()


class PrincipalQuery(StatementQuery):

    def run(self, db, stmt_types=None, activity=None, is_active=None,
            agent_count=None, ev_count=None) -> StatementQueryResult:
        raise NotImplementedError()


class RoHashQuery(ReadonlyQuery):
    def __init__(self, stmt_hashes, *args, **kwargs):
        self.stmt_hashes = stmt_hashes
        super(RoHashQuery, self).__init__(*args, **kwargs)

    def run(self, ro, stmt_types=None, activity=None, is_active=None,
            agent_count=None, ev_count=None) -> StatementQueryResult:
        query = self.get_query()
        query.filter(stmt_types)
