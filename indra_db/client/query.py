from indra.statements import stmts_from_json


class StatementQueryResult(object):
    """The result of a statement query.

    This class encapsulates the results of a search for statements in the
    database. This standardizes the results of such searches.

    Attributes
    ----------
    results : dict
        The results of the query keyed by unique IDs (mk_hash for PA Statements,
        IDs for Raw Statements, etc.)
    limit : int
        The limit that was applied to this query.
    query_json : dict
        A description of the query that was used.
    """
    def __init__(self, results: dict, limit: int, offset: int,
                 query_json: dict):
        self.results = results
        self.limit = limit
        self.offset = offset
        self.query_json = query_json

    def json(self):
        return {'results': self.results, 'limit': self.limit,
                'offset': self.offset, 'query': self.query_json}

    def statements(self):
        return stmts_from_json(list(self.results.values()))


class StatementQuery(object):
    def __init__(self, limit=None, offset=0):
        self.limit = limit
        self.offset = offset

    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def _return_result(self, results):
        return StatementQueryResult(results, self.limit, self.offset,
                                    self.to_json())

    def to_json(self) -> dict:
        return {'limit': self.limit, 'offset': self.offset,
                'constraint': self._get_constraint_json()}

    def _get_constraint_json(self) -> dict:
        raise NotImplementedError()

    def _get_stmt_jsons_from_hashes_query(self, mk_hashes_q):
        return self._return_result({})


class HashQuery(StatementQuery):
    def __init__(self, stmt_hashes, *args, **kwargs):
        self.stmt_hashes = stmt_hashes
        super(HashQuery, self).__init__(*args, **kwargs)

    def _get_constraint_json(self) -> dict:
        return {"hash_query": self.stmt_hashes}

    def run(self, *args, **kwargs):
        raise NotImplementedError()


class RoHashQuery(HashQuery):

    def run(self, ro) -> StatementQueryResult:
        if len(self.stmt_hashes) == 0:
            return self._return_result({})
        elif len(self.stmt_hashes) == 1:
            mk_hashes_q = \
                ro.filter_query([ro.PaMeta.mk_hash, ro.PaMeta.ev_count],
                                ro.PaMeta.mk_hash == self.stmt_hashes[0])
        else:
            mk_hashes_q = \
                ro.filter_query([ro.PaMeta.mk_hash, ro.PaMeta.ev_count],
                                ro.PaMeta.mk_hash.in_(self.stmt_hashes))
        return self._get_stmt_jsons_from_hashes_query(mk_hashes_q)


class ComposableQuery(StatementQuery):

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


