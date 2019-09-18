
class IndraDbException(Exception):
    pass


class NoAuthError(IndraDbException):
    def __init__(self, api_key, access):
        msg = "The api key %s does not grand access to %s." % (api_key, access)
        super(NoAuthError, self).__init__(msg)


class BadHashError(IndraDbException):
    def __init__(self, mk_hash):
        self.bad_hash = mk_hash
        msg = 'The matches-key hash %s is not valid.' % mk_hash
        super(BadHashError, self).__init__(msg)
