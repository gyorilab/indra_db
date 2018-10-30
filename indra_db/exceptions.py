
class IndraDbException(Exception):
    pass


class NoAuthError(IndraDbException):
    def __init__(self, api_key, access):
        msg = "The api key %s does not grand access to %s." % (api_key, access)
        super(NoAuthError, self).__init__(msg)
