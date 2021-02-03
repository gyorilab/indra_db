from flask import Response, jsonify


class HttpUserError(ValueError):
    def __init__(self, msg, err_code=400):
        self.err_code = err_code
        self.msg = msg
        super(HttpUserError, self).__init__(msg)

    def to_json(self):
        return {"result": "failure", "reason": self.msg}

    def response(self):
        return Response(jsonify(self.to_json()), self.err_code)


class ResultTypeError(HttpUserError):
    def __init__(self, result_type):
        self.result_type = result_type
        msg = f"Invalid result type: {result_type}"
        super(ResultTypeError, self).__init__(msg)


class InvalidCredentials(HttpUserError):
    def __init__(self, cred_type):
        super(HttpUserError, self).\
            __init__(f"Invalid credentials: {cred_type}", 401)


class InsufficientPermission(HttpUserError):
    def __init__(self, resource):
        super(HttpUserError, self).\
            __init__(f"Insufficient permissions for: {resource}", 403)
