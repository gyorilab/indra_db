import logging
from os import environ
from datetime import datetime
from functools import wraps

from http.cookies import SimpleCookie

from flask_jwt_extended import jwt_optional, get_jwt_identity, \
    create_access_token, set_access_cookies, unset_jwt_cookies, JWTManager
from flask import Blueprint, jsonify, request

from rest_api.models import User, Role, BadIdentity, IntegrityError, \
    start_fresh, AuthLog

auth = Blueprint('auth', __name__)

logger = logging.getLogger(__name__)


def config_auth(app):
    app.config['JWT_SECRET_KEY'] = environ['INDRADB_JWT_SECRET']
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 2592000  # 30 days
    app.config['JWT_TOKEN_LOCATION'] = ['cookies']
    app.config['JWT_COOKIE_SECURE'] = True
    app.config['JWT_SESSION_COOKIE'] = False
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.config['JWT_COOKIE_CSRF_PROTECT'] = False
    SC = SimpleCookie()
    jwt = JWTManager(app)
    return SC, jwt


def auth_wrapper(func):
    @jwt_optional
    @wraps(func)
    def with_auth_log():
        start_fresh()
        logger.info("Handling %s request." % func.__name__)

        user_identity = get_jwt_identity()
        logger.info("Got user identity: %s" % user_identity)

        auth_log = AuthLog(date=datetime.utcnow(), action=func.__name__,
                           attempt_ip=request.remote_addr,
                           input_identity_token=user_identity)
        auth_details = {}

        ret = func(auth_details, user_identity)

        if isinstance(ret, tuple) and len(ret) == 2:
            resp, code = ret
        else:
            resp = ret
            code = 200
        auth_log.response = resp.json
        auth_log.code = code
        auth_log.details = auth_details
        auth_log.success = (func.__name__ in resp.json
                            and resp.json[func.__name__]
                            and code == 200)

        auth_log.save()
        return ret

    return with_auth_log


@auth.route('/register', methods=['POST'])
@auth_wrapper
def register(auth_details, user_identity):
    try:
        user = User.get_by_identity(user_identity)
        auth_details['user_id'] = user.id
        return jsonify({"message": "User is already logged in."}), 400
    except BadIdentity:
        pass

    data = request.json
    missing = [field for field in ['email', 'password']
               if field not in data]
    if missing:
        auth_details['missing'] = missing
        return jsonify({"message": "No email or password provided"}), 400

    auth_details['new_email'] = data['email']

    new_user = User.new_user(
        email=data['email'],
        password=data['password']
    )

    try:
        new_user.save()
        auth_details['new_user_id'] = new_user.id
        return jsonify({'register': True,
                        'message': 'User {} created'.format(data['email'])})
    except IntegrityError:
        return jsonify({'message': 'User {} exists.'.format(data['email'])}), \
               400
    except Exception as e:
        logger.exception(e)
        logger.error("Unexpected error creating user.")
        return jsonify({'message': 'Could not create account. '
                                   'Something unexpected went wrong.'}), 500


@auth.route('/login', methods=['POST'])
@auth_wrapper
def login(auth_details, user_identity):
    try:
        if user_identity:
            user = User.get_by_identity(user_identity)
            auth_details['user_id'] = user.id
            logger.info("User was already logged in.")
            return jsonify({"message": "User is already logged in.",
                            'login': False, 'user_email': user.email})
    except BadIdentity:
        logger.warning("User had malformed identity or invalid.")
    except Exception as e:
        logger.exception(e)
        logger.error("Got an unexpected exception while looking up user.")

    data = request.json
    missing = [field for field in ['email', 'password']
               if field not in data]
    if missing:
        auth_details['missing'] = missing
        return jsonify({"message": "No email or password provided"}), 400

    logger.debug("Looking for user: %s." % data['email'])
    current_user = User.get_by_email(data['email'], verify=data['password'])

    logger.debug("Got user: %s" % current_user)
    if not current_user:
        logger.info("Got no user, username or password was incorrect.")
        return jsonify({'message': 'Username or password was incorrect.'}), 401
    else:
        # note the user id and the new identity.
        auth_details['user_id'] = current_user.id
        auth_details['new_identity'] = current_user.identity()

        # Save some metadata for this login.
        current_user.current_login_at = datetime.utcnow()
        current_user.current_login_ip = request.remote_addr
        current_user.active = True
        current_user.save()

    access_token = create_access_token(identity=current_user.identity())
    logger.info("Produced new access token.")
    resp = jsonify({'login': True, 'user_email': current_user.email})
    set_access_cookies(resp, access_token)
    return resp


@auth.route('/logout', methods=['POST'])
@auth_wrapper
def logout(auth_details, user_identity):
    # Stash user details
    auth_details['user_id'] = None
    if user_identity:
        try:
            user = User.get_by_identity(user_identity)
        except Exception as e:
            logger.exception(e)
            logger.error("Got error while checking identity on logout.")
            user = None
        if user:
            auth_details['user_id'] = user.id
            user.last_login_at = user.current_login_at
            user.current_login_at = None
            user.last_login_ip = user.current_login_ip
            user.current_login_ip = None
            user.active = False
            user.save()
        else:
            logger.warning("Logging out user without entry in the database.")

    resp = jsonify({'logout': True})
    unset_jwt_cookies(resp)
    return resp


def _resolve_auth(query):
    """Get the roles for the current request, either by JWT or API key.

    If by API key, the key must be in the query. If by JWT, @jwt_optional or
    similar must wrap the calling function.

    Returns a tuple with the current user, if applicable, and a list of
    associated roles.
    """
    api_key = query.pop('api_key', None)
    logger.info("Got api key %s" % api_key)
    if api_key:
        logger.info("Using API key role.")
        return None, [Role.get_by_api_key(api_key)]

    user_identity = get_jwt_identity()
    logger.debug("Got user_identity: %s" % user_identity)
    if not user_identity:
        logger.info("No user identity, no role.")
        return None, []

    try:
        current_user = User.get_by_identity(user_identity)
        logger.debug("Got user: %s" % current_user)
    except BadIdentity:
        logger.info("Identity malformed, no role.")
        return None, []
    except Exception as e:
        logger.exception(e)
        logger.error("Unexpected error looking up user.")
        return None, []

    if not current_user:
        logger.info("Identity not mapped to user, no role.")
        return None, []

    logger.info("Identity mapped to the user, returning roles.")
    return current_user, list(current_user.roles)
