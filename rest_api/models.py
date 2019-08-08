import os
from base64 import b64encode
from datetime import datetime
from time import sleep

from rest_api.database import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy import Boolean, DateTime, Column, Integer, \
                       String, ForeignKey, LargeBinary
from sqlalchemy.dialects.postgresql import JSON

from sqlalchemy.exc import IntegrityError

import scrypt


engine = create_engine(os.environ['INDRALAB_USERS_DB'])
DBSession = sessionmaker(bind=engine)
session = DBSession()


class UserDatabaseError(Exception):
    pass


class BadIdentity(UserDatabaseError):
    pass


def start_fresh():
    session.rollback()


class _AuthMixin(object):
    _label = NotImplemented

    def save(self):
        if not self.id:
            session.add(self)
        try:
            session.commit()
        except Exception as e:
            session.rollback()
            raise e

    def __str__(self):
        if isinstance(self._label, list) or isinstance(self._label, set) \
                or isinstance(self._label, tuple):
            label = ' '.join(getattr(self, label) for label in self._label)
        else:
            label = getattr(self, self._label)
        return "< %s: %s - %s >" % (self.__class__.__name__, self.id, label)

    def __repr__(self):
        return str(self)


class RolesUsers(Base):
    __tablename__ = 'roles_users'
    user_id = Column(Integer, ForeignKey('user.id'), primary_key=True)
    role_id = Column(Integer, ForeignKey('role.id'), primary_key=True)


class Role(Base, _AuthMixin):
    __tablename__ = 'role'
    id = Column(Integer(), primary_key=True)
    name = Column(String(80), unique=True)
    api_key = Column(String(40), unique=True)
    api_access_count = Column(Integer, default=0)
    description = Column(String(255))
    permissions = Column(JSON)

    _label = 'name'

    @classmethod
    def get_by_name(cls, name, *args):
        """Look for a role by a given name."""

        if len(args) > 1:
            raise ValueError("Expected at most 1 extra argument.")

        role = cls.query.filter_by(name=name).first()
        if not role:
            if not args:
                raise UserDatabaseError("Role {name} does not exist."
                                        .format(name=name))
            return args[0]
        return role

    @classmethod
    def get_by_api_key(cls, api_key):
        """Get a role from its API Key."""

        # Look for the role.
        role = cls.query.filter_by(api_key=api_key).first()
        if not role:
            raise UserDatabaseError("Api Key {api_key} is not valid."
                                    .format(api_key=api_key))

        # Count the number of times this role has been accessed by API key.
        role.api_access_count += 1
        session.commit()

        return role


class User(Base, _AuthMixin):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True)
    password = Column(LargeBinary)
    last_login_at = Column(DateTime())
    current_login_at = Column(DateTime())
    last_login_ip = Column(String(100))
    current_login_ip = Column(String(100))
    login_count = Column(Integer, default=0)
    active = Column(Boolean())
    confirmed_at = Column(DateTime())
    roles = relationship('Role',
                         secondary='roles_users',
                         backref=backref('users', lazy='dynamic'))

    _label = 'email'
    _identity_cols = {'id', 'email'}

    @classmethod
    def new_user(cls, email, password, **kwargs):
        return cls(email=email.lower(), password=hash_password(password),
                   **kwargs)

    @classmethod
    def get_by_email(cls, email, verify=None):
        user = session.query(cls).filter(cls.email == email.lower()).first()
        if user is None:
            print("User %s not found." % email.lower())
            return None

        if verify:
            if verify_password(user.password, verify):
                user.last_login_at = datetime.now()
                user.login_count += 1
                user.save()
                return user
            else:
                print("User password failed.")
                return None

        return user

    @classmethod
    def get_by_identity(cls, identity):
        """Get a user from the identity JSON."""
        if not isinstance(identity, dict) or set(identity.keys()) != cls._identity_cols:
            raise BadIdentity("'{identity}' is not an identity."
                              .format(identity=identity))

        user = cls.query.get(identity['id'])
        if not user:
            raise BadIdentity("User {} does not exist.".format(identity['id']))
        if user.email.lower() != identity['email'].lower():
            raise BadIdentity("Invalid identity, email on database does "
                              "not match email given.")
        return user

    def reset_password(self, new_password):
        self.password = hash_password(new_password)
        self.save()

    def bestow_role(self, role_name):
        """Give this user a role."""
        role = Role.get_by_name(role_name)
        new_link = RolesUsers(user_id=self.id, role_id=role.id)
        session.add(new_link)
        session.commit()
        return

    def identity(self):
        """Get the user's identity"""
        return {col: getattr(self, col) for col in self._identity_cols}


class AuthLog(Base, _AuthMixin):
    __tablename__ = 'auth_log'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    success = Column(Boolean, nullable=False)
    attempt_ip = Column(String(64), nullable=False)
    action = Column(String(20), nullable=False)
    response = Column(JSON)
    code = Column(Integer)
    input_identity_token = Column(JSON)
    details = Column(JSON)

    _label = ['action', 'date']


def hash_password(password, maxtime=0.5, datalength=64):
    hp = scrypt.encrypt(b64encode(os.urandom(datalength)), password,
                        maxtime=maxtime)
    return hp


def verify_password(hashed_password, guessed_password, maxtime=0.5):
    try:
        scrypt.decrypt(hashed_password, guessed_password, maxtime)
        return True
    except scrypt.error:
        sleep(1)
        return False

