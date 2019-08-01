import os
from base64 import b64encode
from datetime import datetime

from rest_api.database import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy import Boolean, DateTime, Column, Integer, \
                       String, ForeignKey, LargeBinary
from sqlalchemy.dialects.postgresql import JSON

import scrypt


engine = create_engine(os.environ['INDRALAB_USERS_DB'])
DBSession = sessionmaker(bind=engine)
session = DBSession()


class UserDatabaseError(Exception):
    pass


class RolesUsers(Base):
    __tablename__ = 'roles_users'
    user_id = Column(Integer, ForeignKey('user.id'), primary_key=True)
    role_id = Column(Integer, ForeignKey('role.id'), primary_key=True)


class Role(Base):
    __tablename__ = 'role'
    id = Column(Integer(), primary_key=True)
    name = Column(String(80), unique=True)
    api_key = Column(String(40), unique=True)
    api_access_count = Column(Integer, default=0)
    description = Column(String(255))
    permissions = Column(JSON)

    def save(self):
        if not self.id:
            session.add(self)
        session.commit()

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

    def __str__(self):
        return "{id} - {name}".format(id=self.id,
                                      name=self.name)

    def __repr__(self):
        return "< Role: {} >".format(str(self))


class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True)
    username = Column(String(255), unique=True)
    password = Column(LargeBinary)
    last_login_at = Column(DateTime())
    current_login_at = Column(DateTime())
    last_login_ip = Column(String(100))
    current_login_ip = Column(String(100))
    login_count = Column(Integer)
    active = Column(Boolean())
    confirmed_at = Column(DateTime())
    roles = relationship('Role',
                         secondary='roles_users',
                         backref=backref('users', lazy='dynamic'))

    @classmethod
    def new_user(cls, email, password, **kwargs):
        return cls(email=email, password=hash_password(password), **kwargs)

    def save(self):
        session.add(self)
        session.commit()

    @classmethod
    def get_user_by_email(cls, email, password):
        user = session.query(cls).filter(cls.email == email).first()
        if user is None:
            print("User %s not found." % email)
            return None

        if verify_password(user.password, password):
            return user
        else:
            print("User password failed.")
            return None

    def __str__(self):
        return "%s - %s" % (self.id, self.email)

    def __repr__(self):
        return '< User: %s >' % str(self)


def hash_password(password, maxtime=0.5, datalength=64):
    hp = scrypt.encrypt(b64encode(os.urandom(datalength)), password,
                        maxtime=maxtime)
    return hp


def verify_password(hashed_password, guessed_password, maxtime=0.5):
    try:
        scrypt.decrypt(hashed_password, guessed_password, maxtime)
        return True
    except scrypt.error:
        return False

