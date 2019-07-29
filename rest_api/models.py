import os
from base64 import b64encode

from rest_api.database import Base
from flask_security import UserMixin, RoleMixin
from sqlalchemy import create_engine
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy import Boolean, DateTime, Column, Integer, \
                       String, ForeignKey, LargeBinary

import scrypt


engine = create_engine(os.environ['INDRALAB_USERS_DB'])
DBSession = sessionmaker(bind=engine)
session = DBSession()


class RolesUsers(Base):
    __tablename__ = 'roles_users'
    id = Column(Integer(), primary_key=True)
    user_id = Column('user_id', Integer(), ForeignKey('user.id'))
    role_id = Column('role_id', Integer(), ForeignKey('role.id'))


class Role(Base, RoleMixin):
    __tablename__ = 'role'
    id = Column(Integer(), primary_key=True)
    name = Column(String(80), unique=True)
    description = Column(String(255))


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

