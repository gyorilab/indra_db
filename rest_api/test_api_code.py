import unittest

from rest_api.api import app, TESTING
from rest_api.api_test_util import _DbApiTests


TESTING['status'] = True


class TestLocalCode(_DbApiTests):
    def setUp(self):
        app.testing = True
        self.app = app.test_client()

    @staticmethod
    def _get_api_key():
        return 'TESTKEY'


if __name__ == '__main__':
    unittest.main()
