import unittest
import requests

from indra import get_config

from rest_api.api_test_util import _DbApiTests


class WebResponse:
    """Imitate the response from the "app", but from real request responses."""
    def __init__(self, resp):
        self._resp = resp
        self.data = resp.content

    def __getattribute__(self, item):
        """When in doubt, try to just get the item from the actual resp.

        This should work much of the time because the results from the "app" are
        intended to imitate and actual response.
        """
        try:
            return super(WebResponse, self).__getattribute__(item)
        except AttributeError:
            return getattr(self._resp, item)


class WebApp:
    """Mock the behavior of the "app" but on the real service."""
    def __init__(self, url):
        self.base_url = url
        if self.base_url.endswith('/'):
            self.base_url = self.base_url[:-1]

    def __process_url(self, url):
        if not url.startswith('/'):
            url = '/' + url
        return self.base_url + url

    def get(self, url):
        full_url = self.__process_url(url)
        raw_resp = requests.get(full_url)
        return WebResponse(raw_resp)

    def post(self, url, data, headers):
        full_url = self.__process_url(url)
        raw_resp = requests.post(full_url, data=data, headers=headers)
        return WebResponse(raw_resp)


class TestDeployment(_DbApiTests):
    def setUp(self):
        url = get_config('INDRA_DB_REST_URL', failure_ok=False)
        print("URL:", url)
        self.app = WebApp(url)

    @staticmethod
    def _get_api_key():
        api_key = get_config('INDRA_DB_REST_API_KEY', failure_ok=True)
        if api_key is None:
            raise unittest.SkipTest("No API KEY available. Cannot test auth.")
        return api_key


if __name__ == '__main__':
    unittest.main()
