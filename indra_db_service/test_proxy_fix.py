"""Tests for trusting one upstream proxy via ProxyFix.

``ProxyFix`` is applied when the app module is imported. Each case sets
``INDRA_DB_API_BEHIND_PROXY`` and reloads ``config`` and then ``api``.

NOTE: The tests in this file should be run in isolation as they modify the
environment variables and reload the config and api modules.
"""

import importlib
import os
from pathlib import Path

from flask import jsonify, request


def test_forwarded_headers_trusted_when_behind_proxy():
    os.environ["TESTING_DB_APP"] = "1"
    # Satisfies the import-time Vue path check. The probe never reads it.
    os.environ["INDRA_DB_API_VUE_ROOT"] = str(Path(__file__).resolve().parent)
    os.environ.pop("INDRA_DB_API_DEPLOYMENT", None)
    os.environ["INDRA_DB_API_BEHIND_PROXY"] = "1"

    import indra_db_service.config as config
    import indra_db_service.api as api

    # Reload the config and api modules to apply the changes.
    importlib.reload(config)
    importlib.reload(api)

    # Add a test route to check the forwarded headers.
    @api.app.route("/_proxyfix_probe")
    def _proxyfix_probe():
        return jsonify(remote_addr=request.remote_addr, url=request.url)

    response = api.app.test_client().get(
        "/_proxyfix_probe",
        headers={
            "X-Forwarded-For": "198.51.100.10, 203.0.113.50",
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "db.indra.bio",
        },
    )
    assert response.status_code == 200
    view = response.get_json()
    # x_for=1 keeps the rightmost address, the one added by nginx.
    assert view["remote_addr"] == "203.0.113.50"
    assert view["url"] == "https://db.indra.bio/_proxyfix_probe"


def test_forwarded_headers_ignored_without_behind_proxy():
    os.environ["TESTING_DB_APP"] = "1"
    # Satisfies the import-time Vue path check. The probe never reads it.
    os.environ["INDRA_DB_API_VUE_ROOT"] = str(Path(__file__).resolve().parent)
    os.environ.pop("INDRA_DB_API_DEPLOYMENT", None)
    os.environ.pop("INDRA_DB_API_BEHIND_PROXY", None)

    import indra_db_service.config as config
    import indra_db_service.api as api

    # Reload the config and api modules to apply the changes.
    importlib.reload(config)
    importlib.reload(api)

    # Add a test route to check the forwarded headers.
    @api.app.route("/_proxyfix_probe")
    def _proxyfix_probe():
        return jsonify(remote_addr=request.remote_addr, url=request.url)

    response = api.app.test_client().get(
        "/_proxyfix_probe",
        headers={
            "X-Forwarded-For": "198.51.100.10, 203.0.113.50",
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "db.indra.bio",
        },
    )
    assert response.status_code == 200
    view = response.get_json()
    assert view["remote_addr"] == "127.0.0.1"
    assert view["url"] == "http://localhost/_proxyfix_probe"
