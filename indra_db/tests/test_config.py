from indra_db.config import build_db_url


def test_build_db_url():
    """Test the build of a database URL from typical inputs."""
    res_url = build_db_url(host="host", password="password", dialect="postgres",
                           username="user", port=10, name="db")
    assert res_url == "postgres://user:password@host:10/db", res_url
