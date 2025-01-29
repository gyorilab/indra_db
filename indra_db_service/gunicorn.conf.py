"""Gunicorn configuration file for the INDRA DB service

https://docs.gunicorn.org/en/stable/settings.html#config-file
"""

import threading
from indralab_auth_tools.src.database import monitor_database_connection


def post_fork(server, worker):
    """Function to run after forking a worker

    See: https://docs.gunicorn.org/en/stable/settings.html#post-fork

    This function is called after a worker is forked. It starts a thread to monitor
    the database connection and reset it if it is lost.
    """
    thread = threading.Thread(target=monitor_database_connection, args=(60,), daemon=True)
    thread.start()
    print(f"Started database connection monitor thread in worker {worker.pid}.")
