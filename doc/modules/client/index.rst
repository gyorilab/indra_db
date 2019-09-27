The Client
==========
The purpose of the client is to be the gateway for external access to the
content of the databases. Here we define high level access functions for
getting data out of the database in a natural way. This is where the queries
used by the REST API are defined, and most users looking to access knowledge on
the database should use the client if they can, as it is heavily optimized.

Our system utilizes 2 databases, one which represents the "ground truth", as
we know it, and is structured naturally for performing updates on our
knowledge; it will always be the most up to date. We also have a "readonly"
database that we used for our outward facing services. This database is
optimized for fast queries and the content in it is updated weekly. Each
database has its own set of access tools.


.. toctree::
   :maxdepth: 3

   principal/index.rst
   readonly/index.rst
   misc.rst


