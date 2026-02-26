=========================
INDRA Database REST API
=========================

The INDRA Database software is designed to create and maintain a
database of text references, content, reading results, and ultimately INDRA
Statements extracted from those reading results. The software also manages
the generation and update process of cleaning, deduplicating, and finding
relations between the raw Statement extractions, into what are called
pre-assembled Statements. All INDRA Statements can be represented as JSON,
which is the format returned by the API.

This web API provides the code necessary to support a REST service which
allows access to the pre-assembled Statements in a database. The API supports the
following endpoints for getting Statements:

- :ref:`statements/from_agents <from-agents>`, getting Statements by agents, using
  various ids or names, by statement type (e.g. Phosphorylation)
- :ref:`statements/from_hash <from-hash>` and
  :ref:`statements/from_hashes <from-hashes>`, getting Statements by Statement
  hash, either singly or in batches
- :ref:`statements/from_papers <from-papers>`, getting Statements using the paper
  ids from which they were extracted

The API also supports curating Statements via the
:ref:`curation endpoint <curation>`, helping us improve the quality and accuracy
of our content. You can also list curations (all, or by statement or evidence)
via the :ref:`curation list endpoints <curation-list-all>`.

You will optionally need the following information to access a running web service:

- An API key which needs to be sent in the header of each request to the
  service for restricted endpoints, or
- Login credentials to access the web interface, which is required to curate
  statements in the HTML interface.

You can create an account at https://db.indra.bio/search by clicking the "Login"
button in the top right corner, and then clicking "Register". To obtain an API key,
please contact the `Gyorilab <https://gyorilab.github.io>`_.

The code for the REST service can be found
`here <https://github.com/gyorilab/indra_db/blob/master/indra_db_service/api.py>`_.

See the OpenAPI specification at:
https://db.indra.bio/openapi.yaml

or the Swagger UI at:
https://db.indra.bio/api-docs

The Statement Endpoints
=======================

Below is detailed documentation for the different endpoints of the API that return
statements (i.e. those with the root ``/statements``). All endpoints that return
statements have the following options to control the size and order of the response:

- **format**: The endpoint is capable of returning both HTML and JSON content
  by setting the format parameter to "html" or "json", respectively. See the
  :ref:`section on output formats <output-formats>` below.
- **max_stmts**: Set the maximum number of statements you wish to receive.
  The REST API maximum is 1000, which cannot be overridden by this argument
  (to prevent request timeouts).
- **ev_limit**: The default varies, but in general the amount of Evidence
  returned for each statement is limited. A single statement can have upwards of
  10,000 pieces of evidence, so this allows queries to be run reliably. There
  is no limitation on this value, so use with caution. Setting too high a value
  may cause a request to time out or be too large to return.
- **best_first**: This is set to "true" by default, so statements with the
  most evidence are returned first. These are generally the most reliable,
  however they are also generally the most canonical. Set this parameter to
  "false" to get statements in an arbitrary order. This can also speed up a
  query. You may however find you get a lot of low-quality content.

.. _output-formats:

The output formats
------------------

The output format is controlled by the **format** option described above,
with options to return JSON or HTML.

**JSON:** The default value, intended for programmatic use, is "json". The
JSON that is returned is of the following form (with many made-up but reasonable
numbers filled in):

.. code-block:: python

   {
     "statements": {  # Dict of statement JSONs keyed by hash
       "12345234234": {...},  # Statement JSON 1
       "-246809323482": {...},  # Statement JSON 2
       ...},
     "offset": 2000,  # offset of SQL query
     "evidence_limit": 10,  # evidence limit used
     "statement_limit": 1000,  # REST API Limit
     "evidence_totals": {  # dict of available evidence for each statement keyed by hash
       "12345234234": 7657,
       "-246809323482": 870,
       ...},
     "total_evidence": 163708, # The total amount of evidence available
     "evidence_returned": 10000  # The total amount of evidence returned
   }

where the ``"statements"`` element contains a dictionary of INDRA Statement
JSONs keyed by a shallow statement hash (see :ref:`here <from-hash>` for more
details on these hashes). You can look at the
`JSON schema <https://github.com/gyorilab/indra/blob/master/indra/resources/statements_schema.json>`_
on github for details on the Statement JSON. To learn more about INDRA
Statements, you can read the
`documentation <https://indra.readthedocs.io/en/latest/modules/statements.html>`_.

**HTML:** The other ``format`` parameter option, designed for easier manual
usage, is "html". The service will then return an HTML document that, when
opened in a web browser and if logged in, provides a graphical user interface
for viewing and curating statements at the evidence level. The web page also
allows you to easily query for more evidence for a given statement.
Documentation for the html output (produced by INDRA's HTML assembler) can be found
`here <https://indra.readthedocs.io/en/latest/modules/assemblers/index.html>`_.

.. _from-agents:

Get Statements by agents and type
---------------------------------

.. openapi:: ../../indra_db_service/static/openapi.yaml
   :paths: /statements/from_agents

This endpoint allows you to get statements filtering by their agents and
the type of Statement. The query parameters are as follows:

- **subject**, **object**: The HGNC gene symbol of the subject or
  object of the Statement. **Note**: only one of each of ``subject`` and
  ``object`` will be accepted per query.

  - Example 1: if looking for Statements where MAP2K1 is a subject
    (e.g. "What does MAP2K1 phosphorylate?"), specify ``subject=MAP2K1`` as a
    query parameter
  - Example 2: if looking for Statements where MAP2K1 is the subject and
    MAPK1 is the object, add both ``subject=MAP2K1`` and ``object=MAPK1`` as
    query parameters.
  - Example 3: you can specify the agent id namespace by appending
    ``@<namespace>`` to the agent id in the parameter, e.g. ``subject=6871@HGNC``.

- **agent\***: This parameter is used if the specific role of the agent
  (subject or object) is irrelevant, or the distinction doesn't apply to the
  type of Statement of interest (e.g. Complex, Translocation, ActiveForm).
  **Note**: You can include as many ``agent*`` queries as you like, however you
  will only get Statements that include all agents you query, in addition to
  those queried for ``subject`` and ``object``. Furthermore, to include multiple
  agents on our particular implementation, you must include a suffix to each agent
  key, such as ``agent0`` and ``agent1``, or else all but one agent will be
  stripped out. Note that you need not use integers, you can add any suffix you
  like, e.g. ``agentOfDestruction=TP53`` would be entirely valid.

  - Example 1: To obtain Statements that involve SMAD2 in any role, add
    ``agent=SMAD2`` to the query.
  - Example 2: As with ``subject`` and ``object``, you can specify the
    namespace for an agent by appending ``@<namespace>`` to the agent's id, e.g.
    ``agent=ERK@TEXT``.
  - Example 3: If you wanted to query multiple statements, you could include
    ``agent0=MEK@FPLX`` and ``agent1=ERK@FPLX``. Note that the value of the
    integers has no real bearing on the ordering, and only serves to make the
    agents uniquely keyed. Thus ``agent1=MEK@FPLX`` and ``agent0=ERK@FPLX`` will
    give exactly the same result.

- **type**: This parameter can be used to specify what type of Statement
  (e.g. Phosphorylation, Activation, Complex) to query for.

  - Example: To answer the question "Does MAP2K1 phosphorylate MAPK1?"
    the parameter ``type=Phosphorylation`` can be included in your query.
    Note that this field is not case sensitive, so ``type=phosphorylation`` would
    give the same result.

.. _from-hash:

Get a Statement by hash
-----------------------

.. openapi:: ../../indra_db_service/static/openapi.yaml
   :paths: /statements/from_hash/{hash}

INDRA Statement objects have a method, ``get_hash``, which produces hash from
the content of the Statement. A shallow hash only considers the meaning of
the statement (agents, type, modifications, etc.), whereas a deeper hash also
considers the list of evidence available for that Statement. The shallow hash
is what is used in this application, as it has the same uniqueness properties
used in deduplication. As mentioned above, the Statements are returned keyed by
their hash. In addition, if you construct a Statement in python, you may get
its hash and quickly find any evidence for that Statement in the database.

This endpoint has no extra parameters, but rather takes an extension to the
path. So, to look up the hash 123456789, you would use
``statements/from_hash/123456789``.

Because this only returns one statement, the default evidence limit is
extremely generous, set to 10,000. Thus you are most likely to get all the
evidence for a given statement this way. As described above, the evidence
limit can also be raised, at the risk of a timed out request.

.. _from-hashes:

Get Statements from many hashes
-------------------------------

.. openapi:: ../../indra_db_service/static/openapi.yaml
   :paths: /statements/from_hashes

Like the previous endpoint, this endpoint uses hashes to retrieve Statements,
however instead of only being allowed one at a time, a batch of hashes may be
sent as json data. There are no special parameters for this endpoint. The json
data for the POST request should be formatted as:

.. code-block:: json

   {"hashes": [12345, 246810]}

with up to 1,000 hashes given in the list.

.. _from-papers:

Get Statements from paper ids
-----------------------------

.. openapi:: ../../indra_db_service/static/openapi.yaml
   :paths: /statements/from_papers

Using this endpoint, you can pretend you have a fleet of text extraction tools
that run in seconds! Specifically, you can get the INDRA Statements with evidence from
a given list of papers by passing one of the ids of those papers as JSON data in a POST
request. The papers ids should be formatted as:

.. code-block:: json

   {"ids": [{"id": "12345", "type": "pmid"},
            {"id": "234525", "type": "tcid"},
            {"id": "PMC23423", "type": "pmcid"}]}

a list of dicts, each containing id type and and id value.

.. _curation:

Curation
========

Because the mechanisms represented by our Statements come in large part from
automatic extractions, there can often be errors. For this reason, we always
provide the sentences from which a Statement was extracted (if we extracted
it as some of our content comes from other curated databases), as well as
provenance to lead back to the content (abstract, full text, etc.) that was
read, which allows you to use your own judgement regarding the validity of
a Statement.

If you find something wrong with a Statement, you can use this curation
endpoint to record your observation. This will not necessarily have any
immediate effect on the output, however, over time it will help us improve the
readers we use, our methods for extracting Statements from those reader
outputs, could help us filter erroneous content, and will help us improve our
pre-assembly algorithms.

Further instruction on curation best practices can be found
`here <https://indra.readthedocs.io/en/latest/tutorials/html_curation.html#curation-guidelines>`_.

Curate statements
-----------------

.. openapi:: ../../indra_db_service/static/openapi.yaml
   :paths: /curation/submit/{hash_val}

If you wish to curate a Statement, you must first decide whether you are
curating the Statement as generally incorrect, or whether a particular
sentence supports a given Statement. This is the "level" of your curation:

- **pa**: At this level, you are curating the knowledge in a
  :strong:`p`re-:strong:`a`ssembled Statement. For example, if a Statement
  indicates that "differentiation binds apoptosis", regardless of whether the
  reader(s) made a valid extraction, it is clearly wrong.
- **raw**: At this level, you are curating a particular raw extraction, in
  other words stating that an automatic reader made an error. Even more
  explicitly, you can judge whether the sentence supports the extracted
  Statement. For example the (hypothetical) sentence "KRAS was found to actively
  inhibit BRAF" does not support the Statement "KRAS activates BRAF". Another
  example would be that the sentence "IR causes cell death", where IR refers to
  Ionizing Radiation does not support the extraction "'Insulin Receptor' causes cell
  death". In this case, the reader made an error in extracting "IR" as "Insulin
  Receptor" rather than "Ionizing Radiation".

The two different levels also have different hashes. At the *pa* level, the
hashes discussed :ref:`above <from-hash>` are used, as they are calculated from the
knowledge contained in the statement, independent of the evidence. At the *raw*
level, a different hash must be included: the ``source_hash``, which identifies
a specific piece of evidence, without considering the Statement extracted.
Within a Statement JSON, there is a key "evidence", with a list of Evidence
JSON, which includes an entry for "source_hash":

.. code-block:: python

   {"evidence": [{"source_hash": 98687578576598, ...}, ...], ...}

Once you know the level, and you have the correct hash(es) (the shallow
pre-assembly hash and, optionally, the source hash), you can curate a statement by
POSTing a request with JSON data to the endpoint, as shown in the heading. The
JSON data should contain the following fields:

- **tag**: A very short word or phrase categorizing the error, for example
  "grounding" for a grounding error.
- **text**: A brief description of what you think is most wrong.
- **curator**: Your name, initials, email, or other way to identify yourself.
  Whichever you choose, please be consistent.

Note that you can also indicate that a Statement is *correct*. In particular,
if you find that a Statement has some evidence that supports the Statement and
some that does not, curating examples of both is valuable. In general, flagging
correct Statements can be just as valuable as flagging incorrect Statements.

.. _curation-list-all:

List all curations
------------------

.. openapi:: ../../indra_db_service/static/openapi.yaml
   :paths: /curation/list

This authenticated endpoint returns all curations in the database. Curator
names are anonymized if the caller does not have the correct permissions.
Authentication is done via an API key in the query parameters.

.. _curation-list-stmt:

List curations for a statement
------------------------------

.. openapi:: ../../indra_db_service/static/openapi.yaml
   :paths: /curation/list/{stmt_hash}

This public endpoint returns curations for the given pre-assembly statement
hash. Authentication is not required for this endpoint.

.. _curation-list-stmt-src:

List curations for a statement and evidence
-------------------------------------------

.. openapi:: ../../indra_db_service/static/openapi.yaml
   :paths: /curation/list/{stmt_hash}/{src_hash}

This public endpoint returns all curations for a given statement and evidence. 
The curations are filtered by both the statement hash and the source (evidence) hash.
Authentication is not required for this endpoint.

Usage examples
==============

The web service accepts standard HTTP requests, and any client that can
send such requests can be used to interact with the service. Here we
provide usage examples with the ``curl`` command line tool and ``python`` of
some of the endpoints. This is by no means a comprehensive list, but rather
demonstrates some of the crucial features discussed above.

In the examples, we assume the path to the web API is ``https://db.indra.bio/``, and
that the API key is ``12345``.

Example 1:
----------

``curl`` is a command line tool on Linux and Mac, making it a convenient tool
for making calls to this web API.

Using ``curl`` to query Statements about "MAP2K1 phosphorylates MAPK1":

.. code-block:: bash

   curl -X GET "https://db.indra.bio/statements/from_agents?subject=MAP2K1&object=MAPK1&type=phosphorylation&limit=1&ev_limit=1"

.. dropdown:: JSON returned

   .. code-block:: json

    {
      "results": {
        "-10700327527421433": {
          "type": "Phosphorylation",
          "enz": {
            "name": "MAP2K1",
            "db_refs": {
              "UP": "Q02750",
              "HGNC": "6840",
              "EGID": "5604"
            }
          },
          "sub": {
            "name": "MAPK1",
            "db_refs": {
              "UP": "P28482",
              "EGID": "5594",
              "HGNC": "6871"
            }
          },
          "belief": 0.9181686,
          "evidence": [
            {
              "source_api": "reach",
              "text": "MEK1 and MEK2 are dual specificity kinases and responsible for the phosphorylation and activation of the ERK1 and ERK2.",
              "annotations": {
                "found_by": "Phosphorylation_syntax_8_noun",
                "agents": {
                  "raw_text": [
                    "MEK1",
                    "ERK2"
                  ]
                },
                "prior_uuids": [
                  "596bd319-e794-427c-b010-5bf36eeb4566"
                ],
                "content_source": "elsevier"
              },
              "epistemics": {
                "direct": true,
                "raw_sections": []
              },
              "text_refs": {
                "PMID": "30954694",
                "TRID": 30073993,
                "PMID_NUM": 30954694,
                "DOI": "10.1016/J.JINORGBIO.2019.03.022",
                "DOI_NS": 1016,
                "DOI_ID": "J.JINORGBIO.2019.03.022",
                "PII": "S0162-0134(18)30676-7",
                "TCID": 88199895,
                "SOURCE": "elsevier",
                "RID": 10300088199895,
                "READER": "REACH"
              },
              "source_hash": 5040129478119479862,
              "pmid": "30954694"
            }
          ],
          "id": "80b49514-b402-4c76-8fac-281453a81d98",
          "matches_hash": "-10700327527421433"
        }
      },
      "limit": 1,
      "offset": null,
      "next_offset": 1,
      "query_json": {
        "class": "Intersection",
        "constraint": {
          "query_list": [
            {
              "class": "HasAgent",
              "constraint": {
                "agent_id": "MAPK1",
                "namespace": "NAME",
                "_regularized_id": "MAPK1",
                "role": "OBJECT",
                "agent_num": null
              },
              "inverted": false
            },
            {
              "class": "HasType",
              "constraint": {
                "stmt_types": [
                  "Phosphorylation"
                ]
              },
              "inverted": false
            },
            {
              "class": "HasAgent",
              "constraint": {
                "agent_id": "MAP2K1",
                "namespace": "NAME",
                "_regularized_id": "MAP2K1",
                "role": "SUBJECT",
                "agent_num": null
              },
              "inverted": false
            }
          ]
        },
        "inverted": false
      },
      "evidence_counts": {
        "-10700327527421433": 180
      },
      "belief_scores": {
        "-10700327527421433": 0.9181686
      },
      "source_counts": {
        "-10700327527421433": {
          "creeds": 0,
          "bel_lc": 0,
          "sparser": 60,
          "acsn": 0,
          "hprd": 0,
          "ubibrowser": 0,
          "crog": 0,
          "conib": 0,
          "geneways": 0,
          "vhn": 0,
          "medscan": 2,
          "pe": 0,
          "tees": 0,
          "signor": 2,
          "dgi": 0,
          "semrep": 0,
          "wormbase": 0,
          "cbn": 0,
          "isi": 0,
          "rlimsp": 31,
          "ctd": 0,
          "eidos": 0,
          "pc": 5,
          "minerva": 0,
          "drugbank": 0,
          "trips": 0,
          "trrust": 0,
          "omnipath": 0,
          "reach": 80,
          "tas": 0,
          "psp": 0,
          "biogrid": 0,
          "tkg": 0,
          "gnbr": 0
        }
      },
      "total_evidence": 180,
      "result_type": "statements",
      "offset_comp": 1,
      "returned_evidence": 1,
      "statement_limit": 500,
      "statements_returned": 1,
      "end_of_statements": true,
      "statements_removed": 0,
      "evidence_returned": 1,
      "statements": {
        "-10700327527421433": {
          "type": "Phosphorylation",
          "enz": {
            "name": "MAP2K1",
            "db_refs": {
              "UP": "Q02750",
              "HGNC": "6840",
              "EGID": "5604"
            }
          },
          "sub": {
            "name": "MAPK1",
            "db_refs": {
              "UP": "P28482",
              "EGID": "5594",
              "HGNC": "6871"
            }
          },
          "belief": 0.9181686,
          "evidence": [
            {
              "source_api": "reach",
              "text": "MEK1 and MEK2 are dual specificity kinases and responsible for the phosphorylation and activation of the ERK1 and ERK2.",
              "annotations": {
                "found_by": "Phosphorylation_syntax_8_noun",
                "agents": {
                  "raw_text": [
                    "MEK1",
                    "ERK2"
                  ]
                },
                "prior_uuids": [
                  "596bd319-e794-427c-b010-5bf36eeb4566"
                ],
                "content_source": "elsevier"
              },
              "epistemics": {
                "direct": true,
                "raw_sections": []
              },
              "text_refs": {
                "PMID": "30954694",
                "TRID": 30073993,
                "PMID_NUM": 30954694,
                "DOI": "10.1016/J.JINORGBIO.2019.03.022",
                "DOI_NS": 1016,
                "DOI_ID": "J.JINORGBIO.2019.03.022",
                "PII": "S0162-0134(18)30676-7",
                "TCID": 88199895,
                "SOURCE": "elsevier",
                "RID": 10300088199895,
                "READER": "REACH"
              },
              "source_hash": 5.04012947811948e+18,
              "pmid": "30954694"
            }
          ],
          "id": "80b49514-b402-4c76-8fac-281453a81d98",
          "matches_hash": "-10700327527421433"
        }
      }
  }

Python is another convenient way to use this web API, and has the important
advantage that Statements returned from the service can be used directly with
INDRA tools.

You can use python to get JSON Statements for the same query:

.. code-block:: python

   import requests
   resp = requests.get('https://db.indra.bio/statements/from_agents',
                       params={'subject': 'MAP2K1',
                               'object': 'MAPK1',
                               'type': 'phosphorylation')
   resp_json = resp.json()

which can now be turned into INDRA Statement objects using ``stmts_from_json``:

.. code-block:: python

   from indra.statements import stmts_from_json
   stmts = stmts_from_json(resp_json['statements'].values())

For those familiar with pre-assembled INDRA Statements, note that the
``supports`` and ``supported_by`` lists of the python Statement objects are not
populated.

INDRA also supports a client to this API, which is documented in detail
`elsewhere <https://indra.readthedocs.io/en/latest/modules/sources/indra_db_rest/index.html>`_,
however using that client, the above query is simply:

.. code-block:: python

   from indra.sources import indra_db_rest as idbr
   processor = idbr.get_statements(subject='MAP2K1', object='MAPK1', stmt_type='phosphorylation')
   stmts = processor.statements

Where the URL and, optionally, API key are located in a config file. A key advantage of
this client is that queries that return more than 1000 statement are paged behind
the scenes, so that all the statements which match the query are retrieved with
a single command.

Example 2:
----------

By setting the ``format`` option to ``html`` in the web API address, an HTML
document that presents a graphical user interface when displayed in a web
browser will be returned. The example below queries for statements where
BRCA1 is subject and BRCA2 is object:

.. code-block:: text

   https://db.indra.bio/statements/from_agents?subject=BRCA1&object=BRCA2&format=html

The queried statements will be loaded and you will be able to curate statements on the
level of individual evidences. Links to various source databases (depending on
availability) are available for each piece of evidence to facilitate accurate curation.
Find out more about the HTML interface in the
`HTML assembler documentation <https://indra.readthedocs.io/en/latest/modules/assemblers/html_assembler.html>`_.
For instructions on how to use it and more about the login restriction, see
the
`manual <https://indra.readthedocs.io/en/latest/tutorials/html_curation.html>`_.

Example 3:
----------

Use curl to query for any kind of interaction between SMURF2 and SMAD2,
returning at most 10 statements with 3 evidence each:

.. code-block:: bash

   curl -X GET "https://db.indra.bio/statements/from_agents?agent0=SMURF2&agent1=SMAD2&limit=10&ev_limit=3"

As above, in python this could be handled using the ``requests`` module, or with
the client:

.. code-block:: python

   import requests
   from indra.statements import stmts_from_json
   from indra.sources import indra_db_rest as idbr

   # With requests
   resp = requests.get('https://db.indra.bio/statements/from_agents',
                       params={'agent0': 'SMURF2', 'agent1': 'SMAD',
                               'api_key': 12345, 'limit': 10,
                               'ev_limit': 3})
   resp_json = resp.json()
   stmts = stmts_from_json(resp_json['statements'].values())

   # With the client
   stmts = idbr.get_statements(agents=['SMURF2', 'SMAD'], max_stmts=10,
                               ev_limit=3, simple_response=True)

Example 4:
----------

Note the use of the ``@FPLX`` suffix to denote the namespace used in identifying
the agent to query for things that inhibit MEK, using curl:

.. code-block:: bash

   curl -X GET "https://db.indra.bio/statements/from_agents?object=MEK@FPLX&type=inhibition"

Python requests:

.. code-block:: python

   resp = requests.get('https://db.indra.bio/statements/from_agents',
                       params={'agent': 'MEK@FPLX', 'type': 'inhibition',
                               'api_key': 12345})

and INDRA's client:

.. code-block:: python

   stmts = idbr.get_statements(agents=['MEK@FPLX'], stmt_type='inhibition')

Example 5:
----------

Query for a statement with the hash -1072112758478440, retrieving at most 1000
evidence, using curl:

.. code-block:: bash

   curl -X GET "https://db.indra.bio/statements/from_hash/-1072112758478440?ev_limit=1000"

or INDRA's client:

.. code-block:: python

   stmts = idbr.get_statements_by_hash([-1072112758478440], ev_limit=1000)

Note that client does not actually use the same endpoint here, but rather uses
the ``/from_hashes`` endpoint.

Example 6:
----------

Get the statements from a paper with the pmid 22174878, and
another paper with the doi 10.1259/0007-1285-34-407-693, first create the json
file, call it ``papers.json`` with the following:

.. code-block:: json

   {
     "ids": [
       {"id": "22174878", "type": "pmid"},
       {"id": "10.1259/0007-1285-34-407-693", "type": "doi"}
     ]
   }

and post it to the REST API with curl:

.. code-block:: bash

   curl -X POST "https://db.indra.bio/statements/from_papers" -d @papers.json -H "Content-Type: application/json"

or just use INDRA's client:

.. code-block:: python

   stmts = idbr.get_statments_for_paper([('pmid', '22174878'),
                                         ('doi', '10.1259/0007-1285-34-407-693')])

Example 7:
----------

Curate a Statement at the pre-assembled (pa) level for a Statement with hash
-1072112758478440, using curl:

.. code-block:: bash

   curl -X POST "https://db.indra.bio/curation/submit/-1072112758478440?api_key=12345" -d '{"tag": "correct", "text": "This Statement is OK.", "curator": "Alice"}' -H "Content-Type: application/json"

or INDRA's client:

.. code-block:: python

   idbr.submit_curation(-1072112758478440, 'correct', 'This Statement is OK.', 'Alice')

Note that submitting curations requires authentication, so the API key must be included
in the request. The client handles this for you, as long as you have the API key in
your config file.
