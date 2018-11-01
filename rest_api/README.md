# INDRA Database REST API

The INDRA Database software has been developed to create and maintain a
database of text references, content, reading results, and ultimately INDRA
Statements extracted from those reading results. The software also manages
the generation and update process of cleaning, deduplicating, and finding
relations between the raw Statement extractions, into what are called
pre-assembled Statements. All INDRA Statements can be represented as JSON, 
which is the format returned by the API.

This web API provides the code necessary to support a REST service which
allows access to the pre-assembled Statements in a database. The system is
still under heavy development so capabilities are always expanding, but as
of this writing, the API supports:
- [`statements/from_agents`](#from-agents), getting Statements by agents, using
 various ids or names, by statement type (e.g. Phosphorylation), or
- [`statements/from_hash`](#from-hash) and
 [`statements/from_hashes`](#from-hashes), getting Statements by Statement 
 hash, either singly or in batches, and
- [`statements/from_papers`](#from-papers), getting Statements using the paper 
 ids from which they were extracted, and
- [`curation/submit/<level>/<hash>`](#curation) you can also curate Statements, 
 helping us improve the quality and accuracy of our content.

As mentioned, the service is changing rapidly, and this documentation may at
times be out of date. For the latest, check github or contact us.

You need the following information to access a running web service:
- The address of the web service (below shown with the placeholder `api.host`)
- An API key which needs to be sent in the header of each request to the
 service, or any other credentials that are implemented.

If you want to use our implementation of the web API, you can contact us for
the path and an API key.

The code to support the REST service can be found in `api.py`, implemented
using the Flask Python package. The means of hosting this api are left to
the user. We have had success using [Zappa](https://github.com/Miserlou/Zappa)
and AWS Lambda, and recommend it for a quick and efficient way to get the API
up and running.

## The Statement Endpoints

For all queries, an API key is required, which is passed as a parameter
`api-key` to any/all queries. Below is detailed documentation for the
different endpoints of the API that return statements (i.e. those with the root
`/statements`). All endpoints that return statements have the following
options to control the size and order of the response:
- **`max_stmts`**: Set the maximum number of statements you wish to receive.
 The REST API maximum is 1000, which cannot be overridden by this argument
 (to prevent request timeouts).
- **`ev_limit`**: The default varies, but in general the amount of Evidence 
 returned for each statement is limited. A single statement can have upwards of
 10,000 pieces of evidence, so this allows queries to be run reliably. There
 is no limitation on this value, so use with caution. Setting too high a value
 may cause a request to time out or be too large to return.
- **`best_first`**: This is set to "true" by default, so statements with the
 most evidence are returned first. These are generally the most reliable, 
 however they are also generally the most canonical. Set this parameter to 
 "false" to get statements in an arbitrary order. This can also speed up a 
 query. You may however find you get a lot of low-quality content.

The output of the statement endpoint is JSON. Specifically, the endpoints 
all return a json dict of the following form (with many made-up but reasonable
numbers):
```python
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
```
where the `"statements"` element contains a dictionary of INDRA Statement
JSONs keyed by a shallow statement hash (see [here](#from-hash) for more
details on these hashes). You can look at the
[JSON schema](https://github.com/sorgerlab/indra/blob/master/indra/resources/statements_schema.json)
on github for details on the Statement JSON. To learn more about INDRA
Statement, you can read the
[documentation](https://indra.readthedocs.io/en/latest/modules/statements.html).

<a name="from-agents"></a>
### Get Statements by agents (and type): `GET api.host/statements/from_agents`

This endpoint allows you to get statements filtering by their agents and 
the type of Statement. The query parameters are as follows:
- **subject**, **object**: The HGNC gene symbol of the subject or
 object of the Statement. **Note**: only one of each of `subject` and 
 `object` will be accepted per query.
  - Example 1: if looking for Statements where MAP2K1 is a subject
   (e.g. "What does MAP2K1 phosphorylate?"), specify `subject=MAP2K1` as a 
   query parameter
  - Example 2: if looking for Statements where MAP2K1 is the subject and
   MAPK1 is the object, add both `subject=MAP2K1` and `object=MAPK1` as
   query parameters.
  - Example 3: you can specify the agent id namespace by appending
   `@<namespace>` to the agent id in the parameter, e.g. `subject=6871@HGNC`.

- **agent***: This parameter is used if the specific role of the agent
 (subject or object) is irrelevant, or the distinction doesn't apply to the
 type of Statement of interest (e.g. Complex, Translocation, ActiveForm).
 **Note**: You can include as many `agent*` queries as you like, however you
 will only get Statements that include all agents you query, in addition to
 those queried for `subject` and `object`. Furthermore, to include multiple
 agents on our particular implementation, which uses the AWS API Gateway,
 you must include a suffix to each agent key, such as `agent0` and `agent1`,
 or else all but one agent will be stripped out. Note that you need not use
 integers, you can add any suffix you like, e.g. `agentOfDestruction=TP53`
 would be entirely valid.
  - Example 1: To obtain Statements that involve SMAD2 in any role, add
   `agent=SMAD2` to the query.
  - Example 2: As with `subject` and `object`, you can specify the
   namespace for an agent by appending `@<namespace>` to the agent's id, e.g.
   `agent=ERK@TEXT`.
  - Example 3: If you wanted to query multiple statements, you could include
   `agent0=MEK@FPLX` and `agent1=ERK@FPLX`. Note that the value of the
   integers has no real bearing on the ordering, and only serves to make the
   agents uniquely keyed. Thus `agent1=MEK@FPLX` and `agent0=ERK@FPLX` will
   give exactly the same result.

- **type**: This parameter can be used to specify what type of Statement
 of interest (e.g. Phosphorylation, Activation, Complex).
  - Example: To answer the question "Does MAP2K1 phosphorylate MAPK1?"
   the parameter `type=Phosphorylation` can be included in your query.
   Note that this field is not case sensitive, so `type=phosphorylation` would
   give the same result.

<a name="from-hash"></a> 
### Get a Statement by hash: `GET api.host/statements/from_hash/<hash>`

INDRA Statement objects have a method, `get_hash`, which produces hash from 
the content of the Statement. A shallow hash only considers the meaning of 
the statement (agents, type, modifications, etc.), whereas a deeper hash also
considers the list of evidence available for that Statement. The shallow hash
is what is used in this application, as it has the same uniqueness properties
used in deduplication. As mentioned above, the Statements are returned keyed by
their hash. In addition, if you construct a Statement in python, you may get
its hash and quickly find any evidence for that Statement in the database.

This endpoint has no extra parameters, but rather takes an extension to the 
path. So, to look up the hash 123456789, you would use 
`statements/from_hash/123456789`.

Because this only returns one statement, the default evidence limit is 
extremely generous, set to 10,000. Thus you are most likely to get all the 
evidence for a given statement this way. As described above, the evidence 
limit can also be raised, at the risk of a timed out request.

<a name="from-hashes"></a>
### Get Statements from many hashes: `POST api.host/statements/from_hashes`

Like the previous endpoint, this endpoint uses hashes to retrieve Statements,
however instead of only being allowed one at a time, a bach of 
hashes may be sent as json data. Because data is sent, this is a POST request,
even though you are in practice "getting" information. There are no special 
parameters for this endpoint. The json data should be formatted as:
```json
{"hashes": [12345, 246810]}
```
with up to 1,000 hashes given in the list.

<a name="from-papers"></a>
### Get Statements from paper ids: `POST api.host/statements/from_papers`

Using this endpoint, you can pretend you have a fleet of text extraction tools
that run in seconds! Specifically, you can get the INDRA Statements with 
evidence from a given list of papers by passing one of the ids of those papers.
As with the above method, the fact that data (paper ids) is send requires 
this to be a POST request. The papers ids should be formatted as:
```json
{"ids": [{"id": "12345", "type": "pmid"},
         {"id": "234525", "type": "tcid"},
         {"id": "PMC23423", "type": "pmcid"}]}
```
a list of dicts, each containing id type and and id value.


<a name="curation"></a>
## Curation

Because the mechanisms represented by our Statements come in large part from
automatic extractions, there can often be errors. For this reason, we always
provide the sentences from which a Statement was extracted (if we extracted
it, some of our content comes from other curated databases), as well as
provenance to lead back to the content (abstract, full text, etc.) that was
read, which allows you to use your own judgement regarding the validity of
a Statement.

If you find something wrong with a Statement, you can use this curation
endpoint to record your observation. This will not necessarily have any
immediate effect on the output, however, over time it will help us improve the
readers we use, our methods for extracting Statements from those reader
outputs, could help us filter erroneous content, and will help us improve our
pre-assembly algorithms.

### Curate statements: `POST api.host/curation/submit/<level>/<hash>`

If you wish to curate a Statement, you must first decide whether you are
curating the Statement as generally incorrect, or whether a particular
sentence supports a given Statement. This is the "level" of your curation:
- __pa__: At this level, you are curating the knowledge in a
 ***p**re-**a**ssembled* Statement. For example, if a Statement
 indicates that "differentiation binds apoptosis", regardless of whether the
 reader(s) made a valid extraction, it is clearly wrong.
- **raw**: At this level, you are curating a particular *raw* extraction, in
 other words stating that an automatic reader made an error. Even more
 explicitly, you can judge whether the sentence supports the extracted
 Statement. For example the (hypothetical) sentence "KRAS was found to actively
 inhibit BRAF" does not support the Statement "KRAS activates BRAF". As another
 example (here a grounding error), would be that the sentence "IR causes cell
 death", where IR is Ionizing Radiation does not support the extraction
 "'Insulin Receptor' causes cell death".

The two different levels also have different hashes. At the *pa* level, the 
hashes discussed [above](#from-hash) are used, as they are calculated from the 
knowledge contained in the statement, independent of the evidence. At the *raw*
level, a different hash, the `source_hash` is used, which identifies a specific
piece of evidence, without considering the Statement extracted. Within a
Statement JSON, there is a key "evidence", with a list of Evidence JSON, which
includes an entry for "source_hash":
```python
{"evidence": [{"source_hash": 98687578576598, ...}, ...], ...}
```
Once you know the level ("pa" or "raw"), and you have the correct hash (the 
shallow pre-assembly hash or the source hash), you can curate a statement by
POSTing a request with JSON data to the endpoint, as shown in the heading. The
JSON data should contain the following fields:
- **tag**: A very short word or phrase categorizing the error, for example 
 "grounding" for a grounding error.
- **text**: A brief description of what you think is most wrong.
- **curator**: Your name, initials, email, or other way to identify yourself.
 Whichever you choose, please be consistent.
 
Note that you can also indicate that a Statement is _correct_. In particular,
if you find that a Statement has some evidence that supports the Statement and
some that does not, curating examples of both is valuable. In general, flagging
correct Statements can be just as valuable as flagging incorrect Statements.

## Usage examples

The web service accepts standard GET requests, and any client that can
send such requests can be used to interact with the service. Note, however,
that access directly via a browser can be complicated by the
use of API keys which have to be included in the request header. Here we
provide usage examples with the `curl` command line tool and `python`.

In the examples, let's assume the path to the web API is
`https://api.host/`, and that the API key is `12345`.

### Curl
`curl` is a command line tool on Linux and Mac, making it a convenient tool
for making calls to this web API.

Example 1: Using `curl`, a query for "MAP2K1 phophorylates MAPK1" can be sent
as:
```bash
curl -i -H "x-api-key:12345" -X GET "https://api.host/statements/?subject=MAP2K1&object=MAPK1&type=phosphorylation"
```
which will return a JSON list of Statements. Note that if there is no API key,
you can simply remove `-H "x-api-key:12345"` from the command.

Example 2: If we instead want to know if there is any support for the
proposition there is some kind of interaction between SMURF2 and SMAD2,
your `curl` query would be:
```bash
curl -i -H "x-api-key:12345" -X GET "https://api.host/statements/?
agent=SMURF2&agent=SMAD2"
```
Or, if you are using our implementation of the REST service, you would
need to use
```bash
curl -i -H "x-api-key:12345" -X GET "https://api.host/statements/?
agent0=SMURF2&agentbondjamesbond=SMAD2"
```


### Python
Python is a convenient way to use this web API and has the important
advantage that Statements returned from the service can directly be used
by INDRA in the same environment. If again we want Statements that are
relevant for "MEK phosphorylates ERK", you can get the Statement JSONs
as follows:
```python
import requests
resp = requests.get('https://api.host/statements/',
                    headers={'x-api-key': '12345'},
                    params={'subject': 'MAP2K1',
                            'object': 'MAPK1',
                            'type': 'phosphorylation'})
stmts_json = resp.json()
```
You can also instantiate INDRA Statement objects by calling `stmts_from_json`
from `indra.statements` as :
```python
from indra.statements import stmts_from_json
stmts = stmts_from_json(stmts_json)
```
As with the `curl` example, if there is no API key required, you simply omit
that keyword argument `headers={'x-api-key': '12345'}` from the example above.
For those familiar with useing preassembled INDRA Statements, note that the
`supports` and `supported_by` lists of the python Statement objects can
have instances of `Unresolved` Statements, which are placeholders
of referenced Statements that are not included and resolved in detail in
the response.

Now suppose you want to query for interactions between SMURF2 and SMAD2 but
without specifying their specific roles.
This requires specifying two `agent*` parameters with the request which cannot
be represented with a Python `dict` as was used in the previous example.
The agent arguments can be set directly as a string in the `params`
keyword argument:
```python
resp = requests.get('https://api.host/statements/',
                    headers={'x-api-key': '12345'},
                    params=u'agent0=SMAD2&agent1=SMURF2')
```

### INDRA
Completing the circle of life, you can also access the REST API using a client
implemented in INDRA, specifically `indra.sources.indra_db_rest`. The URL and
API Key (if applicable) are configured in INDRA's config file (usually 
`~/.config/indra/config.ini`), and once added, you can get Statements by
 simply running
```python
from indra.sources import indra_db_rest as dbr
stmts = dbr.get_statements('MEK@FPLX', 'ERK@FPLX', stmt_type='Phosphorylation')
```
This API also handles more complex functionality such as implementing paging to
resolve queries that result in large amounts of content.

### Browser
You can only use a browser if there is no API key required to use the API.
If that is the case, you can simply enter the link:
`https://api.host/statements/?subject=MAP2K1&object=MAPK1`
into your browser's address bar to ge the JSON response which can
be saved.
