import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from hashlib import md5

from flask_cors import CORS
from flask_compress import Compress
from flask import url_for as base_url_for
from jinja2 import Environment, ChoiceLoader
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request
from flask import Flask, request, abort, Response, redirect, jsonify

from indra.statements import get_all_descendants, Statement, Event, Influence, Association, Unresolved
from indra.assemblers.html.assembler import (
    loader as indra_loader,
    SOURCE_INFO,
    DEFAULT_SOURCE_COLORS,
    all_sources,
)
from indra.util.statement_presentation import internal_source_mappings

from indra_db.exceptions import BadHashError
from indra_db.client.principal.curation import *
from indra_db.client.readonly import AgentJsonExpander, Query
from indra_db.util.constructors import get_ro_host

from indralab_auth_tools.auth import auth, resolve_auth, config_auth
from indralab_auth_tools.log import note_in_log, set_log_service_name, user_log_endpoint
from indra_db_service.errors import (
    HttpUserError,
    ResultTypeError,
    InvalidCredentials,
)

from indra_db_service.config import *
from indra_db_service.call_handlers import *
from indra_db_service.util import (
    sec_since,
    get_s3_client,
    gilda_ground,
    _make_english_from_meta,
)


# =========================
# A lot of config and setup
# =========================

# Get a logger, and assert the logging level.
logger = logging.getLogger("db rest api")
logger.setLevel(logging.INFO)

rev_source_mapping = {v: k for k, v in internal_source_mappings.items()}


def _source_index(source: str) -> int:
    if source in all_sources:
        return all_sources.index(source)
    else:
        logger.warning(f"Source {source} is not in the list of all sources.")
        return len(all_sources)


sources_dict = {
    category: sorted([source for source in data["sources"].keys()], key=_source_index)
    for category, data in DEFAULT_SOURCE_COLORS
}

# Set the name of this service for the usage logs.
if not TESTING["status"]:
    set_log_service_name(f"db-rest-api-{DEPLOYMENT if DEPLOYMENT else 'stable'}")
else:
    from functools import wraps
    from werkzeug.exceptions import HTTPException

    logger.warning("TESTING: No logging will be performed.")

    def note_in_log(*args, **kwargs):
        logger.info(f"Faux noting in the log: {args}, {kwargs}")

    def end_log(status):
        logger.info(f"Faux ending log with status {status}")

    def user_log_endpoint(func):
        @wraps(func)
        def run_logged(*args, **kwargs):
            logger.info(f"Faux running logged: {args}, {kwargs}")
            try:
                resp = func(*args, **kwargs)
                if isinstance(resp, str):
                    status = 200
                elif isinstance(resp, tuple) and isinstance(resp[1], int):
                    status = resp[1]
                else:
                    status = resp.status_code
            except HTTPException as e:
                end_log(e.code)
                raise e
            except Exception as e:
                logger.warning("Request experienced internal error. Returning 500.")
                logger.exception(e)
                end_log(500)
                raise
            end_log(status)
            return resp

        return run_logged


# Define a custom flask class to handle the deployment name prefix.
class MyFlask(Flask):
    def route(self, url, *args, **kwargs):
        if DEPLOYMENT is not None:
            url = f"/{DEPLOYMENT}{url}"
        flask_dec = super(MyFlask, self).route(url, **kwargs)
        return flask_dec


# Propagate the deployment name to the static path and auth URLs.
static_url_path = None
if DEPLOYMENT is not None:
    static_url_path = f"/{DEPLOYMENT}/static"
    auth.url_prefix = f"/{DEPLOYMENT}"


# Initialize the flask application (with modified static path).
app = MyFlask(__name__, static_url_path=static_url_path)


# Register the auth application, and config it if we are not testing.
app.register_blueprint(auth)
app.config["DEBUG"] = True
if not TESTING["status"]:
    SC, jwt = config_auth(app)
else:
    logger.warning("TESTING: No auth will be enabled.")

# Apply wrappers to the app that will compress responses and enable CORS.
Compress(app)
CORS(app)

# The directory path to this location (works in any file system).
HERE = Path(__file__).parent.absolute()

# Instantiate a jinja2 env.
env = Environment(
    loader=ChoiceLoader([app.jinja_loader, auth.jinja_loader, indra_loader])
)


# Overwrite url_for function in jinja to handle DEPLOYMENT prefix gracefully.
def url_for(*args, **kwargs):
    """Generate a url for a given endpoint, applying the DEPLOYMENT prefix."""
    res = base_url_for(*args, **kwargs)
    if DEPLOYMENT is not None:
        if not res.startswith(f"/{DEPLOYMENT}"):
            res = f"/{DEPLOYMENT}" + res
    return res


env.globals.update(url_for=url_for)


# Define a useful helper function.
def render_my_template(template, title, **kwargs):
    """Render a Jinja2 template wrapping in identity and other details."""
    kwargs["title"] = TITLE + ": " + title
    if not TESTING["status"]:
        verify_jwt_in_request(optional=True)
        kwargs["identity"] = get_jwt_identity()

    # Set nav elements as inactive by default.
    for nav_element in ["search", "old_search"]:
        key = f"{nav_element}_active"
        kwargs[key] = kwargs.pop(key, False)

    kwargs["simple"] = False
    return env.get_template(template).render(**kwargs)


# ==========================
# Here begins the API proper
# ==========================


@app.route("/", methods=["GET"])
def root():
    return redirect(url_for("search"), code=302)


@app.route("/healthcheck", methods=["GET"])
def i_am_alive():
    return jsonify({"status": "testing" if TESTING["status"] else "healthy"})


@app.route("/ground", methods=["GET"])
def ground():
    ag = request.args["agent"]
    res_json = gilda_ground(ag)
    return jsonify(res_json)


@app.route("/search", methods=["GET"])
@jwt_nontest_optional
@user_log_endpoint
def search():
    stmt_classes = set(get_all_descendants(Statement))
    non_biology_roots = {Event, Influence, Association, Unresolved}
    non_biology_classes = set(non_biology_roots)
    for root in non_biology_roots:
        non_biology_classes |= set(get_all_descendants(root))
    stmt_types_json = json.dumps(sorted(c.__name__ for c in stmt_classes - non_biology_classes))
    if TESTING["status"]:
        if not TESTING["deployment"]:
            vue_src = url_for("serve_indralab_vue", file="IndralabVue.umd.js")
            vue_style = url_for("serve_indralab_vue", file="IndralabVue.css")
        else:
            vue_root = TESTING["vue-root"]
            logging.info(f"Testing deployed vue files at: {vue_root}")
            vue_src = f"{vue_root}/IndralabVue.umd.js"
            vue_style = f"{vue_root}/IndralabVue.css"

    else:
        vue_src = f"{VUE_ROOT}/IndralabVue.umd.js"
        vue_style = f"{VUE_ROOT}/IndralabVue.css"
    return render_my_template(
        "search.html",
        "Search",
        main_tag_class="container-fluid",
        source_colors=DEFAULT_SOURCE_COLORS,
        source_info=SOURCE_INFO,
        search_active=True,
        vue_src=vue_src,
        vue_style=vue_style,
        stmt_types_json=stmt_types_json,
        reverse_source_mapping=rev_source_mapping,
        sources_dict=sources_dict,
    )


suf_ct_map = {".js": "application/javascript", ".css": "text/css"}


@app.route("/data-vis/<path:file_path>")
def serve_data_vis(file_path):
    full_path = HERE / "data-vis" / "dist" / file_path
    logger.info("data-vis: " + str(full_path))
    if not full_path.exists():
        return abort(404)
    with full_path.open("rb") as f:
        return Response(f.read(), content_type=suf_ct_map.get(full_path.suffix))


if TESTING["status"] and not TESTING["deployment"]:
    assert VUE_ROOT and VUE_ROOT.exists(), (
        "Local Vue package needs to be specified if no S3 deployment is used. Set "
        "INDRA_DB_API_VUE_ROOT in the environment to specify the path to the local Vue "
        "package."
    )
    assert VUE_ROOT.is_absolute(), "Cannot test API without absolute path to Vue packages."

    @app.route("/ilv/<path:file>")
    def serve_indralab_vue(file):
        # Sort out where the VUE directory is.
        full_path = VUE_ROOT / file
        if not full_path.exists():
            return abort(404, f"File {full_path.name} not found.")
        with full_path.open(mode="rb") as f:
            return Response(f.read(), content_type=suf_ct_map.get(full_path.suffix))

@app.route("/summary")
def get_summary():
    return render_my_template(
        "summary.html",
        "DB Summary",
        source_colors=DEFAULT_SOURCE_COLORS,
        source_info=SOURCE_INFO,
        reverse_source_mapping=rev_source_mapping
    )


@app.route("/summary/data/stats")
def serve_db_stats():
    """Serve database statistics for the monitor page."""
    stats_file = HERE / "static" / "data" / "db_stats.json"
    if stats_file.exists():
        with stats_file.open("r") as f:
            stats = json.load(f)

    return jsonify(stats)

@app.route("/monitor")
def get_data_explorer():
    return render_my_template("monitor.html", "Monitor")


@app.route("/monitor/data/runtime")
def serve_runtime():
    from indra_db.util.data_gatherer import S3_DATA_LOC

    s3 = get_s3_client()
    res = s3.get_object(
        Bucket=S3_DATA_LOC["bucket"], Key=S3_DATA_LOC["prefix"] + "runtimes.json"
    )
    return jsonify(json.loads(res["Body"].read()))


@app.route("/monitor/data/liststages")
def list_stages():
    from indra_db.util.data_gatherer import S3_DATA_LOC

    s3 = get_s3_client()
    res = s3.list_objects_v2(
        Bucket=S3_DATA_LOC["bucket"], Prefix=S3_DATA_LOC["prefix"], Delimiter="/"
    )

    ret = [
        k[: -len(".json")]
        for k in (e["Key"][len(S3_DATA_LOC["prefix"]) :] for e in res["Contents"])
        if k.endswith(".json") and not k.startswith("runtimes")
    ]
    print(ret)
    return jsonify(ret)


@app.route("/monitor/data/<stage>")
def serve_stages(stage):
    from indra_db.util.data_gatherer import S3_DATA_LOC

    s3 = get_s3_client()
    res = s3.get_object(
        Bucket=S3_DATA_LOC["bucket"], Key=S3_DATA_LOC["prefix"] + stage + ".json"
    )

    return jsonify(json.loads(res["Body"].read()))

@app.route("/statements", methods=["GET"])
@jwt_nontest_optional
@user_log_endpoint
def old_search():
    # Create a template object from the template file, load once
    url_base = request.url_root
    if DEPLOYMENT is not None:
        url_base = f"{url_base}{DEPLOYMENT}/"
    return render_my_template(
        "search_statements.html",
        "Search",
        message="Welcome! Try asking a question.",
        old_search_active=True,
        source_info=SOURCE_INFO,
        source_colors=DEFAULT_SOURCE_COLORS,
        endpoint=url_base,
        reverse_source_mapping=rev_source_mapping,
    )

@app.route("/<result_type>/<path:method>", methods=["GET", "POST"])
@app.route("/metadata/<result_type>/<path:method>", methods=["GET", "POST"])
@user_log_endpoint
def get_statements(result_type, method):
    """Get some statements constrained by query

    Constraints:
    - Agents involved
    - statement hashes
    - paper ids (PMC/PMID, DOI...)
    - agent json (Used in RelationSearch.vue)

    Possible result types:
    - statements
    - interactions
    - agents
    - hashes

    Possible methods:
    - from_agents
    - from_hashes
    - from_hash/<hash>
    - from_papers
    - from_paper/<paper_id>
    - from_agent_json
    - from_simple_json
    """
    if result_type not in ApiCall.valid_result_types:
        return Response("Page not found.", 404)

    note_in_log(method=method, result_type=result_type)
    note_in_log(db_host=get_ro_host("primary"))

    if method == "from_agents" and request.method == "GET":
        call = FromAgentsApiCall(env)
    elif method == "from_hashes" and request.method == "POST":
        call = FromHashesApiCall(env)
    elif method.startswith("from_hash/") and request.method == "GET":
        call = FromHashApiCall(env)
        call.web_query["hash"] = method[len("from_hash/"):]
    elif method == "from_papers" and request.method == "POST":
        call = FromPapersApiCall(env)
    elif method.startswith("from_paper/") and request.method == "GET":
        try:
            _, id_type, id_val = method.split("/")
        except Exception as e:
            logger.error(f"Failed to parse paper ID: {method}")
            logger.exception(e)
            return abort(Response("Page not found.", 404))
        call = FromPapersApiCall(env)
        call.web_query["paper_ids"] = [{"type": id_type, "id": id_val}]
    elif method == "from_agent_json" and request.method == "POST":
        call = FromAgentJsonApiCall(env)
    elif method == "from_simple_json" and request.method == "POST":
        call = FromSimpleJsonApiCall(env)
    else:
        logger.error(f"Invalid URL: {request.url}")
        return abort(404)

    return call.run(result_type=result_type)


@app.route("/expand", methods=["POST"])
@jwt_nontest_optional
def expand_meta_row():
    # Used in AgentPair.vue when an "agent pair" is expanded and that level
    # of data needs to be fetched.
    start_time = datetime.now()

    # Get the agent_json and hashes
    agent_json = request.json.get("agent_json")
    if not agent_json:
        logger.error("No agent_json provided!")
        return Response("No agent_json in request!", 400)
    stmt_type = request.json.get("stmt_type")
    hashes = request.json.get("hashes")
    logger.info(
        f"Expanding on agent_json={agent_json}, stmt_type={stmt_type}, "
        f"hashes={hashes}"
    )

    w_cur_counts = request.args.get("with_cur_counts", "False").lower() == "true"

    # Figure out authorization.
    has_medscan = False
    if not TESTING["status"]:
        user, roles = resolve_auth(request.args.copy())
        for role in roles:
            has_medscan |= role.permissions.get("medscan", False)
    else:
        api_key = request.args.get("api_key", None)
        has_medscan = api_key is not None
    logger.info(f"Auths for medscan: {has_medscan}")

    # Get the sorting parameter.
    sort_by = request.args.get("sort_by", "ev_count")

    # Get the more detailed results.
    q = AgentJsonExpander(agent_json, stmt_type=stmt_type, hashes=hashes)
    result = q.expand(sort_by=sort_by)

    # Filter out any medscan content, and construct english.
    entry_hash_lookup = defaultdict(list)
    for key, entry in result.results.copy().items():
        # Filter medscan...
        if not has_medscan:
            result.evidence_counts[key] -= entry["source_counts"].pop("medscan", 0)
            entry["total_count"] = result.evidence_counts[key]
            if not entry["source_counts"]:
                logger.warning("Censored content present. Removing it.")
                result.results.pop(key)
                result.evidence_counts.pop(key)
                continue

        # Add english...
        eng = _make_english_from_meta(entry)
        if not eng:
            logger.warning(f"English not formed for {key}:\n" f"{entry}")
        entry["english"] = eng

        # Prep curation counts
        if w_cur_counts:
            if "hashes" in entry:
                for h in entry["hashes"]:
                    entry["cur_count"] = 0
                    entry_hash_lookup[h].append(entry)
            elif "hash" in entry:
                entry["cur_count"] = 0
                entry_hash_lookup[entry["hash"]].append(entry)
            else:
                assert False, "Entry has no hash info."

        # Stringify hashes for JavaScript
        if "hashes" in entry:
            entry["hashes"] = [str(h) for h in entry["hashes"]]
        else:
            entry["hash"] = str(entry["hash"])

    if w_cur_counts:
        curations = get_curations(pa_hash=set(entry_hash_lookup.keys()))
        for cur in curations:
            for entry in entry_hash_lookup[cur["pa_hash"]]:
                entry["cur_count"] += 1

    res_json = result.json()
    res_json["relations"] = list(res_json["results"].values())
    res_json.pop("results")
    resp = Response(json.dumps(res_json), mimetype="application/json")
    logger.info(
        f"Returning expansion with {len(result.results)} meta results "
        f"that represent {result.total_evidence} total evidence. Size "
        f"is {sys.getsizeof(resp.data) / 1e6} MB after "
        f"{sec_since(start_time)} seconds."
    )
    return resp


@app.route("/query/<result_type>", methods=["GET", "POST"])
@user_log_endpoint
def get_statements_by_query_json(result_type):
    # Used in indra_db_rest in indra
    note_in_log(result_type=result_type)
    try:
        return DirectQueryApiCall(env).run(result_type)
    except ResultTypeError as e:
        return Response(f"Invalid result type: {e.result_type}", 400)


@app.route("/compile/<fmt>", methods=["POST"])
def compile_query(fmt):
    # Used in indra_db_rest in indra
    if pop_request_bool(dict(request.args), "simple", True):
        q = Query.from_simple_json(request.json)
    else:
        q = Query.from_json(request.json)
    if fmt == "json":
        return jsonify(q.to_json())
    elif fmt == "string":
        return str(q)
    else:
        return Response(f"Invalid format name: {fmt}!", 400)


@app.route("/curation", methods=["GET"])
def describe_curation():
    # Used in indra_db_rest in indra
    return redirect("/statements", code=302)


def auth_curation():
    if not TESTING["status"]:
        failure_reason = {}
        user, roles = resolve_auth(request.args.copy(), failure_reason)
        if failure_reason:
            raise InvalidCredentials(failure_reason["auth_attempted"])
    else:
        api_key = request.args.get("api_key", None)
        if api_key is None:  # any key will do for testing.
            raise InvalidCredentials("API key")
        user = None

        class MockRole:
            permissions = {"get_curations": api_key == "GET_CURATIONS"}

        roles = [MockRole()]
    return user, roles


@app.route("/curation/submit/<hash_val>", methods=["POST"])
@jwt_nontest_optional
@user_log_endpoint
def submit_curation_endpoint(hash_val):
    user, _ = auth_curation()
    if user:
        email = user.email
    else:
        email = request.json.get("email")
        if not email:
            raise HttpUserError("POST with API key requires a user email.")

    logger.info("Adding curation for statement %s." % hash_val)
    ev_hash = request.json.get("ev_hash")
    source_api = request.json.pop("source", "DB REST API")
    tag = request.json.get("tag")
    ip = request.remote_addr
    text = request.json.get("text")
    pa_json = request.json.get("pa_json")
    ev_json = request.json.get("ev_json")
    is_test = "test" in request.args
    if not is_test:
        assert tag != "test"
        try:
            dbid = submit_curation(
                hash_val, tag, email, ip, text, ev_hash, source_api, pa_json, ev_json
            )
        except BadHashError as e:
            raise HttpUserError(f"Invalid hash: {e.mk_hash}")
        res = {"result": "success", "ref": {"id": dbid}}
    else:
        res = {"result": "test passed", "ref": None}
    logger.info("Got result: %s" % str(res))
    return jsonify(res)


@app.route("/curation/list/<stmt_hash>", methods=["GET"], defaults={"src_hash": None})
@app.route("/curation/list/<stmt_hash>/<src_hash>", methods=["GET"])
@user_log_endpoint
def list_curations(stmt_hash, src_hash):
    """Public endpoint to get curations for a specific statement hash (and optionally source hash)."""
    params = {"pa_hash": stmt_hash}
    if src_hash is not None:
        params["source_hash"] = src_hash

    # Get the curations - no authentication required, curator names visible
    curations = get_curations(**params)
    return jsonify(curations), 200


@app.route("/curation/list", methods=["GET"])
@jwt_nontest_optional
@user_log_endpoint
def list_all_curations():
    """Authenticated endpoint to get all curations. Requires get_curations permission for non-anonymized data."""
    # The user needs to have an account to get curations at all.
    user, roles = auth_curation()

    # The user needs extra permission to load all curations without anonymization.
    can_load = False
    for role in roles:
        can_load |= role.permissions.get("get_curations", False)

    # Get all curations
    curations = get_curations()

    # Anonymize curator names if user doesn't have permission
    if not can_load:
        for curation in curations:
            s = curation["curator"]
            if CURATOR_SALT:
                s += CURATOR_SALT
            curation["curator"] = md5(s.encode("utf-8")).hexdigest()[:16]

    return jsonify(curations), 200


# =====================
# Define Error Handlers
# =====================


@app.errorhandler(HttpUserError)
def handle_user_error(error):
    logger.error(f"Got user error ({error.err_code}): {error.msg}")
    return error.response()
