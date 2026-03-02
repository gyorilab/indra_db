import argparse
import logging

import csv
import gzip
import json
import os
import pickle
from itertools import combinations
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sqlalchemy import text
from tqdm import tqdm

from indra_db import get_db, get_ro
from indra_db.readonly_dumping.locations import *
from indra_db.readonly_dumping.util import clean_json_loads
from indra.databases.mesh_client import get_mesh_name, is_disease
from indra.statements import stmt_from_json, Complex
from indra.ontology.bio import bio_ontology

logger = logging.getLogger(__name__)


GROUP_MAP = {
    # 1) Human gene/protein/RNA
    "human_gene_protein": "human gene/protein",
    "human_rna": "human gene/protein",
    "human_gene_other": "human gene/protein",
    "human_gene_protein_fragment": "human gene/protein",
    "protein_family_complex": "human gene/protein",
    # 2) Non-human gene/protein
    "nonhuman_gene_protein": "nonhuman gene/protein",
    "nonhuman_gene_protein_fragment": "nonhuman gene/protein",
    # 3) Small molecules
    "small_molecule": "small molecule",
    # 4) Biological processes
    "biological_process": "biological process",
    # 5) Disease / phenotype
    "disease": "disease or phenotype",
    # 6) Experimental factor
    "experimental_factor": "experimental factor",
    # 7) Other
    "organism": "other",
    "anatomical_region": "other",
    "cellular_location": "other"
}


def statement_type_distribution_graph():
    """Generate the statement distribution in terms of statement types"""
    stmt_type_counter = Counter()

    with gzip.open(unique_stmts_fpath.as_posix(), 'rt') as file:
        reader = csv.reader(file, delimiter='\t')
        for sh_str, stmt_json_str in tqdm(reader, total=47_956_726):
            stmt_json = clean_json_loads(stmt_json_str)
            stmt_type = stmt_json["type"]
            stmt_type_counter[stmt_type] += 1

    df = pd.DataFrame(stmt_type_counter.most_common(),
                      columns=["stmt_type", "count"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["stmt_type"], df["count"])
    ax.set_yscale("log")

    ax.set_xlabel("Statement type")
    ax.set_ylabel("Number of statements (log scale)")
    ax.set_title("Distribution of INDRA Statement Types")

    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    fig.tight_layout()
    return fig, ax, df


def abstract_fulltext_trends_by_year_graph():
    """Generate visualization for the abstract and full text count trends by
    year from 1970 to 2025. Before year 1970, starting year 1780, there are 52779
    abstract and 226925 full text"""

    db = get_db("primary")

    #10 min to run
    sql = """
    SELECT
      tr.pub_year,
      tc.text_type,
      COUNT(*) AS row_count
    FROM text_ref tr
    JOIN text_content tc
      ON tc.text_ref_id = tr.id
    WHERE tr.pub_year IS NOT NULL
      AND tc.text_type IN ('abstract', 'fulltext')
    GROUP BY tr.pub_year, tc.text_type
    ORDER BY tr.pub_year;
    """

    with db._DatabaseManager__engine.connect() as connection:
        result = connection.execute(sql)
        rows = result.fetchall()
        columns = result.keys()
    df = pd.DataFrame(rows, columns=columns)

    pivot = (
        df.pivot(index="pub_year", columns="text_type", values="row_count")
        .fillna(0)
        .sort_index()
    )

    # only keep paper count after 1970+
    pivot = pivot[pivot.index >= 1970]

    years = pivot.index.values
    x = np.arange(len(years))

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 8),
        sharex=True
    )
    abstract_color = "#DD8452"
    fulltext_color = "#4C72B0"
    # --- Abstract ---
    axes[0].bar(x, pivot["abstract"], color=abstract_color)
    axes[0].set_ylabel("Count")
    axes[0].text(
        0.02, 0.90,
        "Abstract records per year",
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight="bold",
        va="top"
    )
    # --- Full text ---
    axes[1].bar(x, pivot["fulltext"], color=fulltext_color)
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Publication year")
    axes[1].text(
        0.02, 0.90,
        "Full text records per year",
        transform=axes[1].transAxes,
        fontsize=12,
        fontweight="bold",
        va="top"
    )

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45)
        ax.tick_params(axis="x", labelbottom=True)

    if len(years) > 0:
        last_year = int(years[-1])
        idx = len(years) - 1

        axes[0].text(
            idx,
            pivot.loc[last_year, "abstract"] * 1.05,
            "*",
            ha="center",
            va="bottom",
            fontsize=14
        )
        axes[1].text(
            idx,
            pivot.loc[last_year, "fulltext"] * 1.05,
            "*",
            ha="center",
            va="bottom",
            fontsize=14
        )
    plt.tight_layout()

    return fig, axes, df, pivot


def belief_score_distribution_graph():
    """
    Load belief scores from `belief_scores_pkl_fpath` and build a log-scaled bar plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object save/embed via `fig.savefig(...)`
    ax : matplotlib.axes.Axes
        The axes object for further edits/annotations
    df : pandas.DataFrame
        Columns: bin_start, bin_end, count
    """

    with open(belief_scores_pkl_fpath, "rb") as f:
        belief_scores = pickle.load(f)

    n_bins = 100
    hist = np.zeros(n_bins, dtype=np.int64)

    for belief in belief_scores.values():
        if belief is None:
            continue

        idx = int(belief * n_bins)
        if idx == n_bins:
            idx = n_bins - 1

        hist[idx] += 1
    bin_width = 1.0 / n_bins

    df = pd.DataFrame({
        "bin_start": np.arange(len(hist)) * bin_width,
        "bin_end": (np.arange(len(hist)) + 1) * bin_width,
        "count": hist
    })

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        df["bin_start"],
        df["count"],
        width=bin_width,
        align="edge",
        color="#bdbdbd",
        edgecolor="black",
        linewidth=0.3
    )

    xticks = np.linspace(0, 1, 21)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.2f}" for x in xticks])

    ax.set_yscale("log")
    ax.set_xlabel("Belief score")
    ax.set_ylabel("Number of statements (log scale)")
    ax.set_title("Distribution of INDRA Statement Belief Scores")
    fig.tight_layout()

    return fig, ax, df



def get_stmt_hahes_for_gene_from_db(gene):
    readonly = get_ro("primary")
    sql = text("""
    SELECT DISTINCT mk_hash
    FROM readonly.agent_interactions
    WHERE agent_json::text LIKE :g;
    """)

    with readonly._DatabaseManager__engine.connect() as connection:
        result = connection.execute(sql, {"g": f'%"{gene}"%'})
        rows = result.fetchall()

    hashes = [r[0] for r in rows]
    stmt_hashes = set(hashes)
    return stmt_hashes


def get_mesh_ids_for_stmt_hashes(stmt_hashes, *, db):
    """
       Return {mk_hash: ["D000123", ...]} by joining:
         readonly.fast_raw_pa_link (mk_hash, id=raw_stmt_id)
         readonly.raw_stmt_mesh_terms (sid=raw_stmt_id, mesh_num)
    """
    stmt_hashes = list(map(int, stmt_hashes))

    q = (
        db.session.query(db.FastRawPaLink.mk_hash, db.RawStmtMeshTerms.mesh_num)
        .join(db.RawStmtMeshTerms,
              db.RawStmtMeshTerms.sid == db.FastRawPaLink.id)
        .filter(db.FastRawPaLink.mk_hash.in_(stmt_hashes))
        .distinct()
    )

    out = defaultdict(set)
    for h, mesh_num in q.all():
        if mesh_num is not None:
            out[int(h)].add(f"D{int(mesh_num):06d}")

    return {h: sorted(s) for h, s in out.items()}


def mesh_distribution_by_gene(gene: str, mesh_type: str = "disease", top_n: int = 30, plot: bool = True):
    """
    mesh_type: 'all' or 'disease'
    return: df with columns ['mesh_id','count'] sorted by count desc
    """
    assert mesh_type in ("all", "disease")
    db = get_ro('primary')

    stmt_hashes = get_stmt_hahes_for_gene_from_db(gene)
    logger.info(f"{gene}: total hashes: {len(stmt_hashes)}")

    mesh_by_hash = get_mesh_ids_for_stmt_hashes(stmt_hashes, db=db)

    all_mesh_ids = []
    for ids in mesh_by_hash.values():
        all_mesh_ids.extend(ids)

    logger.info(f"{gene} total mesh id occurrences: {len(all_mesh_ids)}")

    if mesh_type == "disease":
        mesh_ids_used = [m for m in all_mesh_ids if is_disease(m)]
    else:
        mesh_ids_used = all_mesh_ids

    counts = Counter(mesh_ids_used)
    df = pd.DataFrame(counts.items(), columns=["mesh_id", "count"]).sort_values("count", ascending=False)

    if plot:
        top = df.head(top_n)

        labels = [get_mesh_name(m) or m for m in top["mesh_id"]][::-1]
        vals = top["count"].values[::-1]

        fig = plt.figure(figsize=(10, 8))
        plt.barh(range(len(labels)), vals)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Count")
        title_type = "Disease MeSH Terms" if mesh_type == "disease" else "MeSH Terms"
        plt.title(f"Top {top_n} {title_type} for {gene} (PMID associations)")
        plt.tight_layout()
        return df, fig

    return df, None

def generate_entity_pair_stats(output_path):
    """Generate entity pair statistics from unique statements."""
    type_pair_counter = Counter()

    logger.info("Counting entity pairs...")
    with gzip.open(unique_stmts_fpath.as_posix(), "rt") as fi:
        reader = csv.reader(fi, delimiter="\t")
        for _, sjs in tqdm(reader, total=47_956_726, desc="Entity Pairs"):
            try:
                stmt = stmt_from_json(clean_json_loads(sjs))
            except Exception:
                continue

            agent_types = []
            for agent in stmt.agent_list():
                if agent is None:
                    continue
                ns, _id = agent.get_grounding()
                if not (ns and _id):
                    continue
                t = bio_ontology.get_type(ns, _id) or "ungrounded_or_unknown"
                agent_types.append(t)

            n = len(agent_types)
            if n < 2:
                continue

            if isinstance(stmt, Complex) or n > 2:
                uniq = sorted(set(agent_types))
                for a, b in combinations(uniq, 2):
                    type_pair_counter[(a, b)] += 1
                    type_pair_counter[(b, a)] += 1
                counts = Counter(agent_types)
                for t, k in counts.items():
                    if k >= 2:
                        type_pair_counter[(t, t)] += 1

            else:
                type_pair_counter[(agent_types[0], agent_types[1])] += 1

    def group_of(t):
        return GROUP_MAP.get(t)

    directed = Counter()
    for (t1, t2), c in type_pair_counter.items():
        g1, g2 = group_of(t1), group_of(t2)
        if g1 is None or g2 is None:
            continue
        directed[(g1, g2)] += c

    data = [{"source": a, "target": b, "value": c}
            for (a, b), c in directed.most_common()]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved directed entity pair stats to {output_path}")


def compute_unique_stmt_stats():
    """Scan unique statements and return counts for total + grounding."""
    counts = {
        "unique_statement": 0,
        "grounding_full": 0,
        "grounding_partial": 0,
        "grounding_ungrounded": 0,
    }

    logger.info("Scanning unique statements for total + grounding stats...")
    with gzip.open(unique_stmts_fpath.as_posix(), "rt") as file:
        reader = csv.reader(file, delimiter="\t")
        for _, stmt_json_str in tqdm(reader, total=47_956_726, desc="Unique stmts"):
            counts["unique_statement"] += 1

            stmt = stmt_from_json(clean_json_loads(stmt_json_str))
            real_agents = stmt.real_agent_list()
            if not real_agents:
                continue

            grounded_flags = [
                bool(set(a.db_refs.keys()) - {"TEXT", "TEXT_NORM"})
                for a in real_agents
            ]
            num_grounded = sum(grounded_flags)
            num_agents = len(real_agents)

            if num_grounded == num_agents:
                counts["grounding_full"] += 1
            elif num_grounded == 0:
                counts["grounding_ungrounded"] += 1
            else:
                counts["grounding_partial"] += 1

    return counts

def grounding_distribution_graph(stats=None):
    """
    Plot grounding distribution using precomputed stats.
    stats can be a dict with grounding_full/partial/ungrounded.
    """
    if stats is None:
        raise ValueError("grounding_distribution_graph requires stats dict now.")

    full = stats.get("grounding_full", 0)
    ungrounded = stats.get("grounding_ungrounded", 0)
    partial = stats.get("grounding_partial", 0)

    total = full + ungrounded + partial
    if total == 0:
        logger.warning("No statements found for grounding distribution.")
        return None

    sizes = [full / total * 100, ungrounded / total * 100, partial / total * 100]
    labels = ["Fully Grounded", "Ungrounded", "Partially Grounded"]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title("Grounding Distribution", fontsize=12)
    plt.tight_layout()
    return fig




def evidence_vs_statement_graph():
    """
    Plot the distribution of statements by total evidence count.

    X-axis: total evidence count per statement (log scale).
    Y-axis: number of statements with that evidence count (log scale).

    Returns
    -------
    matplotlib.figure.Figure
        The created matplotlib Figure object.
    """
    with open(source_counts_fpath.as_posix(), 'rb') as f:
        source_count = pickle.load(f)
    with open(belief_scores_pkl_fpath, "rb") as f:
        belief_scores = pickle.load(f)
    logger.info("belief scores and source counts loaded ")

    total_evs = {h: sum(v.values()) for h, v in source_count.items()}
    freq = Counter(total_evs.values())

    x = sorted(freq)
    y = [freq[k] for k in x]

    beliefs_by_evcount = defaultdict(list)

    for h, evc in total_evs.items():
        b = belief_scores.get(h)
        if b is not None:
            beliefs_by_evcount[evc].append(b)

    fig = plt.figure()
    plt.scatter(x, y, s=8, c="black")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Evidence")
    plt.ylabel("Number of statements")
    plt.tight_layout()
    return fig


def pmid_vs_statement_graph():
    """
        Plot the distribution of statements by unique PubMed ID (PMID) count.

        X-axis: number of unique PMIDs per statement (log scale).
        Y-axis: number of statements with that PMID count (log scale).

        Returns
        -------
        matplotlib.figure.Figure
            The created matplotlib Figure object.
        """
    pmids_by_hash = defaultdict(set)

    with gzip.open(processed_stmts_fpath.as_posix(), "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for sh_str, stmt_json_str in tqdm(reader, total=47_956_726,
                                          desc="Collecting PMIDs"):
            stmt_json = clean_json_loads(stmt_json_str)
            h = stmt_json.get("matches_hash")
            if not h:
                continue
            ev = stmt_json.get('evidence')
            tr = ev[0].get('text_refs')
            if tr:
                pmid = tr.get("PMID")
            if pmid:
                pmids_by_hash[h].add(str(pmid))

    pmid_counts = {h: len(s) for h, s in pmids_by_hash.items()}
    freq = Counter(pmid_counts.values())
    x = np.array(sorted(k for k in freq.keys()))
    y = np.array([freq[k] for k in x])

    fig = plt.figure()
    plt.scatter(x, y, s=7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Unique PMIDs per statement")
    plt.ylabel("Number of statements")
    plt.tight_layout()
    plt.show()
    return fig

def compute_total_evidence():
    with open(source_counts_fpath.as_posix(), "rb") as f:
        source_count = pickle.load(f)
    return sum(sum(v.values()) for v in source_count.values())

def generate_db_stats(json_path):
    """Generate db_stats.json with counts of text content types + unique stmt stats."""
    db = get_db("primary")
    sql = text(
        "select count(*) as count, text_type from text_content "
        "where text_type in ('abstract', 'fulltext', 'title') group by text_type;"
    )

    logger.info("Generating DB stats...")
    stats = {}

    with db._DatabaseManager__engine.connect() as connection:
        result = connection.execute(sql)
        for row in result:
            stats[row[1]] = int(row[0])

    unique_counts = compute_unique_stmt_stats()
    stats.update(unique_counts)
    stats["total_evidence"] = compute_total_evidence()

    denom = (
                    stats["grounding_full"]
                    + stats["grounding_partial"]
                    + stats["grounding_ungrounded"]
            ) or 1
    stats["grounding_full_pct"] = stats["grounding_full"] / denom * 100
    stats["grounding_partial_pct"] = stats["grounding_partial"] / denom * 100
    stats["grounding_ungrounded_pct"] = stats["grounding_ungrounded"] / denom * 100

    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved db stats to {json_path}")

def generate_source_stats(output_dir):
    """
    Generate simple source statistics JSON file (src -> count).
    """
    db_stats = []

    ro = get_ro("primary")
    sql = text("""
               SELECT COALESCE(f.src, r.src) AS src,
                      COUNT(*) AS count
               FROM readonly.fast_raw_pa_link AS f
                   LEFT JOIN readonly.raw_stmt_src AS r
               ON f.id = r.sid
               GROUP BY COALESCE (f.src, r.src)
               ORDER BY count DESC;
               """)

    logger.info("Querying source statistics...")
    with ro._DatabaseManager__engine.connect() as connection:
        result = connection.execute(sql)
        for row in result:
            db_stats.append({"src": row[0], "count": row[1]})
    with open(output_dir, 'w') as f:
        json.dump(db_stats, f, indent=2)
    logger.info(f"Saved db stats to {output_dir}")


def generate_all_plots(output_dir, refresh=False, stats_dir=None):

    """Generate all static plots + stats json for the monitor page."""
    matplotlib.use('Agg')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")

    if stats_dir is None or stats_dir is None:
        static_dir = os.path.dirname(output_dir)
        stats_dir = os.path.join(static_dir, "data")

    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
        logger.info(f"Created directory: {stats_dir}")

    logger.info(f"Saving plots to {output_dir} (Refresh={refresh})")
    logger.info(f"Saving stats to {stats_dir} (Refresh={refresh})")

    def should_skip(filepath):
        if os.path.exists(filepath) and not refresh:
            logger.info(f"Skipping {os.path.basename(filepath)} (already exists).")
            return True
        return False

    db_stats_path = os.path.join(stats_dir, "db_stats.json")
    if not should_skip(db_stats_path):
        generate_db_stats(db_stats_path)
    with open(db_stats_path, "r") as f:
        db_stats = json.load(f)

    # ---------- plots ----------
    tasks = [
        ('stmt_type_dist.png', statement_type_distribution_graph, "Statement Type Distribution"),
        ('belief_score_dist.png', belief_score_distribution_graph, "Belief Score Distribution"),
        ('paper_trends.png', abstract_fulltext_trends_by_year_graph, "Paper Trends"),
        ('grounding_dist.png', lambda: grounding_distribution_graph(db_stats), "Grounding Distribution"),
        ('evidence_vs_stmt.png', evidence_vs_statement_graph, "Evidence vs Statement"),
        ('pmid_vs_stmt.png', pmid_vs_statement_graph, "PMID vs Statement"),
    ]

    for filename, func, description in tasks:
        save_path = os.path.join(output_dir, filename)
        if should_skip(save_path):
            continue

        logger.info(f"Generating {description}...")
        try:
            result = func()
            fig = result[0] if isinstance(result, tuple) else result
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved {save_path}")
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error generating {description}: {e}")

    # ---------- gene mesh plots ----------
    target_genes = ['EGFR', 'TP53', 'MAPT', 'SNCA', 'BRCA1', 'BRAF']
    logger.info(f"Generating MeSH distributions for genes: {target_genes}")

    for gene in target_genes:
        filename = f'mesh_dist_{gene}.png'
        save_path = os.path.join(output_dir, filename)
        if should_skip(save_path):
            continue

        logger.info(f"Processing {gene}...")
        try:
            df, fig = mesh_distribution_by_gene(gene, mesh_type="disease", plot=True)
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved {save_path}")
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error generating MeSH plot for {gene}: {e}")

    # ---------- stats json ----------
    stats_tasks = [
        ("source_num.json", generate_source_stats, "Source Stats"),
        ("entity_pairs.json", generate_entity_pair_stats, "Entity Pair Stats"),
    ]

    for filename, func, description in stats_tasks:
        save_path = os.path.join(stats_dir, filename)
        if should_skip(save_path):
            continue

        logger.info(f"Generating {description}...")
        try:
            func(save_path)
            logger.info(f"Saved {save_path}")
        except Exception as e:
            logger.error(f"Error generating {description}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plots for INDRA DB Monitor")

    # Path: ../../indra_db_service/static/plots
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    default_path = os.path.join(root_dir, 'indra_db_service', 'static', 'plots')

    parser.add_argument('--output-dir', '-o', default=default_path,
                        help=f'Directory to save plots. Default: {default_path}')


    parser.add_argument('--refresh', '-r', action='store_true',
                        help='Force regenerate all plots even if they exist.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    generate_all_plots(args.output_dir, refresh=args.refresh)

