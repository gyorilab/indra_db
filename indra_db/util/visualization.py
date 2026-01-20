import csv
import gzip
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import text
from tqdm import tqdm

from indra_db import get_db, get_ro
from indra_db.readonly_dumping.locations import *
from indra_db.readonly_dumping.util import clean_json_loads
from indra.databases.mesh_client import get_mesh_name, is_disease

from indra_cogex.client import Neo4jClient
from indra_cogex.client.queries import get_mesh_ids_for_pmids, get_pmids_for_stmt_hash, logger



def statement_type_distribution():
    """Generate the statement distribution in terms of statements types"""
    stmt_type_counter = Counter()

    with gzip.open(unique_stmts_fpath.as_posix(), 'rt') as file:
        reader = csv.reader(file, delimiter='\t')
        for sh_str, stmt_json_str in tqdm(reader, total=47_956_726):
            stmt_json = clean_json_loads(stmt_json_str)
            stmt_type = stmt_json['type']
            stmt_type_counter[stmt_type] += 1

    df = pd.DataFrame(stmt_type_counter.most_common(),
                      columns=["stmt_type", "count"])

    plt.figure(figsize=(12, 6))

    plt.bar(df["stmt_type"], df["count"])
    plt.yscale("log")

    plt.xlabel("Statement type")
    plt.ylabel("Number of statements (log scale)")
    plt.title("Distribution of INDRA Statement Types")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def abstract_fulltext_trends_by_year():
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

    year_2025_idx = np.where(years == 2025)[0]
    if len(year_2025_idx) > 0:
        idx = year_2025_idx[0]

        axes[0].text(
            idx,
            pivot.loc[2025, "abstract"] * 1.05,
            "*",
            ha="center",
            va="bottom",
            fontsize=14
        )
        axes[1].text(
            idx,
            pivot.loc[2025, "fulltext"] * 1.05,
            "*",
            ha="center",
            va="bottom",
            fontsize=14
        )
    plt.tight_layout()
    fig.text(
        0.5,
        -0.08,
        "* Data for 2025 reflect content by May 2025 and do not represent a complete calendar year.",
        ha="center",
        fontsize=12
    )

    plt.show()


def belief_score_distribution():
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

    bin_width = df["bin_end"].iloc[0] - df["bin_start"].iloc[0]

    plt.figure(figsize=(10, 5))

    plt.bar(
        df["bin_start"],
        df["count"],
        width=bin_width,
        align="edge",
        color="#bdbdbd",
        edgecolor="black",
        linewidth=0.3
    )

    xticks = np.linspace(0, 1, 21)
    plt.xticks(xticks, [f"{x:.2f}" for x in xticks])

    plt.yscale("log")
    plt.xlabel("Belief score")
    plt.ylabel("Number of statements (log scale)")
    plt.title("Distribution of INDRA Statement Belief Scores")

    plt.tight_layout()
    plt.show()



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


def get_mesh_distribution_by_gene(gene: str, mesh_type: str = "disease", top_n: int = 30, plot: bool = True):
    """
    mesh_type: 'all' or 'disease'
    return: df with columns ['mesh_id','count'] sorted by count desc
    """
    assert mesh_type in ("all", "disease")

    stmt_hashes = get_stmt_hahes_for_gene_from_db(gene)
    logger.info(f"Got {len(stmt_hashes)} stmt_hashes for gene {gene}")

    pmid_set = set()
    client = Neo4jClient()

    for h in tqdm(list(stmt_hashes), desc=f"{gene}: collecting PMIDs"):
        pmids = get_pmids_for_stmt_hash(h, client=client)
        pmid_set.update(pmids)

    logger.info(f"Got {len(pmid_set)} unique PMIDs for gene {gene}")

    pmid_to_mesh = get_mesh_ids_for_pmids(list(pmid_set), client=client)

    all_mesh_ids = []
    for mesh_ids in tqdm(pmid_to_mesh.values(), desc=f"{gene}: collecting MeSH IDs"):
        all_mesh_ids.extend(mesh_ids)

    logger.info(f"{gene}: total mesh id occurrences: {len(all_mesh_ids)}")

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

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(labels)), vals)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Count")
        title_type = "Disease MeSH Terms" if mesh_type == "disease" else "MeSH Terms"
        plt.title(f"Top {top_n} {title_type} for {gene} (PMID associations)")
        plt.tight_layout()
        plt.show()

    return df