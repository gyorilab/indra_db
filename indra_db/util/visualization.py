import pandas as pd
from indra_db import get_db
import matplotlib.pyplot as plt
import numpy as np


def abstract_fulltext_trends_by_year():
    """Generate visualization for the abstract and full text count trends by
    year from 1970 to 2025. Between year 1780 and 1970 there are 52779 abstract
    and 226925 full text"""

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
