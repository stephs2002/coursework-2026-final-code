"""
07_clean_label_network.py

Purpose:
    Build a cleaned shared-label network from manually commented label data.

Input:
    outputs/tables/artist_label_summary_with_comments.csv

This file should be based on artist_label_summary.csv produced by:
    06_label_feasibility_check.py

Expected columns:
    artist
    spotify_id
    normalized_label
    commentary
    n_releases_on_label
    raw_label_examples
    release_years

Optional input:
    outputs/tables/nodes_clean.csv

If nodes_clean.csv is available, the script keeps the same node set as the
genre/collaboration pipeline, including artists without usable label ties.
This is important for comparing the three networks on the same set of artists.

Default cleaning logic:
    Included in the main clean label network:
        - commentary == "real label"
        - commentary starts with "edge case"
        - commentary starts with "probably the same"

    Excluded by default:
        - self-released
        - default name by distrokid
        - unknown
        - radio/broadcasting
        - "bear on a bicycle..." case, unless explicitly included

Why self-released labels are excluded by default:
    They may be meaningful descriptively, but they usually do not represent
    shared institutional ties between artists. They are therefore excluded from
    the main institutional shared-label network unless --include-self-released
    is used.

Main outputs:
    outputs/tables/artist_label_summary_clean.csv
    outputs/tables/artist_label_summary_excluded.csv
    outputs/tables/top_labels_clean.csv
    outputs/tables/shared_label_edges_clean.csv
    outputs/tables/label_network_node_metrics_clean.csv
    outputs/tables/label_network_summary_clean.csv
    outputs/networks/shared_label_network_clean.graphml
    outputs/figures/shared_label_network_clean.png

Typical run:
    python 07_clean_label_network.py

If you want a broader version including self-releases:
    python 07_clean_label_network.py --include-self-released --suffix broad

If you want to include the "bear on a bicycle..." doubtful case:
    python 07_clean_label_network.py --include-bear-case
"""

import argparse
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Helper functions
# ---------------------------------------------------------

def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def save_csv_safely(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def normalize_commentary(x):
    if x is None or pd.isna(x):
        return ""
    return str(x).strip().lower()


def should_include_label(commentary, include_self_released=False, include_bear_case=False):
    """
    Decides whether an artist-label row should be included in the clean network.

    This function is intentionally transparent and easy to modify.
    """
    c = normalize_commentary(commentary)

    if c == "real label":
        return True

    if c.startswith("edge case"):
        return True

    if c.startswith("probably the same"):
        return True

    if include_self_released and c == "self-released":
        return True

    if include_bear_case and c.startswith("bear on a bicycle"):
        return True

    return False


def exclusion_reason(commentary):
    c = normalize_commentary(commentary)

    if c == "self-released":
        return "excluded_self_released"
    if c == "default name by distrokid":
        return "excluded_distrokid_default"
    if c == "unknown":
        return "excluded_unknown"
    if c == "radio/broadcasting":
        return "excluded_radio_broadcasting"
    if c.startswith("bear on a bicycle"):
        return "excluded_doubtful_label_case"
    if c == "":
        return "excluded_missing_commentary"

    return "excluded_other_commentary"


def make_suffix(suffix):
    suffix = str(suffix).strip()
    if suffix:
        if not suffix.startswith("_"):
            suffix = "_" + suffix
    return suffix


# ---------------------------------------------------------
# 2. Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-labels",
        default="outputs/tables/artist_label_summary_with_comments.csv",
        help="Path to manually commented artist-label summary."
    )

    parser.add_argument(
        "--input-nodes",
        default="outputs/tables/nodes_clean.csv",
        help="Path to nodes_clean.csv. If unavailable, nodes are inferred from label data."
    )

    parser.add_argument(
        "--output-tables-dir",
        default="outputs/tables",
        help="Directory for output tables."
    )

    parser.add_argument(
        "--output-figures-dir",
        default="outputs/figures",
        help="Directory for figures."
    )

    parser.add_argument(
        "--output-networks-dir",
        default="outputs/networks",
        help="Directory for GraphML network files."
    )

    parser.add_argument(
        "--suffix",
        default="clean",
        help="Suffix for output files. Default: clean."
    )

    parser.add_argument(
        "--include-self-released",
        action="store_true",
        help="Include self-released labels in the network."
    )

    parser.add_argument(
        "--include-bear-case",
        action="store_true",
        help="Include the doubtful bear-on-a-bicycle/endorphins label case."
    )

    parser.add_argument(
        "--min-shared-labels",
        type=int,
        default=1,
        help="Minimum number of shared labels required to create an edge."
    )

    args = parser.parse_args()

    input_labels_path = Path(args.input_labels)
    input_nodes_path = Path(args.input_nodes)

    output_tables_dir = Path(args.output_tables_dir)
    output_figures_dir = Path(args.output_figures_dir)
    output_networks_dir = Path(args.output_networks_dir)

    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    output_networks_dir.mkdir(parents=True, exist_ok=True)

    suffix = make_suffix(args.suffix)

    if not input_labels_path.exists():
        raise FileNotFoundError(f"Input label file not found: {input_labels_path}")

    labels = pd.read_csv(input_labels_path)

    required_columns = [
        "artist",
        "spotify_id",
        "normalized_label",
        "commentary",
        "n_releases_on_label",
        "raw_label_examples",
        "release_years"
    ]

    missing = [c for c in required_columns if c not in labels.columns]
    if missing:
        raise ValueError(f"Input label file is missing columns: {missing}")

    labels["commentary_normalized"] = labels["commentary"].apply(normalize_commentary)

    labels["include_in_clean_label_network"] = labels["commentary"].apply(
        lambda x: should_include_label(
            x,
            include_self_released=args.include_self_released,
            include_bear_case=args.include_bear_case
        )
    )

    labels["cleaning_decision"] = labels["include_in_clean_label_network"].map({
        True: "included",
        False: "excluded"
    })

    labels["exclusion_reason"] = labels.apply(
        lambda row: "" if row["include_in_clean_label_network"] else exclusion_reason(row["commentary"]),
        axis=1
    )

    included = labels[labels["include_in_clean_label_network"]].copy()
    excluded = labels[~labels["include_in_clean_label_network"]].copy()

    # -----------------------------------------------------
    # 2.1. Load node set
    # -----------------------------------------------------

    if input_nodes_path.exists():
        nodes = pd.read_csv(input_nodes_path)

        if "query_name" not in nodes.columns:
            raise ValueError("nodes_clean.csv must contain column: query_name")

        node_names = list(nodes["query_name"])

    else:
        print(f"Warning: nodes file not found: {input_nodes_path}")
        print("Inferring nodes from artist_label_summary_with_comments.csv.")
        nodes = pd.DataFrame({
            "query_name": sorted(labels["artist"].dropna().unique())
        })
        node_names = list(nodes["query_name"])

    # -----------------------------------------------------
    # 2.2. Save clean / excluded artist-label rows
    # -----------------------------------------------------

    clean_path = output_tables_dir / f"artist_label_summary{suffix}.csv"
    excluded_path = output_tables_dir / f"artist_label_summary_excluded{suffix}.csv"

    save_csv_safely(included, clean_path)
    save_csv_safely(excluded, excluded_path)

    # -----------------------------------------------------
    # 2.3. Top clean labels
    # -----------------------------------------------------

    if len(included) > 0:
        top_labels = (
            included
            .groupby("normalized_label")
            .agg(
                n_artists=("artist", "nunique"),
                n_artist_label_pairs=("artist", "count"),
                n_releases=("n_releases_on_label", lambda x: sum(safe_int(v) for v in x)),
                artists=("artist", lambda x: "; ".join(sorted(set([str(v) for v in x if pd.notna(v)]))[:30])),
                raw_label_examples=("raw_label_examples", lambda x: "; ".join(sorted(set([str(v) for v in x if pd.notna(v)]))[:15])),
                commentaries=("commentary", lambda x: "; ".join(sorted(set([str(v) for v in x if pd.notna(v)]))[:10]))
            )
            .reset_index()
            .sort_values(["n_artists", "n_releases"], ascending=False)
        )
    else:
        top_labels = pd.DataFrame(columns=[
            "normalized_label",
            "n_artists",
            "n_artist_label_pairs",
            "n_releases",
            "artists",
            "raw_label_examples",
            "commentaries"
        ])

    top_labels_path = output_tables_dir / f"top_labels{suffix}.csv"
    save_csv_safely(top_labels, top_labels_path)

    # -----------------------------------------------------
    # 2.4. Build artist-label dictionary
    # -----------------------------------------------------

    artist_to_labels = defaultdict(set)
    artist_to_label_release_counts = defaultdict(dict)
    artist_to_raw_examples = defaultdict(dict)

    for _, row in included.iterrows():
        artist = row["artist"]
        label = str(row["normalized_label"]).strip()

        if not artist or not label:
            continue

        artist_to_labels[artist].add(label)

        artist_to_label_release_counts[artist][label] = (
            artist_to_label_release_counts[artist].get(label, 0)
            + safe_int(row["n_releases_on_label"])
        )

        artist_to_raw_examples[artist][label] = row.get("raw_label_examples", "")

    # -----------------------------------------------------
    # 2.5. Build shared-label edges
    # -----------------------------------------------------

    edge_records = []

    for a, b in combinations(node_names, 2):
        labels_a = artist_to_labels.get(a, set())
        labels_b = artist_to_labels.get(b, set())

        shared = labels_a.intersection(labels_b)
        union = labels_a.union(labels_b)

        if len(shared) >= args.min_shared_labels:
            label_jaccard = len(shared) / len(union) if len(union) > 0 else 0

            # Optional supplementary weight:
            # sum of minimum release counts across shared labels.
            min_release_count_sum = 0
            for lab in shared:
                count_a = artist_to_label_release_counts[a].get(lab, 0)
                count_b = artist_to_label_release_counts[b].get(lab, 0)
                min_release_count_sum += min(count_a, count_b)

            edge_records.append({
                "source": a,
                "target": b,
                "shared_label_count": len(shared),
                "shared_labels": "; ".join(sorted(shared)),
                "source_label_count": len(labels_a),
                "target_label_count": len(labels_b),
                "label_jaccard_similarity": round(label_jaccard, 4),
                "shared_label_min_release_count_sum": min_release_count_sum
            })

    shared_edges = pd.DataFrame(edge_records)

    if len(shared_edges) > 0:
        shared_edges = shared_edges.sort_values(
            ["shared_label_count", "label_jaccard_similarity", "shared_label_min_release_count_sum"],
            ascending=False
        )

    shared_edges_path = output_tables_dir / f"shared_label_edges{suffix}.csv"
    save_csv_safely(shared_edges, shared_edges_path)

    # -----------------------------------------------------
    # 2.6. Build graph
    # -----------------------------------------------------

    G = nx.Graph()

    for _, row in nodes.iterrows():
        artist = row["query_name"]
        labels_for_artist = sorted(artist_to_labels.get(artist, set()))

        G.add_node(
            artist,
            spotify_id=row.get("spotify_id", ""),
            spotify_name=row.get("spotify_name", artist),
            genres=row.get("genres", "") if "genres" in row and not pd.isna(row.get("genres", "")) else "",
            n_clean_labels=len(labels_for_artist),
            clean_labels="; ".join(labels_for_artist),
            popularity=safe_int(row.get("popularity", 0)),
            followers=safe_int(row.get("followers", 0))
        )

    for _, row in shared_edges.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            weight=safe_int(row["shared_label_count"], default=1),
            shared_labels=row["shared_labels"],
            label_jaccard_similarity=safe_float(row["label_jaccard_similarity"]),
            shared_label_min_release_count_sum=safe_int(row["shared_label_min_release_count_sum"])
        )

    # -----------------------------------------------------
    # 2.7. Network metrics
    # -----------------------------------------------------

    if G.number_of_nodes() > 1:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, weight=None)
    else:
        degree_centrality = {node: 0 for node in G.nodes()}
        betweenness_centrality = {node: 0 for node in G.nodes()}

    weighted_degree = dict(G.degree(weight="weight"))

    modularity_score = None

    if G.number_of_edges() > 0:
        communities = nx.algorithms.community.greedy_modularity_communities(
            G,
            weight="weight"
        )

        try:
            modularity_score = nx.algorithms.community.quality.modularity(
                G,
                communities,
                weight="weight"
            )
        except Exception:
            modularity_score = None

        community_map = {}
        for community_id, community_nodes in enumerate(communities, start=1):
            for node in community_nodes:
                community_map[node] = community_id
    else:
        community_map = {node: 1 for node in G.nodes()}

    node_metrics = []

    for node in G.nodes():
        node_metrics.append({
            "artist": node,
            "degree": G.degree(node),
            "weighted_degree": round(weighted_degree.get(node, 0), 4),
            "degree_centrality": round(degree_centrality.get(node, 0), 4),
            "betweenness_centrality": round(betweenness_centrality.get(node, 0), 4),
            "community": community_map.get(node),
            "n_clean_labels": G.nodes[node]["n_clean_labels"],
            "clean_labels": G.nodes[node]["clean_labels"],
            "genres": G.nodes[node]["genres"],
            "popularity": G.nodes[node]["popularity"],
            "followers": G.nodes[node]["followers"]
        })

    node_metrics = pd.DataFrame(node_metrics)

    if len(node_metrics) > 0:
        node_metrics = node_metrics.sort_values(
            ["degree_centrality", "weighted_degree", "betweenness_centrality"],
            ascending=False
        )

    node_metrics_path = output_tables_dir / f"label_network_node_metrics{suffix}.csv"
    save_csv_safely(node_metrics, node_metrics_path)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_isolates = len(list(nx.isolates(G)))
    largest_component_size = max((len(c) for c in nx.connected_components(G)), default=0)

    n_artists = len(node_names)
    n_artists_with_any_label_rows = int(labels["artist"].nunique())
    n_artists_with_clean_labels = int(included["artist"].nunique()) if len(included) > 0 else 0

    summary = pd.DataFrame([{
        "network_version": args.suffix,
        "n_artists": n_artists,
        "n_artist_label_rows_total": len(labels),
        "n_artist_label_rows_included": len(included),
        "n_artist_label_rows_excluded": len(excluded),
        "n_artists_with_any_label_rows": n_artists_with_any_label_rows,
        "n_artists_with_clean_labels": n_artists_with_clean_labels,
        "clean_label_artist_coverage_percent": round(n_artists_with_clean_labels / n_artists * 100, 2) if n_artists else 0,
        "n_unique_clean_labels": int(included["normalized_label"].nunique()) if len(included) > 0 else 0,
        "n_shared_label_edges": len(shared_edges),
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": round(nx.density(G), 4) if n_nodes > 1 else 0,
        "n_connected_components": nx.number_connected_components(G) if n_nodes > 0 else 0,
        "largest_component_size": largest_component_size,
        "largest_component_share": round(largest_component_size / n_nodes, 4) if n_nodes else 0,
        "n_isolates": n_isolates,
        "isolate_share": round(n_isolates / n_nodes, 4) if n_nodes else 0,
        "modularity": round(modularity_score, 4) if modularity_score is not None else None,
        "min_shared_labels": args.min_shared_labels,
        "include_self_released": args.include_self_released,
        "include_bear_case": args.include_bear_case
    }])

    summary_path = output_tables_dir / f"label_network_summary{suffix}.csv"
    save_csv_safely(summary, summary_path)

    # Exclusion summary.
    exclusion_summary = (
        excluded
        .groupby(["exclusion_reason", "commentary_normalized"], dropna=False)
        .agg(
            n_rows=("artist", "count"),
            n_artists=("artist", "nunique"),
            n_labels=("normalized_label", "nunique"),
            example_artists=("artist", lambda x: "; ".join(sorted(set([str(v) for v in x if pd.notna(v)]))[:15])),
            example_labels=("normalized_label", lambda x: "; ".join(sorted(set([str(v) for v in x if pd.notna(v)]))[:15]))
        )
        .reset_index()
        .sort_values("n_rows", ascending=False)
        if len(excluded) > 0 else
        pd.DataFrame(columns=[
            "exclusion_reason",
            "commentary_normalized",
            "n_rows",
            "n_artists",
            "n_labels",
            "example_artists",
            "example_labels"
        ])
    )

    exclusion_summary_path = output_tables_dir / f"label_cleaning_exclusion_summary{suffix}.csv"
    save_csv_safely(exclusion_summary, exclusion_summary_path)

    # -----------------------------------------------------
    # 2.8. Save network and figure
    # -----------------------------------------------------

    graphml_path = output_networks_dir / f"shared_label_network{suffix}.graphml"
    nx.write_graphml(G, graphml_path)

    plt.figure(figsize=(14, 10))

    pos = nx.spring_layout(G, seed=42, weight="weight")

    node_sizes = [
        200 + G.nodes[node]["popularity"] * 10
        for node in G.nodes()
    ]

    node_colors = [
        community_map.get(node, 0)
        for node in G.nodes()
    ]

    edge_widths = [
        1 + G[u][v]["weight"]
        for u, v in G.edges()
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.85
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.45
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8
    )

    plt.title(
        f"Clean Shared Label Network among Selected Artists",
        fontsize=14
    )

    plt.axis("off")
    plt.tight_layout()

    figure_path = output_figures_dir / f"shared_label_network{suffix}.png"
    plt.savefig(figure_path, dpi=300)
    plt.show()

    # -----------------------------------------------------
    # 2.9. Print summary
    # -----------------------------------------------------

    print("\nClean label network completed.")
    print("\nSummary:")
    print(summary)

    print("\nTop clean labels:")
    if len(top_labels) > 0:
        print(top_labels.head(15)[[
            "normalized_label",
            "n_artists",
            "n_releases",
            "artists"
        ]])
    else:
        print("No clean labels included.")

    print("\nExclusion summary:")
    print(exclusion_summary)

    print("\nTop artists in clean shared-label network:")
    if len(node_metrics) > 0:
        print(node_metrics.head(10)[[
            "artist",
            "degree",
            "weighted_degree",
            "degree_centrality",
            "community",
            "n_clean_labels",
            "clean_labels"
        ]])
    else:
        print("No node metrics.")

    print("\nSaved files:")
    print(f"Clean label rows: {clean_path}")
    print(f"Excluded label rows: {excluded_path}")
    print(f"Top clean labels: {top_labels_path}")
    print(f"Shared label edges: {shared_edges_path}")
    print(f"Node metrics: {node_metrics_path}")
    print(f"Summary: {summary_path}")
    print(f"Exclusion summary: {exclusion_summary_path}")
    print(f"GraphML: {graphml_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
