"""
08_multilayer_network_comparison.py

Purpose:
    Compare the community structures and edge overlaps across three network layers:

    1. Spotify genre similarity network
    2. Spotify track-level collaboration network
    3. Clean shared-label network

This script is meant to extend the original 04_community_comparison_ari.py.
Instead of comparing only genre communities and collaboration communities, it
compares all three layers pairwise:

    - genre vs collaboration
    - genre vs label
    - collaboration vs label

It also creates a dyadic edge table that can later be used for additional QAP
models, for example:

    collaboration ~ genre_similarity + shared_label_tie

Expected inputs:
    outputs/tables/genre_network_node_metrics.csv
    outputs/tables/collaboration_network_node_metrics.csv
    outputs/tables/label_network_node_metrics_clean.csv

    outputs/tables/artist_pairs_genre_similarity.csv
    outputs/tables/collaboration_edges.csv
    outputs/tables/shared_label_edges_clean.csv

Main outputs:
    outputs/tables/multilayer_community_assignments.csv
    outputs/tables/multilayer_community_summary.csv
    outputs/tables/multilayer_community_ari.csv
    outputs/tables/multilayer_edge_overlap.csv
    outputs/tables/multilayer_pairwise_edge_table.csv

Typical run:
    python 08_multilayer_network_comparison.py

If your clean label network uses another suffix:
    python 08_multilayer_network_comparison.py --label-node-metrics outputs/tables/label_network_node_metrics_broad.csv --label-edges outputs/tables/shared_label_edges_broad.csv

Notes:
    - ARI compares community partitions, not edges directly.
    - Edge overlap compares binary ties in the three networks.
    - For genre edges, the script uses the same threshold as the genre network
      by default: Jaccard >= 0.5. You can change it with --genre-threshold.
"""

import argparse
from pathlib import Path
from itertools import combinations

import pandas as pd
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------
# 1. Helper functions
# ---------------------------------------------------------

def save_csv_safely(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def find_artist_column(df):
    """
    Detects the artist-name column in different pipeline outputs.
    """
    possible = ["artist", "query_name", "source_artist", "name"]
    for col in possible:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find artist column. Available columns: {list(df.columns)}"
    )


def load_community_table(path, network_name):
    """
    Loads node metrics and returns:
        artist, <network_name>_community
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Node metrics file not found: {path}")

    df = pd.read_csv(path)

    artist_col = find_artist_column(df)

    if "community" not in df.columns:
        raise ValueError(
            f"File {path} must contain a 'community' column. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[[artist_col, "community"]].copy()
    out = out.rename(columns={
        artist_col: "artist",
        "community": f"{network_name}_community"
    })

    out["artist"] = out["artist"].astype(str).str.strip()
    out = out[out["artist"] != ""].drop_duplicates("artist")

    # ARI works with categorical labels. Convert to string to avoid confusion
    # when different networks both use community IDs 1, 2, 3...
    out[f"{network_name}_community"] = (
        network_name + "_c" + out[f"{network_name}_community"].astype(str)
    )

    return out


def summarize_partition(assignments, community_col, network_name):
    """
    Summarizes one community partition.
    """
    community_sizes = (
        assignments
        .groupby(community_col)
        .agg(
            n_artists=("artist", "count"),
            artists=("artist", lambda x: "; ".join(sorted(set(x))[:30]))
        )
        .reset_index()
        .sort_values("n_artists", ascending=False)
    )

    return pd.DataFrame([{
        "network": network_name,
        "n_nodes": assignments["artist"].nunique(),
        "n_communities": community_sizes[community_col].nunique(),
        "largest_community_size": int(community_sizes["n_artists"].max()) if len(community_sizes) else 0,
        "largest_community_share": round(
            community_sizes["n_artists"].max() / assignments["artist"].nunique(),
            4
        ) if len(community_sizes) and assignments["artist"].nunique() else 0,
        "community_size_distribution": "; ".join(
            [str(int(x)) for x in community_sizes["n_artists"].tolist()]
        )
    }])


def canonical_pair(a, b):
    return tuple(sorted([str(a).strip(), str(b).strip()]))


def pair_key(a, b):
    a, b = canonical_pair(a, b)
    return f"{a}|||{b}"


def load_genre_edges_from_pairs(path, threshold):
    """
    Loads all genre pair similarities and returns:
        pair_key -> jaccard_similarity
    Only pairs with jaccard_similarity >= threshold are considered genre edges.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Genre pairs file not found: {path}")

    df = pd.read_csv(path)

    required = ["source", "target", "jaccard_similarity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Genre pairs file is missing columns: {missing}")

    df["pair_key"] = df.apply(
        lambda row: pair_key(row["source"], row["target"]),
        axis=1
    )

    # Keep max similarity if duplicates exist.
    pair_to_jaccard = (
        df.groupby("pair_key")["jaccard_similarity"]
        .max()
        .to_dict()
    )

    edge_keys = set(
        df.loc[df["jaccard_similarity"] >= threshold, "pair_key"].tolist()
    )

    return edge_keys, pair_to_jaccard


def load_simple_edges(path, weight_col=None):
    """
    Loads source-target edge list and returns:
        edge_keys: set(pair_key)
        pair_to_weight: dict(pair_key -> weight)
    """
    path = Path(path)

    if not path.exists():
        print(f"Warning: edge file not found: {path}")
        return set(), {}

    df = pd.read_csv(path)

    if len(df) == 0:
        return set(), {}

    required = ["source", "target"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Edge file {path} is missing columns: {missing}")

    df["pair_key"] = df.apply(
        lambda row: pair_key(row["source"], row["target"]),
        axis=1
    )

    edge_keys = set(df["pair_key"].tolist())

    if weight_col and weight_col in df.columns:
        pair_to_weight = (
            df.groupby("pair_key")[weight_col]
            .max()
            .to_dict()
        )
    else:
        pair_to_weight = {k: 1 for k in edge_keys}

    return edge_keys, pair_to_weight


def edge_overlap_stats(edge_sets, network_a, network_b):
    """
    Computes pairwise binary edge overlap statistics between two networks.
    """
    a = edge_sets[network_a]
    b = edge_sets[network_b]

    intersection = a.intersection(b)
    union = a.union(b)

    smaller = min(len(a), len(b))

    return {
        "network_a": network_a,
        "network_b": network_b,
        "n_edges_a": len(a),
        "n_edges_b": len(b),
        "n_overlap_edges": len(intersection),
        "edge_jaccard": round(len(intersection) / len(union), 4) if len(union) else 0,
        "overlap_coefficient": round(len(intersection) / smaller, 4) if smaller else 0,
        "overlap_edge_examples": "; ".join(sorted(list(intersection))[:20])
    }


# ---------------------------------------------------------
# 2. Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--genre-node-metrics",
        default="outputs/tables/genre_network_node_metrics.csv",
        help="Node metrics for genre similarity network."
    )

    parser.add_argument(
        "--collaboration-node-metrics",
        default="outputs/tables/collaboration_network_node_metrics.csv",
        help="Node metrics for collaboration network."
    )

    parser.add_argument(
        "--label-node-metrics",
        default="outputs/tables/label_network_node_metrics_clean.csv",
        help="Node metrics for clean shared-label network."
    )

    parser.add_argument(
        "--genre-pairs",
        default="outputs/tables/artist_pairs_genre_similarity.csv",
        help="Pairwise genre similarity table."
    )

    parser.add_argument(
        "--collaboration-edges",
        default="outputs/tables/collaboration_edges.csv",
        help="Collaboration edge list."
    )

    parser.add_argument(
        "--label-edges",
        default="outputs/tables/shared_label_edges_clean.csv",
        help="Clean shared-label edge list."
    )

    parser.add_argument(
        "--output-tables-dir",
        default="outputs/tables",
        help="Output directory for comparison tables."
    )

    parser.add_argument(
        "--genre-threshold",
        type=float,
        default=0.5,
        help="Jaccard threshold used to define binary genre network edges."
    )

    args = parser.parse_args()

    output_tables_dir = Path(args.output_tables_dir)
    output_tables_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 2.1. Load community assignments
    # -----------------------------------------------------

    genre = load_community_table(
        args.genre_node_metrics,
        "genre"
    )

    collab = load_community_table(
        args.collaboration_node_metrics,
        "collaboration"
    )

    label = load_community_table(
        args.label_node_metrics,
        "label"
    )

    assignments = (
        genre
        .merge(collab, on="artist", how="outer")
        .merge(label, on="artist", how="outer")
        .sort_values("artist")
        .reset_index(drop=True)
    )

    community_cols = [
        "genre_community",
        "collaboration_community",
        "label_community"
    ]

    # The networks should normally have the same node set, but we are careful.
    complete_assignments = assignments.dropna(subset=community_cols).copy()

    save_csv_safely(
        assignments,
        output_tables_dir / "multilayer_community_assignments.csv"
    )

    # -----------------------------------------------------
    # 2.2. Community summaries
    # -----------------------------------------------------

    summary_parts = []

    summary_parts.append(
        summarize_partition(
            complete_assignments[["artist", "genre_community"]].copy(),
            "genre_community",
            "genre"
        )
    )

    summary_parts.append(
        summarize_partition(
            complete_assignments[["artist", "collaboration_community"]].copy(),
            "collaboration_community",
            "collaboration"
        )
    )

    summary_parts.append(
        summarize_partition(
            complete_assignments[["artist", "label_community"]].copy(),
            "label_community",
            "label"
        )
    )

    community_summary = pd.concat(summary_parts, ignore_index=True)

    community_summary["n_nodes_compared_all_layers"] = len(complete_assignments)

    save_csv_safely(
        community_summary,
        output_tables_dir / "multilayer_community_summary.csv"
    )

    # -----------------------------------------------------
    # 2.3. Pairwise ARI
    # -----------------------------------------------------

    ari_records = []

    pairs = [
        ("genre", "collaboration"),
        ("genre", "label"),
        ("collaboration", "label")
    ]

    for a, b in pairs:
        col_a = f"{a}_community"
        col_b = f"{b}_community"

        subset = assignments.dropna(subset=[col_a, col_b]).copy()

        ari = adjusted_rand_score(
            subset[col_a].astype(str),
            subset[col_b].astype(str)
        )

        ari_records.append({
            "network_a": a,
            "network_b": b,
            "n_nodes_compared": len(subset),
            "adjusted_rand_index": round(ari, 4),
            "interpretation_hint": (
                "0 means close to random overlap; 1 means identical partitions. "
                "Negative values are possible when overlap is below random expectation."
            )
        })

    ari_df = pd.DataFrame(ari_records)

    save_csv_safely(
        ari_df,
        output_tables_dir / "multilayer_community_ari.csv"
    )

    # -----------------------------------------------------
    # 2.4. Edge sets and pairwise edge overlap
    # -----------------------------------------------------

    genre_edge_keys, pair_to_jaccard = load_genre_edges_from_pairs(
        args.genre_pairs,
        args.genre_threshold
    )

    collab_edge_keys, pair_to_collab_weight = load_simple_edges(
        args.collaboration_edges,
        weight_col="collaboration_weight"
    )

    label_edge_keys, pair_to_label_weight = load_simple_edges(
        args.label_edges,
        weight_col="shared_label_count"
    )

    edge_sets = {
        "genre": genre_edge_keys,
        "collaboration": collab_edge_keys,
        "label": label_edge_keys
    }

    overlap_records = []

    for a, b in pairs:
        overlap_records.append(
            edge_overlap_stats(edge_sets, a, b)
        )

    edge_overlap = pd.DataFrame(overlap_records)

    edge_overlap["genre_threshold"] = args.genre_threshold

    save_csv_safely(
        edge_overlap,
        output_tables_dir / "multilayer_edge_overlap.csv"
    )

    # -----------------------------------------------------
    # 2.5. Full dyadic edge table for future QAP
    # -----------------------------------------------------

    # Use all artists available in complete assignments, so all layers are aligned.
    artists = sorted(complete_assignments["artist"].tolist())

    pair_records = []

    for a, b in combinations(artists, 2):
        pk = pair_key(a, b)

        pair_records.append({
            "source": a,
            "target": b,
            "pair_key": pk,
            "genre_edge": 1 if pk in genre_edge_keys else 0,
            "collaboration_edge": 1 if pk in collab_edge_keys else 0,
            "label_edge": 1 if pk in label_edge_keys else 0,
            "jaccard_similarity": pair_to_jaccard.get(pk, 0),
            "collaboration_weight": pair_to_collab_weight.get(pk, 0),
            "shared_label_count": pair_to_label_weight.get(pk, 0)
        })

    pairwise_edges = pd.DataFrame(pair_records)

    save_csv_safely(
        pairwise_edges,
        output_tables_dir / "multilayer_pairwise_edge_table.csv"
    )

    # Simple descriptive correlations, not QAP.
    if len(pairwise_edges) > 0:
        corr_cols = [
            "genre_edge",
            "collaboration_edge",
            "label_edge",
            "jaccard_similarity",
            "collaboration_weight",
            "shared_label_count"
        ]

        correlations = pairwise_edges[corr_cols].corr().reset_index()
        correlations = correlations.rename(columns={"index": "variable"})

        save_csv_safely(
            correlations,
            output_tables_dir / "multilayer_pairwise_edge_correlations_descriptive.csv"
        )

    # -----------------------------------------------------
    # 2.6. Print summary
    # -----------------------------------------------------

    print("\nMultilayer community comparison completed.")

    print("\nCommunity summary:")
    print(community_summary)

    print("\nPairwise ARI:")
    print(ari_df)

    print("\nEdge overlap:")
    print(edge_overlap)

    print("\nDyadic edge table summary:")
    print(pairwise_edges[[
        "genre_edge",
        "collaboration_edge",
        "label_edge"
    ]].sum())

    print("\nSaved files:")
    print(output_tables_dir / "multilayer_community_assignments.csv")
    print(output_tables_dir / "multilayer_community_summary.csv")
    print(output_tables_dir / "multilayer_community_ari.csv")
    print(output_tables_dir / "multilayer_edge_overlap.csv")
    print(output_tables_dir / "multilayer_pairwise_edge_table.csv")
    print(output_tables_dir / "multilayer_pairwise_edge_correlations_descriptive.csv")


if __name__ == "__main__":
    main()
