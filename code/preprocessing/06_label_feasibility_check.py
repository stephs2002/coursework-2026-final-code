"""
06_label_feasibility_check.py

Purpose:
    Feasibility check for adding a label layer to the coursework pipeline.

    The script tests whether record-label data can be collected for the current
    artist sample and whether it is meaningful enough to build a shared-label
    artist network.

How it fits into the existing pipeline:
    01_spotify_data_collection_from_csv.py
        -> data/raw/artists_raw_readable.csv

    02_genre_similarity_network.py
        -> outputs/tables/nodes_clean.csv
        -> outputs/tables/artist_pairs_genre_similarity.csv

    03_collaboration_network_robust.py
        -> collaboration network outputs

    06_label_feasibility_check.py
        -> label feasibility tables
        -> shared-label network outputs

Expected input:
    outputs/tables/nodes_clean.csv

Main outputs:
    outputs/tables/artist_release_labels.csv
    outputs/tables/artist_label_summary.csv
    outputs/tables/top_labels.csv
    outputs/tables/shared_label_edges.csv
    outputs/tables/label_network_node_metrics.csv
    outputs/tables/label_network_summary.csv
    outputs/tables/label_failed_artists.csv
    outputs/tables/label_failed_albums.csv
    outputs/networks/shared_label_network.graphml
    outputs/figures/shared_label_network.png

Typical run:
    python 06_label_feasibility_check.py

For weak internet:
    python 06_label_feasibility_check.py --timeout 60 --sleep 1.0 --max-attempts 5

Resume after interruption:
    python 06_label_feasibility_check.py --resume --timeout 60 --sleep 1.0 --max-attempts 5

Important methodological choice:
    By default, the script uses include_groups="album,single".
    It does NOT use "appears_on" and "compilation" by default, because labels on
    those releases may refer to another artist's release, a compilation, or a
    distributor rather than the focal artist's institutional context.

    You can change this if needed:
    python 06_label_feasibility_check.py --include-groups album,single,appears_on
"""

import os
import re
import time
import argparse
import unicodedata
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException


# ---------------------------------------------------------
# 1. Helper functions
# ---------------------------------------------------------

def spotify_call_with_retries(func, *args, max_attempts=5, base_sleep=3, **kwargs):
    """
    Calls Spotify API with retries.

    Useful for weak internet:
    - retries temporary connection/time-out/server problems;
    - does not keep retrying 404, because 404 usually means wrong ID or removed resource.
    """
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)

        except SpotifyException as e:
            last_error = e
            status = getattr(e, "http_status", None)

            if status == 404:
                print(f"Spotify request returned 404. No retry: {e}")
                raise

            print(f"Spotify request failed, attempt {attempt}/{max_attempts}: {e}")

            if attempt == max_attempts:
                raise

            time.sleep(base_sleep * attempt)

        except Exception as e:
            last_error = e
            print(f"Spotify request failed, attempt {attempt}/{max_attempts}: {e}")

            if attempt == max_attempts:
                raise

            time.sleep(base_sleep * attempt)

    raise last_error


def get_release_year(release_date):
    """
    Extracts year from Spotify release_date.
    Spotify dates may look like:
        2020
        2020-05
        2020-05-12
    """
    if release_date is None or pd.isna(release_date):
        return None

    try:
        return int(str(release_date)[:4])
    except Exception:
        return None


def normalize_label(label):
    """
    Conservative label-name normalization.

    It does NOT remove words like 'records', 'recordings', or 'music',
    because these words may be part of meaningful label names
    (for example, 'PC Music').

    It mainly:
    - lowercases;
    - removes extra whitespace;
    - normalizes unicode;
    - standardizes '&' as 'and';
    - removes simple punctuation;
    - removes legal suffixes such as ltd / llc / inc.
    """
    if label is None or pd.isna(label):
        return ""

    x = str(label).strip()

    if not x or x.lower() in ["nan", "none", "null", "unknown"]:
        return ""

    x = unicodedata.normalize("NFKD", x)
    x = x.encode("ascii", "ignore").decode("ascii")
    x = x.lower()

    x = x.replace("&", " and ")
    x = x.replace("©", " ").replace("℗", " ")
    x = re.sub(r"[^a-z0-9]+", " ", x)

    legal_suffixes = {
        "ltd", "limited", "llc", "inc", "corp", "corporation",
        "co", "company", "gmbh", "plc", "sa", "sarl", "bv"
    }

    tokens = [t for t in x.split() if t not in legal_suffixes]
    x = " ".join(tokens)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def save_csv_safely(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def canonical_pair(a, b):
    return tuple(sorted([a, b]))


# ---------------------------------------------------------
# 2. Spotify collection functions
# ---------------------------------------------------------

def get_artist_albums(sp, artist_id, include_groups, max_albums, sleep_seconds,
                      max_attempts, base_sleep):
    """Collects simplified Spotify releases for one artist."""
    albums = []
    seen_album_ids = set()
    offset = 0
    limit = 50

    while True:
        result = spotify_call_with_retries(
            sp.artist_albums,
            artist_id,
            include_groups=include_groups,
            limit=limit,
            offset=offset,
            max_attempts=max_attempts,
            base_sleep=base_sleep
        )

        items = result.get("items", [])

        if not items:
            break

        for album in items:
            album_id = album.get("id")

            if album_id not in seen_album_ids:
                seen_album_ids.add(album_id)
                albums.append(album)

            if len(albums) >= max_albums:
                return albums

        offset += limit

        if result.get("next") is None:
            break

        time.sleep(sleep_seconds)

    return albums


def get_full_album(sp, album_id, max_attempts, base_sleep):
    """
    Collects full Spotify album object.
    The simplified album object from artist_albums does not always contain label.
    Full album object normally contains the 'label' field.
    """
    return spotify_call_with_retries(
        sp.album,
        album_id,
        max_attempts=max_attempts,
        base_sleep=base_sleep
    )


# ---------------------------------------------------------
# 3. Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-nodes",
        default="outputs/tables/nodes_clean.csv",
        help="Path to cleaned artist nodes from 02_genre_similarity_network.py."
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
        help="Directory for network exports."
    )

    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)

    parser.add_argument(
        "--include-groups",
        default="album,single",
        help="Spotify include_groups for artist_albums. Default: album,single."
    )

    parser.add_argument("--max-albums-per-artist", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--base-sleep", type=float, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-shared-labels", type=int, default=1)

    args = parser.parse_args()

    input_nodes_path = Path(args.input_nodes)
    output_tables_dir = Path(args.output_tables_dir)
    output_figures_dir = Path(args.output_figures_dir)
    output_networks_dir = Path(args.output_networks_dir)

    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    output_networks_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_release_labels_path = output_tables_dir / "label_release_checkpoint.csv"
    artist_progress_path = output_tables_dir / "label_artist_progress.csv"
    failed_artists_path = output_tables_dir / "label_failed_artists.csv"
    failed_albums_path = output_tables_dir / "label_failed_albums.csv"

    artist_release_labels_path = output_tables_dir / "artist_release_labels.csv"
    artist_label_summary_path = output_tables_dir / "artist_label_summary.csv"
    top_labels_path = output_tables_dir / "top_labels.csv"
    shared_label_edges_path = output_tables_dir / "shared_label_edges.csv"
    label_node_metrics_path = output_tables_dir / "label_network_node_metrics.csv"
    label_network_summary_path = output_tables_dir / "label_network_summary.csv"

    graphml_path = output_networks_dir / "shared_label_network.graphml"
    figure_path = output_figures_dir / "shared_label_network.png"

    load_dotenv()

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError("Spotify credentials are missing. Check your .env file.")

    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ),
        requests_timeout=args.timeout,
        retries=5,
        status_retries=5,
        status_forcelist=(429, 500, 502, 503, 504)
    )

    if not input_nodes_path.exists():
        raise FileNotFoundError(f"Input nodes file not found: {input_nodes_path}")

    nodes = pd.read_csv(input_nodes_path)
    nodes = nodes.dropna(subset=["spotify_id"]).copy()

    if "query_name" not in nodes.columns:
        raise ValueError("Input nodes file must contain column: query_name")

    print(f"Loaded {len(nodes)} artists from {input_nodes_path}")
    print(f"Label feasibility period: {args.start_year}-{args.end_year}")
    print(f"include_groups: {args.include_groups}")
    print(f"Spotify timeout: {args.timeout} seconds")
    print(f"Sleep between requests: {args.sleep} seconds")
    print(f"Max attempts per request: {args.max_attempts}")

    release_records = []
    failed_artists = []
    failed_albums = []
    completed_artist_ids = set()
    progress_records = []

    if args.resume:
        if checkpoint_release_labels_path.exists():
            checkpoint_df = pd.read_csv(checkpoint_release_labels_path)
            release_records = checkpoint_df.to_dict("records")
            print(f"Resume mode: loaded {len(release_records)} release-label records.")

        if artist_progress_path.exists():
            progress_df = pd.read_csv(artist_progress_path)
            progress_records = progress_df.to_dict("records")
            completed_artist_ids = set(
                progress_df.loc[
                    progress_df["status"] == "completed",
                    "spotify_id"
                ].dropna().astype(str)
            )
            print(f"Resume mode: {len(completed_artist_ids)} artists already completed.")

        if failed_artists_path.exists():
            failed_artists = pd.read_csv(failed_artists_path).to_dict("records")

        if failed_albums_path.exists():
            failed_albums = pd.read_csv(failed_albums_path).to_dict("records")

    album_details_cache = {}

    for idx, row in nodes.iterrows():
        artist_name = row["query_name"]
        artist_id = str(row["spotify_id"])
        spotify_name = row.get("spotify_name", artist_name)

        if args.resume and artist_id in completed_artist_ids:
            print(f"\nSkipping already completed artist: {artist_name}")
            continue

        print(f"\n[{idx + 1}/{len(nodes)}] Collecting label data for: {artist_name}")

        artist_error = ""

        try:
            albums = get_artist_albums(
                sp=sp,
                artist_id=artist_id,
                include_groups=args.include_groups,
                max_albums=args.max_albums_per_artist,
                sleep_seconds=args.sleep,
                max_attempts=args.max_attempts,
                base_sleep=args.base_sleep
            )

        except Exception as e:
            artist_error = str(e)
            print(f"Could not collect releases for {artist_name}: {e}")

            failed_artists.append({
                "artist_name": artist_name,
                "spotify_id": artist_id,
                "error": str(e)
            })

            progress_records.append({
                "artist_name": artist_name,
                "spotify_id": artist_id,
                "status": "failed_release_collection",
                "n_releases_found": 0,
                "n_releases_in_year_range": 0,
                "n_releases_with_label": 0,
                "n_unique_labels": 0,
                "error": artist_error
            })

            save_csv_safely(pd.DataFrame(progress_records), artist_progress_path)
            save_csv_safely(pd.DataFrame(failed_artists), failed_artists_path)
            continue

        albums_in_range = []
        skipped_no_year = 0
        skipped_outside_range = 0

        for album in albums:
            year = get_release_year(album.get("release_date"))

            if year is None:
                skipped_no_year += 1
                continue

            if args.start_year <= year <= args.end_year:
                albums_in_range.append(album)
            else:
                skipped_outside_range += 1

        print(
            f"Found {len(albums)} releases; "
            f"{len(albums_in_range)} in {args.start_year}-{args.end_year}; "
            f"skipped {skipped_outside_range} outside range; "
            f"skipped {skipped_no_year} with missing/invalid year."
        )

        labels_for_artist = set()
        n_releases_with_label = 0

        for album in albums_in_range:
            album_id = album.get("id")

            try:
                if album_id in album_details_cache:
                    full_album = album_details_cache[album_id]
                else:
                    full_album = get_full_album(
                        sp=sp,
                        album_id=album_id,
                        max_attempts=args.max_attempts,
                        base_sleep=args.base_sleep
                    )
                    album_details_cache[album_id] = full_album
                    time.sleep(args.sleep)

            except Exception as e:
                print(f"Could not collect full album object for {album.get('name')}: {e}")

                failed_albums.append({
                    "artist_name": artist_name,
                    "artist_id": artist_id,
                    "album_id": album_id,
                    "album_name": album.get("name"),
                    "album_type": album.get("album_type"),
                    "release_date": album.get("release_date"),
                    "error": str(e)
                })

                save_csv_safely(pd.DataFrame(failed_albums), failed_albums_path)
                continue

            raw_label = full_album.get("label")
            normalized_label = normalize_label(raw_label)

            if normalized_label:
                labels_for_artist.add(normalized_label)
                n_releases_with_label += 1

            album_artist_names = [
                a.get("name") for a in full_album.get("artists", []) if a.get("name")
            ]

            release_records.append({
                "artist": artist_name,
                "spotify_id": artist_id,
                "spotify_name": spotify_name,
                "album_id": album_id,
                "album_name": full_album.get("name"),
                "album_type": full_album.get("album_type") or album.get("album_type"),
                "album_group": album.get("album_group"),
                "release_date": full_album.get("release_date") or album.get("release_date"),
                "release_year": get_release_year(full_album.get("release_date") or album.get("release_date")),
                "raw_label": raw_label,
                "normalized_label": normalized_label,
                "label_missing": 0 if normalized_label else 1,
                "total_tracks": full_album.get("total_tracks"),
                "album_artist_names": "; ".join(album_artist_names),
                "spotify_url": full_album.get("external_urls", {}).get("spotify")
            })

        progress_records.append({
            "artist_name": artist_name,
            "spotify_id": artist_id,
            "status": "completed",
            "n_releases_found": len(albums),
            "n_releases_in_year_range": len(albums_in_range),
            "n_releases_with_label": n_releases_with_label,
            "n_unique_labels": len(labels_for_artist),
            "error": artist_error
        })

        save_csv_safely(pd.DataFrame(release_records), checkpoint_release_labels_path)
        save_csv_safely(pd.DataFrame(progress_records), artist_progress_path)
        save_csv_safely(pd.DataFrame(failed_artists), failed_artists_path)
        save_csv_safely(pd.DataFrame(failed_albums), failed_albums_path)

        print(f"Checkpoint saved. Release-label records so far: {len(release_records)}")

    release_labels = pd.DataFrame(release_records)

    if len(release_labels) == 0:
        release_labels = pd.DataFrame(columns=[
            "artist", "spotify_id", "spotify_name", "album_id", "album_name",
            "album_type", "album_group", "release_date", "release_year",
            "raw_label", "normalized_label", "label_missing", "total_tracks",
            "album_artist_names", "spotify_url"
        ])

    save_csv_safely(release_labels, artist_release_labels_path)

    valid_labels = release_labels[
        release_labels["normalized_label"].fillna("").astype(str).str.strip() != ""
    ].copy()

    if len(valid_labels) > 0:
        artist_label_summary = (
            valid_labels
            .groupby(["artist", "spotify_id", "normalized_label"], dropna=False)
            .agg(
                n_releases_on_label=("album_id", "nunique"),
                raw_label_examples=("raw_label", lambda x: "; ".join(sorted(set([str(v) for v in x if pd.notna(v)]))[:5])),
                release_years=("release_year", lambda x: "; ".join([str(int(v)) for v in sorted(set(x.dropna()))[:10]]))
            )
            .reset_index()
            .sort_values(["artist", "n_releases_on_label"], ascending=[True, False])
        )
    else:
        artist_label_summary = pd.DataFrame(columns=[
            "artist", "spotify_id", "normalized_label", "n_releases_on_label",
            "raw_label_examples", "release_years"
        ])

    save_csv_safely(artist_label_summary, artist_label_summary_path)

    if len(artist_label_summary) > 0:
        top_labels = (
            artist_label_summary
            .groupby("normalized_label")
            .agg(
                n_artists=("artist", "nunique"),
                n_artist_label_pairs=("artist", "count"),
                n_releases=("n_releases_on_label", "sum"),
                artists=("artist", lambda x: "; ".join(sorted(set(x))[:20])),
                raw_label_examples=("raw_label_examples", lambda x: "; ".join(sorted(set([str(v) for v in x if pd.notna(v)]))[:10]))
            )
            .reset_index()
            .sort_values(["n_artists", "n_releases"], ascending=False)
        )
    else:
        top_labels = pd.DataFrame(columns=[
            "normalized_label", "n_artists", "n_artist_label_pairs",
            "n_releases", "artists", "raw_label_examples"
        ])

    save_csv_safely(top_labels, top_labels_path)

    artist_to_labels = defaultdict(set)

    for _, row in artist_label_summary.iterrows():
        label = str(row["normalized_label"]).strip()
        if label:
            artist_to_labels[row["artist"]].add(label)

    edge_records = []
    node_names = list(nodes["query_name"])

    for a, b in combinations(node_names, 2):
        labels_a = artist_to_labels.get(a, set())
        labels_b = artist_to_labels.get(b, set())

        shared = labels_a.intersection(labels_b)
        union = labels_a.union(labels_b)

        if len(shared) >= args.min_shared_labels:
            if len(union) == 0:
                jaccard_label_similarity = 0
            else:
                jaccard_label_similarity = len(shared) / len(union)

            edge_records.append({
                "source": a,
                "target": b,
                "shared_label_count": len(shared),
                "shared_labels": "; ".join(sorted(shared)),
                "source_label_count": len(labels_a),
                "target_label_count": len(labels_b),
                "label_jaccard_similarity": round(jaccard_label_similarity, 4)
            })

    shared_label_edges = pd.DataFrame(edge_records)

    if len(shared_label_edges) > 0:
        shared_label_edges = shared_label_edges.sort_values(
            ["shared_label_count", "label_jaccard_similarity"],
            ascending=False
        )

    save_csv_safely(shared_label_edges, shared_label_edges_path)

    G = nx.Graph()

    for _, row in nodes.iterrows():
        artist = row["query_name"]
        labels = sorted(artist_to_labels.get(artist, set()))

        G.add_node(
            artist,
            spotify_id=row.get("spotify_id"),
            spotify_name=row.get("spotify_name", artist),
            genres=row.get("genres", "") if not pd.isna(row.get("genres", "")) else "",
            n_labels=len(labels),
            labels="; ".join(labels),
            popularity=safe_int(row.get("popularity")),
            followers=safe_int(row.get("followers"))
        )

    for _, row in shared_label_edges.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            weight=int(row["shared_label_count"]),
            shared_labels=row["shared_labels"],
            label_jaccard_similarity=float(row["label_jaccard_similarity"])
        )

    if G.number_of_nodes() > 1:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, weight=None)
    else:
        degree_centrality = {node: 0 for node in G.nodes()}
        betweenness_centrality = {node: 0 for node in G.nodes()}

    weighted_degree = dict(G.degree(weight="weight"))
    modularity_score = None

    if G.number_of_edges() > 0:
        communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
        try:
            modularity_score = nx.algorithms.community.quality.modularity(G, communities, weight="weight")
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
            "degree_centrality": round(degree_centrality[node], 4),
            "betweenness_centrality": round(betweenness_centrality[node], 4),
            "community": community_map.get(node),
            "n_labels": G.nodes[node]["n_labels"],
            "labels": G.nodes[node]["labels"],
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

    save_csv_safely(node_metrics, label_node_metrics_path)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_isolates = len(list(nx.isolates(G)))
    largest_component_size = max((len(c) for c in nx.connected_components(G)), default=0)

    n_artists = len(nodes)
    n_artists_with_label_data = int(artist_label_summary["artist"].nunique()) if len(artist_label_summary) > 0 else 0
    n_releases_checked = len(release_labels)
    n_releases_with_label = int((release_labels["normalized_label"].fillna("").astype(str).str.strip() != "").sum()) if len(release_labels) > 0 else 0
    n_unique_raw_labels = int(valid_labels["raw_label"].dropna().nunique()) if len(valid_labels) > 0 else 0
    n_unique_normalized_labels = int(valid_labels["normalized_label"].dropna().nunique()) if len(valid_labels) > 0 else 0
    n_artist_label_pairs = len(artist_label_summary)

    label_network_summary = pd.DataFrame([{
        "n_artists": n_artists,
        "n_artists_with_label_data": n_artists_with_label_data,
        "label_coverage_percent": round(n_artists_with_label_data / n_artists * 100, 2) if n_artists else 0,
        "n_releases_checked": n_releases_checked,
        "n_releases_with_label": n_releases_with_label,
        "release_label_coverage_percent": round(n_releases_with_label / n_releases_checked * 100, 2) if n_releases_checked else 0,
        "n_unique_raw_labels": n_unique_raw_labels,
        "n_unique_normalized_labels": n_unique_normalized_labels,
        "n_artist_label_pairs": n_artist_label_pairs,
        "n_shared_label_edges": len(shared_label_edges),
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": round(nx.density(G), 4) if n_nodes > 1 else 0,
        "n_connected_components": nx.number_connected_components(G) if n_nodes > 0 else 0,
        "largest_component_size": largest_component_size,
        "largest_component_share": round(largest_component_size / n_nodes, 4) if n_nodes else 0,
        "n_isolates": n_isolates,
        "isolate_share": round(n_isolates / n_nodes, 4) if n_nodes else 0,
        "modularity": round(modularity_score, 4) if modularity_score is not None else None,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "include_groups": args.include_groups,
        "min_shared_labels": args.min_shared_labels,
        "n_failed_artists": len(failed_artists),
        "n_failed_albums": len(failed_albums)
    }])

    save_csv_safely(label_network_summary, label_network_summary_path)
    nx.write_graphml(G, graphml_path)

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, weight="weight")

    node_sizes = [200 + G.nodes[node]["popularity"] * 10 for node in G.nodes()]
    node_colors = [community_map.get(node, 0) for node in G.nodes()]
    edge_widths = [1 + G[u][v]["weight"] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.45)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"Shared Label Network among Selected Artists ({args.start_year}-{args.end_year})", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.show()

    print("\nLabel feasibility check completed.")
    print("\nLabel network summary:")
    print(label_network_summary)

    print("\nTop labels:")
    if len(top_labels) > 0:
        print(top_labels.head(15)[["normalized_label", "n_artists", "n_releases", "artists"]])
    else:
        print("No label data found.")

    print("\nTop artists in shared-label network:")
    if len(node_metrics) > 0:
        print(node_metrics.head(10)[["artist", "degree", "weighted_degree", "degree_centrality", "community", "n_labels", "labels"]])
    else:
        print("No node metrics.")

    print("\nSaved files:")
    print(f"Artist release labels: {artist_release_labels_path}")
    print(f"Artist label summary: {artist_label_summary_path}")
    print(f"Top labels: {top_labels_path}")
    print(f"Shared label edges: {shared_label_edges_path}")
    print(f"Label network summary: {label_network_summary_path}")
    print(f"GraphML: {graphml_path}")
    print(f"Figure: {figure_path}")
    print(f"Failed artists: {failed_artists_path}")
    print(f"Failed albums: {failed_albums_path}")


if __name__ == "__main__":
    main()
