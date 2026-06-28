"""
03_collaboration_network_robust.py

Robust version of 03_collaboration_network.py for weak internet connection.

Changes compared to the pilot version:
1. Spotify client uses longer timeout and retries.
2. Spotify API calls are wrapped in retry logic.
3. Release filtering by year is added: default 2010-2024.
4. Failed artists and failed albums are saved to CSV.
5. Checkpoint files are saved during the run.
6. Optional --resume mode skips artists already marked as completed.
7. Output names remain compatible with the existing pipeline.

Typical run:
    python 03_collaboration_network_robust.py

For weak internet:
    python 03_collaboration_network_robust.py --timeout 60 --sleep 1.0 --max-attempts 5

Resume after interruption:
    python 03_collaboration_network_robust.py --resume --timeout 60 --sleep 1.0
"""

import os
import time
import argparse
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
# Helper functions
# ---------------------------------------------------------

def spotify_call_with_retries(func, *args, max_attempts=4, base_sleep=3, **kwargs):
    """
    Calls Spotify API with retries.
    Does not retry 404, because 404 usually means a wrong/missing Spotify resource.
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
    Spotify dates may look like: 2020, 2020-05, 2020-05-12.
    """
    if release_date is None or pd.isna(release_date):
        return None

    try:
        return int(str(release_date)[:4])
    except Exception:
        return None


def canonical_pair(a, b):
    return tuple(sorted([a, b]))


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


# ---------------------------------------------------------
# Spotify data collection functions
# ---------------------------------------------------------

def get_artist_albums(sp, artist_id, include_groups, max_albums, sleep_seconds,
                      max_attempts, base_sleep):
    """
    Collects albums/singles/appearances/compilations for one artist.
    """
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
            base_sleep=base_sleep,
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


def get_album_tracks(sp, album, sleep_seconds, max_attempts, base_sleep):
    """
    Collects tracks from one Spotify album/release.
    """
    album_id = album.get("id")
    tracks = []
    offset = 0
    limit = 50

    while True:
        result = spotify_call_with_retries(
            sp.album_tracks,
            album_id,
            limit=limit,
            offset=offset,
            max_attempts=max_attempts,
            base_sleep=base_sleep,
        )

        items = result.get("items", [])

        if not items:
            break

        for track in items:
            tracks.append({
                "track_id": track.get("id"),
                "track_name": track.get("name"),
                "album_id": album_id,
                "album_name": album.get("name"),
                "album_type": album.get("album_type"),
                "release_date": album.get("release_date"),
                "release_year": get_release_year(album.get("release_date")),
                "track_artist_ids": [a.get("id") for a in track.get("artists", [])],
                "track_artist_names": [a.get("name") for a in track.get("artists", [])],
                "track_url": track.get("external_urls", {}).get("spotify"),
            })

        offset += limit

        if result.get("next") is None:
            break

        time.sleep(sleep_seconds)

    return tracks


# ---------------------------------------------------------
# Main workflow
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-nodes", default="outputs/tables/nodes_clean.csv")
    parser.add_argument("--input-genre-pairs", default="outputs/tables/artist_pairs_genre_similarity.csv")
    parser.add_argument("--output-tables-dir", default="outputs/tables")
    parser.add_argument("--output-figures-dir", default="outputs/figures")
    parser.add_argument("--output-networks-dir", default="outputs/networks")

    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--include-groups", default="album,single,appears_on,compilation")
    parser.add_argument("--max-albums-per-artist", type=int, default=200)

    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--base-sleep", type=float, default=3)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    input_nodes_path = Path(args.input_nodes)
    input_genre_pairs_path = Path(args.input_genre_pairs)

    output_tables_dir = Path(args.output_tables_dir)
    output_figures_dir = Path(args.output_figures_dir)
    output_networks_dir = Path(args.output_networks_dir)

    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    output_networks_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_tracks_path = output_tables_dir / "collaboration_tracks_checkpoint.csv"
    artist_progress_path = output_tables_dir / "collaboration_artist_progress.csv"
    failed_artists_path = output_tables_dir / "collaboration_failed_artists.csv"
    failed_albums_path = output_tables_dir / "collaboration_failed_albums.csv"

    # -----------------------------------------------------
    # Load Spotify credentials
    # -----------------------------------------------------

    load_dotenv()

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError("Spotify credentials are missing. Check your .env file.")

    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
        ),
        requests_timeout=args.timeout,
        retries=5,
        status_retries=5,
        status_forcelist=(429, 500, 502, 503, 504),
    )

    # -----------------------------------------------------
    # Load cleaned artist nodes
    # -----------------------------------------------------

    if not input_nodes_path.exists():
        raise FileNotFoundError(f"Input nodes file not found: {input_nodes_path}")

    nodes = pd.read_csv(input_nodes_path)
    nodes = nodes.dropna(subset=["spotify_id"]).copy()

    id_to_query_name = dict(zip(nodes["spotify_id"], nodes["query_name"]))
    sample_artist_ids = set(nodes["spotify_id"])

    print(f"Loaded {len(nodes)} artists for collaboration search.")
    print(f"Release year filter: {args.start_year}-{args.end_year}")
    print(f"include_groups: {args.include_groups}")
    print(f"Spotify timeout: {args.timeout} seconds")
    print(f"Sleep between requests: {args.sleep} seconds")
    print(f"Max attempts per request: {args.max_attempts}")

    # -----------------------------------------------------
    # Resume from checkpoint if requested
    # -----------------------------------------------------

    all_track_records = []
    seen_tracks = set()
    failed_artists = []
    failed_albums = []
    completed_artist_ids = set()

    if args.resume:
        if checkpoint_tracks_path.exists():
            checkpoint_df = pd.read_csv(checkpoint_tracks_path)
            all_track_records = checkpoint_df.to_dict("records")
            seen_tracks = set(checkpoint_df["track_id"].dropna().astype(str))
            print(f"Resume mode: loaded {len(all_track_records)} track records from checkpoint.")

        if artist_progress_path.exists():
            progress_df = pd.read_csv(artist_progress_path)
            completed_artist_ids = set(
                progress_df.loc[progress_df["status"] == "completed", "spotify_id"]
                .dropna()
                .astype(str)
            )
            print(f"Resume mode: {len(completed_artist_ids)} artists already completed.")

        if failed_artists_path.exists():
            failed_artists = pd.read_csv(failed_artists_path).to_dict("records")

        if failed_albums_path.exists():
            failed_albums = pd.read_csv(failed_albums_path).to_dict("records")

    progress_records = []

    if artist_progress_path.exists():
        try:
            progress_records = pd.read_csv(artist_progress_path).to_dict("records")
        except Exception:
            progress_records = []

    # -----------------------------------------------------
    # Collect track-level evidence
    # -----------------------------------------------------

    for idx, row in nodes.iterrows():
        artist_name = row["query_name"]
        artist_id = str(row["spotify_id"])

        if args.resume and artist_id in completed_artist_ids:
            print(f"\nSkipping already completed artist: {artist_name}")
            continue

        print(f"\n[{idx + 1}/{len(nodes)}] Collecting releases for: {artist_name}")

        artist_error = ""

        try:
            albums = get_artist_albums(
                sp=sp,
                artist_id=artist_id,
                include_groups=args.include_groups,
                max_albums=args.max_albums_per_artist,
                sleep_seconds=args.sleep,
                max_attempts=args.max_attempts,
                base_sleep=args.base_sleep,
            )

        except Exception as e:
            artist_error = str(e)
            print(f"Could not collect albums for {artist_name}: {e}")

            failed_artists.append({
                "artist_name": artist_name,
                "spotify_id": artist_id,
                "error": str(e),
            })

            progress_records.append({
                "artist_name": artist_name,
                "spotify_id": artist_id,
                "status": "failed_album_collection",
                "n_albums_found": 0,
                "n_albums_in_year_range": 0,
                "n_tracks_added": 0,
                "error": artist_error,
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
            f"Found {len(albums)} releases/appearances; "
            f"{len(albums_in_range)} in {args.start_year}-{args.end_year}; "
            f"skipped {skipped_outside_range} outside range; "
            f"skipped {skipped_no_year} with missing/invalid year."
        )

        tracks_added_for_artist = 0

        for album in albums_in_range:
            try:
                tracks = get_album_tracks(
                    sp=sp,
                    album=album,
                    sleep_seconds=args.sleep,
                    max_attempts=args.max_attempts,
                    base_sleep=args.base_sleep,
                )

            except Exception as e:
                print(f"Could not collect tracks for album {album.get('name')}: {e}")

                failed_albums.append({
                    "artist_name": artist_name,
                    "artist_id": artist_id,
                    "album_id": album.get("id"),
                    "album_name": album.get("name"),
                    "album_type": album.get("album_type"),
                    "release_date": album.get("release_date"),
                    "error": str(e),
                })

                save_csv_safely(pd.DataFrame(failed_albums), failed_albums_path)
                continue

            for track in tracks:
                track_id = str(track["track_id"])

                if track_id in seen_tracks:
                    continue

                seen_tracks.add(track_id)

                present_sample_artist_ids = [
                    aid for aid in track["track_artist_ids"]
                    if aid in sample_artist_ids
                ]

                present_sample_artist_names = [
                    id_to_query_name[aid]
                    for aid in present_sample_artist_ids
                ]

                all_track_records.append({
                    "track_id": track["track_id"],
                    "track_name": track["track_name"],
                    "album_id": track["album_id"],
                    "album_name": track["album_name"],
                    "album_type": track["album_type"],
                    "release_date": track["release_date"],
                    "release_year": track["release_year"],
                    "all_track_artist_names": "; ".join(track["track_artist_names"]),
                    "sample_artist_names": "; ".join(present_sample_artist_names),
                    "n_sample_artists_on_track": len(present_sample_artist_names),
                    "track_url": track["track_url"],
                })

                tracks_added_for_artist += 1

            time.sleep(args.sleep)

        progress_records.append({
            "artist_name": artist_name,
            "spotify_id": artist_id,
            "status": "completed",
            "n_albums_found": len(albums),
            "n_albums_in_year_range": len(albums_in_range),
            "n_tracks_added": tracks_added_for_artist,
            "error": artist_error,
        })

        save_csv_safely(pd.DataFrame(all_track_records), checkpoint_tracks_path)
        save_csv_safely(pd.DataFrame(progress_records), artist_progress_path)
        save_csv_safely(pd.DataFrame(failed_artists), failed_artists_path)
        save_csv_safely(pd.DataFrame(failed_albums), failed_albums_path)

        print(f"Checkpoint saved. Total unique tracks so far: {len(all_track_records)}")

    # -----------------------------------------------------
    # Save checked tracks and collaboration evidence
    # -----------------------------------------------------

    tracks_df = pd.DataFrame(all_track_records)

    if len(tracks_df) == 0:
        tracks_df = pd.DataFrame(columns=[
            "track_id", "track_name", "album_id", "album_name", "album_type",
            "release_date", "release_year", "all_track_artist_names",
            "sample_artist_names", "n_sample_artists_on_track", "track_url",
        ])

    tracks_output_path = output_tables_dir / "spotify_tracks_checked_for_collaborations.csv"
    save_csv_safely(tracks_df, tracks_output_path)

    collab_tracks = tracks_df[
        tracks_df["n_sample_artists_on_track"].fillna(0).astype(int) >= 2
    ].copy()

    collab_tracks_output_path = output_tables_dir / "collaboration_tracks_evidence.csv"
    save_csv_safely(collab_tracks, collab_tracks_output_path)

    print(f"\nTotal unique tracks checked: {len(tracks_df)}")
    print(f"Tracks with at least two sample artists: {len(collab_tracks)}")

    # -----------------------------------------------------
    # Build collaboration edge list
    # -----------------------------------------------------

    pair_to_tracks = defaultdict(list)

    for _, row in collab_tracks.iterrows():
        artists_on_track = [
            x.strip()
            for x in str(row["sample_artist_names"]).split(";")
            if x.strip()
        ]

        for a, b in combinations(sorted(artists_on_track), 2):
            pair = canonical_pair(a, b)
            pair_to_tracks[pair].append({
                "track_name": row["track_name"],
                "album_name": row["album_name"],
                "release_date": row["release_date"],
                "release_year": row.get("release_year"),
                "track_url": row["track_url"],
            })

    edge_records = []

    for (a, b), evidence_list in pair_to_tracks.items():
        edge_records.append({
            "source": a,
            "target": b,
            "collaboration": 1,
            "collaboration_weight": len(evidence_list),
            "evidence_tracks": "; ".join([str(e["track_name"]) for e in evidence_list[:10]]),
            "evidence_albums": "; ".join([str(e["album_name"]) for e in evidence_list[:10]]),
            "evidence_release_dates": "; ".join([str(e["release_date"]) for e in evidence_list[:10]]),
            "evidence_release_years": "; ".join([str(e.get("release_year")) for e in evidence_list[:10]]),
            "evidence_urls": "; ".join([str(e["track_url"]) for e in evidence_list[:10]]),
        })

    collab_edges = pd.DataFrame(edge_records)

    if len(collab_edges) > 0:
        collab_edges = collab_edges.sort_values(by="collaboration_weight", ascending=False)

    collab_edges_path = output_tables_dir / "collaboration_edges.csv"
    save_csv_safely(collab_edges, collab_edges_path)

    print(f"\nCollaboration edges found: {len(collab_edges)}")

    if len(collab_edges) > 0:
        print(collab_edges.head(20)[["source", "target", "collaboration_weight", "evidence_tracks"]])
    else:
        print("No collaboration edges were found among the sample artists.")

    # -----------------------------------------------------
    # Build collaboration network
    # -----------------------------------------------------

    G = nx.Graph()

    for _, row in nodes.iterrows():
        G.add_node(
            row["query_name"],
            spotify_id=row["spotify_id"],
            spotify_name=row["spotify_name"],
            popularity=safe_int(row.get("popularity")),
            followers=safe_int(row.get("followers")),
            genres=row["genres"] if "genres" in row and not pd.isna(row["genres"]) else "",
        )

    for _, row in collab_edges.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            weight=safe_int(row["collaboration_weight"], default=1),
            evidence_tracks=row["evidence_tracks"],
        )

    # -----------------------------------------------------
    # Network metrics
    # -----------------------------------------------------

    if G.number_of_nodes() > 1:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, weight=None)
    else:
        degree_centrality = {n: 0 for n in G.nodes()}
        betweenness_centrality = {n: 0 for n in G.nodes()}

    weighted_degree = dict(G.degree(weight="weight"))

    if G.number_of_edges() > 0:
        communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
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
            "genres": G.nodes[node]["genres"],
            "popularity": G.nodes[node]["popularity"],
            "followers": G.nodes[node]["followers"],
        })

    node_metrics = pd.DataFrame(node_metrics)

    if len(node_metrics) > 0:
        node_metrics = node_metrics.sort_values(
            by=["degree_centrality", "betweenness_centrality"],
            ascending=False,
        )

    node_metrics_path = output_tables_dir / "collaboration_network_node_metrics.csv"
    save_csv_safely(node_metrics, node_metrics_path)

    n_isolates = len(list(nx.isolates(G)))
    largest_component_size = max((len(c) for c in nx.connected_components(G)), default=0)

    network_summary = pd.DataFrame([{
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": round(nx.density(G), 4) if G.number_of_nodes() > 1 else 0,
        "n_connected_components": nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0,
        "largest_component_size": largest_component_size,
        "largest_component_share": round(largest_component_size / G.number_of_nodes(), 4) if G.number_of_nodes() > 0 else 0,
        "n_isolates": n_isolates,
        "isolate_share": round(n_isolates / G.number_of_nodes(), 4) if G.number_of_nodes() > 0 else 0,
        "n_tracks_checked": len(tracks_df),
        "n_collaboration_tracks": len(collab_tracks),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "include_groups": args.include_groups,
        "n_failed_artists": len(failed_artists),
        "n_failed_albums": len(failed_albums),
    }])

    network_summary_path = output_tables_dir / "collaboration_network_summary.csv"
    save_csv_safely(network_summary, network_summary_path)

    print("\nCollaboration network summary:")
    print(network_summary)

    print("\nTop central artists in collaboration network:")
    if len(node_metrics) > 0:
        print(node_metrics.head(10)[[
            "artist", "degree", "weighted_degree", "degree_centrality", "betweenness_centrality", "community"
        ]])

    # -----------------------------------------------------
    # Compare collaboration network with genre similarity
    # -----------------------------------------------------

    if not input_genre_pairs_path.exists():
        print(f"\nGenre pairs file not found: {input_genre_pairs_path}")
        print("Skipping network comparison pairs.")
    else:
        genre_pairs = pd.read_csv(input_genre_pairs_path)
        genre_pairs["pair_key"] = genre_pairs.apply(
            lambda row: "|||".join(sorted([row["source"], row["target"]])),
            axis=1,
        )

        if len(collab_edges) > 0:
            collab_edges["pair_key"] = collab_edges.apply(
                lambda row: "|||".join(sorted([row["source"], row["target"]])),
                axis=1,
            )

            comparison = genre_pairs.merge(
                collab_edges[["pair_key", "collaboration", "collaboration_weight", "evidence_tracks"]],
                on="pair_key",
                how="left",
            )
            comparison["collaboration"] = comparison["collaboration"].fillna(0).astype(int)
            comparison["collaboration_weight"] = comparison["collaboration_weight"].fillna(0).astype(int)
        else:
            comparison = genre_pairs.copy()
            comparison["collaboration"] = 0
            comparison["collaboration_weight"] = 0
            comparison["evidence_tracks"] = ""

        comparison_path = output_tables_dir / "network_comparison_pairs.csv"
        save_csv_safely(comparison, comparison_path)

        if len(comparison) > 0:
            mean_jaccard_collab = comparison.loc[comparison["collaboration"] == 1, "jaccard_similarity"].mean()
            mean_jaccard_no_collab = comparison.loc[comparison["collaboration"] == 0, "jaccard_similarity"].mean()

            comparison_summary = pd.DataFrame([{
                "n_artist_pairs": len(comparison),
                "n_collaboration_pairs": int(comparison["collaboration"].sum()),
                "mean_jaccard_for_collaboration_pairs": round(mean_jaccard_collab, 4) if pd.notna(mean_jaccard_collab) else None,
                "mean_jaccard_for_non_collaboration_pairs": round(mean_jaccard_no_collab, 4) if pd.notna(mean_jaccard_no_collab) else None,
            }])
        else:
            comparison_summary = pd.DataFrame([{
                "n_artist_pairs": 0,
                "n_collaboration_pairs": 0,
                "mean_jaccard_for_collaboration_pairs": None,
                "mean_jaccard_for_non_collaboration_pairs": None,
            }])

        comparison_summary_path = output_tables_dir / "network_comparison_summary.csv"
        save_csv_safely(comparison_summary, comparison_summary_path)

        print("\nNetwork comparison summary:")
        print(comparison_summary)

    # -----------------------------------------------------
    # Save graph and visualization
    # -----------------------------------------------------

    graphml_path = output_networks_dir / "collaboration_network.graphml"
    nx.write_graphml(G, graphml_path)

    plt.figure(figsize=(14, 10))

    pos = nx.spring_layout(G, seed=42, weight="weight")

    node_sizes = [200 + G.nodes[node]["popularity"] * 10 for node in G.nodes()]
    node_colors = [community_map.get(node, 0) for node in G.nodes()]
    edge_widths = [1 + G[u][v]["weight"] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.45)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"Spotify Collaboration Network among Selected Artists ({args.start_year}-{args.end_year})", fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    figure_path = output_figures_dir / "collaboration_network.png"
    plt.savefig(figure_path, dpi=300)
    plt.show()

    print("\nDone.")
    print(f"Tracks table: {tracks_output_path}")
    print(f"Collaboration edges: {collab_edges_path}")
    print(f"Network summary: {network_summary_path}")
    print(f"GraphML: {graphml_path}")
    print(f"Figure: {figure_path}")
    print(f"Failed artists: {failed_artists_path}")
    print(f"Failed albums: {failed_albums_path}")


if __name__ == "__main__":
    main()
