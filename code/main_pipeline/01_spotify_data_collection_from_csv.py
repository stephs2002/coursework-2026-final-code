"""
01_spotify_data_collection_from_csv.py

Adapted version of 01_spotify_data_collection.py for the coursework dataset.

What is changed compared to the pilot version:
    1. Reads artists from a CSV file instead of a hard-coded list.
    2. Uses spotify_id when it is provided.
    3. Falls back to Spotify search by artist_name if spotify_id is missing or invalid.
    4. Uses longer timeout, retries, and pauses for unstable internet connection.
    5. Saves outputs in the same format as the old pipeline:
        data/raw/artists_raw.csv
        data/raw/artists_raw_readable.csv

Expected input CSV columns:
    Required:
        artist_name
    Optional but recommended:
        spotify_id

The script also preserves useful manual columns if they exist:
    manual_scene_status
    manual_evidence_source
    manual_note

Example runs:
    python 01_spotify_data_collection_from_csv.py --input data/input/accepted_artists_core.csv --run-name core

    python 01_spotify_data_collection_from_csv.py --input data/input/accepted_artists_expanded.csv --run-name expanded

Important:
    The standard files data/raw/artists_raw.csv and data/raw/artists_raw_readable.csv
    are overwritten at each run, because the next scripts in the existing pipeline expect these exact paths.

    If --run-name is provided, the script additionally saves run-specific copies, for example:
        data/raw/artists_raw_core.csv
        data/raw/artists_raw_readable_core.csv
"""

import os
import time
import argparse
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
from dotenv import load_dotenv

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException


# ---------------------------------------------------------
# 1. Helper functions
# ---------------------------------------------------------

def similarity(a, b):
    """Approximate string similarity for artist-name matching."""
    if a is None or b is None:
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def normalize_optional_id(x):
    """
    Converts empty / NaN / 'nan' values to None.
    Keeps non-empty Spotify IDs as clean strings.
    """
    if x is None:
        return None
    if pd.isna(x):
        return None

    x = str(x).strip()
    if x == "" or x.lower() in ["nan", "none", "null"]:
        return None

    return x


def spotify_call_with_retries(func, *args, max_attempts=4, base_sleep=3, **kwargs):
    """
    Calls Spotify API with retries.

    Useful for weak internet:
        - retries temporary connection/time-out/server problems;
        - does NOT keep retrying 404, because 404 usually means wrong Spotify ID.
    """
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)

        except SpotifyException as e:
            last_error = e
            status = getattr(e, "http_status", None)

            # 404 = wrong/missing resource. Retrying will not help.
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


def load_candidates(input_path):
    """
    Loads artist candidates from CSV.

    Required:
        artist_name

    Optional:
        spotify_id

    Also preserves manual validation columns when available.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if "artist_name" not in df.columns:
        # Soft fallback in case the file uses another common column name.
        possible_name_cols = ["query_name", "name", "artist", "spotify_name"]
        found = [c for c in possible_name_cols if c in df.columns]
        if not found:
            raise ValueError(
                "Input CSV must contain column 'artist_name' "
                "or one of: query_name, name, artist, spotify_name"
            )
        df = df.rename(columns={found[0]: "artist_name"})

    if "spotify_id" not in df.columns:
        df["spotify_id"] = None

    df["artist_name"] = df["artist_name"].astype(str).str.strip()
    df = df[df["artist_name"] != ""].copy()

    # Avoid duplicated candidate rows.
    df = df.drop_duplicates(subset=["artist_name", "spotify_id"])

    return df


# ---------------------------------------------------------
# 2. Spotify lookup
# ---------------------------------------------------------

def get_spotify_artist(candidate_name, candidate_spotify_id, sp, max_attempts, base_sleep):
    """
    Main lookup logic:
        1. If spotify_id is provided, try direct ID lookup.
        2. If ID fails or is missing, fall back to search by artist name.
    """
    candidate_spotify_id = normalize_optional_id(candidate_spotify_id)

    # -----------------------------------------------------
    # 2.1. Prefer manually provided / pre-validated Spotify ID
    # -----------------------------------------------------

    if candidate_spotify_id:
        try:
            artist = spotify_call_with_retries(
                sp.artist,
                candidate_spotify_id,
                max_attempts=max_attempts,
                base_sleep=base_sleep
            )

            return {
                "query_name": candidate_name,
                "input_spotify_id": candidate_spotify_id,
                "spotify_id": artist.get("id"),
                "spotify_name": artist.get("name"),
                "match_score": 1.0,
                "genres": artist.get("genres", []),
                "n_genres": len(artist.get("genres", [])),
                "popularity": artist.get("popularity"),
                "followers": artist.get("followers", {}).get("total"),
                "spotify_url": artist.get("external_urls", {}).get("spotify"),
                # Keep this value compatible with 02_genre_similarity_network.py,
                # which keeps clear_match and manual_verified.
                "data_quality_note": "manual_verified",
                "spotify_lookup_method": "spotify_id_lookup",
                "spotify_lookup_error": ""
            }

        except Exception as e:
            print(f"Spotify ID lookup failed for {candidate_name}: {e}")
            print("Falling back to Spotify search by artist name...")

            id_lookup_error = str(e)

    else:
        id_lookup_error = ""

    # -----------------------------------------------------
    # 2.2. Fallback: search by artist name
    # -----------------------------------------------------

    try:
        result = spotify_call_with_retries(
            sp.search,
            q=candidate_name,
            type="artist",
            limit=5,
            max_attempts=max_attempts,
            base_sleep=base_sleep
        )
    except Exception as e:
        return {
            "query_name": candidate_name,
            "input_spotify_id": candidate_spotify_id,
            "spotify_id": None,
            "spotify_name": None,
            "match_score": None,
            "genres": [],
            "n_genres": 0,
            "popularity": None,
            "followers": None,
            "spotify_url": None,
            "data_quality_note": "spotify_search_failed",
            "spotify_lookup_method": "search_failed",
            "spotify_lookup_error": f"ID error: {id_lookup_error}; search error: {e}"
        }

    items = result.get("artists", {}).get("items", [])

    if not items:
        return {
            "query_name": candidate_name,
            "input_spotify_id": candidate_spotify_id,
            "spotify_id": None,
            "spotify_name": None,
            "match_score": None,
            "genres": [],
            "n_genres": 0,
            "popularity": None,
            "followers": None,
            "spotify_url": None,
            "data_quality_note": "not_found",
            "spotify_lookup_method": "search_no_results",
            "spotify_lookup_error": id_lookup_error
        }

    best = max(items, key=lambda x: similarity(candidate_name, x.get("name")))
    match_score = similarity(candidate_name, best.get("name"))

    if match_score >= 0.95:
        note = "clear_match"
    elif match_score >= 0.75:
        note = "possible_match_check_manually"
    else:
        note = "ambiguous_check_manually"

    if id_lookup_error:
        note = f"id_failed_then_{note}"

    return {
        "query_name": candidate_name,
        "input_spotify_id": candidate_spotify_id,
        "spotify_id": best.get("id"),
        "spotify_name": best.get("name"),
        "match_score": round(match_score, 3),
        "genres": best.get("genres", []),
        "n_genres": len(best.get("genres", [])),
        "popularity": best.get("popularity"),
        "followers": best.get("followers", {}).get("total"),
        "spotify_url": best.get("external_urls", {}).get("spotify"),
        "data_quality_note": note,
        "spotify_lookup_method": "search_by_name",
        "spotify_lookup_error": id_lookup_error
    }


# ---------------------------------------------------------
# 3. Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/input/accepted_artists_core.csv",
        help="Path to input CSV with artist_name and spotify_id columns."
    )

    parser.add_argument(
        "--run-name",
        default="",
        help="Optional suffix for run-specific output files, e.g. core or expanded."
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Spotify request timeout in seconds. Use 30-60 for weak internet."
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Pause between Spotify artist lookups in seconds."
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="Number of retry attempts for temporary Spotify failures."
    )

    parser.add_argument(
        "--base-sleep",
        type=float,
        default=3,
        help="Base waiting time between retries. Actual wait increases by attempt number."
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # 3.1. Load Spotify credentials
    # -----------------------------------------------------

    load_dotenv()

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError("Spotify credentials are missing. Check your .env file.")

    # More robust client for unstable internet.
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

    # -----------------------------------------------------
    # 3.2. Load candidates
    # -----------------------------------------------------

    candidates = load_candidates(args.input)

    print(f"Loaded {len(candidates)} candidate artists from {args.input}")
    print(f"Spotify timeout: {args.timeout} seconds")
    print(f"Sleep between artists: {args.sleep} seconds")

    records = []

    # Keep these columns if they exist in the candidate file.
    manual_cols = [
        "manual_scene_status",
        "manual_evidence_source",
        "manual_note",
        "country_scene_association",
        "country_evidence"
    ]

    for idx, row in candidates.iterrows():
        artist_name = row["artist_name"]
        spotify_id = row.get("spotify_id")

        print(f"\n[{len(records) + 1}/{len(candidates)}] Processing: {artist_name}")

        spotify_record = get_spotify_artist(
            candidate_name=artist_name,
            candidate_spotify_id=spotify_id,
            sp=sp,
            max_attempts=args.max_attempts,
            base_sleep=args.base_sleep
        )

        # Preserve manual/context columns from accepted artist files.
        for col in manual_cols:
            if col in candidates.columns:
                spotify_record[col] = row.get(col)

        records.append(spotify_record)

        time.sleep(args.sleep)

    artists_df = pd.DataFrame(records)

    # -----------------------------------------------------
    # 3.3. Save outputs
    # -----------------------------------------------------

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Standard outputs expected by the next scripts.
    raw_path = output_dir / "artists_raw.csv"
    readable_path = output_dir / "artists_raw_readable.csv"

    artists_df.to_csv(raw_path, index=False)

    artists_readable = artists_df.copy()
    artists_readable["genres"] = artists_readable["genres"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else ""
    )

    artists_readable.to_csv(readable_path, index=False)

    # Optional run-specific copies, useful if you compare core vs expanded.
    if args.run_name:
        safe_run_name = args.run_name.strip().replace(" ", "_")

        raw_run_path = output_dir / f"artists_raw_{safe_run_name}.csv"
        readable_run_path = output_dir / f"artists_raw_readable_{safe_run_name}.csv"

        artists_df.to_csv(raw_run_path, index=False)
        artists_readable.to_csv(readable_run_path, index=False)

    # -----------------------------------------------------
    # 3.4. Print quality summary
    # -----------------------------------------------------

    print("\nData collection completed.")
    print(f"Saved standard raw table: {raw_path}")
    print(f"Saved standard readable table: {readable_path}")

    if args.run_name:
        print(f"Saved run-specific files with suffix: {args.run_name}")

    print("\nMatch quality summary:")
    print(artists_df["data_quality_note"].value_counts(dropna=False))

    print("\nGenre coverage:")
    n = len(artists_df)
    n_with_genres = int((artists_df["n_genres"] > 0).sum())
    pct = round((n_with_genres / n) * 100, 2) if n else 0
    print(f"Artists with at least one Spotify genre: {n_with_genres}/{n} ({pct}%)")

    print("\nArtists requiring manual check:")
    manual_check_notes = [
        "possible_match_check_manually",
        "ambiguous_check_manually",
        "not_found",
        "spotify_search_failed",
        "search_failed"
    ]

    manual_check = artists_readable[
        (
            artists_readable["data_quality_note"].isin(manual_check_notes)
        )
        | (
            artists_readable["data_quality_note"].astype(str).str.contains(
                "id_failed", na=False
            )
        )
    ].copy()

    if len(manual_check) == 0:
        print("No Spotify matches require manual check according to current rules.")
    else:
        print(
            manual_check[
                [
                    "query_name",
                    "input_spotify_id",
                    "spotify_name",
                    "match_score",
                    "genres",
                    "spotify_url",
                    "data_quality_note",
                    "spotify_lookup_method",
                    "spotify_lookup_error"
                ]
            ]
        )


if __name__ == "__main__":
    main()
