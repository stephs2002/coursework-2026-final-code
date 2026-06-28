"""
10_snowball_expansion_candidates.py

Purpose:
    Generate candidate artists for collaboration-based snowball expansion.

What the script does:
    1. Takes the current seed sample from outputs/tables/nodes_clean.csv.
    2. Takes track evidence from outputs/tables/spotify_tracks_checked_for_collaborations.csv.
    3. Re-queries Spotify track objects to recover all track artist IDs.
    4. Finds artists who co-appear on tracks with seed artists.
    5. Collects comparable metadata for the candidate artists:
        - Spotify artist metadata;
        - Spotify genres / popularity / followers;
        - MusicBrainz country / area;
        - Wikidata geographic fields.
    6. Creates automatic preliminary flags: UK / probable UK / non-UK or unclear.
    7. Does NOT delete or exclude anyone automatically.
    8. Supports future wave expansion through the --wave argument and alternative seed/track inputs.

How it fits into the current pipeline:
    01 -> 02 -> 03 -> 06 -> 07 -> 08 -> 09
    10_snowball_expansion_candidates.py
        -> outputs/tables/snowball_candidate_artists_wave1.csv
        -> manual validation by the researcher
        -> accepted_artists_wave1.csv
        -> rerun 01-09 on the expanded sample

Expected inputs:
    outputs/tables/nodes_clean.csv
    outputs/tables/spotify_tracks_checked_for_collaborations.csv

Main outputs:
    outputs/tables/snowball_candidate_artists_wave1.csv
    outputs/tables/snowball_priority_candidates_wave1.csv
    outputs/tables/snowball_candidate_artist_edges_wave1.csv
    outputs/tables/snowball_candidate_track_evidence_wave1.csv
    outputs/tables/snowball_candidate_summary_wave1.csv
    outputs/tables/snowball_candidate_failed_tracks_wave1.csv
    outputs/tables/snowball_candidate_metadata_failures_wave1.csv

Weak internet:
    The script uses long Spotify timeout, retries, pauses, and checkpoints.
    For a quick test run, use --max-tracks-to-fetch 500.

Typical run:
    python 10_snowball_expansion_candidates.py --wave 1 --timeout 60 --sleep 1.0 --max-attempts 5

Resume metadata collection after interruption:
    python 10_snowball_expansion_candidates.py --wave 1 --resume --timeout 60 --sleep 1.0 --max-attempts 5

Fast test without external country metadata:
    python 10_snowball_expansion_candidates.py --skip-external-metadata --max-candidates-metadata 50

Important:
    This script creates a candidate list, not the final sample.
    It adds automatic indicators, but final inclusion/exclusion should be done manually.
"""

import os
import re
import time
import argparse
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import requests
from dotenv import load_dotenv

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException


# ---------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------

HEADERS = {
    "User-Agent": "DASS-Term-Paper-Snowball-Expansion/0.1"
}

TARGET_COUNTRY_CODE_MB = "GB"
TARGET_AREA_KEYWORDS = [
    "united kingdom", "uk", "great britain", "britain",
    "england", "scotland", "wales", "northern ireland",
    "london", "manchester", "bristol", "glasgow", "edinburgh",
    "leeds", "liverpool", "birmingham", "brighton", "sheffield",
    "cardiff", "belfast", "south london", "north london", "east london", "west london",
]

SCENE_GENRE_KEYWORDS = [
    "indie", "alternative", "electronic", "experimental", "art pop", "hyperpop",
    "post-punk", "rock", "rap", "grime", "garage", "bass", "ambient",
    "club", "dance", "pop", "punk", "synth", "deconstructed", "uk", "london",
]


# ---------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------

def similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def get_release_year(release_date):
    if release_date is None or pd.isna(release_date):
        return None
    try:
        return int(str(release_date)[:4])
    except Exception:
        return None


def split_semicolon_field(x):
    if x is None or pd.isna(x):
        return []
    return [v.strip() for v in str(x).split(";") if v.strip()]


def clean_join(values, max_items=None):
    vals = []
    for v in values:
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        v = str(v).strip()
        if v and v.lower() != "nan":
            vals.append(v)
    vals = sorted(set(vals))
    if max_items:
        vals = vals[:max_items]
    return "; ".join(vals)



def log(message):
    """Timestamped console logging."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def log_section(title):
    """Readable section divider for long runs."""
    line = "=" * 72
    log(line)
    log(title)
    log(line)


def extract_track_id_from_url(url):
    if url is None or pd.isna(url):
        return None
    m = re.search(r"/track/([A-Za-z0-9]+)", str(url))
    return m.group(1) if m else None


def contains_target_keyword(*values):
    text = " | ".join([
        str(v).lower() for v in values
        if v is not None and not pd.isna(v)
    ])
    return any(k in text for k in TARGET_AREA_KEYWORDS)


def has_scene_keyword(genres):
    if genres is None:
        return 0
    text = " | ".join(genres).lower() if isinstance(genres, list) else str(genres).lower()
    return int(any(k in text for k in SCENE_GENRE_KEYWORDS))


def save_csv_safely(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def spotify_call_with_retries(func, *args, max_attempts=5, base_sleep=3, **kwargs):
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


def safe_request_json(url, params=None, headers=None, timeout=45):
    try:
        response = requests.get(
            url,
            params=params,
            headers=headers or HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Request failed: {url} | {e}")
        return None


# ---------------------------------------------------------
# 3. Metadata collection
# ---------------------------------------------------------

def get_spotify_artist_by_id(sp, artist_id, max_attempts, base_sleep):
    artist = spotify_call_with_retries(
        sp.artist,
        artist_id,
        max_attempts=max_attempts,
        base_sleep=base_sleep,
    )
    return {
        "candidate_spotify_id": artist.get("id"),
        "candidate_artist_name_spotify": artist.get("name"),
        "spotify_genres": artist.get("genres", []),
        "spotify_n_genres": len(artist.get("genres", [])),
        "spotify_popularity": artist.get("popularity"),
        "spotify_followers": artist.get("followers", {}).get("total"),
        "spotify_url": artist.get("external_urls", {}).get("spotify"),
        "spotify_metadata_status": "found_by_id",
    }


def search_musicbrainz_artist(artist_name, sleep_seconds):
    url = "https://musicbrainz.org/ws/2/artist/"
    params = {
        "query": f'artist:"{artist_name}"',
        "fmt": "json",
        "limit": 5,
    }
    data = safe_request_json(url, params=params, headers=HEADERS, timeout=45)
    time.sleep(sleep_seconds)

    if not data or not data.get("artists"):
        return {
            "mbid": None,
            "mb_name": None,
            "mb_match_score": None,
            "mb_search_score": None,
            "mb_type": None,
            "mb_country": None,
            "mb_area": None,
            "mb_begin_area": None,
            "mb_disambiguation": None,
            "mb_data_quality_note": "musicbrainz_not_found",
        }

    candidates = data.get("artists", [])
    best = max(candidates, key=lambda x: similarity(artist_name, x.get("name", "")))
    match_score = similarity(artist_name, best.get("name", ""))

    if match_score >= 0.95:
        note = "mb_clear_match"
    elif match_score >= 0.75:
        note = "mb_possible_match_check_manually"
    else:
        note = "mb_ambiguous_check_manually"

    return {
        "mbid": best.get("id"),
        "mb_name": best.get("name"),
        "mb_match_score": round(match_score, 3),
        "mb_search_score": best.get("score"),
        "mb_type": best.get("type"),
        "mb_country": best.get("country"),
        "mb_area": (best.get("area") or {}).get("name"),
        "mb_begin_area": (best.get("begin-area") or {}).get("name"),
        "mb_disambiguation": best.get("disambiguation"),
        "mb_data_quality_note": note,
    }


def run_wikidata_sparql(query, sleep_seconds):
    url = "https://query.wikidata.org/sparql"
    headers = {
        **HEADERS,
        "Accept": "application/sparql-results+json",
    }
    data = safe_request_json(
        url,
        params={"query": query, "format": "json"},
        headers=headers,
        timeout=60,
    )
    time.sleep(sleep_seconds)
    return data.get("results", {}).get("bindings", []) if data else []


def wikidata_lookup_by_property(property_id, value, sleep_seconds):
    if value is None or pd.isna(value) or not str(value).strip():
        return None

    value = str(value).replace('"', '\\"').strip()

    query = f"""
    SELECT
      ?item ?itemLabel
      (GROUP_CONCAT(DISTINCT ?countryOriginLabel; separator="; ") AS ?country_of_origin)
      (GROUP_CONCAT(DISTINCT ?citizenshipLabel; separator="; ") AS ?country_of_citizenship)
      (GROUP_CONCAT(DISTINCT ?formationLabel; separator="; ") AS ?location_of_formation)
      (GROUP_CONCAT(DISTINCT ?birthPlaceLabel; separator="; ") AS ?place_of_birth)
      (GROUP_CONCAT(DISTINCT ?workLocationLabel; separator="; ") AS ?work_location)
    WHERE {{
      ?item wdt:{property_id} "{value}" .
      OPTIONAL {{ ?item wdt:P495 ?countryOrigin. }}
      OPTIONAL {{ ?item wdt:P27 ?citizenship. }}
      OPTIONAL {{ ?item wdt:P740 ?formation. }}
      OPTIONAL {{ ?item wdt:P19 ?birthPlace. }}
      OPTIONAL {{ ?item wdt:P937 ?workLocation. }}
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
        ?item rdfs:label ?itemLabel.
        ?countryOrigin rdfs:label ?countryOriginLabel.
        ?citizenship rdfs:label ?citizenshipLabel.
        ?formation rdfs:label ?formationLabel.
        ?birthPlace rdfs:label ?birthPlaceLabel.
        ?workLocation rdfs:label ?workLocationLabel.
      }}
    }}
    GROUP BY ?item ?itemLabel
    LIMIT 5
    """

    rows = run_wikidata_sparql(query, sleep_seconds=sleep_seconds)
    if not rows:
        return None

    row = rows[0]

    def get_value(key):
        return row.get(key, {}).get("value")

    return {
        "wikidata_qid": get_value("item").split("/")[-1] if get_value("item") else None,
        "wikidata_label": get_value("itemLabel"),
        "wd_country_of_origin": get_value("country_of_origin"),
        "wd_country_of_citizenship": get_value("country_of_citizenship"),
        "wd_location_of_formation": get_value("location_of_formation"),
        "wd_place_of_birth": get_value("place_of_birth"),
        "wd_work_location": get_value("work_location"),
        "wd_n_matches": len(rows),
    }


def get_wikidata_metadata(mbid=None, spotify_id=None, sleep_seconds=0.25):
    result = None
    source = None

    if mbid:
        result = wikidata_lookup_by_property("P434", mbid, sleep_seconds=sleep_seconds)
        source = "wikidata_by_musicbrainz_id" if result else None

    if result is None and spotify_id:
        result = wikidata_lookup_by_property("P1902", spotify_id, sleep_seconds=sleep_seconds)
        source = "wikidata_by_spotify_id" if result else None

    if result is None:
        return {
            "wikidata_qid": None,
            "wikidata_label": None,
            "wd_country_of_origin": None,
            "wd_country_of_citizenship": None,
            "wd_location_of_formation": None,
            "wd_place_of_birth": None,
            "wd_work_location": None,
            "wd_n_matches": 0,
            "wikidata_lookup_source": "wikidata_not_found",
        }

    result["wikidata_lookup_source"] = source
    return result


# ---------------------------------------------------------
# 4. Preliminary country / scene flags
# ---------------------------------------------------------

def classify_country_scene_association(row):
    mb_country = row.get("mb_country")
    mb_area = row.get("mb_area")
    mb_begin_area = row.get("mb_begin_area")
    wd_fields = [
        row.get("wd_country_of_origin"),
        row.get("wd_country_of_citizenship"),
        row.get("wd_location_of_formation"),
        row.get("wd_place_of_birth"),
        row.get("wd_work_location"),
    ]

    has_any_geo_metadata = any(
        pd.notna(x) and str(x).strip() != ""
        for x in [mb_country, mb_area, mb_begin_area] + wd_fields
    )

    if str(mb_country).strip().upper() == TARGET_COUNTRY_CODE_MB:
        return "confirmed_target_country"

    if contains_target_keyword(mb_area, mb_begin_area, *wd_fields):
        return "confirmed_or_probable_target_country"

    if has_any_geo_metadata:
        return "metadata_found_non_target_or_unclear"

    return "no_country_metadata"


def build_country_evidence(row):
    return clean_join([
        f"MB country={row.get('mb_country')}" if pd.notna(row.get("mb_country")) else None,
        f"MB area={row.get('mb_area')}" if pd.notna(row.get("mb_area")) else None,
        f"MB begin-area={row.get('mb_begin_area')}" if pd.notna(row.get("mb_begin_area")) else None,
        f"WD origin={row.get('wd_country_of_origin')}" if pd.notna(row.get("wd_country_of_origin")) else None,
        f"WD citizenship={row.get('wd_country_of_citizenship')}" if pd.notna(row.get("wd_country_of_citizenship")) else None,
        f"WD formation={row.get('wd_location_of_formation')}" if pd.notna(row.get("wd_location_of_formation")) else None,
        f"WD birth={row.get('wd_place_of_birth')}" if pd.notna(row.get("wd_place_of_birth")) else None,
        f"WD work={row.get('wd_work_location')}" if pd.notna(row.get("wd_work_location")) else None,
    ])


def preliminary_scene_flag(row):
    country_status = row.get("country_scene_association")
    scene_genre_flag = row.get("has_relevant_scene_genre_keyword", 0)

    if country_status in ["confirmed_target_country", "confirmed_or_probable_target_country"]:
        return "likely_uk_or_uk_scene"

    if scene_genre_flag == 1:
        return "non_uk_or_unclear_but_genre_relevant"

    if country_status == "metadata_found_non_target_or_unclear":
        return "probably_non_uk_or_unclear"

    return "needs_manual_check"


# ---------------------------------------------------------
# 5. Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wave", type=int, default=1)
    parser.add_argument("--seed-nodes", default="outputs/tables/nodes_clean.csv")
    parser.add_argument("--tracks", default="outputs/tables/spotify_tracks_checked_for_collaborations.csv")
    parser.add_argument("--output-tables-dir", default="outputs/tables")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--mb-sleep", type=float, default=1.05)
    parser.add_argument("--wd-sleep", type=float, default=0.25)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--base-sleep", type=float, default=3)
    parser.add_argument("--max-candidates-metadata", type=int, default=0)
    parser.add_argument(
        "--max-tracks-to-fetch",
        type=int,
        default=0,
        help=(
            "Optional cap for the number of Spotify track objects to fetch. "
            "Useful for test runs. 0 means fetch all prefiltered tracks."
        )
    )
    parser.add_argument(
        "--skip-first-tracks",
        type=int,
        default=0,
        help=(
            "Skip the first N prefiltered track rows before fetching. "
            "Use this only to continue a run that was interrupted before the script had a full track-stage checkpoint."
        )
    )
    parser.add_argument(
        "--log-every-tracks",
        type=int,
        default=100,
        help="Print progress after every N Spotify track requests."
    )
    parser.add_argument(
        "--log-every-candidates",
        type=int,
        default=10,
        help="Print progress after every N candidate metadata records."
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-external-metadata", action="store_true")
    parser.add_argument(
        "--no-track-prefilter",
        action="store_true",
        help="Disable prefilter that skips tracks with no apparent non-seed artist names."
    )

    args = parser.parse_args()

    wave = args.wave
    suffix = f"wave{wave}"
    output_tables_dir = Path(args.output_tables_dir)
    output_tables_dir.mkdir(parents=True, exist_ok=True)

    seed_nodes_path = Path(args.seed_nodes)
    tracks_path = Path(args.tracks)

    if not seed_nodes_path.exists():
        raise FileNotFoundError(f"Seed nodes file not found: {seed_nodes_path}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks file not found: {tracks_path}")

    # Outputs.
    track_evidence_path = output_tables_dir / f"snowball_candidate_track_evidence_{suffix}.csv"
    candidate_edges_path = output_tables_dir / f"snowball_candidate_artist_edges_{suffix}.csv"
    candidates_path = output_tables_dir / f"snowball_candidate_artists_{suffix}.csv"
    priority_candidates_path = output_tables_dir / f"snowball_priority_candidates_{suffix}.csv"
    summary_path = output_tables_dir / f"snowball_candidate_summary_{suffix}.csv"
    failed_tracks_path = output_tables_dir / f"snowball_candidate_failed_tracks_{suffix}.csv"
    metadata_failures_path = output_tables_dir / f"snowball_candidate_metadata_failures_{suffix}.csv"
    metadata_checkpoint_path = output_tables_dir / f"snowball_candidate_metadata_checkpoint_{suffix}.csv"

    # Spotify client.
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

    # Load seed and tracks.
    seed = pd.read_csv(seed_nodes_path)
    tracks = pd.read_csv(tracks_path)

    if "spotify_id" not in seed.columns or "query_name" not in seed.columns:
        raise ValueError("Seed nodes file must contain columns: spotify_id, query_name")

    if "track_id" not in tracks.columns:
        if "track_url" in tracks.columns:
            tracks["track_id"] = tracks["track_url"].apply(extract_track_id_from_url)
        else:
            raise ValueError("Tracks file must contain track_id or track_url")

    seed["spotify_id"] = seed["spotify_id"].astype(str)
    seed_artist_ids = set(seed["spotify_id"].dropna().astype(str))
    seed_id_to_name = dict(zip(seed["spotify_id"], seed["query_name"]))

    tracks["release_year"] = tracks["release_year"].apply(get_release_year)
    tracks = tracks[
        (tracks["release_year"].notna())
        & (tracks["release_year"] >= args.start_year)
        & (tracks["release_year"] <= args.end_year)
    ].copy()

    tracks = tracks.dropna(subset=["track_id"]).copy()
    tracks["track_id"] = tracks["track_id"].astype(str)
    tracks = tracks.drop_duplicates("track_id").copy()

    before_prefilter = len(tracks)
    if not args.no_track_prefilter and {"all_track_artist_names", "sample_artist_names"}.issubset(tracks.columns):
        tracks["all_artist_count_approx"] = tracks["all_track_artist_names"].apply(lambda x: len(split_semicolon_field(x)))
        tracks["sample_artist_count_approx"] = tracks["sample_artist_names"].apply(lambda x: len(split_semicolon_field(x)))
        tracks = tracks[tracks["all_artist_count_approx"] > tracks["sample_artist_count_approx"]].copy()

    if args.max_tracks_to_fetch and args.max_tracks_to_fetch > 0:
        tracks = tracks.head(args.max_tracks_to_fetch).copy()
        log(f"Track fetching capped at first {len(tracks)} tracks for this test run.")

    log("Script version: v8_logged_track_resume")
    log_section("STAGE 1: INPUTS LOADED")
    log(f"Loaded seed artists: {len(seed_artist_ids)}")
    log(f"Loaded unique track records in {args.start_year}-{args.end_year}: {before_prefilter}")
    log(f"Tracks after collaborator prefilter/cap: {len(tracks)}")
    log(f"Snowball wave: {wave}")
    log(f"Track log frequency: every {args.log_every_tracks} track requests")
    log(f"Candidate metadata log frequency: every {args.log_every_candidates} candidates")

    # -----------------------------------------------------
    # Recover full track artist IDs and build evidence
    # -----------------------------------------------------

    track_evidence_records = []
    failed_tracks = []

    # Resume support for the track-fetching stage.
    # v7 resumed only the metadata stage; v8 can keep old evidence rows and continue from a manual offset.
    if args.resume and track_evidence_path.exists():
        try:
            existing_evidence_df = pd.read_csv(track_evidence_path)
            if len(existing_evidence_df) > 0:
                track_evidence_records = existing_evidence_df.to_dict("records")
            log(f"Resume mode: loaded existing track evidence rows: {len(track_evidence_records)}")
        except Exception as e:
            log(f"Resume warning: could not load existing track evidence file {track_evidence_path}: {e}")

    if args.resume and failed_tracks_path.exists():
        try:
            failed_df = pd.read_csv(failed_tracks_path)
            if len(failed_df) > 0:
                failed_tracks = failed_df.to_dict("records")
            log(f"Resume mode: loaded existing failed track rows: {len(failed_tracks)}")
        except Exception as e:
            log(f"Resume warning: could not load existing failed tracks file {failed_tracks_path}: {e}")

    original_tracks_after_prefilter = len(tracks)
    if args.skip_first_tracks and args.skip_first_tracks > 0:
        skip_n = min(args.skip_first_tracks, len(tracks))
        tracks = tracks.iloc[skip_n:].copy()
        log(f"Manual continuation: skipped first {skip_n} prefiltered track rows; remaining tracks to fetch: {len(tracks)}")

    total_tracks_to_fetch = len(tracks)

    for track_counter, (_, row) in enumerate(tracks.iterrows(), start=1):
        track_id = str(row["track_id"])
        if not track_id or track_id.lower() == "nan":
            continue

        if (
            track_counter == 1
            or (args.log_every_tracks > 0 and track_counter % args.log_every_tracks == 0)
            or track_counter == total_tracks_to_fetch
        ):
            log(
                f"Track fetch progress: {track_counter}/{total_tracks_to_fetch} "
                f"({round(track_counter / total_tracks_to_fetch * 100, 1) if total_tracks_to_fetch else 0}%). "
                f"Evidence rows so far: {len(track_evidence_records)}; "
                f"failed tracks: {len(failed_tracks)}"
            )

        try:
            track = spotify_call_with_retries(
                sp.track,
                track_id,
                max_attempts=args.max_attempts,
                base_sleep=args.base_sleep,
            )
            time.sleep(args.sleep)
        except Exception as e:
            log(f"Could not fetch track {track_id}: {e}")
            failed_tracks.append({
                "track_id": track_id,
                "track_name": row.get("track_name"),
                "track_url": row.get("track_url"),
                "error": str(e),
            })
            save_csv_safely(pd.DataFrame(failed_tracks), failed_tracks_path)
            continue

        track_artists = track.get("artists", [])
        track_artist_ids = [a.get("id") for a in track_artists if a.get("id")]
        track_artist_names = [a.get("name") for a in track_artists if a.get("name")]
        artist_id_to_name = {
            a.get("id"): a.get("name")
            for a in track_artists
            if a.get("id") and a.get("name")
        }

        present_seed_ids = [aid for aid in track_artist_ids if aid in seed_artist_ids]
        present_candidate_ids = [aid for aid in track_artist_ids if aid not in seed_artist_ids]

        if len(present_seed_ids) == 0 or len(present_candidate_ids) == 0:
            continue

        for candidate_id in present_candidate_ids:
            for seed_id in present_seed_ids:
                track_evidence_records.append({
                    "source_wave": wave,
                    "candidate_spotify_id": candidate_id,
                    "candidate_artist_name": artist_id_to_name.get(candidate_id),
                    "seed_spotify_id": seed_id,
                    "seed_artist_name": seed_id_to_name.get(seed_id),
                    "track_id": track.get("id"),
                    "track_name": track.get("name"),
                    "release_year": row.get("release_year"),
                    "album_id": row.get("album_id"),
                    "album_name": row.get("album_name"),
                    "album_type": row.get("album_type"),
                    "track_url": track.get("external_urls", {}).get("spotify") or row.get("track_url"),
                    "all_track_artist_names": "; ".join(track_artist_names),
                    "evidence_type": "spotify_track_coappearance",
                })

        if track_counter % 250 == 0:
            checkpoint_evidence_df = pd.DataFrame(track_evidence_records)
            if len(checkpoint_evidence_df) > 0:
                checkpoint_evidence_df = checkpoint_evidence_df.drop_duplicates(
                    subset=["candidate_spotify_id", "seed_spotify_id", "track_id"],
                    keep="last",
                )
                track_evidence_records = checkpoint_evidence_df.to_dict("records")
            save_csv_safely(checkpoint_evidence_df, track_evidence_path)
            print(f"Track checkpoint {track_counter}/{len(tracks)}: evidence rows = {len(track_evidence_records)}")

    track_evidence = pd.DataFrame(track_evidence_records)
    if len(track_evidence) > 0:
        track_evidence = track_evidence.drop_duplicates(
            subset=["candidate_spotify_id", "seed_spotify_id", "track_id"],
            keep="last",
        )
    if len(track_evidence) == 0:
        track_evidence = pd.DataFrame(columns=[
            "source_wave", "candidate_spotify_id", "candidate_artist_name",
            "seed_spotify_id", "seed_artist_name", "track_id", "track_name",
            "release_year", "album_id", "album_name", "album_type", "track_url",
            "all_track_artist_names", "evidence_type",
        ])

    save_csv_safely(track_evidence, track_evidence_path)
    save_csv_safely(pd.DataFrame(failed_tracks), failed_tracks_path)

    # Candidate-seed edges.
    if len(track_evidence) > 0:
        candidate_edges = (
            track_evidence
            .groupby(["candidate_spotify_id", "candidate_artist_name", "seed_spotify_id", "seed_artist_name"], dropna=False)
            .agg(
                n_tracks_with_seed_artist=("track_id", "nunique"),
                evidence_tracks=("track_name", lambda x: clean_join(x, max_items=10)),
                evidence_years=("release_year", lambda x: clean_join(x, max_items=10)),
                evidence_urls=("track_url", lambda x: clean_join(x, max_items=5)),
            )
            .reset_index()
            .sort_values(["n_tracks_with_seed_artist", "candidate_artist_name"], ascending=[False, True])
        )
    else:
        candidate_edges = pd.DataFrame(columns=[
            "candidate_spotify_id", "candidate_artist_name", "seed_spotify_id", "seed_artist_name",
            "n_tracks_with_seed_artist", "evidence_tracks", "evidence_years", "evidence_urls",
        ])
    save_csv_safely(candidate_edges, candidate_edges_path)

    # Candidate aggregation.
    if len(track_evidence) > 0:
        candidate_base = (
            track_evidence
            .groupby(["candidate_spotify_id", "candidate_artist_name"], dropna=False)
            .agg(
                n_tracks_with_seed_sample=("track_id", "nunique"),
                n_seed_artists_connected=("seed_spotify_id", "nunique"),
                connected_seed_artists=("seed_artist_name", lambda x: clean_join(x, max_items=30)),
                evidence_tracks=("track_name", lambda x: clean_join(x, max_items=20)),
                evidence_years=("release_year", lambda x: clean_join(x, max_items=20)),
                evidence_urls=("track_url", lambda x: clean_join(x, max_items=10)),
                all_track_artist_names_examples=("all_track_artist_names", lambda x: clean_join(x, max_items=10)),
            )
            .reset_index()
            .sort_values(["n_seed_artists_connected", "n_tracks_with_seed_sample", "candidate_artist_name"], ascending=[False, False, True])
        )
    else:
        candidate_base = pd.DataFrame(columns=[
            "candidate_spotify_id", "candidate_artist_name", "n_tracks_with_seed_sample",
            "n_seed_artists_connected", "connected_seed_artists", "evidence_tracks", "evidence_years",
            "evidence_urls", "all_track_artist_names_examples",
        ])

    candidate_base["already_in_seed_sample"] = candidate_base["candidate_spotify_id"].isin(seed_artist_ids)
    candidate_base["source_wave"] = wave
    candidate_base = candidate_base[~candidate_base["already_in_seed_sample"]].copy()

    # -----------------------------------------------------
    # Metadata collection
    # -----------------------------------------------------

    completed_metadata_ids = set()
    metadata_records = []
    metadata_failures = []

    if args.resume and metadata_checkpoint_path.exists():
        checkpoint_df = pd.read_csv(metadata_checkpoint_path)
        metadata_records = checkpoint_df.to_dict("records")
        completed_metadata_ids = set(checkpoint_df["candidate_spotify_id"].dropna().astype(str))
        log(f"Resume mode: loaded metadata for {len(completed_metadata_ids)} candidates.")

    if args.resume and metadata_failures_path.exists():
        metadata_failures = pd.read_csv(metadata_failures_path).to_dict("records")

    metadata_candidates = candidate_base.copy()
    if args.max_candidates_metadata and args.max_candidates_metadata > 0:
        metadata_candidates = metadata_candidates.head(args.max_candidates_metadata).copy()
        log(f"Metadata collection capped at {len(metadata_candidates)} candidates.")

    total_metadata_candidates = len(metadata_candidates)

    for metadata_counter, (_, row) in enumerate(metadata_candidates.iterrows(), start=1):
        candidate_id = str(row["candidate_spotify_id"])
        candidate_name_from_track = row["candidate_artist_name"]

        if candidate_id in completed_metadata_ids:
            continue

        log(f"Metadata progress: {metadata_counter}/{total_metadata_candidates}. Candidate: {candidate_name_from_track} ({candidate_id})")

        try:
            spotify_meta = get_spotify_artist_by_id(
                sp=sp,
                artist_id=candidate_id,
                max_attempts=args.max_attempts,
                base_sleep=args.base_sleep,
            )
            time.sleep(args.sleep)
        except Exception as e:
            log(f"Could not collect Spotify metadata for {candidate_name_from_track}: {e}")
            metadata_failures.append({
                "candidate_spotify_id": candidate_id,
                "candidate_artist_name": candidate_name_from_track,
                "stage": "spotify_artist_metadata",
                "error": str(e),
            })
            save_csv_safely(pd.DataFrame(metadata_failures), metadata_failures_path)
            continue

        mb_meta = {
            "mbid": None, "mb_name": None, "mb_match_score": None,
            "mb_search_score": None, "mb_type": None, "mb_country": None,
            "mb_area": None, "mb_begin_area": None, "mb_disambiguation": None,
            "mb_data_quality_note": "external_metadata_skipped",
        }
        wd_meta = {
            "wikidata_qid": None, "wikidata_label": None,
            "wd_country_of_origin": None, "wd_country_of_citizenship": None,
            "wd_location_of_formation": None, "wd_place_of_birth": None,
            "wd_work_location": None, "wd_n_matches": 0,
            "wikidata_lookup_source": "external_metadata_skipped",
        }

        if not args.skip_external_metadata:
            artist_name_for_external = spotify_meta.get("candidate_artist_name_spotify") or candidate_name_from_track

            try:
                mb_meta = search_musicbrainz_artist(artist_name_for_external, sleep_seconds=args.mb_sleep)
            except Exception as e:
                log(f"MusicBrainz lookup failed for {artist_name_for_external}: {e}")
                mb_meta["mb_data_quality_note"] = "musicbrainz_request_failed"

            try:
                wd_meta = get_wikidata_metadata(
                    mbid=mb_meta.get("mbid"),
                    spotify_id=candidate_id,
                    sleep_seconds=args.wd_sleep,
                )
            except Exception as e:
                log(f"Wikidata lookup failed for {artist_name_for_external}: {e}")
                wd_meta["wikidata_lookup_source"] = "wikidata_request_failed"

        combined = {**row.to_dict(), **spotify_meta, **mb_meta, **wd_meta}
        combined["spotify_genres_readable"] = (
            "; ".join(spotify_meta.get("spotify_genres", []))
            if isinstance(spotify_meta.get("spotify_genres"), list)
            else ""
        )
        combined["has_relevant_scene_genre_keyword"] = has_scene_keyword(spotify_meta.get("spotify_genres", []))

        combined_series = pd.Series(combined)
        combined["country_scene_association"] = classify_country_scene_association(combined_series)
        combined["country_evidence"] = build_country_evidence(combined_series)
        combined["preliminary_scene_flag"] = preliminary_scene_flag(combined)

        combined["manual_scene_status"] = ""
        combined["manual_evidence_source"] = ""
        combined["manual_note"] = ""

        metadata_records.append(combined)
        save_csv_safely(pd.DataFrame(metadata_records), metadata_checkpoint_path)

    metadata_df = pd.DataFrame(metadata_records)

    if len(metadata_df) > 0:
        metadata_cols = [c for c in metadata_df.columns if c not in candidate_base.columns or c == "candidate_spotify_id"]
        final_candidates = candidate_base.merge(metadata_df[metadata_cols], on="candidate_spotify_id", how="left")
    else:
        final_candidates = candidate_base.copy()

    for col in ["manual_scene_status", "manual_evidence_source", "manual_note"]:
        if col not in final_candidates.columns:
            final_candidates[col] = ""

    final_candidates["priority_2plus_seed_artists"] = (
        final_candidates["n_seed_artists_connected"].fillna(0).astype(int) >= 2
    ).astype(int)
    final_candidates["priority_2plus_tracks"] = (
        final_candidates["n_tracks_with_seed_sample"].fillna(0).astype(int) >= 2
    ).astype(int)
    final_candidates["priority_candidate"] = (
        (final_candidates["priority_2plus_seed_artists"] == 1)
        | (final_candidates["priority_2plus_tracks"] == 1)
    ).astype(int)

    final_candidates = final_candidates.sort_values(
        ["priority_candidate", "n_seed_artists_connected", "n_tracks_with_seed_sample", "candidate_artist_name"],
        ascending=[False, False, False, True],
    )

    save_csv_safely(final_candidates, candidates_path)
    priority_candidates = final_candidates[final_candidates["priority_candidate"] == 1].copy()
    save_csv_safely(priority_candidates, priority_candidates_path)
    save_csv_safely(pd.DataFrame(metadata_failures), metadata_failures_path)

    # Summary.
    summary = pd.DataFrame([
        {"metric": "source_wave", "value": wave, "interpretation": "Snowball wave number."},
        {"metric": "n_seed_artists", "value": len(seed_artist_ids), "interpretation": "Number of artists in seed sample."},
        {"metric": "n_tracks_before_prefilter", "value": before_prefilter, "interpretation": "Unique track records before collaborator prefilter."},
        {"metric": "n_tracks_after_prefilter_or_cap_total", "value": original_tracks_after_prefilter, "interpretation": "Unique track records after prefilter/cap before manual continuation skip."},
        {"metric": "n_tracks_fetch_attempted_this_run", "value": len(tracks), "interpretation": "Unique track records re-queried from Spotify during this run."},
        {"metric": "n_tracks_manually_skipped_at_start", "value": int(args.skip_first_tracks or 0), "interpretation": "Prefiltered track rows skipped at the start for manual continuation."},
        {"metric": "n_candidate_track_evidence_rows", "value": len(track_evidence), "interpretation": "Candidate-seed-track evidence rows."},
        {"metric": "n_unique_candidate_artists", "value": len(final_candidates), "interpretation": "Unique non-seed artists found through coappearance."},
        {"metric": "n_priority_candidates", "value": len(priority_candidates), "interpretation": "Candidates with 2+ seed artists or 2+ tracks with seed sample."},
        {"metric": "n_likely_uk_or_uk_scene", "value": int((final_candidates.get("preliminary_scene_flag", pd.Series(dtype=str)) == "likely_uk_or_uk_scene").sum()) if "preliminary_scene_flag" in final_candidates.columns else 0, "interpretation": "Automatic preliminary UK/UK-scene flag."},
        {"metric": "n_non_uk_or_unclear_but_genre_relevant", "value": int((final_candidates.get("preliminary_scene_flag", pd.Series(dtype=str)) == "non_uk_or_unclear_but_genre_relevant").sum()) if "preliminary_scene_flag" in final_candidates.columns else 0, "interpretation": "Automatic preliminary genre relevance flag."},
        {"metric": "n_failed_tracks", "value": len(failed_tracks), "interpretation": "Spotify track requests that failed."},
        {"metric": "n_metadata_failures", "value": len(metadata_failures), "interpretation": "Candidate metadata collection failures."},
    ])
    save_csv_safely(summary, summary_path)

    log_section("SNOWBALL CANDIDATE GENERATION COMPLETED")
    log("Summary:")
    print(summary)

    log("Top candidate artists:")
    display_cols = [
        "candidate_artist_name", "candidate_spotify_id", "n_seed_artists_connected",
        "n_tracks_with_seed_sample", "connected_seed_artists",
        "preliminary_scene_flag", "country_scene_association", "spotify_genres_readable",
    ]
    existing = [c for c in display_cols if c in final_candidates.columns]
    if len(final_candidates) > 0:
        print(final_candidates.head(30)[existing])
    else:
        log("No candidates found.")

    log("Saved files:")
    print(f"Candidate artists: {candidates_path}")
    print(f"Priority candidates: {priority_candidates_path}")
    print(f"Candidate-seed edges: {candidate_edges_path}")
    print(f"Track evidence: {track_evidence_path}")
    print(f"Summary: {summary_path}")
    print(f"Failed tracks: {failed_tracks_path}")
    print(f"Metadata failures: {metadata_failures_path}")


if __name__ == "__main__":
    main()
