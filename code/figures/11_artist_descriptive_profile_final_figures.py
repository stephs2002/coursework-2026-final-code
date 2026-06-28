# code/11_artist_descriptive_profile_updated.py
#
# Updated descriptive profile script for the term paper.
# Run from the project root:
#   python code\11_artist_descriptive_profile_updated.py
#
# Outputs:
#   outputs/tables/artist_master_descriptive_96.csv
#   outputs/tables/sample_visibility_summary.csv
#   outputs/tables/genre_tag_distribution.csv
#   outputs/tables/top_genre_tags.csv
#   outputs/tables/label_count_distribution.csv
#   outputs/tables/top_release_label_names_by_artists.csv
#   outputs/tables/centrality_summary_by_layer.csv
#   outputs/tables/top_artists_by_followers.csv
#   outputs/tables/top_artists_by_popularity.csv
#   outputs/tables/top_artists_by_collaboration_degree.csv
#   outputs/tables/top_artists_by_label_degree.csv
#   outputs/tables/top_artists_by_genre_degree.csv
#
#   outputs/figures/fig_spotify_popularity_distribution.png
#   outputs/figures/fig_followers_log_distribution.png
#   outputs/figures/fig_genre_tag_count_distribution.png
#   outputs/figures/fig_label_count_distribution.png
#   outputs/figures/fig_top_release_label_names.png
#   outputs/figures/fig_followers_vs_collaboration_degree.png
#   outputs/figures/fig_collaboration_degree_distribution.png
#   outputs/figures/fig_top_artists_by_followers.png
#   outputs/figures/fig_top_artists_by_popularity.png
#   outputs/figures/fig_top_artists_by_collaboration_degree.png
#   outputs/figures/fig_top_genre_tags.png

from pathlib import Path
import ast
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(".")
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

TOP_N = 10


# -----------------------------
# Helpers
# -----------------------------

def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        warnings.warn(f"Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def find_col(df: pd.DataFrame, candidates):
    """Find first existing column by flexible lowercase matching."""
    if df.empty:
        return None

    lower_map = {c.lower().strip(): c for c in df.columns}

    for cand in candidates:
        cand_l = cand.lower().strip()
        if cand_l in lower_map:
            return lower_map[cand_l]

    # partial fallback
    for cand in candidates:
        cand_l = cand.lower().strip()
        for c in df.columns:
            if cand_l in c.lower():
                return c

    return None


def normalize_artist_name(s):
    if pd.isna(s):
        return np.nan
    return str(s).strip().lower()


def parse_genres(value):
    """Parse Spotify genre field robustly."""
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()
    if text in ["", "[]", "nan", "None", "null"]:
        return []

    # Try Python-list format: ['art pop', 'indie']
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    # Try separators
    if ";" in text:
        parts = text.split(";")
    elif "|" in text:
        parts = text.split("|")
    elif "," in text:
        parts = text.split(",")
    else:
        parts = [text]

    return [p.strip().strip("'").strip('"') for p in parts if p.strip()]


def simplify_genre_group(genres):
    """Rough genre grouping for a readable scatterplot legend."""
    if not genres:
        return "No Spotify genre tag"

    joined = " ".join(genres).lower()

    if any(x in joined for x in ["rap", "hip hop", "grime", "drill"]):
        return "Rap / grime / hip-hop"

    if any(x in joined for x in [
        "electronic", "electronica", "dance", "club", "techno",
        "house", "ambient", "bass", "deconstructed", "edm", "idm",
        "breakcore", "glitch"
    ]):
        return "Electronic / club"

    if any(x in joined for x in [
        "pop", "art pop", "hyperpop", "bedroom pop", "experimental pop"
    ]):
        return "Pop / experimental pop"

    if any(x in joined for x in [
        "indie", "rock", "post-punk", "punk", "shoegaze", "alternative"
    ]):
        return "Indie / rock / post-punk"

    if any(x in joined for x in [
        "folk", "singer-songwriter", "songwriter"
    ]):
        return "Folk / singer-songwriter"

    return "Other / mixed"


def choose_merge_keys(left: pd.DataFrame, right: pd.DataFrame):
    """Prefer spotify_id, otherwise artist_name_norm."""
    if "spotify_id" in left.columns and "spotify_id" in right.columns:
        return ["spotify_id"]
    if "artist_name_norm" in left.columns and "artist_name_norm" in right.columns:
        return ["artist_name_norm"]
    return None


def prefix_metrics(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Standardize and prefix metric columns."""
    if df.empty:
        return df

    df = df.copy()

    name_col = find_col(df, ["artist_name", "name", "artist", "matched_name"])
    id_col = find_col(df, ["spotify_id", "artist_id"])

    if name_col and name_col != "artist_name":
        df = df.rename(columns={name_col: "artist_name"})

    if id_col and id_col != "spotify_id":
        df = df.rename(columns={id_col: "spotify_id"})

    if "artist_name" in df.columns:
        df["artist_name_norm"] = df["artist_name"].apply(normalize_artist_name)

    keep_key_cols = {"spotify_id", "artist_name", "artist_name_norm"}
    rename = {}

    for c in df.columns:
        if c in keep_key_cols:
            continue
        rename[c] = f"{prefix}_{c}"

    df = df.rename(columns=rename)
    return df


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def save_hist(series, title, xlabel, filename, bins=20):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        print(f"Skipping {filename}: no numeric data")
        return

    plt.figure(figsize=(8.5, 5.2))
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of artists")
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=300)
    plt.close()


def save_bar_counts(series, title, xlabel, filename):
    s = series.dropna()
    if s.empty:
        print(f"Skipping {filename}: no data")
        return

    counts = s.value_counts().sort_index()

    plt.figure(figsize=(8.5, 5.2))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of artists")
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=300)
    plt.close()


def save_horizontal_bar(
    df,
    label_col,
    value_col,
    title,
    xlabel,
    filename,
    top_n=10,
    scale=1.0,
):
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        print(f"Skipping {filename}: required columns not found")
        return

    # Select columns robustly and avoid duplicate-name DataFrames.
    plot_df = df.loc[:, [label_col, value_col]].copy()
    plot_df = plot_df.loc[:, ~plot_df.columns.duplicated()]
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col])
    plot_df = plot_df.sort_values(value_col, ascending=False).head(top_n)

    if plot_df.empty:
        print(f"Skipping {filename}: no data after filtering")
        return

    plot_df["plot_value"] = plot_df[value_col] / scale
    plot_df = plot_df.iloc[::-1]

    plt.figure(figsize=(8.5, 5.4))
    plt.barh(plot_df[label_col], plot_df["plot_value"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=300)
    plt.close()


def find_degree_col(master: pd.DataFrame, prefix: str):
    candidates = [
        f"{prefix}_degree",
        f"{prefix}_Degree",
        f"{prefix}_degree_centrality",
        f"{prefix}_weighted_degree",
        f"{prefix}_strength",
    ]
    return find_col(master, candidates)


def pick_outlier_artists(master, collab_degree_col, n_by_degree=6, n_by_followers=5):
    """Return a small fixed set of analytically important artists for scatterplot labels.

    The goal is to avoid label clutter. The selected artists represent:
    - high collaboration degree within the sample;
    - high platform visibility;
    - useful contrast between visibility and network centrality.
    """
    preferred = [
        "Charli XCX",
        "Dean Blunt",
        "Katy J Pearson",
        "A. G. Cook",
        "GAIKA",
        "FKA twigs",
        "The 1975",
        "PinkPantheress",
    ]

    available = set(master["artist_name"].astype(str))
    return {name for name in preferred if name in available}


def scatter_label_offset(artist_name):
    """Manual label offsets for the scatterplot to reduce overlap."""
    offsets = {
        "Charli XCX": (6, 5),
        "Dean Blunt": (6, 5),
        "Katy J Pearson": (6, 5),
        "A. G. Cook": (6, 5),
        "GAIKA": (6, 5),
        "FKA twigs": (6, 5),
        "The 1975": (6, -16),
        "PinkPantheress": (6, 12),
    }
    return offsets.get(str(artist_name), (6, 5))


# -----------------------------
# 1. Load base artist data
# -----------------------------

nodes = read_csv_safe(TABLES / "nodes_clean.csv")

if nodes.empty:
    raise FileNotFoundError("nodes_clean.csv not found or empty. Cannot build master dataset.")

nodes = nodes.copy()

name_col = find_col(nodes, ["artist_name", "name", "artist", "matched_name"])
id_col = find_col(nodes, ["spotify_id", "artist_id"])

if name_col and name_col != "artist_name":
    nodes = nodes.rename(columns={name_col: "artist_name"})

if id_col and id_col != "spotify_id":
    nodes = nodes.rename(columns={id_col: "spotify_id"})

if "artist_name" not in nodes.columns:
    raise ValueError("Could not identify artist name column in nodes_clean.csv")

nodes["artist_name_norm"] = nodes["artist_name"].apply(normalize_artist_name)

# Spotify attributes
pop_col = find_col(nodes, ["popularity", "spotify_popularity", "artist_popularity"])
followers_col = find_col(nodes, ["followers", "followers_total", "spotify_followers"])
genres_col = find_col(nodes, ["genres", "spotify_genres", "artist_genres"])

if pop_col:
    nodes["spotify_popularity"] = pd.to_numeric(nodes[pop_col], errors="coerce")
else:
    nodes["spotify_popularity"] = np.nan

if followers_col:
    nodes["followers"] = pd.to_numeric(nodes[followers_col], errors="coerce")
else:
    nodes["followers"] = np.nan

nodes["followers_log10"] = np.log10(nodes["followers"].fillna(0) + 1)

if genres_col:
    nodes["genre_list"] = nodes[genres_col].apply(parse_genres)
else:
    nodes["genre_list"] = [[] for _ in range(len(nodes))]

nodes["n_genre_tags"] = nodes["genre_list"].apply(len)
nodes["all_genres"] = nodes["genre_list"].apply(lambda x: "; ".join(x))
nodes["primary_genre"] = nodes["genre_list"].apply(lambda x: x[0] if x else "No Spotify genre tag")
nodes["primary_genre_group"] = nodes["genre_list"].apply(simplify_genre_group)


# -----------------------------
# 2. Load and merge node metrics
# -----------------------------

genre_metrics = prefix_metrics(
    read_csv_safe(TABLES / "genre_network_node_metrics.csv"),
    "genre"
)

collab_metrics = prefix_metrics(
    read_csv_safe(TABLES / "collaboration_network_node_metrics.csv"),
    "collaboration"
)

label_metrics = prefix_metrics(
    read_csv_safe(TABLES / "label_network_node_metrics_clean_auto.csv"),
    "label"
)

community = prefix_metrics(
    read_csv_safe(TABLES / "multilayer_community_assignments.csv"),
    "community"
)

master = nodes.copy()

for df in [genre_metrics, collab_metrics, label_metrics, community]:
    if df.empty:
        continue

    keys = choose_merge_keys(master, df)
    if keys is None:
        print("Skipping merge: no compatible keys found.")
        print("Columns:", df.columns.tolist())
        continue

    cols_to_use = keys + [c for c in df.columns if c not in keys and c not in ["artist_name"]]
    df_small = df[cols_to_use].drop_duplicates(subset=keys)

    master = master.merge(df_small, on=keys, how="left")


# -----------------------------
# 3. Add clean label counts and label lists
# -----------------------------

label_summary = read_csv_safe(TABLES / "artist_label_summary_clean_auto.csv")

if label_summary.empty:
    label_summary = read_csv_safe(TABLES / "artist_label_summary_clean.csv")

if not label_summary.empty:
    label_summary = label_summary.copy()

    label_name_col = find_col(label_summary, ["normalized_label", "clean_label", "label", "label_name"])
    ls_artist_col = find_col(label_summary, ["artist_name", "name", "artist", "matched_name"])
    ls_id_col = find_col(label_summary, ["spotify_id", "artist_id"])

    if ls_artist_col and ls_artist_col != "artist_name":
        label_summary = label_summary.rename(columns={ls_artist_col: "artist_name"})
    if ls_id_col and ls_id_col != "spotify_id":
        label_summary = label_summary.rename(columns={ls_id_col: "spotify_id"})

    if "artist_name" in label_summary.columns:
        label_summary["artist_name_norm"] = label_summary["artist_name"].apply(normalize_artist_name)

    if label_name_col:
        if "spotify_id" in label_summary.columns and "spotify_id" in master.columns:
            label_group_key = "spotify_id"
        else:
            label_group_key = "artist_name_norm"

        label_agg = (
            label_summary
            .dropna(subset=[label_name_col])
            .groupby(label_group_key)[label_name_col]
            .agg(lambda x: sorted(set(str(v).strip() for v in x if str(v).strip())))
            .reset_index()
        )

        label_agg["n_labels"] = label_agg[label_name_col].apply(len)
        label_agg["labels"] = label_agg[label_name_col].apply(lambda x: "; ".join(x))
        label_agg = label_agg.drop(columns=[label_name_col])

        master = master.merge(label_agg, on=label_group_key, how="left")
    else:
        master["n_labels"] = np.nan
        master["labels"] = ""
else:
    master["n_labels"] = np.nan
    master["labels"] = ""

master["n_labels"] = master["n_labels"].fillna(0).astype(int)
master["labels"] = master["labels"].fillna("")


# -----------------------------
# 4. Identify useful degree columns
# -----------------------------

genre_degree_col = find_degree_col(master, "genre")
collab_degree_col = find_degree_col(master, "collaboration")
label_degree_col = find_degree_col(master, "label")

collab_weighted_col = find_col(master, [
    "collaboration_weighted_degree",
    "collaboration_strength",
    "collaboration_weighted_degree_centrality",
    "collaboration_collaboration_weight",
])

print("\nDetected important columns:")
print("popularity:", "spotify_popularity")
print("followers:", "followers")
print("followers_log10:", "followers_log10")
print("genre_degree:", genre_degree_col)
print("collaboration_degree:", collab_degree_col)
print("collaboration_weighted:", collab_weighted_col)
print("label_degree:", label_degree_col)


# -----------------------------
# 5. Save master dataset
# -----------------------------

master_out = TABLES / "artist_master_descriptive_96.csv"
master.to_csv(master_out, index=False)
print(f"\nSaved: {master_out}")


# -----------------------------
# 6. Summary tables
# -----------------------------

summary_rows = []

for col, label in [
    ("spotify_popularity", "Spotify popularity"),
    ("followers", "Followers"),
    ("followers_log10", "Followers (log10)"),
    ("n_genre_tags", "Number of Spotify genre tags"),
    ("n_labels", "Number of labels"),
]:
    if col in master.columns:
        s = pd.to_numeric(master[col], errors="coerce").dropna()
        if not s.empty:
            summary_rows.append({
                "metric": label,
                "n": int(s.shape[0]),
                "min": s.min(),
                "q25": s.quantile(0.25),
                "median": s.median(),
                "mean": s.mean(),
                "q75": s.quantile(0.75),
                "max": s.max(),
                "sd": s.std(),
            })

sample_summary = pd.DataFrame(summary_rows)
sample_summary.to_csv(TABLES / "sample_visibility_summary.csv", index=False)


genre_dist = (
    master["n_genre_tags"]
    .value_counts()
    .rename_axis("n_genre_tags")
    .reset_index(name="n_artists")
    .sort_values("n_genre_tags")
)
genre_dist.to_csv(TABLES / "genre_tag_distribution.csv", index=False)


# Top genre tags
all_tags = []
for tags in master["genre_list"]:
    all_tags.extend(tags)

top_genres = (
    pd.Series(all_tags, dtype="object")
    .value_counts()
    .reset_index()
)
top_genres.columns = ["genre_tag", "n_artists"]
top_genres.to_csv(TABLES / "top_genre_tags.csv", index=False)


label_dist = (
    master["n_labels"]
    .value_counts()
    .rename_axis("n_labels")
    .reset_index(name="n_artists")
    .sort_values("n_labels")
)
label_dist.to_csv(TABLES / "label_count_distribution.csv", index=False)


# Top release-label names from artist-label summary, counted by unique artists.
top_labels_by_artists = pd.DataFrame()

if not label_summary.empty:
    label_name_col = find_col(label_summary, ["normalized_label", "clean_label", "label", "label_name"])
    artist_key_col = None

    if "spotify_id" in label_summary.columns:
        artist_key_col = "spotify_id"
    elif "artist_name_norm" in label_summary.columns:
        artist_key_col = "artist_name_norm"

    if label_name_col and artist_key_col:
        top_labels_by_artists = (
            label_summary
            .dropna(subset=[label_name_col, artist_key_col])
            .assign(label_name=lambda d: d[label_name_col].astype(str).str.strip())
            .query("label_name != ''")
            .groupby("label_name")[artist_key_col]
            .nunique()
            .reset_index(name="n_artists")
            .sort_values("n_artists", ascending=False)
        )
        top_labels_by_artists.to_csv(TABLES / "top_release_label_names_by_artists.csv", index=False)


# Centrality / node metric summary
centrality_rows = []

metric_files = [
    ("genre", genre_metrics),
    ("collaboration", collab_metrics),
    ("label", label_metrics),
]

exclude_patterns = ["id", "community", "component", "cluster", "name"]

for layer, df in metric_files:
    if df.empty:
        continue

    for col in df.columns:
        if col in ["artist_name", "artist_name_norm", "spotify_id"]:
            continue
        if any(p in col.lower() for p in exclude_patterns):
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        centrality_rows.append({
            "layer": layer,
            "metric": col,
            "n": int(s.shape[0]),
            "min": s.min(),
            "q25": s.quantile(0.25),
            "median": s.median(),
            "mean": s.mean(),
            "q75": s.quantile(0.75),
            "max": s.max(),
            "sd": s.std(),
            "n_zero": int((s == 0).sum()),
        })

centrality_summary = pd.DataFrame(centrality_rows)
centrality_summary.to_csv(TABLES / "centrality_summary_by_layer.csv", index=False)


# Top artists tables
def save_top_artists(master_df, col, filename, n=10):
    if col is None or col not in master_df.columns:
        print(f"Skipping {filename}: column not found")
        return pd.DataFrame()

    # Avoid duplicate column names when `col` is already one of the descriptive columns
    base_cols = ["artist_name", "spotify_popularity", "followers", "followers_log10", "n_genre_tags", "n_labels"]
    out_cols = []
    for c in base_cols + [col]:
        if c in master_df.columns and c not in out_cols:
            out_cols.append(c)

    top = master_df[out_cols].copy()
    top["metric_value"] = pd.to_numeric(master_df[col], errors="coerce")
    top = (
        top
        .sort_values("metric_value", ascending=False)
        .head(n)
        .drop(columns=["metric_value"])
    )

    top.to_csv(TABLES / filename, index=False)
    return top


top_followers = save_top_artists(master, "followers", "top_artists_by_followers.csv", TOP_N)
top_popularity = save_top_artists(master, "spotify_popularity", "top_artists_by_popularity.csv", TOP_N)
top_collab_degree = save_top_artists(master, collab_degree_col, "top_artists_by_collaboration_degree.csv", TOP_N)
top_label_degree = save_top_artists(master, label_degree_col, "top_artists_by_label_degree.csv", TOP_N)
top_genre_degree = save_top_artists(master, genre_degree_col, "top_artists_by_genre_degree.csv", TOP_N)


# -----------------------------
# 7. Figures
# -----------------------------

save_hist(
    master["spotify_popularity"],
    "Distribution of Spotify artist popularity",
    "Spotify popularity",
    "fig_spotify_popularity_distribution.png",
    bins=20,
)

save_hist(
    master["followers_log10"],
    "Distribution of followers (log)",
    "Followers (log)",
    "fig_followers_log_distribution.png",
    bins=20,
)

save_bar_counts(
    master["n_genre_tags"],
    "Number of Spotify genre tags per artist",
    "Number of genre tags",
    "fig_genre_tag_count_distribution.png",
)

save_bar_counts(
    master["n_labels"],
    "Number of labels per artist",
    "Number of labels",
    "fig_label_count_distribution.png",
)


# Scatter: followers vs collaboration degree, with outlier labels.
if collab_degree_col is not None and collab_degree_col in master.columns:
    plot_df = master.copy()
    plot_df["collab_degree_plot"] = pd.to_numeric(plot_df[collab_degree_col], errors="coerce")
    plot_df["popularity_plot"] = pd.to_numeric(plot_df["spotify_popularity"], errors="coerce")
    plot_df["size_plot"] = plot_df["popularity_plot"].fillna(plot_df["popularity_plot"].median())

    # keep point sizes readable
    plot_df["size_plot"] = 20 + 3 * plot_df["size_plot"].fillna(0)

    plt.figure(figsize=(9.5, 6.2))

    for group, g in plot_df.groupby("primary_genre_group"):
        plt.scatter(
            g["followers_log10"],
            g["collab_degree_plot"],
            s=g["size_plot"],
            alpha=0.68,
            label=group,
        )

    names_to_label = pick_outlier_artists(master, collab_degree_col)

    # Add a little extra space for labels near the right and top borders.
    x_max = pd.to_numeric(plot_df["followers_log10"], errors="coerce").max()
    y_max = pd.to_numeric(plot_df["collab_degree_plot"], errors="coerce").max()
    if pd.notna(x_max):
        plt.xlim(right=x_max + 0.35)
    if pd.notna(y_max):
        plt.ylim(top=y_max + 0.9)

    for _, row in plot_df.iterrows():
        artist_name = str(row["artist_name"])
        if artist_name in names_to_label:
            x = row["followers_log10"]
            y = row["collab_degree_plot"]
            if pd.notna(x) and pd.notna(y):
                dx, dy = scatter_label_offset(artist_name)
                plt.annotate(
                    artist_name,
                    (x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8,
                    annotation_clip=False,
                )

    plt.title("Spotify followers and collaboration degree")
    plt.xlabel("Followers (log)")
    plt.ylabel("Collaboration degree")
    plt.legend(fontsize=8, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_followers_vs_collaboration_degree.png", dpi=300)
    plt.close()

    save_hist(
        plot_df["collab_degree_plot"],
        "Distribution of collaboration degree",
        "Collaboration degree",
        "fig_collaboration_degree_distribution.png",
        bins=20,
    )
else:
    print("Skipping scatterplot: collaboration degree column not found")


# Top genre tags figure, top 10
if not top_genres.empty:
    tg = top_genres.head(TOP_N).iloc[::-1]

    plt.figure(figsize=(8.5, 5.4))
    plt.barh(tg["genre_tag"], tg["n_artists"])
    plt.title("Top Spotify genre tags")
    plt.xlabel("Number of artists")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_top_genre_tags.png", dpi=300)
    plt.close()


# Top release-label names, top 10, by unique artists
if not top_labels_by_artists.empty:
    tl = top_labels_by_artists.head(TOP_N).iloc[::-1]

    plt.figure(figsize=(8.5, 5.4))
    plt.barh(tl["label_name"], tl["n_artists"])
    plt.title("Top release-label names in Spotify metadata")
    plt.xlabel("Number of artists")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_top_release_label_names.png", dpi=300)
    plt.close()


# Top artists by visibility and centrality
if not top_followers.empty:
    save_horizontal_bar(
        top_followers,
        "artist_name",
        "followers",
        f"Top {TOP_N} artists by Spotify followers",
        "Followers, millions",
        "fig_top_artists_by_followers.png",
        top_n=TOP_N,
        scale=1_000_000,
    )

if not top_popularity.empty:
    save_horizontal_bar(
        top_popularity,
        "artist_name",
        "spotify_popularity",
        f"Top {TOP_N} artists by Spotify popularity",
        "Spotify popularity",
        "fig_top_artists_by_popularity.png",
        top_n=TOP_N,
    )

if not top_collab_degree.empty and collab_degree_col in top_collab_degree.columns:
    save_horizontal_bar(
        top_collab_degree,
        "artist_name",
        collab_degree_col,
        f"Top {TOP_N} artists by collaboration degree",
        "Collaboration degree",
        "fig_top_artists_by_collaboration_degree.png",
        top_n=TOP_N,
    )


print("\nDone.")
print("Created / updated main table:")
print(" - outputs/tables/artist_master_descriptive_96.csv")
print("\nCreated / updated summary tables:")
print(" - outputs/tables/sample_visibility_summary.csv")
print(" - outputs/tables/genre_tag_distribution.csv")
print(" - outputs/tables/top_genre_tags.csv")
print(" - outputs/tables/label_count_distribution.csv")
print(" - outputs/tables/top_release_label_names_by_artists.csv")
print(" - outputs/tables/centrality_summary_by_layer.csv")
print(" - outputs/tables/top_artists_by_followers.csv")
print(" - outputs/tables/top_artists_by_popularity.csv")
print(" - outputs/tables/top_artists_by_collaboration_degree.csv")
print(" - outputs/tables/top_artists_by_label_degree.csv")
print(" - outputs/tables/top_artists_by_genre_degree.csv")
print("\nCreated / updated figures:")
print(" - outputs/figures/fig_spotify_popularity_distribution.png")
print(" - outputs/figures/fig_followers_log_distribution.png")
print(" - outputs/figures/fig_genre_tag_count_distribution.png")
print(" - outputs/figures/fig_label_count_distribution.png")
print(" - outputs/figures/fig_top_release_label_names.png")
print(" - outputs/figures/fig_followers_vs_collaboration_degree.png")
print(" - outputs/figures/fig_collaboration_degree_distribution.png")
print(" - outputs/figures/fig_top_artists_by_followers.png")
print(" - outputs/figures/fig_top_artists_by_popularity.png")
print(" - outputs/figures/fig_top_artists_by_collaboration_degree.png")
print(" - outputs/figures/fig_top_genre_tags.png")
