import os
import ast
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations


# ---------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------

INPUT_PATH = "data/raw/artists_raw_readable.csv"

OUTPUT_TABLES_DIR = "outputs/tables"
OUTPUT_FIGURES_DIR = "outputs/figures"
OUTPUT_NETWORKS_DIR = "outputs/networks"

os.makedirs(OUTPUT_TABLES_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_NETWORKS_DIR, exist_ok=True)

JACCARD_THRESHOLD = 0.5


# ---------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------

artists = pd.read_csv(INPUT_PATH)


# ---------------------------------------------------------
# 3. Helper function: parse genre strings
# ---------------------------------------------------------

def parse_genres(x):
    """
    Converts Spotify genre string into a clean Python list.
    Example: 'hyperpop; electroclash' -> ['hyperpop', 'electroclash']
    """
    if pd.isna(x):
        return []

    x = str(x).strip()

    if x == "" or x.lower() == "nan":
        return []

    return [g.strip().lower() for g in x.split(";") if g.strip()]


artists["genre_list"] = artists["genres"].apply(parse_genres)
artists["n_genres_clean"] = artists["genre_list"].apply(len)


# ---------------------------------------------------------
# 4. Exclude ambiguous Spotify matches
# ---------------------------------------------------------

manual_check = artists[artists["data_quality_note"] != "clear_match"].copy()

manual_check.to_csv(
    f"{OUTPUT_TABLES_DIR}/artists_requiring_manual_check.csv",
    index=False
)

artists_clean = artists[
    artists["data_quality_note"].isin(["clear_match", "manual_verified"])
].copy()

artists_clean.to_csv(
    f"{OUTPUT_TABLES_DIR}/nodes_clean.csv",
    index=False
)


# ---------------------------------------------------------
# 5. Build pairwise genre similarity table
# ---------------------------------------------------------

pair_records = []

for _, a in artists_clean.iterrows():
    for _, b in artists_clean.iterrows():
        pass

for i, j in combinations(artists_clean.index, 2):
    artist_a = artists_clean.loc[i]
    artist_b = artists_clean.loc[j]

    genres_a = set(artist_a["genre_list"])
    genres_b = set(artist_b["genre_list"])

    shared_genres = genres_a.intersection(genres_b)
    all_genres = genres_a.union(genres_b)

    if len(all_genres) == 0:
        jaccard = 0
    else:
        jaccard = len(shared_genres) / len(all_genres)

    pair_records.append({
        "source": artist_a["query_name"],
        "target": artist_b["query_name"],
        "source_spotify_id": artist_a["spotify_id"],
        "target_spotify_id": artist_b["spotify_id"],
        "source_genres": "; ".join(sorted(genres_a)),
        "target_genres": "; ".join(sorted(genres_b)),
        "shared_genres": "; ".join(sorted(shared_genres)),
        "shared_genres_count": len(shared_genres),
        "jaccard_similarity": round(jaccard, 4)
    })


pairs = pd.DataFrame(pair_records)

pairs.to_csv(
    f"{OUTPUT_TABLES_DIR}/artist_pairs_genre_similarity.csv",
    index=False
)


# ---------------------------------------------------------
# 6. Create edge lists
# ---------------------------------------------------------

genre_edges_any_overlap = pairs[pairs["jaccard_similarity"] > 0].copy()

genre_edges_threshold = pairs[
    pairs["jaccard_similarity"] >= JACCARD_THRESHOLD
].copy()

genre_edges_any_overlap.to_csv(
    f"{OUTPUT_TABLES_DIR}/genre_edges_any_overlap.csv",
    index=False
)

genre_edges_threshold.to_csv(
    f"{OUTPUT_TABLES_DIR}/genre_edges_threshold_{str(JACCARD_THRESHOLD).replace('.', '_')}.csv",
    index=False
)


# ---------------------------------------------------------
# 7. Build NetworkX graph
# ---------------------------------------------------------

G = nx.Graph()

for _, row in artists_clean.iterrows():
    G.add_node(
        row["query_name"],
        spotify_id=row["spotify_id"],
        spotify_name=row["spotify_name"],
        popularity=int(row["popularity"]) if not pd.isna(row["popularity"]) else 0,
        followers=int(row["followers"]) if not pd.isna(row["followers"]) else 0,
        genres="; ".join(row["genre_list"]),
        n_genres=int(row["n_genres_clean"])
    )

for _, row in genre_edges_threshold.iterrows():
    G.add_edge(
        row["source"],
        row["target"],
        weight=float(row["jaccard_similarity"]),
        shared_genres=row["shared_genres"],
        shared_genres_count=int(row["shared_genres_count"])
    )


# ---------------------------------------------------------
# 8. Network metrics
# ---------------------------------------------------------

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, weight=None)

# Community detection
if G.number_of_edges() > 0:
    communities = nx.algorithms.community.greedy_modularity_communities(
        G,
        weight="weight"
    )

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
        "degree_centrality": round(degree_centrality[node], 4),
        "betweenness_centrality": round(betweenness_centrality[node], 4),
        "community": community_map.get(node),
        "n_genres": G.nodes[node]["n_genres"],
        "genres": G.nodes[node]["genres"],
        "popularity": G.nodes[node]["popularity"],
        "followers": G.nodes[node]["followers"]
    })

node_metrics = pd.DataFrame(node_metrics)

node_metrics = node_metrics.sort_values(
    by=["degree_centrality", "betweenness_centrality"],
    ascending=False
)

node_metrics.to_csv(
    f"{OUTPUT_TABLES_DIR}/genre_network_node_metrics.csv",
    index=False
)


# ---------------------------------------------------------
# 9. Network-level summary
# ---------------------------------------------------------

manual_verified_count = int((artists["data_quality_note"] == "manual_verified").sum())

unresolved_ambiguous = artists[
    ~artists["data_quality_note"].isin(["clear_match", "manual_verified"])
].copy()

network_summary = pd.DataFrame([{
    "n_nodes": G.number_of_nodes(),
    "n_edges_threshold": G.number_of_edges(),
    "density": round(nx.density(G), 4),
    "n_connected_components": nx.number_connected_components(G),
    "jaccard_threshold": JACCARD_THRESHOLD,
    "n_manual_verified": manual_verified_count,
    "n_unresolved_ambiguous": len(unresolved_ambiguous),
    "n_nodes_without_genres": int((artists_clean["n_genres_clean"] == 0).sum())
}])

network_summary.to_csv(
    f"{OUTPUT_TABLES_DIR}/genre_network_summary.csv",
    index=False
)

print("\nNetwork summary:")
print(network_summary)

print("\nTop central artists:")
print(node_metrics.head(10)[[
    "artist",
    "degree",
    "degree_centrality",
    "betweenness_centrality",
    "community",
    "genres"
]])


# ---------------------------------------------------------
# 10. Save network for Gephi / future R analysis
# ---------------------------------------------------------

nx.write_graphml(
    G,
    f"{OUTPUT_NETWORKS_DIR}/genre_similarity_network_threshold_{str(JACCARD_THRESHOLD).replace('.', '_')}.graphml"
)


# ---------------------------------------------------------
# 11. Simple visualization
# ---------------------------------------------------------

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
    1 + G[u][v]["weight"] * 4
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
    alpha=0.35
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=8
)

plt.title(
    f"Spotify Genre Similarity Network, Jaccard ≥ {JACCARD_THRESHOLD}",
    fontsize=14
)

plt.axis("off")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FIGURES_DIR}/genre_similarity_network_threshold_{str(JACCARD_THRESHOLD).replace('.', '_')}.png",
    dpi=300
)

plt.show()