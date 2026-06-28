import os
import pandas as pd
from sklearn.metrics import adjusted_rand_score

OUTPUT_TABLES_DIR = "outputs/tables"
os.makedirs(OUTPUT_TABLES_DIR, exist_ok=True)

genre = pd.read_csv("outputs/tables/genre_network_node_metrics.csv")
collab = pd.read_csv("outputs/tables/collaboration_network_node_metrics.csv")

df = genre[["artist", "community"]].rename(
    columns={"community": "genre_community"}
).merge(
    collab[["artist", "community"]].rename(
        columns={"community": "collaboration_community"}
    ),
    on="artist",
    how="inner"
)

ari = adjusted_rand_score(
    df["genre_community"],
    df["collaboration_community"]
)

summary = pd.DataFrame([{
    "n_artists_compared": len(df),
    "adjusted_rand_index": round(ari, 4)
}])

df.to_csv(
    "outputs/tables/community_comparison_artist_level.csv",
    index=False
)

summary.to_csv(
    "outputs/tables/community_comparison_ari_summary.csv",
    index=False
)

print("Community comparison completed.")
print(summary)