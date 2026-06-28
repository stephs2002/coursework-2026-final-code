library(readr)
library(sna)

nodes <- read_csv("outputs/tables/nodes_clean.csv")
pairs <- read_csv("outputs/tables/network_comparison_pairs.csv")

artists <- nodes$query_name
n <- length(artists)

genre_mat <- matrix(0, nrow = n, ncol = n, dimnames = list(artists, artists))
collab_mat <- matrix(0, nrow = n, ncol = n, dimnames = list(artists, artists))

for (i in seq_len(nrow(pairs))) {
  a <- pairs$source[i]
  b <- pairs$target[i]
  
  genre_mat[a, b] <- pairs$jaccard_similarity[i]
  genre_mat[b, a] <- pairs$jaccard_similarity[i]
  
  collab_mat[a, b] <- pairs$collaboration[i]
  collab_mat[b, a] <- pairs$collaboration[i]
}

diag(genre_mat) <- 0
diag(collab_mat) <- 0

qap_model <- netlm(
  collab_mat,
  list(genre_mat),
  nullhyp = "qap",
  reps = 5000
)

sink("outputs/tables/qap_results.txt")
print(summary(qap_model))
sink()

print(summary(qap_model))