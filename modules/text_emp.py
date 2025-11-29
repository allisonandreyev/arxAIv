import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

CSV_PATH = Path("papers.csv")
THRESHOLD_GRAPH = Path("paper_graph_threshold.json")
KNN_GRAPH = Path("paper_graph2.json")
K_NEIGHBORS = 6
THRESHOLD_STD_SCALE = 0.25

def embed_corpus(df: pd.DataFrame, model_name: str = "allenai/specter2_base"):
    """Encode paper text and return embeddings plus node metadata."""
    model = SentenceTransformer(model_name)
    embeddings: List[np.ndarray] = []
    nodes: List[dict] = []

    for idx, row in df.iterrows():
        fname = row.get("filename", f"id_{idx}")
        txt = str(row.get("CONTENT", ""))

        emb = model.encode(txt, normalize_embeddings=True)
        embeddings.append(emb)
        nodes.append({"id": fname, "text": txt, "embedding": emb.tolist()})

    return np.vstack(embeddings), nodes


def exaggerate(emb: np.ndarray, power: float = 3.0, stretch: float = 5.0) -> np.ndarray:
    """Nonlinear scaling to spread embeddings for clearer separation."""
    emb = emb - emb.mean(axis=0)
    emb = normalize(emb)
    emb = np.sign(emb) * (np.abs(emb) ** power)
    return emb * stretch


def similarity_threshold_links(
    embeddings: np.ndarray, nodes: List[dict], std_scale: float = THRESHOLD_STD_SCALE
) -> Tuple[List[dict], float]:
    """Create links above a similarity threshold derived from the distribution."""
    sim_matrix = cosine_similarity(embeddings)
    upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

    threshold = float(upper.mean() + std_scale * upper.std())
    links: List[dict] = []

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            sim = sim_matrix[i][j]
            if sim >= threshold:
                links.append(
                    {"source": nodes[i]["id"], "target": nodes[j]["id"], "weight": float(sim)}
                )

    return links, threshold


def knn_links(
    embeddings: np.ndarray, df: pd.DataFrame, k: int = K_NEIGHBORS, metric: str = "cosine"
) -> Tuple[List[dict], List[dict]]:
    """Build k-NN graph links using cosine distance."""
    filenames = df["filename"]

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    nodes: List[dict] = []
    links: List[dict] = []

    for i, fname in enumerate(filenames):
        nodes.append(
            {
                "id": fname,
                "text": df.loc[i, "CONTENT"],
                "cluster": int(df.loc[i, "cluster"]) if "cluster" in df.columns else None,
            }
        )

        for neighbor_idx, dist in zip(indices[i][1:], distances[i][1:]):
            sim = 1 - dist  # cosine similarity
            links.append(
                {
                    "source": fname,
                    "target": filenames[neighbor_idx],
                    "weight": float(sim),
                    "distance": float(dist),
                }
            )

    return nodes, links


def plot_similarity_histogram(embeddings: np.ndarray) -> None:
    """Quick visualization of pairwise similarity distribution."""
    sims = cosine_similarity(embeddings)
    upper = sims[np.triu_indices_from(sims, k=1)]
    plt.hist(upper, bins=50)
    plt.title("Cosine Similarity Distribution")
    plt.show()


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2))


def main():
    df = pd.read_csv(CSV_PATH)

    embeddings, nodes = embed_corpus(df)
    plot_similarity_histogram(embeddings)

    exaggerated = exaggerate(embeddings, power=3.0, stretch=5.0)
    links, threshold = similarity_threshold_links(exaggerated, nodes)
    save_json({"nodes": nodes, "links": links}, THRESHOLD_GRAPH)

    print(f"Auto-selected threshold: {threshold}")
    print(f"Saved threshold graph: {THRESHOLD_GRAPH}")
    print(f"Nodes: {len(nodes)} | Links: {len(links)}")

    knn_nodes, knn_edges = knn_links(exaggerated, df, k=K_NEIGHBORS)
    save_json({"nodes": knn_nodes, "links": knn_edges}, KNN_GRAPH)
    print(f"Saved k-NN graph: {KNN_GRAPH} (k={K_NEIGHBORS})")


if __name__ == "__main__":
    main()
