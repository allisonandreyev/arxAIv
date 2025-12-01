import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import open_clip
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif")

def load_model(device="cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.to(device)
    model.eval()
    return model, preprocess

def embed_directory(image_dir, out_path="fig_embeddings.npz", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = load_model(device)

    embeddings = []
    filenames = []

    with torch.no_grad():
        for fname in tqdm(sorted(os.listdir(image_dir)), desc="Embedding images"):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue

            path = os.path.join(image_dir, fname)
            img = Image.open(path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            feat = model.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)  # unit norm
            embeddings.append(feat.cpu().numpy()[0])
            filenames.append(fname)

    embeddings = np.stack(embeddings)  # (N, D)

    # Center and z-score to highlight deviations from the "average" figure
    scaler = StandardScaler(with_mean=True, with_std=True)
    z = scaler.fit_transform(embeddings)

    # Extra deviation emphasis:
    # scale each vector by (norm)^alpha so outliers get pushed out
    norms = np.linalg.norm(z, axis=1, keepdims=True)  # shape (N, 1)
    alpha = 0.5  # tweak this: higher alpha = more exaggeration
    z_emph = z * (norms ** alpha)

    # Reduce to 3D for initial positions in the graph
    pca = PCA(n_components=3)
    coords3d = pca.fit_transform(z_emph)

    np.savez(
        out_path,
        filenames=np.array(filenames),
        embeddings=embeddings,
        z=z,
        z_emph=z_emph,
        coords3d=coords3d,
    )
    print(f"Saved embeddings to {out_path}")

if __name__ == "__main__":
    # change "figures" to your folder of images
    embed_directory("figures", out_path="fig_embeddings.npz")

# build_figure_graph.py

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def build_graph(npz_path="fig_embeddings.npz",
                out_path="figure_graph.json",
                k_neighbors=5):
    data = np.load(npz_path, allow_pickle=True)
    filenames = data["filenames"]
    z_emph = data["z_emph"]
    coords3d = data["coords3d"]

    n = len(filenames)

    # Cosine distance on the deviation-emphasized embeddings
    dist = cosine_distances(z_emph)

    nodes = []
    for i, name in enumerate(filenames):
        deviation = float(np.linalg.norm(z_emph[i]))  # how "weird" this figure is
        nodes.append({
            "id": int(i),
            "name": str(name),
            "deviation": deviation,
            "x": float(coords3d[i, 0]),
            "y": float(coords3d[i, 1]),
            "z": float(coords3d[i, 2]),
        })

    links = []
    for i in range(n):
        # sort by distance, skip self at index 0
        nn_idx = np.argsort(dist[i])[1:k_neighbors + 1]
        for j in nn_idx:
            d = dist[i, j]
            weight = 1.0 / (1e-6 + d)  # larger weight for more similar figures
            links.append({
                "source": int(i),
                "target": int(j),
                "weight": float(weight),
            })

    graph = {"nodes": nodes, "links": links}

    with open(out_path, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"Wrote graph JSON to {out_path}")

if __name__ == "__main__":
    build_graph("fig_embeddings.npz", "figure_graph.json", k_neighbors=6)
