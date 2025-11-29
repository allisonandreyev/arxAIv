import csv
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

# Default locations (kept from the original Colab setup)
REAL_DIR = Path("drive/MyDrive/Real Figures")
AI_DIR = Path("drive/MyDrive/CVPR AI Figures")
OUTPUT_CSV = Path("figure_metrics.csv")


def load_model(model_name: str = "ViT-H-14", pretrained: str = "laion2b_s32b_b79k"):
    """Create the CLIP model and preprocessing pipeline."""
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    return model, preprocess


def embed_image(path: Path, model: torch.nn.Module, preprocess) -> np.ndarray:
    """Return a normalized CLIP embedding for an image."""
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(img)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


def embed_directory(
    image_dir: Path, model: torch.nn.Module, preprocess, suffix: str = ".png"
) -> np.ndarray:
    """Embed all images in a directory."""
    embeddings: List[np.ndarray] = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(suffix):
            embeddings.append(embed_image(image_dir / fname, model, preprocess))
    if not embeddings:
        raise ValueError(f"No images with suffix '{suffix}' found in {image_dir}")
    return np.vstack(embeddings)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def structural_score(path: Path) -> float:
    """Simple structure proxy combining edge density and entropy."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    edges = cv2.Canny(img, 100, 200)
    edge_density = edges.mean()

    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    return 0.6 * entropy + 0.4 * edge_density


def score_ai_figures(real_dir: Path, ai_dir: Path, output_csv: Path = OUTPUT_CSV) -> None:
    """Compute CLIP similarity and simple structure metrics for AI figures."""
    model, preprocess = load_model()

    real_embeddings = embed_directory(real_dir, model, preprocess)
    centroid = real_embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    scores: List[Tuple[str, float, float]] = []
    for fname in sorted(os.listdir(ai_dir)):
        if fname.endswith(".png"):
            emb = embed_image(ai_dir / fname, model, preprocess)
            sim = cosine(emb, centroid)
            struct = structural_score(ai_dir / fname)
            scores.append((fname, sim, struct))

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "clip_similarity", "structural_complexity"])
        for fname, sim, struct in scores:
            writer.writerow([fname, sim, struct])


if __name__ == "__main__":
    score_ai_figures(REAL_DIR, AI_DIR, OUTPUT_CSV)
