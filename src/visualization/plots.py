# src/viz/plots.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


@dataclass
class TSNEConfig:
    perplexity: int = 30
    seed: int = 42
    connect_k: int = 200


def plot_tsne_pairs(G: np.ndarray, T: np.ndarray, out_path: str, cfg: TSNEConfig):
    X = np.vstack([G, T])  # [2N, D]
    tsne = TSNE(
        n_components=2,
        random_state=cfg.seed,
        perplexity=cfg.perplexity,
        init="random",
        learning_rate="auto",
    )
    X2 = tsne.fit_transform(X)

    N = G.shape[0]
    Xg = X2[:N]
    Xt = X2[N:]

    plt.figure(figsize=(10, 8))
    plt.scatter(Xg[:, 0], Xg[:, 1], s=18, alpha=0.65, marker="o")
    plt.scatter(Xt[:, 0], Xt[:, 1], s=60, alpha=0.65, marker="*")

    for i in range(min(N, cfg.connect_k)):
        plt.plot([Xg[i, 0], Xt[i, 0]], [Xg[i, 1], Xt[i, 1]], linewidth=0.5, alpha=0.2)

    plt.title("t-SNE: Graph (o) vs Text (*) embeddings")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_similarity_heatmap(
    G: np.ndarray,
    T: np.ndarray,
    out_path: str,
    title: str = "Cosine similarity (Graph @ Text^T)",
):
    # embeddings are L2-normalized => dot product = cosine similarity
    S = G @ T.T  # [B, B]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(S, aspect="auto")
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("Text index in batch")
    plt.ylabel("Graph index in batch")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    diag = np.diag(S)
    off = S.copy()
    np.fill_diagonal(off, np.nan)
    stats = {
        "diag_mean": float(np.nanmean(diag)),
        "diag_std": float(np.nanstd(diag)),
        "off_mean": float(np.nanmean(off)),
        "off_std": float(np.nanstd(off)),
    }
    return stats


def plot_diag_hist(G: np.ndarray, T: np.ndarray, out_path: str, bins: int = 50):
    diag = np.sum(G * T, axis=1)  # cosine per matched pair

    plt.figure(figsize=(7, 5))
    plt.hist(diag, bins=bins, alpha=0.8)
    plt.title("Histogram: cosine similarity of matched (graph,text) pairs")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
