from __future__ import annotations

import csv
import os
import random
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

BASE_DIR = Path(r"C:\Users\seeya\Documents\Codex\2026-07-13\395894")
H5_DIR = BASE_DIR / "outputs" / "table1_intersection_h5_table1_samples"
OUT_DIR = BASE_DIR / "outputs" / "comment4_latent_space_umap_all_six_3epoch"
FIG_DIR = OUT_DIR / "figures"

DATASETS = {
    "PC3 high serum": H5_DIR / "PC3_high_table1_intersection_genes_table1_samples.h5",
    "PC3 low serum": H5_DIR / "PC3_low_table1_intersection_genes_table1_samples.h5",
    "C4-2B high serum": H5_DIR / "C42B_high_table1_intersection_genes_table1_samples.h5",
    "C4-2B low serum": H5_DIR / "C42B_low_table1_intersection_genes_table1_samples.h5",
    "Myc-CaP high serum": H5_DIR / "MycCaP_high_table1_intersection_genes_table1_samples.h5",
    "Myc-CaP low serum": H5_DIR / "MycCaP_low_table1_intersection_genes_table1_samples.h5",
}

SEED = 4
EPOCHS = 3
BATCH_SIZE = 256
TEST_SIZE = 0.30
MAX_UMAP_CELLS = 12000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["feature"], dtype=np.float32)
        y = np.asarray(handle["label"]).reshape(-1).astype(int)
    return x, y


def build_full_mlc_ae(input_dim: int) -> tuple[Model, Model]:
    inputs = Input(shape=(input_dim,), name="m_rna_input")
    x = BatchNormalization(name="inputs_4")(inputs)
    encoded = Dense(64, activation="relu", name="encoded")(x)
    decoded_hidden = Dense(512, activation="relu", name="inputs_5")(encoded)
    reconstruction = Dense(input_dim, activation="linear", name="m_rna")(decoded_hidden)
    category = Dense(2, activation="softmax", name="category")(encoded)
    model = Model(inputs=inputs, outputs=[reconstruction, category], name="full_mlc_ae")
    model.compile(optimizer="adam", loss=["mse", "cosine_similarity"], loss_weights=[0.001, 0.5], metrics={"category": ["accuracy"]})
    encoder = Model(inputs=inputs, outputs=encoded, name="encoder")
    return model, encoder


def stratified_sample_indices(y: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if len(y) <= max_n:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    idxs = []
    for cls, count in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        take = max(1, int(round(max_n * count / len(y))))
        take = min(take, len(cls_idx))
        idxs.append(rng.choice(cls_idx, size=take, replace=False))
    out = np.concatenate(idxs)
    if len(out) > max_n:
        out = rng.choice(out, size=max_n, replace=False)
    rng.shuffle(out)
    return out


def plot_umap(dataset: str, emb: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=180)
    colors = np.where(labels == 1, "#b23a48", "#2f6f9f")
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=4, alpha=0.65, linewidths=0)
    ax.set_title(f"MLC-AE latent UMAP: {dataset}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2f6f9f", markersize=6, label="label 0"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#b23a48", markersize=6, label="label 1"),
    ]
    ax.legend(handles=handles, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_dataset(dataset: str, path: Path) -> dict[str, object]:
    print(f"Running {dataset}...", flush=True)
    x, y = load_h5(path)
    y_cat = to_categorical(y, num_classes=2)
    train_x, _test_x, train_y, _test_y, train_y_cat, _test_y_cat = train_test_split(
        x, y, y_cat, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    set_seed(SEED)
    model, encoder = build_full_mlc_ae(x.shape[1])
    model.fit(train_x, [train_x, train_y_cat], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, shuffle=True)
    latent = encoder.predict(x, batch_size=BATCH_SIZE, verbose=0).astype(np.float32)

    n_total_cells = len(y)
    sample_idx = stratified_sample_indices(y, MAX_UMAP_CELLS, SEED)
    latent_sample = latent[sample_idx]
    y_sample = y[sample_idx]

    pca = PCA(n_components=2, random_state=SEED)
    pcs = pca.fit_transform(latent_sample)
    try:
        pca_sil = float(silhouette_score(pcs, y_sample)) if len(np.unique(y_sample)) > 1 else float("nan")
    except Exception:
        pca_sil = float("nan")

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="euclidean", random_state=SEED)
    emb = reducer.fit_transform(latent_sample)
    try:
        umap_sil = float(silhouette_score(emb, y_sample)) if len(np.unique(y_sample)) > 1 else float("nan")
    except Exception:
        umap_sil = float("nan")

    safe = dataset.replace(" ", "_").replace("-", "").replace("/", "_")
    fig_path = FIG_DIR / f"latent_umap_{safe}.png"
    plot_umap(dataset, emb, y_sample, fig_path)

    np.save(OUT_DIR / f"latent_sample_{safe}.npy", latent_sample)
    np.save(OUT_DIR / f"latent_umap_{safe}.npy", emb)
    np.save(OUT_DIR / f"labels_sample_{safe}.npy", y_sample)

    tf.keras.backend.clear_session()
    del x, y, y_cat, train_x, train_y, train_y_cat, latent, latent_sample, emb

    return {
        "dataset": dataset,
        "n_cells_total": int(n_total_cells),
        "n_cells_used_for_umap": int(len(sample_idx)),
        "n_latent_dimensions": 64,
        "epochs": EPOCHS,
        "pca_var_pc1": f"{pca.explained_variance_ratio_[0]:.6f}",
        "pca_var_pc2": f"{pca.explained_variance_ratio_[1]:.6f}",
        "pca_silhouette_by_label": f"{pca_sil:.6f}",
        "umap_silhouette_by_label": f"{umap_sil:.6f}",
        "figure": str(fig_path),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, path in DATASETS.items():
        row = run_dataset(dataset, path)
        rows.append(row)
        out_csv = OUT_DIR / "latent_umap_summary.csv"
        with out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote figure: {row['figure']}", flush=True)
    print(f"Wrote {OUT_DIR / 'latent_umap_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()




