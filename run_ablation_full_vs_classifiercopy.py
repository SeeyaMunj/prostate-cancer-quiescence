from __future__ import annotations

import csv
import os
import random
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

BASE_DIR = Path(r"C:\Users\seeya\Documents\Codex\2026-07-13\395894")
H5_DIR = BASE_DIR / "outputs" / "table1_intersection_h5_table1_samples"
OUT_DIR = BASE_DIR / "outputs" / "comment4_ablation_full_vs_classifier_only_3epoch"

DATASETS = {
    "PC3_high": H5_DIR / "PC3_high_table1_intersection_genes_table1_samples.h5",
    "PC3_low": H5_DIR / "PC3_low_table1_intersection_genes_table1_samples.h5",
    "C42B_high": H5_DIR / "C42B_high_table1_intersection_genes_table1_samples.h5",
    "C42B_low": H5_DIR / "C42B_low_table1_intersection_genes_table1_samples.h5",
    "MycCaP_high": H5_DIR / "MycCaP_high_table1_intersection_genes_table1_samples.h5",
    "MycCaP_low": H5_DIR / "MycCaP_low_table1_intersection_genes_table1_samples.h5",
}

SEED = 4
EPOCHS = 3
BATCH_SIZE = 256
TEST_SIZE = 0.30


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["feature"], dtype=np.float32)
        y = np.asarray(handle["label"]).reshape(-1).astype(int)
    return x, y


def build_full_mlc_ae(input_dim: int) -> Model:
    inputs = Input(shape=(input_dim,), name="m_rna_input")
    x = BatchNormalization(name="inputs_4")(inputs)
    encoded = Dense(64, activation="relu", name="encoded")(x)
    decoded_hidden = Dense(512, activation="relu", name="inputs_5")(encoded)
    reconstruction = Dense(input_dim, activation="linear", name="m_rna")(decoded_hidden)
    category = Dense(2, activation="softmax", name="category")(encoded)
    model = Model(inputs=inputs, outputs=[reconstruction, category], name="full_mlc_ae")
    model.compile(optimizer="adam", loss=["mse", "cosine_similarity"], loss_weights=[0.001, 0.5], metrics={"category": ["accuracy"]})
    return model


def build_classifier_only(input_dim: int) -> Model:
    inputs = Input(shape=(input_dim,), name="m_rna_input")
    x = BatchNormalization(name="inputs_4")(inputs)
    encoded = Dense(64, activation="relu", name="encoded")(x)
    category = Dense(2, activation="softmax", name="category")(encoded)
    model = Model(inputs=inputs, outputs=category, name="classifier_only")
    model.compile(optimizer="adam", loss="cosine_similarity", metrics=["accuracy"])
    return model


def auc_from_probs(y_true: np.ndarray, probs: np.ndarray) -> float:
    auc_1 = roc_auc_score(y_true, probs[:, 1])
    auc_0 = roc_auc_score(y_true, probs[:, 0])
    return float(max(auc_0, auc_1))


def run_dataset(name: str, path: Path) -> list[dict[str, object]]:
    x, y = load_h5(path)
    y_cat = to_categorical(y, num_classes=2)
    train_x, test_x, train_y, test_y, train_y_cat, _ = train_test_split(x, y, y_cat, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    rows: list[dict[str, object]] = []

    set_seed(SEED)
    full_model = build_full_mlc_ae(x.shape[1])
    full_model.fit(train_x, [train_x, train_y_cat], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, shuffle=True)
    _, full_probs = full_model.predict(test_x, batch_size=BATCH_SIZE, verbose=0)
    rows.append({"dataset": name, "model": "full_mlc_ae", "auc": f"{auc_from_probs(test_y, full_probs):.6f}", "n_cells": x.shape[0], "n_genes": x.shape[1], "epochs": EPOCHS, "batch_size": BATCH_SIZE, "seed": SEED})

    tf.keras.backend.clear_session()
    set_seed(SEED)
    classifier = build_classifier_only(x.shape[1])
    classifier.fit(train_x, train_y_cat, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, shuffle=True)
    class_probs = classifier.predict(test_x, batch_size=BATCH_SIZE, verbose=0)
    rows.append({"dataset": name, "model": "classifier_only_no_decoder", "auc": f"{auc_from_probs(test_y, class_probs):.6f}", "n_cells": x.shape[0], "n_genes": x.shape[1], "epochs": EPOCHS, "batch_size": BATCH_SIZE, "seed": SEED})

    tf.keras.backend.clear_session()
    del x, y, y_cat, train_x, test_x, train_y, test_y, train_y_cat
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "ablation_full_vs_classifier_only_3epoch.csv"
    all_rows: list[dict[str, object]] = []
    for name, path in DATASETS.items():
        print(f"Running {name}...", flush=True)
        rows = run_dataset(name, path)
        all_rows.extend(rows)
        for row in rows:
            print(f"  {row['model']}: AUC={row['auc']}", flush=True)
        with out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"Wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
