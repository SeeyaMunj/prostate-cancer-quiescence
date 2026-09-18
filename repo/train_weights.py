"""Train the MLC-AE on each cell-line/serum dataset and save the trained weights.

Reproduces the released model weights in weights/. One model per dataset, manuscript architecture and settings:
    input -> BatchNormalization -> Dense 64 (ReLU, 'encoded')
      decoder branch: Dense 512 (ReLU) -> Dense n_genes (linear)
      classification branch: Dense 2 (softmax, 'category')
    loss = 0.001 x MSE(reconstruction) + 0.5 x cosine_similarity(one-hot G0/non-G0); Adam(1e-3), batch 256, 10 epochs
Cells are split 70/30 (stratified, seed 4); the model is fitted on the 70% training split only and scored once on the
held-out 30%. Saved per dataset: weights, the gene order used as input, and a JSON with the test AUC and settings.

Usage:  python train_weights.py [dataset ...]      (default: all six)
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C                      # noqa: E402
from ribo_ablation import common_genes, load_dense   # noqa: E402

OUT = Path(__file__).resolve().parent / 'weights'
OUT.mkdir(exist_ok=True)
SEED = 4
EPOCHS = 10
BATCH = 256


def build(n_genes, seed=SEED):
    import tensorflow as tf
    from tensorflow.keras.layers import BatchNormalization, Dense, Input
    from tensorflow.keras.models import Model
    tf.keras.backend.clear_session()
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    inp = Input(shape=(n_genes,))
    enc = Dense(64, activation='relu', name='encoded')(BatchNormalization()(inp))
    dec = Dense(n_genes, activation='linear', name='reconstruction')(Dense(512, activation='relu')(enc))
    cat = Dense(2, activation='softmax', name='category')(enc)
    m = Model(inp, [dec, cat])
    m.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=['mse', 'cosine_similarity'], loss_weights=[0.001, 0.5])
    return m


def run_example():
    """Smoke test: trains on example_data/example_pc3_high.npz (800 cells x 1,500 genes, ~1 minute)."""
    d = np.load(Path(__file__).resolve().parent / 'example_data' / 'example_pc3_high.npz', allow_pickle=True)
    X, y = d['X'], d['y'].astype(int)
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    model = build(X.shape[1])
    model.fit(xtr, [xtr, np.eye(2, dtype=np.float32)[ytr]], epochs=EPOCHS, batch_size=BATCH, verbose=2)
    p = model.predict(xte, batch_size=2048, verbose=0)[1][:, 1]
    print(f'example run OK - {X.shape[0]} cells x {X.shape[1]} genes, held-out AUC {roc_auc_score(yte, p):.3f}')


def main(names):
    genes = common_genes()
    for name in names:
        t0 = time.time()
        X, y = load_dense(name, genes)
        xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        model = build(X.shape[1])
        model.fit(xtr, [xtr, np.eye(2, dtype=np.float32)[ytr]], epochs=EPOCHS, batch_size=BATCH, verbose=2, shuffle=True)
        p = model.predict(xte, batch_size=2048, verbose=0)[1][:, 1]
        auc = float(roc_auc_score(yte, p))
        model.save_weights(str(OUT / f'{name}_mlcae.weights.h5'))
        json.dump({'dataset': name, 'n_genes': int(X.shape[1]), 'n_cells': int(len(y)), 'n_G0': int(y.sum()),
                   'train_cells': int(len(ytr)), 'test_cells': int(len(yte)), 'test_auc': round(auc, 4),
                   'epochs': EPOCHS, 'batch_size': BATCH, 'seed': SEED, 'split': '70/30 stratified',
                   'architecture': 'BN -> Dense64 ReLU (encoded) -> [Dense512 ReLU -> Dense n_genes linear] + [Dense2 softmax]',
                   'loss': '0.001*MSE + 0.5*cosine_similarity', 'optimizer': 'Adam(0.001)',
                   'seconds': round(time.time() - t0, 1)},
                  open(OUT / f'{name}_mlcae.json', 'w'), indent=1)
        print(f'{name}: test AUC {auc:.4f}  ({time.time() - t0:.0f}s)', flush=True)
        del X, xtr, xte, model
    (OUT / 'input_genes.txt').write_text('\n'.join(genes), encoding='utf-8')
    print('saved', len(genes), 'input gene symbols to weights/input_genes.txt')


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--example' in args:
        run_example()
    else:
        main(args or list(C.DATASETS))
