"""Checks that every released weights file loads and reproduces the AUC recorded at release.

Usage:  python verify_weights.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C                                   # noqa: E402
from ribo_ablation import common_genes, load_dense   # noqa: E402
from train_weights import SEED, build                # noqa: E402

W = Path(__file__).resolve().parent / 'weights'
genes = common_genes()
ok = True
for name in C.DATASETS:
    meta = json.load(open(W / f'{name}_mlcae.json'))
    X, y = load_dense(name, genes)
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    model = build(X.shape[1])
    model.load_weights(str(W / f'{name}_mlcae.weights.h5'))
    auc = roc_auc_score(yte, model.predict(xte, batch_size=2048, verbose=0)[1][:, 1])
    same = abs(auc - meta['test_auc']) < 0.002
    ok &= same
    print(f"{name:12s} loaded ok, held-out AUC {auc:.4f} (recorded {meta['test_auc']}) {'OK' if same else 'MISMATCH'}")
    del X, xtr, xte, model
print('\nall weights verified' if ok else '\nsome weights did not reproduce their recorded AUC')
