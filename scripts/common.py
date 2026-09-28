"""Shared helpers for the round-2 (R2) reviewer analyses (comments 1, 2, 5).

Design follows BIB-26-0605_R2_Response_to_Reviewers.docx:
  - outer 5-fold label-stratified CV per dataset (seed 4)
  - every data-dependent step (scaling, model fitting, SHAP / gene selection,
    downstream classifier + its hyperparameters) uses outer-training cells only
  - the outer-test fold is scored exactly once
Input space: the 10,697-gene common matrix used as MLC-AE input in the manuscript.
"""
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold

H5_DIR = Path(r'C:\Users\seeya\Documents\Codex\2026-07-13\395894\outputs\table1_intersection_h5_table1_samples')
RAW_DIR = Path(r'C:\Users\seeya\Documents\Codex\2026-07-13\395894\outputs\scRNAseq h5 files 3 17 2025-20260721T013732Z-1-001\scRNAseq h5 files 3 17 2025')
OUT = Path(r'C:\Users\seeya\Documents\claude\shap\r2_analysis\results')

DATASETS = ['PC3_high', 'PC3_low', 'C42B_high', 'C42B_low', 'MycCaP_high', 'MycCaP_low']
RAW_FILES = {
    'PC3_high': ('PC3_HighSerum_G0', 'PC3_HighSerum_NotG0'),
    'PC3_low': ('PC3_LowSerum_G0', 'PC3_LowSerum_NotG0'),
    'C42B_high': ('C42B_HighSerum_G0', 'C42B_HighSerum_NotG0'),
    'C42B_low': ('C42B_LowSerum_G0', 'C42B_LowSerum_NotG0'),
    'MycCaP_high': ('MycCaP_HighSerum_G0', 'MycCaP_HighSerum_NotG0'),
    'MycCaP_low': ('MycCaP_LowSerum_G0', 'MycCaP_LowSerum_NotG0'),
}
SEED = 4
N_FOLDS = 5
TOP_K = 10

# Ribosomal genes removed in the ablation: every symbol starting with RPL/RPS (any suffix, incl. RPLP0-2,
# RPSA, paralogs) plus mitochondrial MRPL/MRPS, case-insensitive. In the 10,697-gene matrix this matches
# 11 genes (RPL11, RPL15, RPL27, RPL36A, RPL37, RPL41, RPL7A, RPS18, RPS21, RPS26, RPS9).
RP_REGEX = re.compile(r'^M?RP[LS]', re.I)


def is_cytosolic_rp(gene):
    return bool(RP_REGEX.match(gene))


def load_dataset(name):
    """Returns X (float32, cells x 10,697), y (0/1), genes, cell names, group ('G0'/'NG0' from cell name).

    MycCaP_high in the source H5 stores every cell twice (suffixes _1/_3 and _2/_4 with identical
    expression rows); duplicates are removed so the same cell can never be in both train and test.
    """
    path = H5_DIR / f'{name}_table1_intersection_genes_table1_samples.h5'
    with h5py.File(path, 'r') as f:
        x = np.asarray(f['feature'], dtype=np.float32)
        y = np.asarray(f['label']).reshape(-1).astype(int)
        genes = [g.decode() for g in f['gene_name'][:]]
        cells = [s.decode() for s in f['sample'][:]]
    group = np.array(['NG0' if ('NGO_' in c or c.startswith('NG0')) else 'G0' for c in cells])
    barcodes = np.array([re.search(r'[ACGT]{16}', c).group(0) for c in cells])
    keys = [f'{g}|{b}' for g, b in zip(group, barcodes)]
    _, first = np.unique(keys, return_index=True)
    n_dup = len(keys) - len(first)
    if n_dup:
        keep = np.sort(first)
        x, y = x[keep], y[keep]
        cells = [cells[i] for i in keep]
        group, barcodes = group[keep], barcodes[keep]
    return dict(X=x, y=y, genes=genes, cells=cells, group=group, barcodes=barcodes, n_duplicates_removed=int(n_dup))


def outer_folds(y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y)]


def inner_split(y_train, fold_id, frac=0.2):
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=SEED + 100 + fold_id)
    return next(sss.split(np.zeros(len(y_train)), y_train))


def fold_dir(dataset, fold_id):
    d = OUT / dataset / f'fold_{fold_id}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


def load_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


# ---------------- DeLong test (Sun & Xu 2014 fast implementation) ----------------
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_paired(y, s1, s2):
    """Two-sided DeLong test for two correlated AUCs on the same cells. Returns auc1, auc2, z, p."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind='mergesort')
    y = y[order]
    preds = np.vstack([np.asarray(s1)[order], np.asarray(s2)[order]])
    m = int(y.sum())
    n = len(y) - m
    tx = np.array([_midrank(p[:m]) for p in preds])
    ty = np.array([_midrank(p[m:]) for p in preds])
    tz = np.array([_midrank(p) for p in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)
