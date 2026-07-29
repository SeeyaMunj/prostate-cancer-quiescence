import os, warnings
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.exceptions import ConvergenceWarning

BASE = r"C:\Users\User\Documents\ "
DATA_DIR = os.path.join(BASE, "outputs", "table1_intersection_h5_table1_samples")
OUT_DIR = os.path.join(BASE, "outputs", "comment4_permuted_label_ablation_10697")
os.makedirs(OUT_DIR, exist_ok=True)
DATASETS = {
    "PC3_high": "PC3_high_table1_intersection_genes_table1_samples.h5",
    "PC3_low": "PC3_low_table1_intersection_genes_table1_samples.h5",
    "C42B_high": "C42B_high_table1_intersection_genes_table1_samples.h5",
    "C42B_low": "C42B_low_table1_intersection_genes_table1_samples.h5",
    "MycCaP_high": "MycCaP_high_table1_intersection_genes_table1_samples.h5",
    "MycCaP_low": "MycCaP_low_table1_intersection_genes_table1_samples.h5",
}

def load_h5(path):
    with h5py.File(path, 'r') as h:
        X = sparse.csr_matrix(h['feature'][()])
        y = np.asarray(h['label'][()]).reshape(-1).astype(int)
    return X, y

def auc_sgd(X, y_train_labels, y_eval_labels, seed=4):
    idx_train, idx_test = train_test_split(np.arange(X.shape[0]), test_size=0.30, random_state=seed, stratify=y_train_labels)
    clf = make_pipeline(MaxAbsScaler(), SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, max_iter=1000, tol=1e-3, random_state=seed))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        clf.fit(X[idx_train], y_train_labels[idx_train])
    scores = clf.predict_proba(X[idx_test])[:, 1]
    return roc_auc_score(y_eval_labels[idx_test], scores), len(idx_train), len(idx_test)

rows=[]
N_PERM=20
for ds, fn in DATASETS.items():
    print('Processing', ds, flush=True)
    X, y = load_h5(os.path.join(DATA_DIR, fn))
    true_auc, ntr, nte = auc_sgd(X, y, y, seed=4)
    rows.append({'dataset': ds, 'analysis': 'true_labels', 'repeat': 0, 'auc': true_auc, 'n_cells': X.shape[0], 'n_genes': X.shape[1], 'n_train': ntr, 'n_test': nte})
    rng = np.random.default_rng(20260721 + len(ds))
    perm_aucs=[]
    for r in range(N_PERM):
        y_perm = rng.permutation(y)
        a, _, _ = auc_sgd(X, y_perm, y_perm, seed=4)
        perm_aucs.append(a)
        rows.append({'dataset': ds, 'analysis': 'permuted_labels', 'repeat': r+1, 'auc': a, 'n_cells': X.shape[0], 'n_genes': X.shape[1], 'n_train': ntr, 'n_test': nte})
    print(ds, 'true=', round(true_auc, 4), 'perm_mean=', round(float(np.mean(perm_aucs)), 4), 'perm_sd=', round(float(np.std(perm_aucs, ddof=1)), 4), flush=True)

raw = pd.DataFrame(rows)
summary_rows=[]
for ds in DATASETS:
    sub = raw[(raw.dataset == ds) & (raw.analysis == 'permuted_labels')]
    true_auc = float(raw[(raw.dataset == ds) & (raw.analysis == 'true_labels')]['auc'].iloc[0])
    summary_rows.append({
        'dataset': ds,
        'true_label_auc': true_auc,
        'permuted_label_mean_auc': float(sub.auc.mean()),
        'permuted_label_sd_auc': float(sub.auc.std(ddof=1)),
        'permuted_label_min_auc': float(sub.auc.min()),
        'permuted_label_max_auc': float(sub.auc.max()),
        'n_permutations': len(sub),
        'n_cells': int(raw[raw.dataset == ds]['n_cells'].iloc[0]),
        'n_genes': int(raw[raw.dataset == ds]['n_genes'].iloc[0]),
    })
summary = pd.DataFrame(summary_rows)
raw.to_csv(os.path.join(OUT_DIR, 'permuted_label_ablation_raw_auc.csv'), index=False)
summary.to_csv(os.path.join(OUT_DIR, 'permuted_label_ablation_summary.csv'), index=False)
print('Saved', OUT_DIR)
print(summary.to_string(index=False))
