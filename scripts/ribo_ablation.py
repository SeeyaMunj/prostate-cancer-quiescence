"""Ribosomal-gene ablation (reviewer comment 5) on a common-gene matrix rebuilt from the raw 10x files.

Everything below was fixed before any result was seen and is identical for the "with" and "without" runs:
  data   : QC-passed cells from the labelled H5 files matched to the raw Cell Ranger libraries (each cell once);
           label = library of origin (G0 = 1); Seurat LogNormalize (scale 10,000) per cell;
           genes detected in >= 3 cells in all six datasets, human/mouse symbols matched ignoring case
  ablate : remove every ribosomal protein gene: symbols starting RPL, RPS, MRPL or MRPS (case-insensitive),
           excluding the RPS6K* kinases and RPS19BP1 (not ribosomal proteins)
  CV     : 5 outer folds, label-stratified, seed 4; every model fit on outer-training cells only,
           scored once on the outer-test fold
  models : MLC-AE (manuscript architecture/loss, 5 epochs as in the manuscript CV);
           SHAP top-10 genes of that MLC-AE (GradientExplainer on training cells) -> L2 logistic regression (C=1);
           elastic-net logistic regression and linear SVM (SGD, library-default alpha/l1_ratio, 10 passes, z-scored);
           random forest and XGBoost at library-default hyperparameters
Run with --part gpu (MLC-AE + SHAP) and --part cpu (conventional models) in separate processes.
"""
import argparse
import gc
import json
import os
import random
import re
import time
import warnings

import h5py
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

import common as C

warnings.filterwarnings('ignore')
OUT = C.OUT.parent / 'results_ribo_ablation'
MIN_CELLS = 3
# ribosomal protein genes: RPL*/RPS* (cytosolic, incl. RPLP0-2, RPSA, paralogs) and MRPL*/MRPS* (mitochondrial);
# excludes the RPS6K* kinases and RPS19BP1, which are not ribosomal proteins
RP = re.compile(r'^(?!RPS6K|RPS19BP)M?RP[LS]', re.I)


def read_raw(stem):
    with h5py.File(C.RAW_DIR / f'{stem}_filtered_feature_bc_matrix.h5', 'r') as f:
        m = f['matrix']
        mat = sp.csc_matrix((m['data'][:], m['indices'][:], m['indptr'][:]), shape=tuple(m['shape'][:]))
        names = [x.decode() for x in m['features/name'][:]]
        ftype = [x.decode() for x in m['features/feature_type'][:]]
        barcodes = [x.decode().split('-')[0] for x in m['barcodes'][:]]
    ge = np.array([t == 'Gene Expression' for t in ftype])
    return mat[ge].tocsc(), [n.upper() for n, g in zip(names, ge) if g], barcodes


def dataset_counts(name):
    """Sparse genes x cells counts for the labelled QC-passed cells, labels, per-cell totals, gene symbols."""
    with h5py.File(C.H5_DIR / f'{name}_table1_intersection_genes_table1_samples.h5', 'r') as f:
        cells = [s.decode() for s in f['sample'][:]]
    group = ['NG0' if ('NGO_' in c or c.startswith('NG0')) else 'G0' for c in cells]
    barc = [re.search(r'[ACGT]{16}', c).group(0) for c in cells]
    unique = list(dict.fromkeys(zip(group, barc)))  # each cell once (Myc-CaP high rows are duplicated)
    mats, labels, totals, genes = [], [], [], None
    for grp, stem, lab in zip(['G0', 'NG0'], C.RAW_FILES[name], [1, 0]):
        mat, names, barcodes = read_raw(stem)
        index = {b: i for i, b in enumerate(barcodes)}
        want = [index[b] for g, b in unique if g == grp and b in index]
        sub = mat[:, want].tocsc()
        genes = genes or names
        mats.append(sub)
        totals.append(np.asarray(sub.sum(axis=0)).ravel())
        labels += [lab] * len(want)
    return sp.hstack(mats).tocsr(), np.array(labels), np.concatenate(totals), genes, len(unique)


def common_genes():
    path = OUT / 'common_genes.txt'
    if path.exists():
        return path.read_text(encoding='utf-8').split('\n')
    sets, info = [], {}
    for name in C.DATASETS:
        mat, y, _, genes, n_h5 = dataset_counts(name)
        det = np.asarray((mat > 0).sum(axis=1)).ravel()
        first = {}
        for i, g in enumerate(genes):
            first.setdefault(g, i)
        sets.append({g for g, i in first.items() if det[i] >= MIN_CELLS})
        info[name] = {'cells_used': int(len(y)), 'cells_in_labelled_h5': n_h5, 'G0': int(y.sum())}
        del mat
        gc.collect()
    genes = sorted(set.intersection(*sets))
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(genes), encoding='utf-8')
    rp = [g for g in genes if RP.match(g)]
    info['n_common_genes'] = len(genes)
    info['removed_ribosomal_genes'] = rp
    (OUT / 'data_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'common genes: {len(genes)}; ribosomal genes removed in ablation: {len(rp)}', flush=True)
    return genes


def load_dense(name, genes):
    mat, y, totals, gnames, _ = dataset_counts(name)
    first = {}
    for i, g in enumerate(gnames):
        first.setdefault(g, i)
    sub = mat[[first[g] for g in genes]].T.tocsr().astype(np.float32)
    sub = sp.diags((1e4 / np.where(totals == 0, 1, totals)).astype(np.float32)) @ sub
    sub.data = np.log1p(sub.data)
    X = sub.toarray()
    del mat, sub
    gc.collect()
    return X, y


def auc(y, s):
    return float(roc_auc_score(y, s))


# ---------------- GPU part: MLC-AE and SHAP top-10 ----------------
def run_gpu(name, X, y, genes, folds, keep_nr):
    import shap
    import tensorflow as tf
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.layers import BatchNormalization, Dense, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import to_categorical
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)

    def fit(x, yy, seed):
        tf.keras.backend.clear_session()
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
        inp = Input(shape=(x.shape[1],))
        enc = Dense(64, activation='relu', name='encoded')(BatchNormalization()(inp))
        dec = Dense(x.shape[1], activation='linear')(Dense(512, activation='relu')(enc))
        cat = Dense(2, activation='softmax', name='category')(enc)
        m = Model(inp, [dec, cat])
        m.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=['mse', 'cosine_similarity'], loss_weights=[0.001, 0.5])
        m.fit(x, [x, to_categorical(yy, 2)], batch_size=256, epochs=5, verbose=0, shuffle=True)
        return m

    for k, (tr, te) in enumerate(folds):
        out = OUT / name / f'fold_{k}_gpu.json'
        if out.exists():
            continue
        t0 = time.time()
        seed = C.SEED + k
        res = {}
        for variant, cols in [('with_RP', None), ('without_RP', keep_nr)]:
            xtr = X[tr] if cols is None else X[np.ix_(tr, cols)]
            xte = X[te] if cols is None else X[np.ix_(te, cols)]
            vg = genes if cols is None else [genes[i] for i in cols]
            m = fit(xtr, y[tr], seed)
            p = m.predict(xte, batch_size=1024, verbose=0)[1][:, 1]
            rng = np.random.default_rng(seed)
            head = Model(m.input, m.get_layer('category').output)
            ex = shap.GradientExplainer(head, xtr[rng.choice(len(tr), 100, replace=False)])
            sv = ex.shap_values(xtr[rng.choice(len(tr), 300, replace=False)])
            imp = np.abs(sv[1] if isinstance(sv, list) else sv[..., 1]).mean(axis=0)
            top = np.argsort(-imp)[:10]
            lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(xtr[:, top], y[tr])
            q = lr.predict_proba(xte[:, top])[:, 1]
            res[variant] = {'mlcae_auc': auc(y[te], p), 'shap_top10_lr_auc': auc(y[te], q),
                            'shap_top10': [vg[i] for i in top],
                            'n_ribosomal_in_shap_top10': int(sum(bool(RP.match(vg[i])) for i in top))}
            np.savez(OUT / name / f'fold_{k}_{variant}_gpu_scores.npz', mlcae=p, shap_top10_lr=q, test_idx=te)
            del m, head, ex, sv, xtr, xte
            gc.collect()
        res['seconds'] = time.time() - t0
        out.write_text(json.dumps(res, indent=2), encoding='utf-8')
        print(f'[gpu] {name} fold {k}: MLC-AE {res["with_RP"]["mlcae_auc"]:.4f} -> {res["without_RP"]["mlcae_auc"]:.4f}; '
              f'SHAP top-10 {res["with_RP"]["shap_top10_lr_auc"]:.4f} -> {res["without_RP"]["shap_top10_lr_auc"]:.4f} '
              f'({res["seconds"]:.0f}s)', flush=True)


# ---------------- CPU part: conventional models ----------------
TOP_SAVE = 50


def _sgd(params, x, yy, seed, chunk=2048, epochs=10):
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(random_state=seed, **params)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        idx = rng.permutation(len(yy))
        for s in range(0, len(idx), chunk):
            b = np.sort(idx[s:s + chunk])
            clf.partial_fit(x[b].astype(np.float64), yy[b], classes=np.array([0, 1]))
    return clf


def _top(importance, names):
    order = np.argsort(-np.nan_to_num(importance, nan=-np.inf))[:TOP_SAVE]
    return [[names[i], float(importance[i])] for i in order]


def fit_conventional(xtr, ytr, xte, seed, fold_id, names):
    """Fits the four conventional models on the given training cells/genes; returns outer-test scores and each
    model's own gene ranking (top-50), computed from training cells only:
      elastic net / linear SVM : |coefficient| (z-scored genes)
      random forest            : permutation importance (AUC drop, 3 repeats) on an inner 20% validation split of
                                 the training cells, using a forest refit on the inner 80%; candidates = top-200
                                 genes by impurity importance
      XGBoost                  : gain
    """
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    mu, sd = xtr.mean(axis=0), xtr.std(axis=0)
    sd[sd == 0] = 1
    ztr, zte = (xtr - mu) / sd, (xte - mu) / sd
    scores, tops = {}, {}
    m = _sgd(dict(loss='log_loss', penalty='elasticnet'), ztr, ytr, seed)
    scores['elastic_net'] = m.decision_function(zte.astype(np.float64))
    tops['elastic_net'] = _top(np.abs(m.coef_.ravel()), names)
    m = _sgd(dict(loss='hinge', penalty='l2'), ztr, ytr, seed)
    scores['linear_svm'] = m.decision_function(zte.astype(np.float64))
    tops['linear_svm'] = _top(np.abs(m.coef_.ravel()), names)
    del ztr, zte, m
    gc.collect()
    scores['random_forest'] = RandomForestClassifier(n_jobs=6, random_state=seed).fit(xtr, ytr).predict_proba(xte)[:, 1]
    itr, iva = C.inner_split(ytr, fold_id)
    rf = RandomForestClassifier(n_jobs=6, random_state=seed).fit(xtr[itr], ytr[itr])
    if len(iva) > 2000:
        iva, _ = train_test_split(iva, train_size=2000, stratify=ytr[iva], random_state=seed)
    xv, yv = xtr[iva].copy(), ytr[iva]
    base = auc(yv, rf.predict_proba(xv)[:, 1])
    perm = np.full(xtr.shape[1], np.nan)
    rng = np.random.default_rng(seed)
    for j in np.argsort(-rf.feature_importances_)[:200]:
        orig = xv[:, j].copy()
        drops = []
        for _ in range(3):
            xv[:, j] = orig[rng.permutation(len(orig))]
            drops.append(base - auc(yv, rf.predict_proba(xv)[:, 1]))
        xv[:, j] = orig
        perm[j] = np.mean(drops)
    tops['random_forest'] = _top(perm, names)
    del rf, xv
    gc.collect()
    xg = xgb.XGBClassifier(tree_method='hist', n_jobs=6, random_state=seed, verbosity=0).fit(xtr, ytr)
    scores['xgboost'] = xg.predict_proba(xte)[:, 1]
    gain = np.zeros(xtr.shape[1])
    for key, val in xg.get_booster().get_score(importance_type='gain').items():
        gain[int(key[1:])] = val
    tops['xgboost'] = _top(gain, names)
    return scores, tops


def run_cpu(name, X, y, genes, folds, keep_nr):
    nr_set = set(keep_nr.tolist())
    for k, (tr, te) in enumerate(folds):
        out = OUT / name / f'fold_{k}_cpu.json'
        if out.exists() and all((OUT / name / f'fold_{k}_{v}_cpu_topgenes.json').exists() for v in ('with_RP', 'without_RP')):
            continue
        t0 = time.time()
        seed = C.SEED + k
        # columns that vary across the outer-training cells (constant columns carry no information)
        xs = X[tr]
        varying = np.where(xs.max(axis=0) > xs.min(axis=0))[0]
        del xs
        res = {}
        for variant in ('with_RP', 'without_RP'):
            cols = varying if variant == 'with_RP' else np.array([c for c in varying if c in nr_set])
            scores, tops = fit_conventional(X[np.ix_(tr, cols)], y[tr], X[np.ix_(te, cols)], seed, k,
                                            [genes[c] for c in cols])
            res[variant] = {m: auc(y[te], s) for m, s in scores.items()}
            res[variant]['n_genes_used'] = int(len(cols))
            np.savez(OUT / name / f'fold_{k}_{variant}_cpu_scores.npz', test_idx=te, **scores)
            (OUT / name / f'fold_{k}_{variant}_cpu_topgenes.json').write_text(json.dumps(tops, indent=1), encoding='utf-8')
            gc.collect()
        res['seconds'] = time.time() - t0
        out.write_text(json.dumps(res, indent=2), encoding='utf-8')
        print(f'[cpu] {name} fold {k}: ' + ', '.join(f'{m} {res["with_RP"][m]:.4f}->{res["without_RP"][m]:.4f}'
                                                     for m in ('elastic_net', 'linear_svm', 'random_forest', 'xgboost'))
              + f' ({res["seconds"]:.0f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['gpu', 'cpu', 'genes'], required=True)
    ap.add_argument('--datasets', nargs='+', default=C.DATASETS)
    args = ap.parse_args()
    genes = common_genes()
    if args.part == 'genes':
        return
    keep_nr = np.array([i for i, g in enumerate(genes) if not RP.match(g)])
    for name in args.datasets:
        (OUT / name).mkdir(parents=True, exist_ok=True)
        X, y = load_dense(name, genes)
        folds = C.outer_folds(y)
        print(f'[{args.part}] {name}: {X.shape}, G0={int(y.sum())}', flush=True)
        (run_gpu if args.part == 'gpu' else run_cpu)(name, X, y, genes, folds, keep_nr)
        del X
        gc.collect()


if __name__ == '__main__':
    main()
