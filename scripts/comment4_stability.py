"""Comment 4: does the reconstruction branch change which genes SHAP selects?

Full MLC-AE vs classifier-only network (input batch norm -> 64-unit ReLU encoded layer -> 2-unit softmax; decoder and
reconstruction loss removed; same optimizer, loss weight on classification, 5 epochs), compared on:
  (a) outer-fold performance: 5 outer folds (seed 4 + fold), same as all other analyses; direct AUC and the AUC of the
      model's top-10 SHAP genes on the evaluation ANN / L2 logistic regression (same downstream code and seeds);
      the full model's fold results are reused from results_ribo_ablation / results_eval_top10 (identical settings)
  (b) stability across seeds: training cells of outer fold 0 fixed, seeds 4-8; pairwise Jaccard of top-10 and top-50
      SHAP gene sets; full-model seed 4 is the existing fold-0 run's gene list recomputed here for its top-50
  (c) stability across data splits: pairwise Jaccard of the five outer folds' top-10 sets
SHAP: GradientExplainer on the classification head, 100 background / 300 explained training cells (as before).
"""
import argparse
import gc
import json
import os
import random
import time
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

import common as C
from ribo_ablation import common_genes, load_dense
from stage3_downstream import downstream

warnings.filterwarnings('ignore')
BASE = C.OUT.parent
OUT = BASE / 'results_comment4_stability'
SEEDS = [4, 5, 6, 7, 8]


def main():
    import shap
    import tensorflow as tf
    from tensorflow.keras.layers import BatchNormalization, Dense, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import to_categorical
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)

    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=C.DATASETS)
    args = ap.parse_args()

    def fit(arch, x, yy, seed):
        tf.keras.backend.clear_session()
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
        inp = Input(shape=(x.shape[1],))
        enc = Dense(64, activation='relu', name='encoded')(BatchNormalization()(inp))
        cat = Dense(2, activation='softmax', name='category')(enc)
        if arch == 'full':
            dec = Dense(x.shape[1], activation='linear')(Dense(512, activation='relu')(enc))
            m = Model(inp, [dec, cat])
            m.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=['mse', 'cosine_similarity'], loss_weights=[0.001, 0.5])
            m.fit(x, [x, to_categorical(yy, 2)], batch_size=256, epochs=5, verbose=0, shuffle=True)
        else:
            m = Model(inp, cat)
            m.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='cosine_similarity', loss_weights=0.5)
            m.fit(x, to_categorical(yy, 2), batch_size=256, epochs=5, verbose=0, shuffle=True)
        return m

    def predict(m, x):
        out = m.predict(x, batch_size=1024, verbose=0)
        return (out[1] if isinstance(out, list) else out)[:, 1]

    def shap_rank(m, xtr, seed):
        rng = np.random.default_rng(seed)
        head = Model(m.input, m.get_layer('category').output)
        ex = shap.GradientExplainer(head, xtr[rng.choice(len(xtr), 100, replace=False)])
        sv = ex.shap_values(xtr[rng.choice(len(xtr), 300, replace=False)])
        imp = np.abs(sv[1] if isinstance(sv, list) else sv[..., 1]).mean(axis=0)
        return np.argsort(-imp)[:50]

    genes = common_genes()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        (OUT / name).mkdir(exist_ok=True)
        X, y = load_dense(name, genes)
        folds = C.outer_folds(y)
        t_ds = time.time()
        # (a) outer folds, classifier-only (full model reused from earlier runs)
        for k, (tr, te) in enumerate(folds):
            f = OUT / name / f'folds_classifier_only_fold{k}.json'
            if f.exists():
                continue
            seed = C.SEED + k
            xtr, xte = X[tr], X[te]
            m = fit('classifier_only', xtr, y[tr], seed)
            p = predict(m, xte)
            top50 = shap_rank(m, xtr, seed)
            del m
            top = top50[:10]
            res = {'fold': k, 'direct_auc': float(roc_auc_score(y[te], p)), 'top50': [genes[i] for i in top50]}
            scores = {}
            for kind in ('ann', 'lr'):
                s, hp = downstream(kind, X[np.ix_(tr, top)], y[tr], X[np.ix_(te, top)], k, seed)
                res[f'top10_{kind}_auc'] = float(roc_auc_score(y[te], s))
                scores[kind] = s.tolist()
            res['scores'] = scores
            res['test_idx'] = te.tolist()
            f.write_text(json.dumps(res), encoding='utf-8')
            del xtr, xte
            gc.collect()
            print(f'[{name}] classifier-only fold {k}: direct {res["direct_auc"]:.3f} top-10 ANN {res["top10_ann_auc"]:.3f}', flush=True)
        # (b) seeds on the fold-0 training cells, both architectures
        tr0 = folds[0][0]
        x0 = X[tr0]
        for arch in ('full', 'classifier_only'):
            for seed in SEEDS:
                f = OUT / name / f'seeds_{arch}_seed{seed}.json'
                if f.exists():
                    continue
                m = fit(arch, x0, y[tr0], seed)
                top50 = shap_rank(m, x0, seed)
                del m
                gc.collect()
                f.write_text(json.dumps({'arch': arch, 'seed': seed, 'top50': [genes[i] for i in top50]}), encoding='utf-8')
                print(f'[{name}] seeds {arch} seed {seed}: top-10 {[genes[i] for i in top50[:10]]}', flush=True)
        del x0, X
        gc.collect()
        print(f'[{name}] COMPLETE ({time.time() - t_ds:.0f}s)', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
