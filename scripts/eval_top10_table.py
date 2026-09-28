"""Tables for eval_top10.py: evaluation-classifier AUC on each model's top-10 genes, per cell line.

AUC mean ± SD over outer folds; Δ = condition − all genes (paired, same test cells);
p = paired DeLong per fold, Stouffer-combined over folds.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

import common as C
from eval_top10 import CONDITIONS, OUT, SELECTORS

NAMES = {'mlcae_shap': 'MLC-AE (SHAP)', 'elastic_net': 'Elastic-net LR (|coef|)', 'linear_svm': 'Linear SVM (|coef|)',
         'random_forest': 'Random forest (permutation)', 'xgboost': 'XGBoost (gain)'}
LABELS = {'PC3_high': 'PC3 high serum', 'PC3_low': 'PC3 low serum', 'C42B_high': 'C4-2B high serum',
          'C42B_low': 'C4-2B low serum', 'MycCaP_high': 'Myc-CaP high serum', 'MycCaP_low': 'Myc-CaP low serum'}
COND_LABEL = {'ribosomal_removed': 'ribosomal removed', 'ribo_cc_top200shap_removed': 'ribo + cell-cycle + top-200 SHAP removed'}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    md = ['# Evaluation classifier on each model\'s own top-10 genes', '',
          'Each model selects its top-10 genes from the training cells under each condition; the same evaluation '
          'classifier is trained on those 10 genes (training cells) and scored on the held-out fold. '
          'AUC mean ± SD over 5 outer folds; Δ vs all genes (paired); p = paired DeLong, Stouffer-combined.', '']
    csv = []
    for kind, klabel in [('ann', 'Evaluation ANN'), ('lr', 'L2 logistic regression')]:
        for name in C.DATASETS:
            f = OUT / f'{name}.json'
            if not f.exists():
                continue
            d = json.loads(f.read_text(encoding='utf-8'))
            df = pd.DataFrame(d['rows'])
            df = df[df.classifier == kind]
            df['sel'] = df.selector
            y = np.array(d['y'])
            rows = []
            for sel in SELECTORS:
                row = {'Top-10 genes from': NAMES[sel]}
                per = {}
                for cond in CONDITIONS:
                    q = df[(df.sel == sel) & (df.condition == cond)].groupby('fold').auc.mean().sort_index().values
                    per[cond] = q
                    row['AUC all genes' if cond == 'all_genes' else f'AUC {COND_LABEL[cond]}'] = \
                        f'{q.mean():.3f} ± {q.std(ddof=1):.3f}'
                for cond in CONDITIONS[1:]:
                    delta = (per[cond] - per['all_genes']).mean()
                    p = ''
                    if sel:
                        zs = []
                        for k in range(C.N_FOLDS):
                            te = np.array(d['test_idx'][str(k)])
                            _, _, z, _ = C.delong_paired(y[te], d['scores'][f'{k}|all_genes|{sel}|{kind}'],
                                                         d['scores'][f'{k}|{cond}|{sel}|{kind}'])
                            zs.append(z)
                        p = f'{2 * stats.norm.sf(abs(np.sum(zs) / np.sqrt(len(zs)))):.2g}'
                    row[f'Δ {COND_LABEL[cond]}'] = f'{delta:+.3f}' + (f' (p={p})' if p else '')
                rows.append(row)
                csv.append({'classifier': klabel, 'cell line': LABELS[name], **row})
            md += [f'## {klabel} - {LABELS[name]}', pd.DataFrame(rows).to_markdown(index=False), '']
    pd.DataFrame(csv).to_csv(C.OUT.parent / 'eval_top10_tables.csv', index=False, encoding='utf-8-sig')
    (C.OUT.parent / 'EVAL_TOP10_TABLES.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
