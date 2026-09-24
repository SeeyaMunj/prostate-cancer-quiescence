"""Writes the executable notebooks in notebooks/ (one per pipeline stage)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'notebooks'
OUT.mkdir(exist_ok=True)


def nb(cells):
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                                        'name': 'python3'},
                                         'language_info': {'name': 'python', 'version': '3.9.16'}},
            'nbformat': 4, 'nbformat_minor': 5}


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip().split('\n')}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': text.strip().split('\n')}


HEADER = """
# {title}

{intro}

**Inputs:** {inputs}
**Outputs:** {outputs}
**Approximate runtime:** {runtime}

Paths are set in `scripts/common.py` (`H5_DIR`, `RAW_DIR`, `OUT`). Edit them once before running any notebook.
"""

SETUP = """
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'scripts'))
import common as C
print('datasets:', list(C.DATASETS))
print('results directory:', C.OUT)
"""

notebooks = {
 '01_qc_and_common_genes.ipynb': dict(
  title='1. Quality control, normalization and the common gene space',
  intro='Reads the Cell Ranger filtered matrices, keeps the QC-passed cells that carry a FACS G0/non-G0 label, '
        'normalizes each cell (Seurat LogNormalize, scale 10,000) and builds the gene space shared by all six datasets.',
  inputs='`<RAW_DIR>/*_filtered_feature_bc_matrix.h5` and the labelled cell lists',
  outputs='`results/common_genes.txt`, `results/data_info.json`',
  runtime='about 15 minutes (slowest step: intersecting genes across datasets)',
  cells=[code(SETUP),
         code("from ribo_ablation import common_genes, dataset_counts\n"
              "genes = common_genes()\n"
              "print(len(genes), 'genes common to all six datasets')\n"
              "print(genes[:10])"),
         md('### Cell counts per dataset'),
         code("for name in C.DATASETS:\n"
              "    mat, y, totals, gnames, n_h5 = dataset_counts(name)\n"
              "    print(f'{name:12s} cells={len(y):6d}  G0={int(y.sum()):6d}  non-G0={int((1-y).sum()):6d}')")]),

 '02_train_mlcae.ipynb': dict(
  title='2. Training the MLC-AE',
  intro='Trains the multi-task autoencoder (reconstruction + G0/non-G0 classification) on each dataset and saves the '
        'weights. The released weights in `weights/` were produced by exactly this step.',
  inputs='normalized matrices, `results/common_genes.txt`',
  outputs='`weights/<dataset>_mlcae.weights.h5`, `weights/<dataset>_mlcae.json`',
  runtime='about 2-8 minutes per dataset on a laptop GPU (NVIDIA RTX 2060); longer on CPU',
  cells=[code(SETUP),
         md('Run the training script for one dataset (or omit the argument to train all six):'),
         code("%run ../train_weights.py PC3_high"),
         md('### Load released weights and score the held-out cells'),
         code("import numpy as np, json\n"
              "from sklearn.model_selection import train_test_split\n"
              "from sklearn.metrics import roc_auc_score\n"
              "from ribo_ablation import common_genes, load_dense\n"
              "sys.path.insert(0, str(Path.cwd().parent))\n"
              "from train_weights import build\n"
              "name = 'PC3_high'\n"
              "genes = common_genes(); X, y = load_dense(name, genes)\n"
              "xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=4, stratify=y)\n"
              "model = build(X.shape[1]); model.load_weights(f'../weights/{name}_mlcae.weights.h5')\n"
              "p = model.predict(xte, batch_size=2048, verbose=0)[1][:, 1]\n"
              "print('held-out AUC:', round(roc_auc_score(yte, p), 4))\n"
              "print('recorded at release:', json.load(open(f'../weights/{name}_mlcae.json'))['test_auc'])")]),

 '03_shap_ranking.ipynb': dict(
  title='3. SHAP gene ranking',
  intro='Ranks genes by their contribution to the classification head using SHAP GradientExplainer '
        '(100 background cells, 300 explained cells, training cells only).',
  inputs='trained model, training split',
  outputs='ranked gene list per dataset (`results/<dataset>/fold_*_gpu.json`)',
  runtime='about 1-3 minutes per dataset',
  cells=[code(SETUP),
         code("import shap, numpy as np\n"
              "from tensorflow.keras.models import Model\n"
              "from ribo_ablation import common_genes, load_dense\n"
              "sys.path.insert(0, str(Path.cwd().parent))\n"
              "from train_weights import build\n"
              "name = 'PC3_high'\n"
              "from sklearn.model_selection import train_test_split\n"
              "genes = common_genes(); X, y = load_dense(name, genes)\n"
              "# SHAP uses the training split only (Supplementary Table S6)\n"
              "xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=4, stratify=y)\n"
              "model = build(X.shape[1]); model.load_weights(f'../weights/{name}_mlcae.weights.h5')\n"
              "head = Model(model.input, model.get_layer('category').output)\n"
              "rng = np.random.default_rng(4)\n"
              "bg = xtr[np.sort(rng.choice(len(xtr), 100, replace=False))]\n"
              "ex = xtr[np.sort(rng.choice(len(xtr), 300, replace=False))]\n"
              "sv = shap.GradientExplainer(head, bg).shap_values(ex)\n"
              "imp = np.abs(sv[1] if isinstance(sv, list) else sv[..., 1]).mean(axis=0)\n"
              "top = np.argsort(-imp)[:10]\n"
              "print('top-10 SHAP genes:', [genes[i] for i in top])")]),

 '04_benchmark_and_evaluation.ipynb': dict(
  title='4. Method benchmark and the evaluation classifier',
  intro='Trains every method on the same full training matrix, takes each method\'s own top-10 genes, and scores all '
        'of them with the same evaluation classifier. Produces the numbers in Supplementary Table S3.',
  inputs='normalized matrices, `results/common_genes.txt`',
  outputs='`results/<dataset>/fold_*_cpu_scores.npz`, `results/<dataset>/fold_*_cpu_topgenes.json`',
  runtime='about 20-40 minutes per dataset (random forest and XGBoost dominate)',
  cells=[code(SETUP),
         code("%run ../scripts/ribo_ablation.py --help"),
         md('Run the benchmark for one dataset:'),
         code("!python ../scripts/ribo_ablation.py --part cpu --dataset PC3_high"),
         md('Collect the results into the table:'),
         code("!python ../scripts/eval_top10_table.py")]),

 '05_ablations.ipynb': dict(
  title='5. Ablations: reconstruction branch and ribosomal genes',
  intro='Two ablations. (a) The decoder and reconstruction loss are removed (Supplementary Tables S4 and S4b). '
        '(b) All ribosomal protein genes are removed from the input matrix (Supplementary Table S8).',
  inputs='normalized matrices, `results/common_genes.txt`',
  outputs='`results/ablation/*.json`',
  runtime='about 30 minutes for all six datasets',
  cells=[code(SETUP),
         md('### Ribosomal genes removed from the input'),
         code("import re\n"
              "from ribo_ablation import common_genes\n"
              "genes = common_genes()\n"
              "ribo = [g for g in genes if re.match(r'^(?!RPS6K|RPS19BP)M?RP[LS]', g)]\n"
              "print(len(ribo), 'ribosomal protein genes removed;', len(genes) - len(ribo), 'genes remain')"),
         code("!python ../scripts/ribo_ablation.py --part gpu --dataset PC3_high"),
         md('### Reconstruction branch removed (classifier-only network)'),
         code("!python ../scripts/comment4_stability.py")]),

 '06_patient_cohort.ipynb': dict(
  title='6. Patient cohort analysis (GSE176031)',
  intro='Reproduces Supplementary Table S1 and Supplementary Figure S8: the fold-change-weighted 26-gene score in the '
        'tumor-enriched luminal epithelium of GSE176031, its association with three proliferation measures, the '
        'per-patient summary and the unweighted sensitivity analysis. This step runs in R.',
  inputs='`gse176031_tumor_epithelial_scored.rds`, `PC3LOW_26GENES.csv`',
  outputs='`results_comment7/comment7_patient_association_results.csv`, `FigureS8_weighted_G0_score.png`',
  runtime='about 3 minutes',
  cells=[md('```bash\nRscript ../scripts/comment7_patient_analysis.R\n```\n'
            'Edit the three paths at the top of the script first (`IN_RDS`, `IN_FC`, `OUT_DIR`).'),
         code("import pandas as pd\n"
              "res = pd.read_csv('../results_comment7/comment7_patient_association_results.csv')\n"
              "res[['score','measure','cell_rho','cell_p','mixed_beta','mixed_p']]")]),
}

for fname, spec in notebooks.items():
    cells = [md(HEADER.format(title=spec['title'], intro=spec['intro'], inputs=spec['inputs'],
                              outputs=spec['outputs'], runtime=spec['runtime']))] + spec['cells']
    json.dump(nb(cells), open(OUT / fname, 'w', encoding='utf-8'), indent=1)
    print('wrote', fname)
