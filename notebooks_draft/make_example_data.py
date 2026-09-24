"""Builds the small example dataset used to test the install (example_data/example_pc3_high.npz).

400 G0 and 400 non-G0 PC3 high-serum cells, restricted to the 1,500 most variable genes of that subsample.
It is a smoke test for the pipeline, not a dataset for analysis: the models in weights/ expect all
9,758 genes listed in weights/input_genes.txt.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ribo_ablation import common_genes, load_dense  # noqa: E402

OUT = Path(__file__).resolve().parent / 'example_data'
OUT.mkdir(exist_ok=True)
N_PER_CLASS, N_GENES, SEED = 400, 1500, 4

genes = common_genes()
X, y = load_dense('PC3_high', genes)
rng = np.random.default_rng(SEED)
idx = np.concatenate([rng.choice(np.where(y == 1)[0], N_PER_CLASS, replace=False),
                      rng.choice(np.where(y == 0)[0], N_PER_CLASS, replace=False)])
idx.sort()
Xs, ys = X[idx], y[idx]
keep = np.argsort(-Xs.var(axis=0))[:N_GENES]
keep.sort()
np.savez_compressed(OUT / 'example_pc3_high.npz', X=Xs[:, keep].astype(np.float32), y=ys.astype(np.int8),
                    genes=np.array([genes[i] for i in keep]))
print(f'wrote example_data/example_pc3_high.npz: {Xs.shape[0]} cells x {N_GENES} genes '
      f'({(OUT / "example_pc3_high.npz").stat().st_size / 1e6:.1f} MB)')
