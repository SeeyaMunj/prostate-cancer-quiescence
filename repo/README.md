# Ribosomal gene signatures of prostate cancer quiescence — reproducible protocol

This repository reproduces every computational result in the manuscript, from the raw Cell Ranger output to the
figures and supplementary tables. It contains the environment files, the ordered pipeline with the exact commands,
executable notebooks, the trained model weights, a small example dataset, and a map from each figure and table to the
code that produces it.

---

## 1. What is here

```
.
├── README.md                     this file
├── environment.yml               conda environment (Python)
├── requirements.txt              pip alternative
├── r_packages.txt                R package list (R 4.5.2)
├── install_r_packages.R          installs the R packages
├── notebooks/                    executable notebooks, one per stage (01-06)
├── weights/                      trained MLC-AE weights, one per dataset + input gene order please find them in https://drive.google.com/drive/u/0/folders/1KoBfRSsgUmdHLVLkFBeYeXXx2HECFx_o
├── example_data/                 small subsampled dataset for a test run
└── results/                      created when you run the pipeline
```

## 2. Installation

```bash
conda env create -f environment.yml
conda activate prostate-quiescence
Rscript install_r_packages.R
```

Without conda: `pip install -r requirements.txt` (Python 3.9 recommended; TensorFlow 2.10 is the last version with
native Windows GPU support).

Check the install:

```bash
python -c "import tensorflow as tf, shap, sklearn; print(tf.__version__, shap.__version__, sklearn.__version__)"
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## 3. Data

| Data | Where to get it | Used by |
|---|---|---|
| Cell-line scRNA-seq (PC3, C4-2B, Myc-CaP; high and low serum) | GEO accession [ADD ACCESSION] — Cell Ranger `filtered_feature_bc_matrix.h5` per library | steps 1–5 |
| FACS G0/non-G0 cell labels | `data/labels/` in this repository | steps 1–5 |
| GSE176031 patient cohort | GEO GSE176031 (raw DGE matrices) | step 6 |
| PC3 low-serum differential expression weights | `data/PC3LOW_26GENES.csv` | step 6 |

Set the three paths at the top of `scripts/common.py` (`H5_DIR`, `RAW_DIR`, `OUT`) once; every script and notebook
reads them from there.

## 4. Run the pipeline

Runtimes are for a laptop with an NVIDIA RTX 2060 (6 GB), Intel i7-10750H and 16 GB RAM.

| Step | Command | Output | Runtime |
|---|---|---|---|
| 1. QC, normalization, common genes | `python scripts/ribo_ablation.py --part genes` | `results/common_genes.txt`, `results/data_info.json` | ~15 min |
| 2. Train the MLC-AE | `python train_weights.py` | `weights/<dataset>_mlcae.weights.h5` | ~30 s per dataset (GPU) |
| 3. SHAP gene ranking | `python scripts/ribo_ablation.py --part gpu` | ranked genes per dataset | ~2 min per dataset |
| 4. Benchmark and evaluation classifier | `python scripts/ribo_ablation.py --part cpu` | Supplementary Table S3 | ~20–40 min per dataset |
| 5. Ablations (reconstruction, ribosomal genes) | `python scripts/comment4_stability.py` | Supplementary Tables S4, S4b, S8 | ~30 min |
| 6. Patient cohort (R) | `Rscript scripts/comment7_patient_analysis.R` | Supplementary Table S1, Figure S8 | ~3 min |

Steps 3–5 write one JSON file per dataset and fold and skip work that is already saved, so an interrupted run can be
restarted with the same command.

To check the install end to end without the full data, run step 2 on the example dataset:

```bash
python train_weights.py --example
```

## 5. Trained weights

`weights/` holds one trained model per dataset, produced by `train_weights.py` with the settings in Supplementary
Table S6. `weights/input_genes.txt` gives the gene order the models expect; a matrix must be normalized the same way
(Seurat LogNormalize, scale 10,000) and its columns ordered by that file.

| File | Dataset | Cells | Held-out AUC |
|---|---|---|---|
| `PC3_high_mlcae.weights.h5` | PC3 high serum | 7,437 | see `PC3_high_mlcae.json` |
| `PC3_low_mlcae.weights.h5` | PC3 low serum | 28,214 | see `PC3_low_mlcae.json` |
| `C42B_high_mlcae.weights.h5` | C4-2B high serum | 12,997 | see `C42B_high_mlcae.json` |
| `C42B_low_mlcae.weights.h5` | C4-2B low serum | 6,925 | see `C42B_low_mlcae.json` |
| `MycCaP_high_mlcae.weights.h5` | Myc-CaP high serum | 20,140 | see `MycCaP_high_mlcae.json` |
| `MycCaP_low_mlcae.weights.h5` | Myc-CaP low serum | 11,765 | see `MycCaP_low_mlcae.json` |

Each `.json` records the cells, the held-out AUC, and the architecture and training settings used.

Loading a model:

```python
from train_weights import build
genes = open('weights/input_genes.txt').read().split('\n')
model = build(len(genes))
model.load_weights('weights/PC3_high_mlcae.weights.h5')
probabilities = model.predict(X)[1][:, 1]   # column 1 = P(G0)
```

## 6. Notebooks

| Notebook | Stage |
|---|---|
| `01_qc_and_common_genes.ipynb` | QC, normalization, common gene space, cell counts (Table 1) |
| `02_train_mlcae.ipynb` | Training the MLC-AE; loading the released weights |
| `03_shap_ranking.ipynb` | SHAP gene ranking (Figures 5a–b, S2, S3) |
| `04_benchmark_and_evaluation.ipynb` | Method benchmark and evaluation classifier (Supplementary Table S3; Figure 8a–b) |
| `05_ablations.ipynb` | Reconstruction-branch and ribosomal-gene ablations (Supplementary Tables S4, S4b, S8) |
| `06_patient_cohort.ipynb` | GSE176031 analysis (Supplementary Table S1, Figure S8) |

## 7. Which code makes which figure

`docs/figure_table_map.csv` lists every figure and table in the manuscript and Supplementary Material with the script
or notebook that produces it, and marks the items that come from external tools (iScanGuide, iPathwayGuide) or from
laboratory experiments rather than from this code.

## 8. Troubleshooting

**"Could not load dynamic library cudart64_110.dll" / no GPU found.** TensorFlow falls back to the CPU and everything
still runs, roughly 5–10× slower. For GPU use, TensorFlow 2.10 needs CUDA 11.2 and cuDNN 8.1.

**Out of memory during training.** The dense matrix for the largest dataset is about 1 GB in float32. Lower the batch
size (`BATCH` in `train_weights.py`) or train one dataset at a time: `python train_weights.py PC3_high`.

**Out of memory in the benchmark step.** Random forest and XGBoost hold the full matrix; run one dataset at a time and
close other Python processes.

**`shap.GradientExplainer` raises a `tf.function` error.** SHAP 0.41 requires TensorFlow 2.10; newer TensorFlow
versions change the gradient API. Pin the versions in `requirements.txt`.

**Seurat `JoinLayers` not found.** That function needs Seurat 5; older objects must be updated with `UpdateSeuratObject`.

**Results differ slightly between runs.** Model training, SHAP sampling and the random forest are stochastic. Seeds are
fixed in every script (seed 4), but GPU kernels are not bit-deterministic, so AUCs can move by a few thousandths.
Reported values are means over folds or seeds.

**Cell Ranger files are missing barcodes present in the label files.** Only QC-passed cells carry labels; the loader
matches by library and barcode and drops the rest, which is expected.

## 9. Citation

If you use this code, please cite the manuscript and this repository release: [ADD DOI AFTER ARCHIVING].
