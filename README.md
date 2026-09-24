# Ribosomal gene signatures of prostate cancer quiescence — analysis protocol

This repository contains the code and the trained model behind *Ribosomal Gene Signatures of Prostate Cancer
Quiescence Revealed by AI-Guided Single-Cell Analysis*. It documents, in order, how the single-cell matrices
are prepared, how the MLC-AE is trained, how genes are ranked with SHAP, and how the ablation, latent-space
and patient-cohort analyses are run, with the command and the inputs and outputs for each step.

---

## 1. Repository layout

```
.
├── README.md                                 this protocol
├── environment.yml / requirements.txt        Python environment
├── r_packages.txt                            R packages (R 4.5.2)
├── create_intersection_h5.py                 Step 1: common gene space and per-dataset H5 files
├── mlc-ae_.py                                Steps 2-3: MLC-AE training, testing and SHAP ranking
├── mlc_ae_cv.py, options.py                  Step 4: five-fold cross-validation
├── run_ablation_full_vs_classifier_only.py   Step 5: full model vs classifier-only network
├── run_permuted_label_ablation.py            Step 5: permuted-label control
├── run_latent_space_umap_all_six.py          Step 6: latent-space UMAP
├── external_dataset/                         Step 7: GSE176031 patient cohort (R)
├── datasets_h5/                              example intersected matrix (C4-2B high serum)
├── MycCaP_low_mlcae.weights.h5               trained model, Myc-CaP low serum
└── docs/figure_table_map.csv                 which code produces each figure and table
```

## 2. Software environment

```bash
conda env create -f environment.yml
conda activate prostate-quiescence
```

Without conda: `pip install -r requirements.txt` (Python 3.9; TensorFlow 2.10 is the last version with native
Windows GPU support, and SHAP 0.41 requires it). R packages for Step 7 are listed in `r_packages.txt`.

Check the installation:

```bash
python -c "import tensorflow as tf, shap, sklearn; print(tf.__version__, shap.__version__, sklearn.__version__)"
```

## 3. Data

| Data | Where | Used by |
|---|---|---|
| Cell-line scRNA-seq (PC3, C4-2B, Myc-CaP; high and low serum) and FACS G0/non-G0 labels | GEO accession [ADD ACCESSION]; processed files at https://drive.google.com/drive/folders/1G0PcZrgVe6RN5WR4F_w12w-4VRew8iwp | Steps 1–6 |
| Intersected matrix, C4-2B high serum | `datasets_h5/` in this repository | Step 1 output example |
| GSE176031 patient cohort | GEO GSE176031 | Step 7 |

Steps 5 and 6 locate the intersected matrices through the `PCQ_BASE` environment variable, which should point
at the folder that contains `outputs/table1_intersection_h5_table1_samples/`:

```bash
export PCQ_BASE=/path/to/project     # Windows: set PCQ_BASE=C:\path\to\project
```

## 4. Protocol

Runtimes are for a laptop with an NVIDIA RTX 2060 (6 GB), Intel i7-10750H and 16 GB RAM.

### Step 1 — Common gene space

```bash
python create_intersection_h5.py --config config/datasets.json --output_dir outputs/table1_intersection_h5_table1_samples
```

*Input:* a JSON file listing, for each dataset, its expression matrix, gene list and FACS labels.
*Output:* one H5 file per dataset restricted to the 10,697 genes common to all six, plus the common gene list
and a manifest. *Runtime:* about 13 minutes.

### Step 2 — Train the MLC-AE

```bash
python mlc-ae_.py --phase train --h5_file outputs/table1_intersection_h5_table1_samples/PC3_high_table1_intersection_genes_table1_samples.h5 --output_dir outputs/PC3_high --max_epoch 10 --seed 4
```

*Input:* one intersected H5 file. *Output:* model weights and training log. *Runtime:* seconds per dataset on a
GPU. Options: `--batch_size` (default 256), `--test_size` (held-out cells), `--seed` (default 4).

### Step 3 — Test and rank genes with SHAP

```bash
python mlc-ae_.py --phase test --h5_file outputs/table1_intersection_h5_table1_samples/PC3_high_table1_intersection_genes_table1_samples.h5 --output_dir outputs/PC3_high --seed 4
```

*Output:* held-out AUC and the SHAP gene ranking used for the top-10 gene sets. Add `--no_shap` to score
without recomputing SHAP values. *Runtime:* about 20–25 s per dataset for the SHAP step.

### Step 4 — Five-fold cross-validation

```bash
python mlc_ae_cv.py
```

Settings are in `options.py`; edit the dataset entry there before running. *Output:* per-fold AUCs for the
PC3 datasets.

### Step 5 — Ablations

```bash
python run_ablation_full_vs_classifier_only.py     # full model vs classifier-only network
python run_permuted_label_ablation.py              # permuted-label control
```

*Input:* the intersected H5 files found through `PCQ_BASE`. *Output:* one CSV per analysis under
`outputs/`, reporting AUC per dataset. These produce Supplementary Table S4.

### Step 6 — Latent-space UMAP

```bash
python run_latent_space_umap_all_six.py
```

*Output:* the 64-dimensional encoded representations projected with UMAP, one panel per dataset
(Supplementary Figure S9).

### Step 7 — Patient cohort (R)

```bash
Rscript external_dataset/01_load_and_merge.R
Rscript external_dataset/02_qc_and_cluster.R
```

*Input:* the GSE176031 digital gene expression matrices. *Output:* the merged, quality-controlled Seurat
object and the tumour-epithelial clusters used for signature scoring (Supplementary Figures S5–S7).

## 5. Trained model

`MycCaP_low_mlcae.weights.h5` is the trained MLC-AE for the Myc-CaP low-serum dataset. Models for the other
five datasets are in the linked Drive folder. A matrix scored with these weights must be normalised the same
way (Seurat LogNormalize, scale factor 10,000) and its columns ordered by the common gene list produced in
Step 1.

## 6. Which code makes which figure

`docs/figure_table_map.csv` lists every figure and table in the manuscript and Supplementary Material with
the script that produces it. Items generated by external tools (iScanGuide, iPathwayGuide) or by laboratory
experiments are marked as such and cannot be reproduced from this repository.

## 7. Troubleshooting

**No GPU found / "Could not load dynamic library cudart64_110.dll".** TensorFlow falls back to the CPU and
everything still runs, roughly 5–10× slower. For GPU use, TensorFlow 2.10 needs CUDA 11.2 and cuDNN 8.1. To
force CPU execution, set `CUDA_VISIBLE_DEVICES=-1`.

**Out of memory.** The dense matrix for the largest dataset is about 1 GB in float32. Run one dataset at a
time, and lower the batch size with `--batch_size 64`.

**`shap.GradientExplainer` raises a `tf.function` error.** SHAP 0.41 requires TensorFlow 2.10; newer
TensorFlow versions change the gradient API. Use the pinned versions.

**`FileNotFoundError` in Steps 5 and 6.** `PCQ_BASE` is unset or does not contain
`outputs/table1_intersection_h5_table1_samples/`. Set it as shown in Section 3.

**Results differ slightly between runs.** Model training and SHAP sampling are stochastic. Seeds are fixed
(seed 4), but GPU kernels are not bit-deterministic, so AUCs can move by a few thousandths.

## 8. Citation

If you use this code, please cite the manuscript and this repository release: [ADD DOI AFTER ARCHIVING].
