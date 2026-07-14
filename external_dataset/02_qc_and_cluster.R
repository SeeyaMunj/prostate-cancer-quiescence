# ============================================================
# 02_qc_and_cluster.R
# Purpose: QC filtering, normalization, clustering, and marker-
#          based inspection so you can identify which cluster(s)
#          are tumor epithelial cells.
# ============================================================

library(Seurat)
library(dplyr)
library(patchwork)

set.seed(42)  # clustering/UMAP have stochastic steps -- fix this for reproducibility,
              # and report it, given how much reviewer attention this manuscript's
              # reproducibility has already received.

combined <- readRDS("gse176031_combined_raw.rds")

# ---- 1. QC metrics ----
combined[["percent.mt"]]   <- PercentageFeatureSet(combined, pattern = "^MT-")
combined[["percent.ribo"]] <- PercentageFeatureSet(combined, pattern = "^RP[SL]")

# IMPORTANT: we deliberately do NOT hard-filter on percent.ribo here.
# The G0 signature you're testing is majority ribosomal-protein genes
# (RPL/RPS), so aggressively filtering cells on %ribosomal content before
# scoring them for a ribosomal-gene signature risks removing exactly the
# biological variation you're trying to detect. We filter on
# nCount/nFeature/%mt only, and instead CHECK percent.ribo as a covariate
# in script 04 (this doubles as material for Reviewer 1's Comment 4, which
# is currently unanswered in the response document).

VlnPlot(combined, features = c("nFeature_RNA", "nCount_RNA", "percent.mt", "percent.ribo"),
        ncol = 4, pt.size = 0)
# Look at this plot before picking thresholds below. Seq-Well data has
# different depth/sparsity characteristics than 10x -- don't reuse 10x-tuned
# cutoffs blindly. The values below are a reasonable starting point for
# Seq-Well/Drop-seq-style data; adjust based on what you actually see.

combined <- subset(combined,
                    subset = nFeature_RNA > 200 & nFeature_RNA < 6000 &
                             nCount_RNA > 500 &
                             percent.mt < 20)

cat("Cells remaining after QC:", ncol(combined), "\n")

# ---- 2. Normalize, reduce, cluster ----
combined <- NormalizeData(combined)
combined <- FindVariableFeatures(combined, selection.method = "vst", nfeatures = 2000)
combined <- ScaleData(combined)
combined <- RunPCA(combined, npcs = 30)
combined <- FindNeighbors(combined, dims = 1:30)
combined <- FindClusters(combined, resolution = 0.6)   # matches the resolution you already
                                                         # used elsewhere in the manuscript (Fig. 3)
combined <- RunUMAP(combined, dims = 1:30)

DimPlot(combined, label = TRUE) + NoLegend()

# ---- 3. Marker-based cluster identity ----
# Mirrors the marker scheme actually used by Song et al. 2022 (the real
# source paper for GSE176031), so cluster calls here are consistent with
# how the original authors annotated this exact dataset.
marker_panel <- list(
  Basal_epithelial = c("KRT5", "KRT15", "KRT17", "TP63"),
  Luminal_tumor     = c("KLK3", "KLK2", "ACPP", "NKX3-1", "PCA3", "AMACR", "TRPM8"),
  Club_OtherEpi     = c("PIGR", "MMP7", "CP", "SCGB1A1", "LTF"),
  Immune_pan        = c("PTPRC"),
  T_cell            = c("CD3D", "IL7R", "CD8A"),
  Myeloid           = c("LYZ", "APOE", "CD68"),
  Endothelial       = c("CLDN5", "SELE"),
  Fibroblast        = c("DCN", "C1S", "C7"),
  SmoothMuscle      = c("ACTA2", "MYH11", "RGS5")
)

DotPlot(combined, features = unlist(marker_panel), cluster.idents = TRUE) +
  RotatedAxis()

# ---- 4. STOP AND LOOK ----
# Identify which cluster number(s) show high Luminal_tumor marker expression
# (KLK3/KLK2/ACPP/NKX3-1/PCA3/AMACR/TRPM8) and LOW Basal/Immune/Stromal
# marker expression. Note: Song et al. went further and separated ERG+/ERG-
# tumor cells from non-malignant luminal cells using ERG expression +
# InferCNV -- a heavier pipeline than needed here. For Comment 6, treating
# the full luminal/tumor-marker-high compartment *within tumor-derived
# samples* (tissue_type != "normal", already excluded in script 01) as
# "tumor epithelial" is a reasonable, clearly-documented simplification --
# state it that way in the Methods rather than implying you replicated the
# full ERG/CNV pipeline.
#
# Write the cluster number(s) you identify into script 03.

saveRDS(combined, "gse176031_clustered.rds")
