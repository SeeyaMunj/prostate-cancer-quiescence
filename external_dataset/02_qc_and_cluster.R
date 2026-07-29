

library(Seurat)
library(dplyr)
library(patchwork)

set.seed(42)  

combined <- readRDS("gse176031_combined_raw.rds")

# ---- 1. QC metrics ----
combined[["percent.mt"]]   <- PercentageFeatureSet(combined, pattern = "^MT-")
combined[["percent.ribo"]] <- PercentageFeatureSet(combined, pattern = "^RP[SL]")



VlnPlot(combined, features = c("nFeature_RNA", "nCount_RNA", "percent.mt", "percent.ribo"),
        ncol = 4, pt.size = 0)


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

#
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



saveRDS(combined, "gse176031_clustered.rds")
