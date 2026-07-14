# ============================================================
# 03_score_signatures.R
# Purpose: subset to tumor-epithelial cells and score each cell
#          for the G0 signature and three proliferation metrics.
# ============================================================

library(Seurat)
library(msigdbr)
library(dplyr)

combined <- readRDS("gse176031_clustered.rds")

# ---- 1. Subset to tumor epithelial cells ----
# Determined via 02b_identify_tumor_clusters.R: clusters where Luminal_tumor
# (KLK3/KLK2/ACPP/NKX3-1/PCA3/AMACR/TRPM8) is the single highest-scoring
# marker category among the 9 tested, with Immune/Endothelial/Fibroblast/
# SmoothMuscle all low. Excludes 3 (Basal-dominant), 10 (Club-dominant),
# and 18 (n=94, no dominant category -- ambiguous).
#
# CAVEAT -- state this in the Methods: this luminal-marker rule cannot
# separate true malignant tumor cells from non-malignant luminal epithelium,
# since both express the same core LE markers. Song et al. 2022 only
# resolve that distinction via ERG expression + InferCNV, which isn't
# reproduced here. Describe this compartment as "tumor-enriched luminal
# epithelium," not purified malignant cells -- consistent with why it's a
# larger share of the tissue (~46%) than Song et al.'s stricter tumor-only
# calls. Cluster 8 shows a secondary Endothelial/Club signal (14.5%/9.3%)
# alongside its dominant Luminal signal (68.5%) -- plausible ambient-RNA/
# doublet contamination; kept in since Luminal still clearly dominates, but
# a reasonable candidate for a with/without sensitivity check.
tumor_epithelial_clusters <- c("1", "4", "5", "7", "8", "9", "14", "17")

tumor_epi <- subset(combined, idents = tumor_epithelial_clusters)
cat("Tumor epithelial cells:", ncol(tumor_epi), "\n")
cat("Patients represented:\n")
print(table(droplevels(tumor_epi$patient_id)))
# If one patient dominates the cell count, flag that now -- it affects how
# much weight to put on the pooled (cell-level) correlation vs. the
# per-patient check in script 04.

# ---- 2. G0 signature score ----
# 26-gene core set from Fig. 8e of the manuscript. NOTE: only 25 symbols are
# printed in the manuscript text against a stated count of "26 genes" --
# cross-check this list against your own analysis code / supplementary
# table before using it here, since a silently incomplete list would
# understate the score.
g0_genes <- c("ARF5","BIRC5","BUB3","HIST1H4C","MIF","NDUFA13","NME2",
              "RPL11","RPL15","RPL36A","RPL37","RPL41","RPL7A","RPS18",
              "RPS21","RPS9","RRM2","SPINT2","SRSF11","TUBA1B","TUBB",
              "TXN","UBE2C","UQCRQ","WDR34")

missing_g0 <- setdiff(g0_genes, rownames(tumor_epi))
if (length(missing_g0) > 0) {
  cat("WARNING - G0 genes not found in this dataset (will be dropped from scoring):\n")
  print(missing_g0)
}

tumor_epi <- AddModuleScore(tumor_epi, features = list(g0_genes), name = "G0_score")
tumor_epi$G0_score  <- tumor_epi$G0_score1   # AddModuleScore appends a numeric suffix
tumor_epi$G0_score1 <- NULL

# ---- 3. Proliferation, metric 1: MKI67 expression ----
tumor_epi$MKI67_expr <- FetchData(tumor_epi, vars = "MKI67")[, 1]

# ---- 4. Proliferation, metric 2: E2F/MYC target module ----
# Pulled from MSigDB Hallmark rather than typed by hand, so the gene list is
# exact and citable (cite as MSigDB Hallmark, Liberzon et al. 2015).
hallmark  <- msigdbr(species = "Homo sapiens", category = "H")
e2f_genes <- hallmark %>% filter(gs_name == "HALLMARK_E2F_TARGETS")   %>% pull(gene_symbol)
myc_genes <- hallmark %>% filter(gs_name == "HALLMARK_MYC_TARGETS_V1") %>% pull(gene_symbol)

tumor_epi <- AddModuleScore(tumor_epi, features = list(e2f_genes), name = "E2F_score")
tumor_epi <- AddModuleScore(tumor_epi, features = list(myc_genes), name = "MYC_score")
tumor_epi$E2F_score <- tumor_epi$E2F_score1; tumor_epi$E2F_score1 <- NULL
tumor_epi$MYC_score <- tumor_epi$MYC_score1; tumor_epi$MYC_score1 <- NULL
tumor_epi$E2F_MYC_score <- (tumor_epi$E2F_score + tumor_epi$MYC_score) / 2

# ---- 5. Proliferation, metric 3: canonical S/G2M cycling score ----
# Uses the same Tirosh et al. gene sets / Seurat function already cited
# elsewhere in the response document (Reviewer 1, Comment 3), so the method
# is consistent across the paper.
data("cc.genes.updated.2019", package = "Seurat")
tumor_epi <- CellCycleScoring(tumor_epi,
                               s.features   = cc.genes.updated.2019$s.genes,
                               g2m.features = cc.genes.updated.2019$g2m.genes)
# Adds: S.Score, G2M.Score, Phase (categorical G1/S/G2M -- useful as an
# alternative, non-continuous check in script 04 if you want one).

saveRDS(tumor_epi, "gse176031_tumor_epithelial_scored.rds")
