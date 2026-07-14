# ============================================================
# 05_visualize.R
# Purpose: build the supplementary figure -- UMAP of the tumor-
#          epithelial compartment colored by G0 score and by
#          proliferation metrics.
# ============================================================

library(Seurat)
library(patchwork)
library(ggplot2)
library(dplyr)

tumor_epi <- readRDS("gse176031_tumor_epithelial_scored.rds")

# Recompute the high/low split locally so this script doesn't depend on
# script 04 having written it back onto the Seurat object.
q <- quantile(tumor_epi$G0_score, probs = c(0.25, 0.75))
tumor_epi$g0_group <- case_when(
  tumor_epi$G0_score >= q[2] ~ "high",
  tumor_epi$G0_score <= q[1] ~ "low",
  TRUE ~ "mid"
)

p1 <- FeaturePlot(tumor_epi, features = "G0_score", cols = c("lightgrey", "darkred")) +
  ggtitle("G0 signature score")

p2 <- FeaturePlot(tumor_epi, features = "MKI67_expr", cols = c("lightgrey", "darkblue")) +
  ggtitle("MKI67 expression")

p3 <- FeaturePlot(tumor_epi, features = "E2F_MYC_score", cols = c("lightgrey", "darkgreen")) +
  ggtitle("E2F/MYC target score")

p4 <- DimPlot(tumor_epi, group.by = "g0_group") +
  ggtitle("Signature-high vs -low (quartile split)")

fig <- (p1 | p2) / (p3 | p4)
ggsave("supplementary_figure_G0_external_validation.pdf", fig, width = 10, height = 8)
fig
