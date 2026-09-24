library(Seurat)
library(msigdbr)
library(dplyr)

combined <- readRDS("gse176031_clustered.rds")


tumor_epithelial_clusters <- c("1", "4", "5", "7", "8", "9", "14", "17")

tumor_epi <- subset(combined, idents = tumor_epithelial_clusters)
cat("Tumor epithelial cells:", ncol(tumor_epi), "\n")
cat("Patients represented:\n")
print(table(droplevels(tumor_epi$patient_id)))

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


hallmark  <- msigdbr(species = "Homo sapiens", category = "H")
e2f_genes <- hallmark %>% filter(gs_name == "HALLMARK_E2F_TARGETS")   %>% pull(gene_symbol)
myc_genes <- hallmark %>% filter(gs_name == "HALLMARK_MYC_TARGETS_V1") %>% pull(gene_symbol)

tumor_epi <- AddModuleScore(tumor_epi, features = list(e2f_genes), name = "E2F_score")
tumor_epi <- AddModuleScore(tumor_epi, features = list(myc_genes), name = "MYC_score")
tumor_epi$E2F_score <- tumor_epi$E2F_score1; tumor_epi$E2F_score1 <- NULL
tumor_epi$MYC_score <- tumor_epi$MYC_score1; tumor_epi$MYC_score1 <- NULL
tumor_epi$E2F_MYC_score <- (tumor_epi$E2F_score + tumor_epi$MYC_score) / 2


data("cc.genes.updated.2019", package = "Seurat")
tumor_epi <- CellCycleScoring(tumor_epi,
                               s.features   = cc.genes.updated.2019$s.genes,
                               g2m.features = cc.genes.updated.2019$g2m.genes)


saveRDS(tumor_epi, "gse176031_tumor_epithelial_scored.rds")
