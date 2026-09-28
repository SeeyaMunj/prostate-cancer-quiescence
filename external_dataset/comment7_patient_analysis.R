# ============================================================
# comment7_patient_analysis.R
# Reviewer 2, comment 7: exploratory association analysis in GSE176031 (tumor-enriched luminal epithelium),
# re-run exactly as described in the Supplement.
#   - G0 signature: all 26 genes (including TCOF1), fold-change-weighted and signed:
#       score = sum over genes of (z-scored normalized expression x log2 fold change), weights from PC3LOW_26GENES.csv
#   - Proliferation measures (computed in 03_score_signatures.R): E2F/MYC module = mean of the AddModuleScore values for
#     HALLMARK_E2F_TARGETS and HALLMARK_MYC_TARGETS_V1; S/G2M cycling score = S.Score + G2M.Score (Seurat
#     CellCycleScoring, Tirosh et al. gene sets); MKI67 = normalized MKI67 expression
#   - Cell-level Spearman correlations; Ki-67+ frequency in signature-high vs -low quartiles (Fisher's exact test)
#   - Patient-adjusted linear mixed-effects models for ALL THREE measures: measure ~ G0 score + (1 | patient)
#   - Patient-level summary: within-patient Spearman correlations for each of the 11 patients, and correlation of
#     patient means (pseudobulk, n = 11)
#   - Sensitivity analysis: unweighted AddModuleScore with the same 26 genes
#   - Figure S8: UMAP of weighted G0 score, MKI67, E2F/MYC module and quartile groups; per-patient correlation plot
# Input : gse176031_tumor_epithelial_scored.rds (scripts 01-03; patient IDs PB1/PB2 already assigned),
#         PC3LOW_26GENES.csv (gene, log2 fold change from the PC3 low-serum G0 vs non-G0 comparison)
# Output: comment7_*.csv in the output directory
# ============================================================
suppressMessages({library(Seurat); library(dplyr); library(lme4); library(lmerTest); library(ggplot2); library(patchwork)})
set.seed(4)

IN_RDS  <- "C:/Users/seeya/Documents/GitHub/prostate-cancer-quiescence/external_dataset/gse176031_tumor_epithelial_scored.rds"
IN_FC   <- "C:/Users/seeya/Downloads/PC3LOW_26GENES.csv"
OUT_DIR <- "C:/Users/seeya/Documents/claude/shap/r2_analysis/results_comment7"
dir.create(OUT_DIR, showWarnings = FALSE)

g0_genes <- c("ARF5","BIRC5","BUB3","HIST1H4C","MIF","NDUFA13","NME2","RPL11","RPL15","RPL36A","RPL37","RPL41","RPL7A",
              "RPS18","RPS21","RPS9","RRM2","SPINT2","SRSF11","TCOF1","TUBA1B","TUBB","TXN","UBE2C","UQCRQ","WDR34")

tumor_epi <- readRDS(IN_RDS)
tumor_epi <- JoinLayers(tumor_epi)
pb1_gsm <- c("GSM5353216", "GSM5353217", "GSM5353218", "GSM5353219", "GSM5353220")
pb2_gsm <- c("GSM5353221", "GSM5353222", "GSM5353223")
tumor_epi$patient_id <- as.character(tumor_epi$patient_id)
tumor_epi$patient_id[tumor_epi$gsm %in% pb1_gsm] <- "PB1"
tumor_epi$patient_id[tumor_epi$gsm %in% pb2_gsm] <- "PB2"
tumor_epi$patient_id <- factor(tumor_epi$patient_id)
cat("Cells:", ncol(tumor_epi), " patients:", nlevels(tumor_epi$patient_id), " NA patient:", sum(is.na(tumor_epi$patient_id)), "\n")

# ---- gene weights ----
fc <- read.csv(IN_FC, stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM")
stopifnot(setequal(fc$Gene, g0_genes))
fc$present_in_GSE176031 <- fc$Gene %in% rownames(tumor_epi)
expr <- t(as.matrix(GetAssayData(tumor_epi, layer = "data")[fc$Gene[fc$present_in_GSE176031], , drop = FALSE]))
sdv <- apply(expr, 2, sd)
fc$nonzero_variance <- fc$Gene %in% names(sdv)[sdv > 0]
use <- fc$Gene[fc$present_in_GSE176031 & fc$nonzero_variance]
expr_z <- scale(expr[, use, drop = FALSE])
w <- fc$Log2_Fold_Change[match(use, fc$Gene)]
tumor_epi$G0_weighted <- as.numeric(expr_z %*% w)
fc$sign <- ifelse(fc$Log2_Fold_Change > 0, "+", "-")
fc$used_in_score <- fc$Gene %in% use
write.csv(fc[order(fc$Log2_Fold_Change), c("Gene", "Log2_Fold_Change", "sign", "FDR_P_Value", "present_in_GSE176031", "used_in_score")],
          file.path(OUT_DIR, "comment7_gene_weights.csv"), row.names = FALSE)
cat("Genes used in weighted score:", length(use), "of 26; not used:", paste(setdiff(g0_genes, use), collapse = ", "), "\n")
if ("G0_score_weighted" %in% colnames(tumor_epi@meta.data))
  cat("Agreement with previously saved weighted score (Pearson r):",
      round(cor(tumor_epi$G0_weighted, tumor_epi$G0_score_weighted), 6), "\n")

# ---- unweighted sensitivity score (AddModuleScore, same 26 genes) ----
tumor_epi <- AddModuleScore(tumor_epi, features = list(intersect(g0_genes, rownames(tumor_epi))), name = "G0_unweighted_26_", seed = 4)
tumor_epi$G0_unweighted26 <- tumor_epi$G0_unweighted_26_1

md <- tumor_epi@meta.data
md$cycling_score <- md$S.Score + md$G2M.Score
measures <- c(E2F_MYC = "E2F_MYC_score", S_G2M = "cycling_score", MKI67 = "MKI67_expr")

analyse <- function(score_col, label) {
  rows <- list()
  for (m in names(measures)) {
    y <- measures[[m]]
    ct <- suppressWarnings(cor.test(md[[score_col]], md[[y]], method = "spearman", exact = FALSE))
    f <- as.formula(paste(y, "~", score_col, "+ (1 | patient_id)"))
    mm <- lmer(f, data = md)
    co <- summary(mm)$coefficients[score_col, ]
    pp <- md %>% group_by(patient_id) %>%
      summarise(rho = suppressWarnings(cor(.data[[score_col]], .data[[y]], method = "spearman")), n = n(), .groups = "drop")
    pb <- md %>% group_by(patient_id) %>% summarise(s = mean(.data[[score_col]]), m = mean(.data[[y]]), .groups = "drop")
    pbt <- suppressWarnings(cor.test(pb$s, pb$m, method = "spearman", exact = FALSE))
    rows[[m]] <- data.frame(score = label, measure = m,
                            cell_rho = unname(ct$estimate), cell_p = ct$p.value,
                            mixed_beta = unname(co["Estimate"]), mixed_se = unname(co["Std. Error"]), mixed_p = unname(co["Pr(>|t|)"]),
                            patients_n = sum(!is.na(pp$rho)), patient_rho_median = median(pp$rho, na.rm = TRUE),
                            patient_rho_min = min(pp$rho, na.rm = TRUE), patient_rho_max = max(pp$rho, na.rm = TRUE),
                            patients_negative = sum(pp$rho < 0, na.rm = TRUE),
                            pseudobulk_rho = unname(pbt$estimate), pseudobulk_p = pbt$p.value)
    write.csv(pp, file.path(OUT_DIR, sprintf("comment7_per_patient_rho_%s_%s.csv", label, m)), row.names = FALSE)
  }
  q <- quantile(md[[score_col]], c(0.25, 0.75))
  grp <- ifelse(md[[score_col]] >= q[2], "high", ifelse(md[[score_col]] <= q[1], "low", "mid"))
  tab <- table(grp[grp != "mid"], md$MKI67_expr[grp != "mid"] > 0)
  pct <- prop.table(tab, 1)[, "TRUE"] * 100
  ft <- fisher.test(tab)
  res <- bind_rows(rows)
  res$ki67_pos_pct_high <- pct["high"]; res$ki67_pos_pct_low <- pct["low"]; res$ki67_fisher_p <- ft$p.value
  ribo <- suppressWarnings(cor.test(md[[score_col]], md$percent.ribo, method = "spearman", exact = FALSE))
  res$rho_vs_percent_ribo <- unname(ribo$estimate)
  res
}

out <- bind_rows(analyse("G0_weighted", "weighted26"), analyse("G0_unweighted26", "unweighted26"))
write.csv(out, file.path(OUT_DIR, "comment7_patient_association_results.csv"), row.names = FALSE)
options(width = 250)
print(out, digits = 3)
pts <- md %>% group_by(patient_id) %>% summarise(n_cells = n(), .groups = "drop")
write.csv(pts, file.path(OUT_DIR, "comment7_patients.csv"), row.names = FALSE)
print(pts)

# ---- Figure S8 (weighted score) ----
q <- quantile(tumor_epi$G0_weighted, c(0.25, 0.75))
tumor_epi$g0_group <- factor(ifelse(tumor_epi$G0_weighted >= q[2], "signature-high",
                                    ifelse(tumor_epi$G0_weighted <= q[1], "signature-low", "middle")),
                             levels = c("signature-high", "middle", "signature-low"))
p1 <- FeaturePlot(tumor_epi, features = "G0_weighted", cols = c("lightgrey", "darkred")) + ggtitle("Weighted 26-gene G0 score")
p2 <- FeaturePlot(tumor_epi, features = "MKI67_expr", cols = c("lightgrey", "darkblue")) + ggtitle("MKI67 expression")
p3 <- FeaturePlot(tumor_epi, features = "E2F_MYC_score", cols = c("lightgrey", "darkgreen")) + ggtitle("E2F/MYC target module")
p4 <- DimPlot(tumor_epi, group.by = "g0_group", cols = c("firebrick", "grey85", "forestgreen")) +
  ggtitle("Quartile groups (weighted score)")
ggsave(file.path(OUT_DIR, "FigureS8_weighted_G0_score.png"), (p1 | p2) / (p3 | p4), width = 10, height = 8, dpi = 200)

# ---- per-patient correlations (weighted score) ----
pp_all <- bind_rows(lapply(names(measures), function(m)
  read.csv(file.path(OUT_DIR, sprintf("comment7_per_patient_rho_weighted26_%s.csv", m))) %>% mutate(measure = m)))
pp_all$measure <- factor(pp_all$measure, levels = c("E2F_MYC", "S_G2M", "MKI67"), labels = c("E2F/MYC module", "S/G2M score", "MKI67"))
pd <- ggplot(pp_all, aes(x = rho, y = reorder(patient_id, n))) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_point(aes(size = n), colour = "steelblue4") + facet_wrap(~ measure, nrow = 1) +
  labs(x = "Within-patient Spearman rho (weighted G0 score vs measure)", y = "Patient", size = "Cells") +
  theme_bw(base_size = 10)
ggsave(file.path(OUT_DIR, "FigureSX_per_patient_correlations.png"), pd, width = 9, height = 3.8, dpi = 200)
