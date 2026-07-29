library(Seurat)
library(dplyr)
library(lme4)       # for the patient-random-effect check
library(lmerTest)   # gives p-values for lme4 models

tumor_epi <- readRDS("gse176031_tumor_epithelial_scored.rds")
md <- tumor_epi@meta.data
md$cycling_score <- md$S.Score + md$G2M.Score

# ---- 1. Pooled (cell-level) correlations, as drafted in the response ----
cor_cycling <- cor.test(md$G0_score, md$cycling_score, method = "spearman")
cor_mki67   <- cor.test(md$G0_score, md$MKI67_expr,    method = "spearman")
cor_e2fmyc  <- cor.test(md$G0_score, md$E2F_MYC_score, method = "spearman")

results <- data.frame(
  comparison = c("G0 vs cycling score", "G0 vs MKI67", "G0 vs E2F/MYC"),
  rho = c(cor_cycling$estimate, cor_mki67$estimate, cor_e2fmyc$estimate),
  p   = c(cor_cycling$p.value,  cor_mki67$p.value,  cor_e2fmyc$p.value)
)
print(results)

# ---- 2. Signature-high vs signature-low: %MKI67+ comparison ----
q <- quantile(md$G0_score, probs = c(0.25, 0.75))
md$g0_group <- case_when(
  md$G0_score >= q[2] ~ "high",
  md$G0_score <= q[1] ~ "low",
  TRUE ~ "mid"
)

md$ki67_pos <- md$MKI67_expr > 0

sub <- md[md$g0_group != "mid", ]
tab <- table(sub$g0_group, sub$ki67_pos)
print(tab)
print(fisher.test(tab))
pct_pos <- prop.table(tab, margin = 1)[, "TRUE"] * 100
cat(sprintf("%%MKI67+ in signature-high: %.1f%% vs signature-low: %.1f%%\n",
            pct_pos["high"], pct_pos["low"]))


# 3a. Pseudobulk: one point per patient
patient_avg <- md %>%
  group_by(patient_id) %>%
  summarise(G0 = mean(G0_score), cycling = mean(cycling_score),
            mki67 = mean(MKI67_expr), n_cells = n(), .groups = "drop")
print(patient_avg)
print(cor.test(patient_avg$G0, patient_avg$cycling, method = "spearman"))


mod <- lmer(cycling_score ~ G0_score + (1 | patient_id), data = md)
print(summary(mod))

# ---- 4. Technical-covariate check ----

cor_nCount <- cor.test(md$G0_score, md$nCount_RNA,   method = "spearman")
cor_nFeat  <- cor.test(md$G0_score, md$nFeature_RNA, method = "spearman")
cor_ribo   <- cor.test(md$G0_score, md$percent.ribo, method = "spearman")

cat(sprintf("G0 score vs nCount_RNA:   rho = %.3f, p = %.3g\n", cor_nCount$estimate, cor_nCount$p.value))
cat(sprintf("G0 score vs nFeature_RNA: rho = %.3f, p = %.3g\n", cor_nFeat$estimate,  cor_nFeat$p.value))
cat(sprintf("G0 score vs %%ribo:        rho = %.3f, p = %.3g\n", cor_ribo$estimate,   cor_ribo$p.value))


saveRDS(list(results = results, patient_avg = patient_avg, tab = tab,
             mixed_model = mod,
             tech_cov = list(nCount = cor_nCount, nFeature = cor_nFeat, ribo = cor_ribo)),
        "stats_results.rds")
