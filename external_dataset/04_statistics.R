# ============================================================
# 04_statistics.R
# Purpose: correlations, high/low group comparison, per-patient
#          robustness check, and a technical-covariate check.
# ============================================================

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

# Ki-67+ defined at the single-cell level as "any detected transcript."
# MKI67 dropout is severe in scRNA-seq, so treat this binary comparison as
# SECONDARY, supporting evidence -- the continuous correlation above is
# the primary evidence and should be led with in the write-up.
md$ki67_pos <- md$MKI67_expr > 0

sub <- md[md$g0_group != "mid", ]
tab <- table(sub$g0_group, sub$ki67_pos)
print(tab)
print(fisher.test(tab))
pct_pos <- prop.table(tab, margin = 1)[, "TRUE"] * 100
cat(sprintf("%%MKI67+ in signature-high: %.1f%% vs signature-low: %.1f%%\n",
            pct_pos["high"], pct_pos["low"]))

# ---- 3. Per-patient robustness check ----
# Addresses the same pseudo-replication concern Reviewer 1 raised in
# Comment 1 -- and the source paper (Song et al. 2022) itself reports that
# tumor cells cluster in a patient-specific manner for ERG+ cases, which is
# exactly the kind of structure that can inflate a pooled cell-level
# correlation. Report this alongside the pooled result, not instead of it.

# 3a. Pseudobulk: one point per patient
patient_avg <- md %>%
  group_by(patient_id) %>%
  summarise(G0 = mean(G0_score), cycling = mean(cycling_score),
            mki67 = mean(MKI67_expr), n_cells = n(), .groups = "drop")
print(patient_avg)
print(cor.test(patient_avg$G0, patient_avg$cycling, method = "spearman"))

# 3b. Mixed model: does G0 score predict cycling score after accounting for
#     patient as a random effect? More rigorous than the pseudobulk check
#     alone -- report both.
mod <- lmer(cycling_score ~ G0_score + (1 | patient_id), data = md)
print(summary(mod))

# ---- 4. Technical-covariate check ----
# Pre-empts Reviewer 1's Comment 4, which is currently blank in the
# response document: does the G0 score just track library size / total
# ribosomal reads, given the signature is majority ribosomal-protein genes?
cor_nCount <- cor.test(md$G0_score, md$nCount_RNA,   method = "spearman")
cor_nFeat  <- cor.test(md$G0_score, md$nFeature_RNA, method = "spearman")
cor_ribo   <- cor.test(md$G0_score, md$percent.ribo, method = "spearman")

cat(sprintf("G0 score vs nCount_RNA:   rho = %.3f, p = %.3g\n", cor_nCount$estimate, cor_nCount$p.value))
cat(sprintf("G0 score vs nFeature_RNA: rho = %.3f, p = %.3g\n", cor_nFeat$estimate,  cor_nFeat$p.value))
cat(sprintf("G0 score vs %%ribo:        rho = %.3f, p = %.3g\n", cor_ribo$estimate,   cor_ribo$p.value))
# If these correlations are as strong as (or stronger than) the G0-vs-
# cycling correlation in section 1, the signature may be tracking a
# technical covariate rather than quiescence specifically. Worth reporting
# either way; if it looks confounded, a useful sensitivity check is to
# re-run scoring with the RPL/RPS genes dropped from g0_genes in script 03
# and see whether the cycling/MKI67 correlation survives on the non-
# ribosomal genes alone (ARF5, BIRC5, BUB3, HIST1H4C, MIF, NDUFA13, NME2,
# RRM2, SPINT2, SRSF11, TUBA1B, TUBB, TXN, UBE2C, UQCRQ, WDR34).

saveRDS(list(results = results, patient_avg = patient_avg, tab = tab,
             mixed_model = mod,
             tech_cov = list(nCount = cor_nCount, nFeature = cor_nFeat, ribo = cor_ribo)),
        "stats_results.rds")
