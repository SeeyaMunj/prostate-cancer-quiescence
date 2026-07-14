# ============================================================
# 06_generate_writeup_numbers.R
# Purpose: format the computed statistics into ready-to-paste
#          sentences matching the existing response-to-reviewers
#          paragraph structure (fills in the "XXX" placeholders).
# ============================================================

stats <- readRDS("stats_results.rds")

cat(sprintf(
"Consistent with the reviewer's hypothesis, cells with high quiescence-signature
scores were significantly enriched among low-proliferation tumor cells: the G0
signature score was inversely correlated with the cycling score (Spearman rho = %.2f,
p = %.2g), with MKI67 expression (rho = %.2f, p = %.2g), and with the E2F/MYC module
score (rho = %.2f, p = %.2g).\n\n",
  stats$results$rho[1], stats$results$p[1],
  stats$results$rho[2], stats$results$p[2],
  stats$results$rho[3], stats$results$p[3]
))

pct <- prop.table(stats$tab, margin = 1)[, "TRUE"] * 100
cat(sprintf(
"Signature-high cells were markedly depleted of Ki-67+ cells (%.1f%% vs. %.1f%%).\n\n",
  pct["low"], pct["high"]
))

cat("--- Additional material not yet in the draft, but worth adding ---\n\n")

cat("Per-patient / pseudobulk check (addresses the same pseudo-replication concern\n")
cat("Reviewer 1 raised in Comment 1):\n")
print(stats$patient_avg)
cat("\n")

cat("Mixed-model check (G0 score predicting cycling score with patient as random effect):\n")
print(summary(stats$mixed_model)$coefficients)
cat("\n")

cat("Technical-covariate check (addresses Reviewer 1's Comment 4, currently blank):\n")
for (nm in names(stats$tech_cov)) {
  x <- stats$tech_cov[[nm]]
  cat(sprintf("  G0 score vs %-12s rho = %.3f, p = %.3g\n", nm, x$estimate, x$p.value))
}
