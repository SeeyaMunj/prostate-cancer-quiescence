# Installs the R packages used by the R steps of the pipeline (R 4.5.2).
pkgs <- c("Seurat", "dplyr", "ggplot2", "patchwork", "lme4", "lmerTest", "msigdbr", "babelgene")
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
for (p in pkgs) cat(p, as.character(utils::packageVersion(p)), "\n")
