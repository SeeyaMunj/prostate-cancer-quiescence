# ============================================================
# 01_load_and_merge.R
# Purpose: Parse sample metadata (patient, tissue type) from
#          filenames, load each sample's digital gene expression
#          matrix, tag each cell, and merge into one Seurat object.
# Prereq: run 00_inspect_raw_download.R first and confirm the file
#         format matches what's assumed below.
# ============================================================

# ---- Environment check ----
# Catches this ahead of time instead of failing confusingly downstream:
#   "namespace 'rlang' 1.1.6 is already loaded, but >= 1.1.7 is required"
# rlang can't be swapped mid-session once something has loaded it -- if
# this stops you, restart R, run install.packages("rlang") BEFORE loading
# anything else, then re-run this script in that fresh session.
if (packageVersion("rlang") < "1.1.7") {
  stop("Installed rlang is ", packageVersion("rlang"), "; Seurat/dplyr need >= 1.1.7.\n",
       "  Fix: restart R, then install.packages('rlang') before loading anything else.")
}

library(Seurat)
library(Matrix)
library(data.table)
library(dplyr)
library(stringr)

extract_dir <- "GSE176031_raw"
files <- list.files(extract_dir, full.names = TRUE)

# ---- 1. Build a sample metadata table from filenames ----
# GEO supplementary filenames follow the convention GSMxxxxxxx_<original_name>.
# Adjust the regexes below once you've confirmed the exact pattern from script 00 --
# these are written against the GSM sample titles visible on the GEO record
# (e.g. "PA_PR5249_T1_S3_L001", "PA_PR5249_N1_S1_L001", "HNW_PR5251_T_org_...",
# "PA_AUG_PB_1A_S1"), but GEO truncates the sample list in the browser view,
# so verify against your full local file listing.

meta <- data.frame(file = files, stringsAsFactors = FALSE) %>%
  mutate(
    basefile    = basename(file),
    gsm         = str_extract(basefile, "^GSM[0-9]+"),
    sample_name = str_remove(basefile, "^GSM[0-9]+_"),

    # Tissue type from naming convention -- verify against your actual filenames:
    #   *org*          -> organoid
    #   *_T[0-9]*      -> RP tumor (paired-tumor arm, if a matching _N exists)
    #   *_N[0-9]*      -> RP paired-normal
    #   *PB*           -> prostate biopsy
    #   anything else  -> unpaired RP tumor-only sample (e.g. PR5186/PR5196/PR5199-style)
    tissue_type = case_when(
      str_detect(sample_name, regex("org", ignore_case = TRUE)) ~ "organoid",
      str_detect(sample_name, "_T[0-9]")                        ~ "tumor",
      str_detect(sample_name, "_N[0-9]")                        ~ "normal",
      str_detect(sample_name, regex("PB", ignore_case = TRUE))  ~ "biopsy",
      TRUE ~ "unpaired_tumor"
    ),

    # Patient ID token (PR##### for prostatectomy patients, AUG for one biopsy
    # patient, PB1/PB2 for the other two -- each biopsied at two anatomical
    # regions, named PB1A/PB1B and PB2A/PB2B in the sample names).
    patient_id = str_extract(sample_name, "PR[0-9]+|AUG|PB[0-9]")
  )

print(table(meta$tissue_type))
print(table(meta$patient_id, meta$tissue_type))

# STOP: eyeball this table before continuing. If tissue_type or patient_id
# look wrong for any rows, fix the regexes above -- everything downstream
# (which cells count as "tumor") depends on getting this right.

# ---- 2. Restrict to samples relevant to Comment 6 ----
# Primary tumor-bearing tissue only. Organoids are excluded here: reviewers
# raised organoid validation as a separate, harder ask (Reviewer 1 Comment 6
# and Reviewer 3 Comment 2 both specifically want a "primary" / "patient"
# dataset), so keep this analysis to actual tissue.
meta_use <- meta %>% filter(tissue_type %in% c("tumor", "unpaired_tumor", "biopsy"))
cat("\nSamples to be loaded (tumor-bearing tissue only):\n")
print(meta_use[, c("gsm", "sample_name", "tissue_type", "patient_id")])

# ---- 3. Read each sample's DGE matrix into a Seurat object ----
read_dge <- function(path) {
  dt <- fread(path, header = TRUE)
  genes <- dt[[1]]
  if (any(duplicated(genes))) {
    cat("  Note:", sum(duplicated(genes)), "duplicate gene symbols; appending suffixes.\n",
        "  (Consider summing duplicate rows instead if this matters for your analysis.)\n")
    genes <- make.unique(genes)
  }
  mat <- as.matrix(dt[, -1, with = FALSE])
  rownames(mat) <- genes
  Matrix(mat, sparse = TRUE)
}

seurat_list <- list()
for (i in seq_len(nrow(meta_use))) {
  row <- meta_use[i, ]
  cat("Loading", row$gsm, "(", row$tissue_type, ")...\n")
  mat <- read_dge(row$file)
  colnames(mat) <- paste0(row$gsm, "_", colnames(mat))  # avoid barcode collisions across samples

  so <- CreateSeuratObject(counts = mat, project = row$gsm, min.cells = 3, min.features = 200)
  so$gsm         <- row$gsm
  so$patient_id  <- row$patient_id
  so$tissue_type <- row$tissue_type
  seurat_list[[row$gsm]] <- so
}

# ---- 4. Merge all samples into one object ----
combined <- merge(seurat_list[[1]], y = seurat_list[-1], add.cell.ids = names(seurat_list))
combined$patient_id  <- factor(combined$patient_id)
combined$tissue_type <- factor(combined$tissue_type)

cat("\nCombined object:", ncol(combined), "cells x", nrow(combined), "genes\n")
cat("Patients:", paste(levels(combined$patient_id), collapse = ", "), "\n")

saveRDS(combined, "gse176031_combined_raw.rds")
