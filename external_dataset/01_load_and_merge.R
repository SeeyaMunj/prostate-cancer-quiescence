
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



meta <- data.frame(file = files, stringsAsFactors = FALSE) %>%
  mutate(
    basefile    = basename(file),
    gsm         = str_extract(basefile, "^GSM[0-9]+"),
    sample_name = str_remove(basefile, "^GSM[0-9]+_"),

    
    tissue_type = case_when(
      str_detect(sample_name, regex("org", ignore_case = TRUE)) ~ "organoid",
      str_detect(sample_name, "_T[0-9]")                        ~ "tumor",
      str_detect(sample_name, "_N[0-9]")                        ~ "normal",
      str_detect(sample_name, regex("PB", ignore_case = TRUE))  ~ "biopsy",
      TRUE ~ "unpaired_tumor"
    ),

    
    patient_id = str_extract(sample_name, "PR[0-9]+|AUG|PB[0-9]")
  )

print(table(meta$tissue_type))
print(table(meta$patient_id, meta$tissue_type))


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
