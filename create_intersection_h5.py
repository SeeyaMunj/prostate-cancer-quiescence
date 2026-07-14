import argparse
import csv
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np


HEADER_TOKENS = {"gene", "genes", "gene_name", "gene names", "gene name", "genename"}


def clean(value):
    return value.strip().lstrip("\ufeff").strip('"').strip()


def read_gene_file(path):
    genes = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row:
                continue
            gene = clean(row[0])
            if not gene:
                continue
            if not genes and gene.lower() in HEADER_TOKENS:
                continue
            genes.append(gene)
    return genes


def read_genes_from_feature_first_column(path):
    genes = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                genes.append(clean(row[0]))
    return genes


def read_genes(dataset):
    if dataset.get("feature_has_gene_col", False):
        return read_genes_from_feature_first_column(dataset["feature_csv"])
    if not dataset.get("gene_csv"):
        raise ValueError(f"{dataset['name']} needs gene_csv when feature_has_gene_col is false.")
    return read_gene_file(dataset["gene_csv"])


def first_index_by_gene(genes):
    mapping = {}
    duplicates = 0
    for idx, gene in enumerate(genes):
        if gene in mapping:
            duplicates += 1
            continue
        mapping[gene] = idx
    return mapping, duplicates


def parse_labels(samples, dataset):
    if dataset.get("label_csv"):
        labels = []
        with open(dataset["label_csv"], newline="", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                value = clean(row[0])
                if value.lower() in {"label", "labels"}:
                    continue
                labels.append(int(value))
        if len(labels) != len(samples):
            raise ValueError(f"{dataset['name']} label_csv has {len(labels)} labels, expected {len(samples)}.")
        return np.asarray(labels, dtype=np.int64)

    regex = dataset.get("label_regex")
    if not regex:
        raise ValueError(f"{dataset['name']} needs either label_csv or label_regex.")

    go_value = int(dataset.get("go_label", 0))
    ngo_value = int(dataset.get("ngo_label", 1))
    labels = []
    for sample in samples:
        match = re.search(regex, clean(sample), flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not infer GO/NGO label from sample name: {sample}")
        group = match.group(1).upper()
        labels.append(ngo_value if group == "NGO" else go_value)
    return np.asarray(labels, dtype=np.int64)


def get_samples(dataset):
    with open(dataset["feature_csv"], newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
    if dataset.get("feature_has_gene_col", False):
        return [clean(x) for x in header[1:]]
    return [clean(x) for x in header]


def collect_intersection(datasets):
    gene_lists = {}
    gene_sets = []

    for dataset in datasets:
        name = dataset["name"]
        genes = read_genes(dataset)
        gene_lists[name] = genes
        gene_set = set(genes)
        gene_sets.append(gene_set)
        print(f"{name}: rows={len(genes)} unique_genes={len(gene_set)} duplicates={len(genes) - len(gene_set)}")

    common_set = set.intersection(*gene_sets)

    # Keep the final feature order from the first dataset.
    first_name = datasets[0]["name"]
    common_genes = [gene for gene in gene_lists[first_name] if gene in common_set]
    seen = set()
    common_genes = [gene for gene in common_genes if not (gene in seen or seen.add(gene))]

    print(f"COMMON_INTERSECTION_GENES={len(common_genes)}")
    return common_genes, gene_lists


def write_h5(dataset, common_genes, gene_lists, output_dir, block_rows=256):
    name = dataset["name"]
    feature_csv = dataset["feature_csv"]
    out_path = Path(output_dir) / f"{name}_common_intersection.h5"
    gene_txt_path = Path(output_dir) / f"{name}_common_intersection_genes.txt"

    gene_to_row, duplicates = first_index_by_gene(gene_lists[name])
    missing = [gene for gene in common_genes if gene not in gene_to_row]
    if missing:
        raise ValueError(f"{name} is missing {len(missing)} common genes unexpectedly.")

    selected_rows = {gene_to_row[gene]: pos for pos, gene in enumerate(common_genes)}
    samples = get_samples(dataset)
    labels = parse_labels(samples, dataset)

    num_samples = len(samples)
    num_features = len(common_genes)
    print(f"\n{name}: writing {out_path}")
    print(f"{name}: samples={num_samples} genes={num_features} duplicate_rows_ignored={duplicates}")

    with h5py.File(out_path, "w") as h5:
        feature_ds = h5.create_dataset(
            "feature",
            shape=(num_samples, num_features),
            dtype="float32",
            chunks=(min(256, num_samples), min(512, num_features)),
            compression="gzip",
            compression_opts=4,
        )
        h5.create_dataset("label", data=labels.reshape(-1, 1), compression="gzip")
        h5.create_dataset("gene_name", data=np.asarray(common_genes, dtype="S"))
        h5.create_dataset("sample", data=np.asarray(samples, dtype="S"))
        h5.attrs["source_csv"] = feature_csv
        h5.attrs["feature_set"] = "gene intersection across input datasets"

        block_values = []
        block_positions = []
        written = 0
        with open(feature_csv, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)
            for row_idx, row in enumerate(reader):
                out_pos = selected_rows.get(row_idx)
                if out_pos is None:
                    continue

                values = row[1:] if dataset.get("feature_has_gene_col", False) else row
                if len(values) != num_samples:
                    raise ValueError(f"{name} row {row_idx + 2} has {len(values)} values, expected {num_samples}.")

                try:
                    numeric_values = np.asarray(values, dtype=np.float32)
                except ValueError:
                    numeric_values = []
                    for value in values:
                        try:
                            numeric_values.append(float(value))
                        except ValueError:
                            numeric_values.append(0.0)
                    numeric_values = np.asarray(numeric_values, dtype=np.float32)

                block_values.append(numeric_values)
                block_positions.append(out_pos)

                if len(block_values) == block_rows:
                    arr = np.vstack(block_values)
                    order = np.argsort(block_positions)
                    positions = np.asarray(block_positions, dtype=np.int64)[order]
                    feature_ds[:, positions] = arr[order].T
                    written += len(block_values)
                    print(f"{name}: wrote {written}/{num_features}", flush=True)
                    block_values = []
                    block_positions = []

            if block_values:
                arr = np.vstack(block_values)
                order = np.argsort(block_positions)
                positions = np.asarray(block_positions, dtype=np.int64)[order]
                feature_ds[:, positions] = arr[order].T
                written += len(block_values)
                print(f"{name}: wrote {written}/{num_features}", flush=True)

    gene_txt_path.write_text("\n".join(common_genes) + "\n", encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Create one H5 file per dataset using the gene intersection across all input CSV matrices."
    )
    parser.add_argument("--config", required=True, help="JSON file describing the datasets.")
    parser.add_argument("--output_dir", required=True, help="Folder where H5 files will be written.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    datasets = config["datasets"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common_genes, gene_lists = collect_intersection(datasets)
    (output_dir / "common_intersection_genes.txt").write_text("\n".join(common_genes) + "\n", encoding="utf-8")

    manifest = []
    for dataset in datasets:
        h5_path = write_h5(dataset, common_genes, gene_lists, output_dir)
        manifest.append(f"{dataset['name']}\t{h5_path}")

    (output_dir / "manifest.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
