import os
import random

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from options import opt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tensorflow.keras.layers import BatchNormalization, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


def set_seed(seed):
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def data_generator(h5_file_path, indices, batch_size, num_classes, is_training=True):
    indices = np.array(indices)
    with h5py.File(h5_file_path, "r") as h5:
        features_ds = h5["feature"]
        labels_ds = h5["label"]
        while True:
            if is_training:
                np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                sorted_indices = np.sort(batch_indices)
                x = features_ds[sorted_indices].astype(np.float32)
                y = labels_ds[sorted_indices].squeeze().astype(np.int64)
                if is_training:
                    perm = np.random.permutation(len(x))
                    x = x[perm]
                    y = y[perm]
                yield x, (x, to_categorical(y, num_classes=num_classes))


def create_model(num_features, num_classes):
    inputs = Input(shape=(num_features,), name="inputs")
    normalized = BatchNormalization(name="inputs_4")(inputs)
    encoded = Dense(units=64, activation="relu", name="encoded")(normalized)
    hidden = Dense(512, activation="relu", name="inputs_5")(encoded)
    decoded = Dense(units=num_features, activation="linear", name="m_rna")(hidden)
    category = Dense(units=num_classes, activation="softmax", name="category")(encoded)
    model = Model(inputs=inputs, outputs=[decoded, category])
    model.compile(
        optimizer="adam",
        loss=["mse", "cosine_similarity"],
        loss_weights=[0.001, 0.5],
        metrics={"m_rna": ["mae", "mse"], "category": "acc"},
    )
    return model


def load_metadata(h5_file):
    with h5py.File(h5_file, "r") as h5:
        x_shape = h5["feature"].shape
        labels = h5["label"][:].squeeze().astype(np.int64)
        genes = h5["gene_name"][:]
        samples = h5["sample"][:]
    return x_shape, labels, genes, samples


def evaluate_split(h5_file, labels, train_idx, test_idx, repeat_id, output_dir):
    seed = opt.seed + repeat_id
    set_seed(seed)

    with h5py.File(h5_file, "r") as h5:
        num_features = h5["feature"].shape[1]

    num_classes = len(np.unique(labels)) if opt.use_all else 2
    model = create_model(num_features, num_classes)

    repeat_dir = os.path.join(output_dir, f"repeat_{repeat_id:03d}")
    os.makedirs(repeat_dir, exist_ok=True)
    checkpoint_path = os.path.join(repeat_dir, f"{opt.dataset_name}_model.weights.h5")

    batch_size = min(opt.batch_size, max(1, len(train_idx)))
    steps_per_epoch = max(1, int(np.ceil(len(train_idx) / batch_size)))
    train_gen = data_generator(h5_file, train_idx, batch_size, num_classes, is_training=True)

    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=opt.max_epoch,
        verbose=2,
    )
    model.save_weights(checkpoint_path)

    sorted_idx = np.argsort(test_idx)
    sorted_test_idx = np.array(test_idx)[sorted_idx]
    with h5py.File(h5_file, "r") as h5:
        x_test = h5["feature"][sorted_test_idx].astype(np.float32)
    unsort_idx = np.argsort(sorted_idx)
    x_test = x_test[unsort_idx]
    y_test = labels[np.array(test_idx)]

    recon, probs = model.predict(x_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    positive_score = probs[:, 1]
    auc = roc_auc_score(y_test, positive_score)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    recon_mae = float(np.mean(np.abs(x_test - recon)))

    np.savetxt(os.path.join(repeat_dir, "label.csv"), y_test, delimiter=",", fmt="%d")
    np.savetxt(os.path.join(repeat_dir, "pred_label.csv"), y_pred, delimiter=",", fmt="%d")
    np.savetxt(os.path.join(repeat_dir, "pred_probability_class1.csv"), positive_score, delimiter=",")

    return {
        "repeat": repeat_id,
        "seed": seed,
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "auc": float(auc),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "reconstruction_mae": recon_mae,
        "weights_file": checkpoint_path,
    }


def write_summary(results, output_dir, title, split_line, sentence_prefix):
    results_csv = os.path.join(output_dir, f"{opt.dataset_name}_results.csv")
    summary_txt = os.path.join(output_dir, f"{opt.dataset_name}_summary.txt")
    results.to_csv(results_csv, index=False)

    auc_mean = results["auc"].mean()
    auc_std = results["auc"].std(ddof=1) if len(results) > 1 else 0.0
    summary = (
        f"{title}\n"
        f"Dataset: {opt.dataset_name}\n"
        f"{split_line}\n"
        f"Epochs per fold/split: {opt.max_epoch}\n"
        f"Mean AUC: {auc_mean:.6f} +/- {auc_std:.6f}\n\n"
        "Per-run AUCs:\n"
        + "\n".join(f"run {int(r.repeat):02d}: {r.auc:.6f}" for r in results.itertuples())
        + "\n\nManuscript sentence:\n"
        f"{sentence_prefix} Across these runs, the model achieved a mean AUC of "
        f"{auc_mean:.3f} +/- {auc_std:.3f} for {opt.serum_label}.\n"
    )
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n" + summary)
    print(f"Wrote results: {results_csv}")
    print(f"Wrote summary: {summary_txt}")


def run_cv():
    h5_file = opt.h5_file
    if not os.path.exists(h5_file):
        raise FileNotFoundError(
            f"HDF5 file not found: {h5_file}\n"
            "First run: python make_pc3high_h5.py --csv_file feature_PC3HighHigh.csv"
        )

    os.makedirs(opt.output_dir, exist_ok=True)
    x_shape, labels, genes, samples = load_metadata(h5_file)
    unique_labels = np.unique(labels)
    if set(unique_labels.tolist()) != {0, 1} and not opt.use_all:
        raise ValueError(
            f"This repeated AUC script expects binary labels 0/1. Found labels: {unique_labels.tolist()}."
        )

    print(f"{opt.dataset_name} H5: {h5_file}")
    print(f"Samples: {x_shape[0]}")
    print(f"Features: {x_shape[1]}")

    rows = []
    all_indices = np.arange(len(labels))
    if opt.cv_method == "kfold":
        print(f"Stratified {opt.folds}-fold CV")
        splitter = StratifiedKFold(n_splits=opt.folds, shuffle=True, random_state=opt.seed)
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(all_indices, labels), start=1):
            if fold_id < opt.start_fold:
                print(f"\n=== Fold {fold_id}/{opt.folds} already completed; skipping ===")
                continue
            print(f"\n=== Fold {fold_id}/{opt.folds} ===")
            rows.append(evaluate_split(h5_file, labels, train_idx, test_idx, fold_id, opt.output_dir))

        results = pd.DataFrame(rows)
        write_summary(
            results,
            opt.output_dir,
            f"Stratified k-fold cross-validation for {opt.serum_label}",
            f"Folds: {opt.folds}",
            f"Rather than relying on a single random 70/30 split, we performed {opt.folds}-fold cross-validation for the {opt.serum_label} within-dataset analysis.",
        )
    else:
        print(f"Repeated 70/30 splits: {opt.repeats}")
        splitter = StratifiedShuffleSplit(
            n_splits=opt.repeats,
            test_size=opt.test_fraction,
            random_state=opt.seed,
        )
        for repeat_id, (train_idx, test_idx) in enumerate(splitter.split(all_indices, labels), start=1):
            if repeat_id < opt.start_repeat:
                print(f"\n=== Repeat {repeat_id}/{opt.repeats} already completed; skipping ===")
                continue
            print(f"\n=== Repeat {repeat_id}/{opt.repeats} ===")
            rows.append(evaluate_split(h5_file, labels, train_idx, test_idx, repeat_id, opt.output_dir))

        results = pd.DataFrame(rows)
        write_summary(
            results,
            opt.output_dir,
            f"Repeated-split validation for {opt.serum_label}",
            f"Repeats: {opt.repeats}; split: {int((1 - opt.test_fraction) * 100)}/{int(opt.test_fraction * 100)} train/test",
            f"Rather than relying on a single random 70/30 split, we repeated the split {opt.repeats} times for the {opt.serum_label} within-dataset analysis.",
        )


if __name__ == "__main__":
    run_cv()
