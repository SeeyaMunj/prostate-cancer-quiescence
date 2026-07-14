import argparse
import os
import random

import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
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
        feature_ds = h5["feature"]
        label_ds = h5["label"]
        while True:
            if is_training:
                np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                sorted_indices = np.sort(batch_indices)
                x = feature_ds[sorted_indices].astype(np.float32)
                y = label_ds[sorted_indices].squeeze().astype(np.int64)
                if is_training:
                    perm = np.random.permutation(len(x))
                    x = x[perm]
                    y = y[perm]
                yield x, (x, to_categorical(y, num_classes=num_classes))


def create_model(num_features, num_classes=2):
    # Original MLC-AE architecture.
    inputs = Input(shape=(num_features,), name="inputs")
    inputs_4 = BatchNormalization(name="inputs_4")(inputs)
    encoded = Dense(units=64, activation="relu", name="encoded")(inputs_4)
    inputs_5 = Dense(512, activation="relu", name="inputs_5")(encoded)
    decoded = Dense(units=num_features, activation="linear", name="m_rna")(inputs_5)
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
        labels = h5["label"][:].squeeze().astype(np.int64)
        num_samples, num_features = h5["feature"].shape
    return labels, num_samples, num_features


def train_and_test(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    labels, num_samples, num_features = load_metadata(args.h5_file)
    num_classes = 2

    print(f"H5 file: {args.h5_file}")
    print(f"Samples: {num_samples}")
    print(f"Genes/features: {num_features}")
    print(f"Labels: {dict(zip(*np.unique(labels, return_counts=True)))}")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_fraction, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(np.arange(num_samples), labels))

    model = create_model(num_features, num_classes=num_classes)
    checkpoint_path = os.path.join(args.output_dir, "mlc_ae_original_model.weights.h5")

    batch_size = min(args.batch_size, len(train_idx))
    steps_per_epoch = max(1, int(np.ceil(len(train_idx) / batch_size)))
    train_gen = data_generator(args.h5_file, train_idx, batch_size, num_classes, is_training=True)

    if args.phase in {"train", "train_test"}:
        model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=args.max_epoch,
            verbose=2,
        )
        model.save_weights(checkpoint_path)
        print(f"Saved weights: {checkpoint_path}")

    if args.phase in {"test", "train_test"}:
        if args.phase == "test":
            model.load_weights(checkpoint_path)

        sorted_order = np.argsort(test_idx)
        sorted_test_idx = test_idx[sorted_order]
        with h5py.File(args.h5_file, "r") as h5:
            x_test = h5["feature"][sorted_test_idx].astype(np.float32)
        unsort_order = np.argsort(sorted_order)
        x_test = x_test[unsort_order]
        y_test = labels[test_idx]

        recon, probs = model.predict(x_test, batch_size=batch_size, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        positive_score = probs[:, 1]

        auc = roc_auc_score(y_test, positive_score)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fpr, tpr, thresholds = roc_curve(y_test, positive_score)
        recon_mae = float(np.mean(np.abs(x_test - recon)))

        np.savetxt(os.path.join(args.output_dir, "label.csv"), y_test, delimiter=",", fmt="%d")
        np.savetxt(os.path.join(args.output_dir, "pred_label.csv"), y_pred, delimiter=",", fmt="%d")
        np.savetxt(os.path.join(args.output_dir, "pred_probability_class1.csv"), positive_score, delimiter=",")
        np.savetxt(os.path.join(args.output_dir, "roc_curve.csv"), np.column_stack([fpr, tpr, thresholds]), delimiter=",")

        summary = (
            f"AUC: {auc:.6f}\n"
            f"Accuracy: {acc:.6f}\n"
            f"Balanced accuracy: {bal_acc:.6f}\n"
            f"Reconstruction MAE: {recon_mae:.6f}\n"
            f"Confusion matrix [labels 0,1]:\n{cm}\n"
        )
        with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary)
        print(summary)


def main():
    parser = argparse.ArgumentParser(description="Original MLC-AE architecture runner for H5 gene-expression files.")
    parser.add_argument("--h5_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--phase", default="train_test", choices=["train", "test", "train_test"])
    parser.add_argument("--max_epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()
    train_and_test(args)


if __name__ == "__main__":
    main()
