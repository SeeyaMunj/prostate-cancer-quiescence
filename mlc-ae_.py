import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
import tensorflow.keras.utils as np_utils
import h5py
import os
import argparse
import shap
import pandas as pd
from matplotlib import colors as plt_colors
import matplotlib.pyplot as plt
import sklearn.metrics as sk  # Ensure sklearn.metrics is imported

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test', choices=['train', 'test'], help='train or test')
parser.add_argument('--h5_file', default=r'C:\Users\User\Documents\PC3_high_table1_intersection_genes_table1_samples.h5')
parser.add_argument('--gene_name_file', default='', help='optional gene-name CSV; if empty, gene_name is read from H5')
parser.add_argument('--output_dir', default=r'C:\Users\User\Documents\mlc_ae_train_shap_corrected_run')
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 10]')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--test_size', type=int, default=2712, help='number of held-out test cells')
parser.add_argument('--seed', type=int, default=4, help='random seed')
parser.add_argument('--use_all', action='store_true', help='use all classes instead of binary labels')
parser.add_argument('--use_argmax', action='store_true', help='used when testing')
parser.add_argument('--no_shap', action='store_true', help='skip SHAP during test')
opt = parser.parse_args()
opt.use_shap = not opt.no_shap
MODEL_DIR = opt.output_dir
os.makedirs(MODEL_DIR, exist_ok=True)

# Check if TensorFlow is using GPU or CPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is using the GPU")
else:
    print("TensorFlow is using the CPU")

def visualize_reconstruction(original_data, reconstructed_data):
    num_samples = min(5, len(original_data), len(reconstructed_data))
    for i in range(num_samples):
        plt.figure(figsize=(10, 5))
        plt.plot(original_data[i], label='Original Data', color='blue')
        plt.plot(reconstructed_data[i], label='Reconstructed Data', color='red')
        plt.title(f'Sample {i+1} Reconstruction')
        plt.xlabel('Gene Expression Features')
        plt.ylabel('Expression Levels')
        plt.legend()
        plt.savefig(os.path.join(MODEL_DIR, f'reconstruction_sample_{i+1}.png'))
        plt.close()

def calculate_reconstruction_error(original_data, reconstructed_data):
    errors = []
    for i in range(len(original_data)):
        error = np.mean(np.abs(original_data[i] - reconstructed_data[i]))
        errors.append(error)
    return np.mean(errors)

def load_feature_rows(h5_file_path, dataset_name, indices):
    """Load H5 rows by arbitrary indices while preserving the requested order."""
    sorted_idx = np.argsort(indices)
    sorted_indices = indices[sorted_idx]
    with h5py.File(h5_file_path, 'r') as f:
        feature_data = f[dataset_name][sorted_indices]
    unsorted_idx = np.argsort(sorted_idx)
    return feature_data[unsorted_idx].astype(np.float32)

def decode_gene_names(gene_values):
    decoded = []
    for gene_value in gene_values:
        if isinstance(gene_value, bytes):
            decoded.append(gene_value.decode('utf-8'))
        else:
            decoded.append(str(gene_value))
    return np.asarray(decoded)

def data_generator(h5_file_path, indices, batch_size, num_classes, is_training=True):
    with h5py.File(h5_file_path, 'r') as f:
        m_rna_dataset_name = 'feature'
        label_dataset_name = 'label'

        m_rna_dataset = f[m_rna_dataset_name]
        label_dataset = f[label_dataset_name]
        num_samples = len(indices)
        while True:
            if is_training:
                np.random.shuffle(indices)
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                # Sort batch_indices to ensure increasing order
                sorted_indices = np.sort(batch_indices)
                # Load data using sorted indices
                batch_m_rna = m_rna_dataset[sorted_indices]
                batch_label = label_dataset[sorted_indices].squeeze()
                # Shuffle within the batch to maintain randomness
                if is_training:
                    perm = np.random.permutation(len(batch_m_rna))
                    batch_m_rna = batch_m_rna[perm]
                    batch_label = batch_label[perm]
                batch_m_rna = batch_m_rna.astype(np.float32)
                batch_categorical_label = np_utils.to_categorical(batch_label, num_classes=num_classes)
                yield batch_m_rna, [batch_m_rna, batch_categorical_label]

def mlc_ae(training=False):
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    load_file = opt.h5_file
    print('HELLO ALL')

    m_rna_dataset_name = 'feature'
    label_dataset_name = 'label'
    gene_name_dataset_name = 'gene_name'
    sample_id_dataset_name = 'sample'

    # Open HDF5 file and get dataset shapes
    with h5py.File(load_file, 'r') as f:
        # List available datasets
        print("Datasets available in the HDF5 file:")
        for name in f:
            print(name)

        # Map 'feature' dataset to 'm_rna' variable in code
        m_rna_dataset = f[m_rna_dataset_name]
        label_dataset = f[label_dataset_name][:]
        gene = f[gene_name_dataset_name][:]
        sample_id = f[sample_id_dataset_name][:]
        label = label_dataset.squeeze()
        gene = gene.squeeze()
        sample_id = sample_id.squeeze()
        num_samples = m_rna_dataset.shape[0]
        num_features = m_rna_dataset.shape[1]

    print('Number of samples:', num_samples)
    print('Number of features:', num_features)
    print('Label shape:', label.shape)

    """First random: train and test sets"""
    np.random.seed(opt.seed)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Shuffle sample_id and gene arrays
    sample_id = sample_id[indices]
    gene = gene  # Assuming gene names are the same for all samples

    # Split indices into train and test
    test_size = opt.test_size
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]

    # Get labels for train and test sets
    label_train = label[train_indices]
    label_test = label[test_indices]

    # PR sample operations
    pr_idx_train = np.array([i for i, e in enumerate(label_train) if e == 1 or e == 0])
    pr_idx_test = np.array([i for i, e in enumerate(label_test) if e == 1 or e == 0])
    print('Healthy samples in training set:', sum([1 for x in label_train if x == 0]))
    print('Healthy samples in testing set:', sum([1 for x in label_test if x == 0]))

    pr_train_indices = train_indices[pr_idx_train]
    pr_test_indices = test_indices[pr_idx_test]

    print('PR train and test size:', pr_train_indices.shape, pr_test_indices.shape)
    print("Data loading has just been finished")

    def create_model():
        inputs = Input(shape=(num_features,), name="inputs")
        inputs_4 = BatchNormalization(name="inputs_4")(inputs)
        encoded = Dense(units=64, activation='relu', name='encoded')(inputs_4)
        inputs_5 = Dense(512, activation="relu", name="inputs_5")(encoded)
        decoded_tcga = Dense(units=num_features, activation='linear', name="m_rna")(inputs_5)
        num_classes = 6 if opt.use_all else 2
        cl_0 = Dense(units=num_classes, activation="softmax", name="category")(encoded)
        m = Model(inputs=inputs, outputs=[decoded_tcga, cl_0])
        m.compile(optimizer='adam',
                  loss=["mse", "cosine_similarity"],
                  loss_weights=[0.001, 0.5],
                  metrics={"m_rna": ["mae", "mse"], "category": "acc"})
        return m

    model = create_model()
    checkpoint_path = os.path.join(MODEL_DIR, 'my_model.h5')
    print("Checkpoint path:")
    print(checkpoint_path)

    batch_size = opt.batch_size
    num_classes = 6 if opt.use_all else 2

    if training:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=MODEL_DIR)
        if opt.use_all:
            print('opt.use_all')
            steps_per_epoch = len(train_indices) // batch_size
            validation_steps = len(test_indices) // batch_size
            train_gen = data_generator(load_file, train_indices, batch_size, num_classes, is_training=True)
            val_gen = data_generator(load_file, test_indices, batch_size, num_classes, is_training=False)
            model.fit(train_gen,
                      steps_per_epoch=steps_per_epoch,
                      epochs=opt.max_epoch,
                      callbacks=[tensorboard_callback],
                      validation_data=val_gen,
                      validation_steps=validation_steps,
                      verbose=2)
        else:
            print('NOT opt.use_all')
            steps_per_epoch = len(pr_train_indices) // batch_size
            train_gen = data_generator(load_file, pr_train_indices, batch_size, num_classes, is_training=True)
            model.fit(train_gen,
                      steps_per_epoch=steps_per_epoch,
                      epochs=opt.max_epoch,
                      callbacks=[tensorboard_callback],
                      verbose=2)
        model.save_weights(filepath=checkpoint_path)
        print("Training has just been finished")

    else:
        model.load_weights(checkpoint_path)
        if not opt.use_all:
            test_indices = pr_test_indices
            label_test = label[test_indices]
        else:
            label_test = label[test_indices]

        m_rna_test = load_feature_rows(load_file, m_rna_dataset_name, test_indices)
        label_test = label[test_indices]

        num_classes = 6 if opt.use_all else 2
        categorical_label_test = np_utils.to_categorical(label_test, num_classes=num_classes)

        data_pred = model.predict(m_rna_test, batch_size=batch_size, verbose=2)

        # Visualization of reconstructed output
        visualize_reconstruction(m_rna_test, data_pred[0])

        # Calculating reconstruction error
        reconstruction_error = calculate_reconstruction_error(m_rna_test, data_pred[0])
        print("Mean Reconstruction Error:", reconstruction_error)

        """ Argmax """
        y_pred = np.argmax(data_pred[1], axis=1)
        y_gt = label_test
        if opt.use_all:
            confusion_0 = sk.confusion_matrix(y_gt, y_pred, labels=[0, 1, 2, 3, 4, 5])
        else:
            confusion_0 = sk.confusion_matrix(y_gt, y_pred, labels=[0, 1])
        print("Confusion Matrix:\n", confusion_0)
        balanced_acc = sk.balanced_accuracy_score(y_gt, y_pred)
        acc = sk.accuracy_score(y_gt, y_pred)

        """ Logits for ROC """
        y_logit = data_pred[1][:, 0]
        x_logit = categorical_label_test[:, 0]

        """ Save for records """
        np.savetxt(X=m_rna_test, fname=MODEL_DIR + "/test_gene.csv", delimiter=",", fmt='%1.3f')
        np.savetxt(X=label_test, fname=MODEL_DIR + "/label.csv", delimiter=",", fmt='%1.3f')
        np.savetxt(X=data_pred[0], fname=MODEL_DIR + "/pred_gene.csv", delimiter=",", fmt='%1.3f')
        np.savetxt(X=y_pred, fname=MODEL_DIR + "/pred_label.csv", delimiter=",", fmt='%1.3f')

        """ Get latent representation """
        layer_name = "encoded"
        encoded_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        encoded_output = encoded_layer_model.predict(m_rna_test)
        np.savetxt(X=encoded_output, fname=MODEL_DIR + "/latent_feat.csv", delimiter=",")

        """ Log """
        log1 = open(os.path.join(MODEL_DIR, 'log_blacc.txt'), 'a')
        log2 = open(os.path.join(MODEL_DIR, 'log_roc.txt'), 'a')
        log3 = open(os.path.join(MODEL_DIR, 'log_acc.txt'), 'a')

        if not opt.use_all:
            print('NOT opt.use_all')
            """Only execute when using PR data (two labels)"""
            auc = sk.roc_auc_score(x_logit, y_logit)
            fpr, tpr, thresh = sk.roc_curve(x_logit, y_logit)
            # Plot ROC curve
            print("fpr, tpr, thresh", fpr, tpr, thresh)

            df1 = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresh})

            # Save DataFrame to Excel
            df1.to_excel(os.path.join(MODEL_DIR, 'roc_data_curve.xlsx'), index=False)

            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()

            # Save the plot as a file (e.g., PNG)
            plt.savefig(os.path.join(MODEL_DIR, 'roc_curve.png'))
            plt.close()
            roc_feat = {'auc': auc, 'fpr': [], 'tpr': []}
            for e in fpr:
                roc_feat['fpr'].append(str(e))
            for e in tpr:
                roc_feat['tpr'].append(str(e))

        if opt.use_shap:
            """ SHAP Explainer: computed from the training split only."""
            print("Processing SHAP using training data only...")
            print('NOT opt.use_all')

            shap_indices = train_indices if opt.use_all else pr_train_indices
            input_feat = load_feature_rows(load_file, m_rna_dataset_name, shap_indices)
            print("SHAP INPUT FEAT", input_feat.shape)

            model.load_weights(checkpoint_path)
            layer_name = "category"
            encoded_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
            e = shap.GradientExplainer(encoded_layer_model, input_feat)
            shap_values = e.shap_values(input_feat)
            print('shap_values', len(shap_values) if isinstance(shap_values, list) else np.asarray(shap_values).shape)

            if opt.gene_name_file and os.path.exists(opt.gene_name_file):
                feat_name = np.loadtxt(opt.gene_name_file, dtype=str, delimiter=",")
            else:
                feat_name = decode_gene_names(gene)
            feat_name = np.asarray(feat_name).squeeze()
            print('feat_name', feat_name.shape)

            class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
            print('class_inds', class_inds.shape)
            print('class_inds', class_inds)
            colors = np.array(['yellowgreen', 'palevioletred'])[class_inds]
            cmap = plt_colors.ListedColormap(colors)
            shap.summary_plot(shap_values, input_feat, feature_names=feat_name, max_display=40,
                              plot_size=(12.0, 16.0, 2.0), plot_type='bar',
                              color=cmap, show=False, sort=True,
                              class_names=['G0', 'not G0'])
            shap_plot_path = os.path.join(MODEL_DIR, 'shap_summary_training_only.png')
            plt.savefig(shap_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            print("Saved SHAP plot:", shap_plot_path)

            for class_idx, class_values in enumerate(shap_values):
                mean_abs_shap = np.abs(class_values).mean(axis=0)
                ranked_idx = np.argsort(mean_abs_shap)[::-1]
                shap_ranked = pd.DataFrame({
                    'rank': np.arange(1, len(ranked_idx) + 1),
                    'gene': feat_name[ranked_idx],
                    'mean_abs_shap': mean_abs_shap[ranked_idx],
                    'class_index': class_idx
                })
                shap_csv_path = os.path.join(MODEL_DIR, f'shap_ranked_genes_training_only_class_{class_idx}.csv')
                shap_ranked.to_csv(shap_csv_path, index=False)
                print("Saved SHAP ranked genes:", shap_csv_path)

        def log_string(out1, out2, out3):
            log1.write(str(out1))
            log1.write('\n')
            log1.flush()
            print(out1)
            if out2:
                log2.write(str(out2['auc']))
                log2.write('\n')
                roc_x, roc_y = ' '.join(out2['fpr']), ' '.join(out2['tpr'])
                log2.write(roc_x)
                log2.write('\n')
                log2.write(roc_y)
                log2.write('\n')
                log2.flush()
                print(out2)
            if out3:
                log3.write(str(out3))
                log3.write('\n')
                log3.flush()

        if hasattr(opt, 'use_argmax') and opt.use_argmax:
            if opt.use_all:
                confusion_1 = ' '.join(list(np.reshape(confusion_0.astype(str), -1)))
                log_string(confusion_1, None, None)
            else:
                confusion_1 = ' '.join(list(np.reshape(confusion_0.astype(str), -1)))
                log_string(confusion_1, roc_feat, acc)
        else:
            log_string(balanced_acc, None, acc)

if __name__ == '__main__':
    if opt.phase == 'train':
        if not os.path.exists(os.path.join(MODEL_DIR, 'code/')):
            os.makedirs(os.path.join(MODEL_DIR, 'code/'))
            # Copy current script to backup folder
            import shutil
            shutil.copy(__file__, os.path.join(MODEL_DIR, 'code/'))
        mlc_ae(True)
    elif opt.phase == 'test':
        mlc_ae(False)
