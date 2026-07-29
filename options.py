import argparse
import os


BASE_DIR = os.getcwd()
DEBUG_INDEX = "PC3High_repeated_cv"
MODEL_DIR = os.path.join(BASE_DIR, "model", DEBUG_INDEX)
os.makedirs(MODEL_DIR, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--phase", default="cv", choices=["train", "test", "cv"])
parser.add_argument("--csv_file", default=r"C:\Users\User\Desktop\rna\High\feature_PC3HighHigh.csv")
parser.add_argument("--h5_file", default=r"C:\Users\User\Desktop\rna\High\PC3HighHigh.h5")
parser.add_argument("--output_dir", default=MODEL_DIR)
parser.add_argument("--dataset_name", default="PC3High")
parser.add_argument("--serum_label", default="PC3 high serum")
parser.add_argument("--max_epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--test_fraction", type=float, default=0.30)
parser.add_argument("--repeats", type=int, default=10)
parser.add_argument("--start_repeat", type=int, default=1)
parser.add_argument("--cv_method", default="repeated_split", choices=["repeated_split", "kfold"])
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--start_fold", type=int, default=1)
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--label_col", default=None)
parser.add_argument("--sample_col", default=None)
parser.add_argument("--use_all", action="store_true")
parser.add_argument("--use_shap", action="store_true")
parser.add_argument("--use_argmax", action="store_true")
opt = parser.parse_args()
