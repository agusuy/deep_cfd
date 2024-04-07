import os
from cfd_dataset import get_dataset
from cfd_model import create_model, training

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
DATASET_FOLDER = os.path.join(ROOT, "dataset")
DATASET_FILE = os.path.join(DATASET_FOLDER, "dataset_2024_04_06.npy")

dataset, X_train, y_train, X_val, y_val =  get_dataset(DATASET_FILE)

input_dimensions = X_train[0].shape
model = create_model(input_dimensions)

training_history, model = training(model, X_train, y_train, X_val, y_val)

