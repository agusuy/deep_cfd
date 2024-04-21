import os
from cfd_dataset import get_dataset
from cfd_model import create_model, training, plot_training_history
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
DATASET_FOLDER = os.path.join(ROOT, "dataset")
DATASET_FILE = os.path.join(DATASET_FOLDER, "dataset_2024_04_06.npy")

PROJECT_FOLDER = os.path.dirname(__file__)
MODELS_FOLDER = os.path.join(PROJECT_FOLDER, "models")

dataset, X_train, y_train, X_val, y_val =  get_dataset(DATASET_FILE)

input_dimensions = X_train[0].shape
model = create_model(input_dimensions)

training_history, model = training(model, X_train, y_train, X_val, y_val)

model_name = MODELS_FOLDER + f"/model_{datetime.now():%Y%m%d_%H%M}"
model.save(model_name + ".h5", save_format="h5")

plot_training_history(training_history, model_name)
