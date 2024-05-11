import os
from cfd_simulation.simulation_constants import HEIGHT, WIDTH, NUM_SEQUENCES, LENGHT_SEQUENCE
from cfd_model.model_constants import WINDOW


###### Folders and Files ######

# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
PROJECT_FOLDER = os.path.dirname(__file__)
OUTPUTS_FOLDER = os.path.join(PROJECT_FOLDER, "outputs")

DATASET_FOLDER = os.path.join(OUTPUTS_FOLDER, "dataset")
MODEL_FOLDER = os.path.join(OUTPUTS_FOLDER, "model")
PLOTS_FOLDER = os.path.join(OUTPUTS_FOLDER, "plots")
STATS_FOLDER = os.path.join(OUTPUTS_FOLDER, "stats")
IMAGES_SIMULATION_FOLDER = os.path.join(OUTPUTS_FOLDER, "images_simulation")
IMAGES_MODEL_FOLDER = os.path.join(OUTPUTS_FOLDER, "images_model")
IMAGES_COMPARED_FOLDER = os.path.join(OUTPUTS_FOLDER, "images_compared")

SIM_CONFIGURATIONS = os.path.join(DATASET_FOLDER, "sim_conf.json")
DATASET_FILE = os.path.join(DATASET_FOLDER, "dataset.npy")
DATASET_GENERATED_FILE = os.path.join(DATASET_FOLDER, "dataset_generated.npy")
SIM_STATS = os.path.join(STATS_FOLDER, "sim_stats.csv")
MODEL_STATS = os.path.join(STATS_FOLDER, "model_stats.csv")
