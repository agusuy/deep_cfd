import os
from project_constants import DATASET_FILE, DATASET_FOLDER, MODEL_FOLDER, PLOTS_FOLDER, SIM_CONFIGURATIONS, SIM_STATS, STATS_FOLDER
from cfd_simulation.simulation import create_dataset
from cfd_model.cfd_model import run_model_training


def _init_folder(path):
    try:
        # create folder to store results
        os.makedirs(path)
    except OSError:
        # folder already exists
        pass

def init_folders():
    _init_folder(DATASET_FOLDER)
    _init_folder(MODEL_FOLDER)
    _init_folder(PLOTS_FOLDER)
    _init_folder(STATS_FOLDER)
    

if __name__ == "__main__":
    
    init_folders()

    create_dataset(DATASET_FILE, SIM_CONFIGURATIONS, SIM_STATS)

    run_model_training(DATASET_FILE, MODEL_FOLDER, PLOTS_FOLDER)

    
