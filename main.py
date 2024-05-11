import os
from project_constants import DATASET_FILE, DATASET_FOLDER, IMAGES_MODEL_FOLDER,\
                        IMAGES_SIMULATION_FOLDER, MODEL_FOLDER, PLOTS_FOLDER,\
                        SIM_CONFIGURATIONS, SIM_STATS, STATS_FOLDER, \
                        IMAGES_COMPARED_FOLDER, DATASET_GENERATED_FILE
from cfd_simulation.simulation import create_dataset
from cfd_model.cfd_model import run_model_training, get_model, generate_sequences
from image_functions import generate_simulation_images, generate_model_images,\
                            generate_compared_images

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
    _init_folder(IMAGES_SIMULATION_FOLDER)
    _init_folder(IMAGES_MODEL_FOLDER)
    _init_folder(IMAGES_COMPARED_FOLDER)


if __name__ == "__main__":
    
    model = None

    init_folders()

    create_dataset(DATASET_FILE, SIM_CONFIGURATIONS, SIM_STATS)

    model = run_model_training(DATASET_FILE, MODEL_FOLDER, PLOTS_FOLDER)
    if model is None:
        model_path = os.path.join(MODEL_FOLDER, "model_20240510_1228.h5")
        model = get_model(model_path)
    
    generate_sequences(model, DATASET_FILE, DATASET_GENERATED_FILE)

    generate_simulation_images()
    generate_model_images()
    generate_compared_images()
