import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

from cfd_simulation.constants import DATASET_FOLDER, IMAGES_FOLDER, DATASET_FILE, LENGHT_SEQUENCE, NUM_SEQUENCES
from cfd_model.cfd_model import generate_sequences
from cfd_model.cfd_dataset import get_dataset

def _get_color_map():
    color_map = plt.cm.jet
    color_map.set_bad(color="black")
    return color_map

def _preprocess_frame(frame, min_val=0.0, max_max=1.0):
    frame[frame<min_val] = np.nan
    frame[frame>max_max] = np.nan

def _plot_frame(frame, sequence_id, frame_id, images_folder_path, save=True):
    plt.clf()

    color_map = _get_color_map()
    
    color_map.set_bad(color='black')
    frame[frame==-1] = np.nan
    
    plt.imshow(np.squeeze(frame).transpose(), cmap=color_map)
    plt.axis("off")
    plt.colorbar(label="Velocity", orientation="horizontal")
    if save:
        sequence_folder_path = images_folder_path + "/sequence_{}/".format(sequence_id)
        file_path = sequence_folder_path + "frame_{0:04d}.png".format(frame_id)
        plt.savefig(file_path)

def generate_sequences_images(sequences, images_folder_path):
    try:
        # create folder to store results
        os.makedirs(images_folder_path)
    except OSError:
        # folder already exists
        # delete previous folder
        shutil.rmtree(images_folder_path)
        os.makedirs(images_folder_path)
    for i in range(NUM_SEQUENCES):
        sequence_folder_path = images_folder_path + "/sequence_{}/".format(i)
        os.makedirs(sequence_folder_path)
 
    # Generate sequence images
    for seq in range(NUM_SEQUENCES):
        sequence = sequences[seq,:] # select sequence from dataset
        for frame_number in range(LENGHT_SEQUENCE):
            frame = sequence[frame_number]
            _preprocess_frame(frame)
            _plot_frame(frame, seq, frame_number, images_folder_path)

def generate_simulation_images():
    sequences = np.load(DATASET_FILE)
    generate_sequences_images(sequences, IMAGES_FOLDER)

def generate_model_images():
    # TODO: Refactor. Constants file
    WINDOW = 5
    DATASET_FILE = os.path.join(DATASET_FOLDER, "dataset_2024_04_25.npy")
    MODEL_FILE = "./cfd_model/models/model_20240426_0724.h5"
    
    original_sequences, _, _, _, _ = get_dataset(DATASET_FILE)

    # TODO: Refactor. Save generated sequence on a file before doing this?
    from keras.models import load_model
    model = load_model(MODEL_FILE)
    sequences = generate_sequences(model, original_sequences, WINDOW)

    # TODO: Refactor. Constants file
    GENERATED_IMAGES_FOLDER = "outputs/model/images/"

    generate_sequences_images(sequences, GENERATED_IMAGES_FOLDER)


if __name__ == "__main__":
    # generate_simulation_images()
    generate_model_images()

