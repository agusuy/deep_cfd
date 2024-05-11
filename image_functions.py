import os
import shutil
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from project_constants import DATASET_FILE, IMAGES_SIMULATION_FOLDER, \
                            IMAGES_MODEL_FOLDER, NUM_SEQUENCES, LENGHT_SEQUENCE,\
                            WINDOW
from cfd_model.cfd_dataset import get_dataset
from cfd_model.cfd_model import generate_sequences


def _get_color_map():
    color_map = plt.cm.jet
    color_map.set_bad(color="black")
    return color_map

def _preprocess_frame(frame, min_val=0.0, max_max=1.0):
    frame[frame<min_val] = np.nan
    frame[frame>max_max] = np.nan
    frame[frame==-1] = np.nan

def _plot_frame(frame, sequence_id, frame_id, images_folder_path, title="", save=True):
    color_map = _get_color_map()

    plt.clf()
    if title:
        plt.title(title)
    plt.imshow(np.squeeze(frame).transpose(), cmap=color_map)
    plt.axis("off")
    plt.colorbar(label="Velocity", orientation="horizontal", 
                 format=ticker.FuncFormatter(lambda x, pos: ''), 
                 ticks=ticker.FixedLocator([])
             )
    plt.tight_layout()
    
    if save:
        sequence_folder_name = "sequence_{}".format(sequence_id)
        sequence_folder_path = os.path.join(images_folder_path, sequence_folder_name)
        image_file_name = "frame_{0:04d}.png".format(frame_id)
        file_path = os.path.join(sequence_folder_path, image_file_name)
        plt.savefig(file_path)

def generate_sequences_images(sequences, images_folder_path, label=""):
    # delete previous folder
    shutil.rmtree(images_folder_path)
    os.makedirs(images_folder_path)

    # Create folders
    for i in range(NUM_SEQUENCES):
        sequence_folder_name = "sequence_{}".format(i)
        sequence_folder_path = os.path.join(images_folder_path, sequence_folder_name)
        os.makedirs(sequence_folder_path)
 
    # Generate sequence images
    for seq_id in range(NUM_SEQUENCES):
        sequence = sequences[seq_id,:] # select sequence from dataset
        
        title = "Sequence {} ".format(seq_id) + label

        for frame_id in range(LENGHT_SEQUENCE):
            frame = sequence[frame_id]
            _preprocess_frame(frame)
            _plot_frame(frame, seq_id, frame_id, images_folder_path, title)

def generate_simulation_images():
    sequences, _, _, _, _ = get_dataset(DATASET_FILE, split=False)
    generate_sequences_images(sequences, IMAGES_SIMULATION_FOLDER, "(simulation)")

def generate_model_images(model):    
    original_sequences, _, _, _, _ = get_dataset(DATASET_FILE, split=False)

    # TODO: Refactor. Save generated sequence on a file before doing this?
    sequences = generate_sequences(model, original_sequences, WINDOW)
    generate_sequences_images(sequences, IMAGES_MODEL_FOLDER, "(model)")
