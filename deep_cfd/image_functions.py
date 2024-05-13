import os
import shutil
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from project_constants import DATASET_FILE, IMAGES_SIMULATION_FOLDER, \
                            IMAGES_MODEL_FOLDER, NUM_SEQUENCES, LENGHT_SEQUENCE,\
                            WINDOW, IMAGES_COMPARED_FOLDER, DATASET_GENERATED_FILE
from cfd_model.cfd_dataset import get_dataset
from cfd_model.cfd_model import get_generated_dataset


def _get_color_map():
    color_map = plt.cm.jet
    color_map.set_bad(color="black")
    return color_map

def preprocess_frame(frame, min_val=0.0, max_max=1.0):
    frame[frame<min_val] = np.nan
    frame[frame>max_max] = np.nan
    frame[frame==-1] = np.nan

def plot_frame(frame, sequence_id=None, frame_id=None, title="", images_folder_path=None):
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
    
    if images_folder_path is not None:
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
            preprocess_frame(frame)
            plot_frame(frame, seq_id, frame_id, title, images_folder_path)

def generate_simulation_images():
    sequences, _, _, _, _ = get_dataset(DATASET_FILE, split=False)
    generate_sequences_images(sequences, IMAGES_SIMULATION_FOLDER, "(simulation)")

def generate_model_images():    
    sequences = get_generated_dataset(DATASET_GENERATED_FILE)
    generate_sequences_images(sequences, IMAGES_MODEL_FOLDER, "(model)")

def plot_compare_frames(frame1, frame2, title1, title2, 
                         sequence_id, frame_id, images_folder_path=None, 
                         orientation="v"):
    
    color_map = _get_color_map()
    plt.clf()

    if orientation == "v":
        _, axes = plt.subplots(1, 2, figsize=(10, 20))
    elif orientation == "h":
        _, axes = plt.subplots(2, 1, figsize=(20, 10))

    axes[0].imshow(np.squeeze(frame1).transpose(), cmap=color_map)
    axes[0].title.set_text(title1)
    axes[0].axis("off")

    axes[1].imshow(np.squeeze(frame2).transpose(), cmap=color_map)
    axes[1].title.set_text(title2)
    axes[1].axis("off")

    plt.tight_layout()

    if images_folder_path is not None:
        sequence_folder_name = "sequence_{}".format(sequence_id)
        sequence_folder_path = os.path.join(images_folder_path, sequence_folder_name)
        image_file_name = "frame_{0:04d}.png".format(frame_id)
        file_path = os.path.join(sequence_folder_path, image_file_name)
        plt.savefig(file_path)

def generate_compared_images():

    # delete previous folder
    shutil.rmtree(IMAGES_COMPARED_FOLDER)
    os.makedirs(IMAGES_COMPARED_FOLDER)

    # Create folders
    for i in range(NUM_SEQUENCES):
        sequence_folder_name = "sequence_{}".format(i)
        sequence_folder_path = os.path.join(IMAGES_COMPARED_FOLDER, sequence_folder_name)
        os.makedirs(sequence_folder_path)

    # Get data
    original_sequences, _, _, _, _ = get_dataset(DATASET_FILE, split=False)
    generate_sequence = get_generated_dataset(DATASET_GENERATED_FILE)

    # Generate images
    for sequence_id, (original, generated) in enumerate(zip(original_sequences, generate_sequence)):
        
        for frame_id, (original_frame, generated_frame) in enumerate(zip(original, generated)):
            preprocess_frame(original_frame)
            preprocess_frame(generated_frame)

            original_title = "Sequence {} (simulation)".format(sequence_id)
            generated_title = "Sequence {} (model)".format(sequence_id)

            plot_compare_frames(original_frame, generated_frame, 
                                 original_title, generated_title, 
                                 sequence_id, frame_id, IMAGES_COMPARED_FOLDER, 
                                 orientation="h")

