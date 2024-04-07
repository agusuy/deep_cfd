import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import json
from constants import DATASET_FILE, IMAGES_FOLDER, NUM_SEQUENCES, \
                        OUTPUTS_FOLDER, SIM_CONFIGURATIONS, WIDTH, HEIGHT
from sklearn.model_selection import ParameterSampler


def _select_parameters(amount, param_grid):
    return list(ParameterSampler(param_grid, n_iter=amount, random_state=42))

def _build_configuration(id, obstacle_type, obstacle_parameters):
    return {
        "id": id,
        "obstacle": {
            "type": obstacle_type,
            "parameters": obstacle_parameters
        }
    }

def create_configurations():
    configurations = []
    id = 0
    
    shapes_number = 2
    amount, extra = divmod(NUM_SEQUENCES, shapes_number)
    amount_circumference = amount
    amount_ellipses = amount + extra

    # amount_circumference = NUM_SEQUENCES
    # amount_ellipses = NUM_SEQUENCES

    # Circumference
    circumference_param_grid = {
        "center_x": range(WIDTH//4, WIDTH//2, 10),
        "center_y": range(HEIGHT//3, 2*HEIGHT//3, 10),
        "radius": range(HEIGHT//9, HEIGHT//5, 3)
    }
    circumference_parameters = _select_parameters(amount_circumference, 
                                                  circumference_param_grid)
    for parameters in circumference_parameters:
        configurations.append(
            _build_configuration(id, "circumference", parameters)
        )
        id+=1

    # Ellipse
    ellipses_param_grid = {
        "center_x": range(WIDTH//4, WIDTH//2, 10),
        "center_y": range(HEIGHT//3, 2*HEIGHT//3, 10),
        "semi_major_axis": range(HEIGHT//9, HEIGHT//3, 10),
        "semi_minor_axis": range(HEIGHT//18, HEIGHT//6, 10),
        "degrees": range(-30, 30, 10)
    }
    ellipses_parameters = _select_parameters(amount_ellipses, ellipses_param_grid)
    for parameters in ellipses_parameters:
        configurations.append(
            _build_configuration(id, "ellipse", parameters)
        )
        id+=1

    try:
        # create folder to store results
        os.makedirs(OUTPUTS_FOLDER)
    except OSError:
        # folder already exists
        pass

    with open(SIM_CONFIGURATIONS, 'w') as f:
        json.dump(configurations, f)

def _plot_frame(frame, sequence_id, frame_id, save=True):
    plt.clf()

    color_map = plt.cm.jet
    
    color_map.set_bad(color='black')
    frame[frame==-1] = np.nan
    
    plt.imshow(frame.transpose(), cmap=color_map)
    plt.axis("off")
    plt.colorbar(label="Velocity", orientation="horizontal")
    if save:
        folder_path = IMAGES_FOLDER + "/sequence_{}/".format(sequence_id)
        file_path = folder_path + "frame_{0:04d}.png".format(frame_id)
        plt.savefig(file_path)

def generate_images(num_sequences, length_sequence, dataset=None):
    try:
        # create folder to store results
        os.makedirs(IMAGES_FOLDER)
    except OSError:
        # folder already exists
        # delete previous folder
        shutil.rmtree(IMAGES_FOLDER)
        os.makedirs(IMAGES_FOLDER)
    for i in range(num_sequences):
        folder_path = IMAGES_FOLDER + "/sequence_{}/".format(i)
        os.makedirs(folder_path)

    if dataset is None:
        dataset = np.load(DATASET_FILE)

    # Generate sequence images
    for seq in range(num_sequences):
        sequence = dataset[seq,:] # select sequence from dataset
        for frame_number in range(length_sequence):
            frame = sequence[frame_number]
            _plot_frame(frame, seq, frame_number)
