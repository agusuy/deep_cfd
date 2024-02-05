import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import json
# from tqdm import tqdm
from math import ceil, floor
from constants import NUM_SEQUENCES, LENGHT_SEQUENCE, WIDTH, HEIGHT
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
    
    # cx, cy, r = WIDTH//4, HEIGHT//2, HEIGHT//9
    # semi_major_axis = r*2
    # semi_minor_axis = r

    # configurations = [
    #     {
    #         "id": 0,
    #         "obstacle": {
    #             "type": "circumference",
    #             "parameters": {
    #                 "center_x": cx,
    #                 "center_y": cy,
    #                 "radius": r
    #             }
    #         }
    #     },
    #     {
    #         "id": 1,
    #         "obstacle": {
    #             "type": "elipse",
    #             "parameters": {
    #                 "center_x": cx,
    #                 "center_y": cy,
    #                 "semi_major_axis": semi_major_axis,
    #                 "semi_minor_axis": semi_minor_axis,
    #                 "degrees": 0
    #             }
    #         }
    #     },
    #     {
    #         "id": 2,
    #         "obstacle": {
    #             "type": "elipse",
    #             "parameters": {
    #                 "center_x": cx,
    #                 "center_y": cy,
    #                 "semi_major_axis": semi_major_axis,
    #                 "semi_minor_axis": semi_minor_axis,
    #                 "degrees": 5
    #             }
    #         }
    #     }
    # ]
    
    configurations = []
    id = 0

    ##########################################

    shapes_number = 2

    amount, extra = divmod(NUM_SEQUENCES, shapes_number)
    
    amount_circumference = amount
    amount_elipses = amount + extra
    
    # circumference
    circumference_param_grid = {
        "center_x": range(WIDTH//4, WIDTH//2, 10),
        "center_y": range(HEIGHT//3, 2*HEIGHT//3, 10),
        "radius": range(HEIGHT//9, HEIGHT//3, 5)
    }
    circumference_parameters = _select_parameters(amount_circumference, 
                                                  circumference_param_grid)
    for parameters in circumference_parameters:
        configurations.append(
            _build_configuration(id, "circumference", parameters)
        )
        id+=1

    # elipse
    elipses_param_grid = {
        "center_x": range(WIDTH//4, WIDTH//2, 10),
        "center_y": range(HEIGHT//3, 2*HEIGHT//3, 10),
        "semi_major_axis": range(HEIGHT//9, HEIGHT//3, 10),
        "semi_minor_axis": range(HEIGHT//18, HEIGHT//6, 10),
        "degrees": range(-30, 30, 10)
    }
    elipses_parameters = _select_parameters(amount_elipses, elipses_param_grid)
    for parameters in elipses_parameters:
        configurations.append(
            _build_configuration(id, "elipse", parameters)
        )
        id+=1


    ##########################################


    configurations_file = "simulation_configurations.json"
    with open(configurations_file, 'w') as f:
        json.dump(configurations, f)

def _plot_frame(frame, sequence_id, frame_id, save=True):
    plt.clf()

    color_map = plt.cm.jet
    
    plt.imshow(frame.transpose(), cmap=color_map)
    plt.colorbar(label="Velocity", orientation="horizontal")
    if save:
        folder_path = "results/sequence_{}/".format(sequence_id)
        file_path = folder_path + "frame_{0:04d}.png".format(frame_id)
        plt.savefig(file_path)

def generate_images(num_sequences, length_sequence, dataset=None):
    results_folder = "results"
    try:
        # create folder to store results
        os.makedirs(results_folder)
    except OSError:
        # folder already exists
        # delete previous folder
        shutil.rmtree(results_folder)
        os.makedirs(results_folder)
    for i in range(num_sequences):
        os.makedirs(results_folder + "/sequence_" + str(i))

    if dataset is None:
        dataset = np.load("dataset.npy")
    print(dataset.shape)

    # Generate sequence images
    for seq in range(num_sequences):
        sequence = dataset[seq,:] # select sequence from dataset
        # for frame_number in tqdm(range(length_sequence)):
        for frame_number in range(length_sequence):
            frame = sequence[frame_number]
            _plot_frame(frame, seq, frame_number)


if __name__ == "__main__":

    create_configurations()

    # generate_images(NUM_SEQUENCES, LENGHT_SEQUENCE)
