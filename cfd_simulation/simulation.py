import os
import numpy as np
from constants import DATASET_FILE, DATASET_FOLDER, \
                        SIM_CONFIGURATIONS, SIM_STATS, \
                        NUM_SEQUENCES, LENGHT_SEQUENCE, WIDTH, HEIGHT
from cfd import run_cfd
from datetime import datetime
from obstacles import *
import json

def create_dataset():
    dataset = np.zeros((NUM_SEQUENCES, LENGHT_SEQUENCE, WIDTH, HEIGHT))
    print(dataset.shape)

    with open(SIM_CONFIGURATIONS, 'r') as f: 
        configurations = json.load(f)

    with open(SIM_STATS, 'w') as f:
        f.write(f"sequence_id, simulation_duration\n")

        for conf in configurations:
            sequence_id = conf["id"]

            start_time = datetime.now()
            print(f">{start_time:%d-%m-%Y %H:%M:%S} Start sequence {sequence_id}")
            
            simulate_configuration(dataset, conf)

            end_time = datetime.now()
            print(f">{end_time:%d-%m-%Y %H:%M:%S} End sequence {sequence_id}")
            simulation_duration = (end_time - start_time).total_seconds()
            f.write(f"{sequence_id}, {simulation_duration}\n")

    try:
        # create folder to store results
        os.makedirs(DATASET_FOLDER)
    except OSError:
        # folder already exists
        pass

    np.save(DATASET_FILE, dataset)
    return dataset

def simulate_configuration(dataset, conf):
    sequence_id = conf["id"]

    obstacle_type = conf["obstacle"]["type"]
    obstacle_parameters = conf["obstacle"]["parameters"]

    if obstacle_type == "circumference":
        obstacle_fun = circumference(**obstacle_parameters) 
    elif obstacle_type == "ellipse":
        obstacle_fun = ellipse(**obstacle_parameters)
            
    obstacle = np.fromfunction(obstacle_fun, (WIDTH,HEIGHT))
    
    sequence = run_cfd(obstacle)
    dataset[sequence_id,:] = sequence

if __name__ == "__main__":
    data = create_dataset()
