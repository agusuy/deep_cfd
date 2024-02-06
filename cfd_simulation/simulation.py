import os
import numpy as np
from constants import DATASET_FILE, DATASET_FOLDER, NUM_SEQUENCES, \
                        LENGHT_SEQUENCE, SIM_CONFIGURATIONS, WIDTH, HEIGHT
from cfd import run_cfd
from datetime import datetime
from obstacles import *
import json

def create_dataset():
    dataset = np.zeros((NUM_SEQUENCES, LENGHT_SEQUENCE, WIDTH, HEIGHT))
    print(dataset.shape)

    with open(SIM_CONFIGURATIONS, 'r') as f: 
        configurations = json.load(f)

    for conf in configurations:
        sequence_id = conf["id"]

        obstacle_type = conf["obstacle"]["type"]
        obstacle_parameters = conf["obstacle"]["parameters"]

        if obstacle_type == "circumference":
            obstacle_fun = circumference(**obstacle_parameters) 
        elif obstacle_type == "elipse":
            obstacle_fun = elipse(**obstacle_parameters)
            
        obstacle = np.fromfunction(obstacle_fun, (WIDTH,HEIGHT))
        
        print(f"{datetime.now():%d-%m-%Y %H:%M:%S}> Start sequence {sequence_id}")
        sequence = run_cfd(obstacle)
        dataset[sequence_id,:] = sequence
        print(f"{datetime.now():%d-%m-%Y %H:%M:%S}> End sequence {sequence_id}")

    try:
        # create folder to store results
        os.makedirs(DATASET_FOLDER)
    except OSError:
        # folder already exists
        pass

    np.save(DATASET_FILE, dataset)
    return dataset

if __name__ == "__main__":
    data = create_dataset()
