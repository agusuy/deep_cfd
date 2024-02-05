import numpy as np
from constants import NUM_SEQUENCES, LENGHT_SEQUENCE, WIDTH, HEIGHT
from cfd import run_cfd
from auxiliary_functions import create_configurations, generate_images
from datetime import datetime
from obstacles import *
import json

def create_dataset():
    dataset = np.zeros((NUM_SEQUENCES, LENGHT_SEQUENCE, WIDTH, HEIGHT))
    print(dataset.shape)

    configurations_file = "simulation_configurations.json"
    with open(configurations_file, 'r') as f: 
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

    np.save("dataset.npy", dataset)
    return dataset

if __name__ == "__main__":

    create_configurations()

    data = create_dataset()
    
    print(f"{datetime.now():%d-%m-%Y %H:%M:%S}> Start generating images")
    generate_images(NUM_SEQUENCES, LENGHT_SEQUENCE, data)
    print(f"{datetime.now():%d-%m-%Y %H:%M:%S}> End generating images")


    # print(f"{datetime.now():%d-%m-%Y %H:%M:%S}> Start generating images")
    # generate_images(NUM_SEQUENCES, LENGHT_SEQUENCE)
    # print(f"{datetime.now():%d-%m-%Y %H:%M:%S}> End generating images")
