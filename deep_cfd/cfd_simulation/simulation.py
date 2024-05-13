import numpy as np
import json
from datetime import datetime
from .configurations import create_configurations
from .simulation_constants import NUM_SEQUENCES, LENGHT_SEQUENCE, WIDTH, HEIGHT
from .cfd import run_cfd
from .obstacles import *

def _simulate_configuration(dataset, conf):
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

def run_simulations(dataset, sim_conf_file, sim_stats_file):
    with open(sim_conf_file, 'r') as f: 
        configurations = json.load(f)

    with open(sim_stats_file, 'w') as f:
        f.write(f"sequence_id, simulation_duration\n")

        for conf in configurations:
            sequence_id = conf["id"]

            start_time = datetime.now()
            print(f">{start_time:%d-%m-%Y %H:%M:%S} Start sequence {sequence_id}")
            
            _simulate_configuration(dataset, conf)

            end_time = datetime.now()
            print(f">{end_time:%d-%m-%Y %H:%M:%S} End sequence {sequence_id}")
            simulation_duration = (end_time - start_time).total_seconds()
            f.write(f"{sequence_id}, {simulation_duration}\n")

def create_dataset(dataset_file, sim_conf_file, sim_stats_file):
    create_configurations(sim_conf_file)

    dataset = np.zeros((NUM_SEQUENCES, LENGHT_SEQUENCE, WIDTH, HEIGHT))
    print(dataset.shape)

    print(f">{datetime.now():%d-%m-%Y %H:%M:%S} Start generating dataset")
    
    run_simulations(dataset, sim_conf_file, sim_stats_file)
    
    np.save(dataset_file, dataset)

    print(f">{datetime.now():%d-%m-%Y %H:%M:%S} End generating dataset")
    
    return dataset
