import os

###### Folders and Files ######

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
PROJECT_FOLDER = os.path.dirname(__file__)

DATASET_FOLDER = os.path.join(ROOT, "dataset")

OUTPUTS_FOLDER = os.path.join(PROJECT_FOLDER, "outputs")
IMAGES_FOLDER = os.path.join(OUTPUTS_FOLDER, "images")

SIM_CONFIGURATIONS = os.path.join(OUTPUTS_FOLDER, "sim_conf.json")
DATASET_FILE = os.path.join(DATASET_FOLDER, "dataset.npy")


###### Dataset Parameters ######

# ITERATIONS = 100000
ITERATIONS = 50000 #<---
# ITERATIONS = 20000
# ITERATIONS = 1000

ITERATIONS_PER_FRAME = 100
LENGHT_SEQUENCE = ITERATIONS//ITERATIONS_PER_FRAME

NUM_SEQUENCES = 140 #<---
# NUM_SEQUENCES = 10
# NUM_SEQUENCES = 1

# WIDTH, HEIGHT = 1000, 500 # Grid dimensions
WIDTH, HEIGHT = 400, 200 # Grid dimensions #<---
# WIDTH, HEIGHT = 100, 100 # Grid dimensions