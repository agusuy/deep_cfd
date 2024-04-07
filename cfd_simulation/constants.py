import os

###### Folders and Files ######

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
PROJECT_FOLDER = os.path.dirname(__file__)

DATASET_FOLDER = os.path.join(ROOT, "dataset")

OUTPUTS_FOLDER = os.path.join(PROJECT_FOLDER, "outputs")
IMAGES_FOLDER = os.path.join(OUTPUTS_FOLDER, "images")

SIM_CONFIGURATIONS = os.path.join(OUTPUTS_FOLDER, "sim_conf.json")
DATASET_FILE = os.path.join(DATASET_FOLDER, "dataset.npy")

SIM_STATS = os.path.join(OUTPUTS_FOLDER, "sim_stats.csv")


###### Dataset Parameters ######

# ITERATIONS = 100000
# ITERATIONS = 50000
ITERATIONS = 40000 #<---

ITERATIONS_PER_FRAME = 100
LENGHT_SEQUENCE = ITERATIONS//ITERATIONS_PER_FRAME

WARM_UP_ITERATIONS = 1000 # Equals first 10 frames

# NUM_SEQUENCES = 140
NUM_SEQUENCES = 40 #<---
# NUM_SEQUENCES = 2

# WIDTH, HEIGHT = 1000, 500 # Grid dimensions
# WIDTH, HEIGHT = 400, 200 # Grid dimensions 
WIDTH, HEIGHT = 200, 100 # Grid dimensions #<---
