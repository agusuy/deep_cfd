from datetime import datetime
from auxiliary_functions import create_configurations, generate_images
from constants import LENGHT_SEQUENCE, NUM_SEQUENCES
from simulation import create_dataset


if __name__ == "__main__":
    create_configurations()

    data = create_dataset()

    print(f"{datetime.now():%d-%m-%Y %H:%M:%S}> Start generating images")
    generate_images(NUM_SEQUENCES, LENGHT_SEQUENCE, data)
    print(f"{datetime.now():%d-%m-%Y %H:%M:%S}> End generating images")

    # print(f"{datetime.now():%d-%m-%Y %H:%M:%S}> Start generating images")
    # generate_images(NUM_SEQUENCES, LENGHT_SEQUENCE)
    # print(f"{datetime.now():%d-%m-%Y %H:%M:%S}> End generating images")