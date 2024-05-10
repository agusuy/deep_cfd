from datetime import datetime
from configurations import create_configurations
from constants import LENGHT_SEQUENCE, NUM_SEQUENCES
from simulation import create_dataset


def run():
    create_configurations()

    print(f">{datetime.now():%d-%m-%Y %H:%M:%S} Start generating dataset")
    data = create_dataset()
    print(f">{datetime.now():%d-%m-%Y %H:%M:%S} End generating dataset")


if __name__ == "__main__":
    run()

    # print(f">{datetime.now():%d-%m-%Y %H:%M:%S} Start generating images")
    # generate_images(NUM_SEQUENCES, LENGHT_SEQUENCE, data)
    # generate_images(NUM_SEQUENCES, LENGHT_SEQUENCE)
    # print(f">{datetime.now():%d-%m-%Y %H:%M:%S} End generating images")
