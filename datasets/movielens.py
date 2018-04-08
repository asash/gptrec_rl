import os
import requests
import logging

from aprec.utils import mkdir_p_local, get_dir, console_logging, shell

MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
MOVIELENS_DIR = "data/movielens"
MOVIELENS_FILE = "movielens.zip"
DATASET_NAME = 'ml-20m'
MOVIELENS_FILE_ABSPATH = os.path.join(get_dir(), MOVIELENS_DIR, MOVIELENS_FILE)
MOVIELENS_DIR_ABSPATH = os.path.join(get_dir(), MOVIELENS_DIR)


def download_movielens_dataset():
    mkdir_p_local(MOVIELENS_DIR)
    if os.path.isfile(MOVIELENS_FILE_ABSPATH):
        logging.info("movielens_dataset already exists, skipping")
        return 
    logging.info("downloading movielens dataset...")
    response = requests.get(MOVIELENS_URL)
    with open(MOVIELENS_FILE_ABSPATH, 'wb') as out_file:
        out_file.write(response.content)
    logging.info("movielens dataset downloaded")
    
def extract_movielens_dataset():
    check_file = os.path.join(MOVIELENS_DIR_ABSPATH, 'ratings.csv')
    if os.path.isfile(check_file):
        logging.info("movielens dataset is already extracted")
        return
    shell("unzip -o {} -d {}".format(MOVIELENS_FILE_ABSPATH, MOVIELENS_DIR_ABSPATH))
    dataset_dir = os.path.join(MOVIELENS_DIR_ABSPATH, DATASET_NAME)
    for filename in os.listdir(dataset_dir):
        shell("mv {} {}".format(os.path.join(dataset_dir, filename), MOVIELENS_DIR_ABSPATH))
    shell("rm -rf {}".format(dataset_dir))

if __name__ == "__main__":
    console_logging()    
    download_movielens_dataset()
    extract_movielens_dataset()
