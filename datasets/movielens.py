import os
from collections import Counter

import requests
import logging

from aprec.utils.os import mkdir_p_local, get_dir, console_logging, shell
from aprec.api.action import Action
from aprec.api.item import Item
from aprec.api.catalog import Catalog

DATASET_NAME = 'ml-20m'
MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/{}.zip".format(DATASET_NAME)
MOVIELENS_DIR = "data/movielens"
MOVIELENS_FILE = "movielens.zip"
MOVIELENS_FILE_ABSPATH = os.path.join(get_dir(), MOVIELENS_DIR, MOVIELENS_FILE)
MOVIELENS_DIR_ABSPATH = os.path.join(get_dir(), MOVIELENS_DIR)
RATINGS_FILE = os.path.join(MOVIELENS_DIR_ABSPATH, 'ratings.csv')
MOVIES_FILE = os.path.join(MOVIELENS_DIR_ABSPATH, 'movies.csv')


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
    if os.path.isfile(RATINGS_FILE):
        logging.info("movielens dataset is already extracted")
        return
    shell("unzip -o {} -d {}".format(MOVIELENS_FILE_ABSPATH, MOVIELENS_DIR_ABSPATH))
    dataset_dir = os.path.join(MOVIELENS_DIR_ABSPATH, DATASET_NAME)
    for filename in os.listdir(dataset_dir):
        shell("mv {} {}".format(os.path.join(dataset_dir, filename), MOVIELENS_DIR_ABSPATH))
    shell("rm -rf {}".format(dataset_dir))


def prepare_data():
    download_movielens_dataset()
    extract_movielens_dataset()


def get_movielens_actions(min_rating=4.0):
    prepare_data()
    with open(RATINGS_FILE, 'r') as data_file:
        header = True
        for line in data_file:
            if header:
                header = False
            else:
                user_id, movie_id, rating_str, timestamp_str = line.strip().split(',')
                rating = float(rating_str)
                timestamp = int(timestamp_str)
                if rating >= min_rating:
                    yield Action(user_id, movie_id, timestamp, {"rating": rating})

def filter_popular_items(actions_generator, max_actions):
    actions = []
    items_counter = Counter()
    for action in actions_generator:
        actions.append(action)
        items_counter[action.item_id] += 1
    popular_items = set([item_id for (item_id, cnt) in items_counter.most_common(max_actions)])
    return filter(lambda action: action.item_id in popular_items, actions)

def get_movies_catalog():
    prepare_data()
    catalog = Catalog()
    with open(MOVIES_FILE, 'r') as data_file:
        header = True
        for line in data_file:
            if header:
                header = False
            else:
                splits = line.strip().split(",")
                movie_id = splits[0]
                genres = splits[-1].split("|")
                title = ",".join(splits[1:-1]).strip('"')
                item = Item(movie_id).with_title(title).with_tags(genres)
                catalog.add_item(item)
    return catalog


if __name__ == "__main__":
    console_logging()
    prepare_data()
