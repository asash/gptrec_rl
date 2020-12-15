import gzip
import json
from collections import Counter

from aprec.api.action import Action
from aprec.utils.os import mkdir_p_local, get_dir, shell

import os
import logging

AMAZON_BOOKS_RAW_FILE = "reviews_Books.json.gz"
AMAZON_BOOKS_FILE = "reviews_Books.csv.gz"
AMAZON_BOOKS_URL = f"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{AMAZON_BOOKS_RAW_FILE}"
AMAZON_BOOKS_DIR = "data/amazon/books"
AMAZON_BOOKS_RAW_FILE_ABSPATH = os.path.join(get_dir(), AMAZON_BOOKS_DIR, AMAZON_BOOKS_RAW_FILE)
AMAZON_BOOKS_FILE_ABSPATH = os.path.join(get_dir(), AMAZON_BOOKS_DIR, AMAZON_BOOKS_FILE)

def download_amazon_books_dataset():
    mkdir_p_local(AMAZON_BOOKS_DIR)
    if os.path.isfile(AMAZON_BOOKS_RAW_FILE_ABSPATH):
        logging.info("amazon_books dataset is already downloaded, skipping")
        return
    logging.info("downloading movielens dataset...")
    shell(f"wget {AMAZON_BOOKS_URL} -O {AMAZON_BOOKS_RAW_FILE_ABSPATH}")
    logging.info("movielens dataset downloaded")

def extract_amazon_books_dataset():
    if os.path.isfile(AMAZON_BOOKS_FILE_ABSPATH):
        logging.info("amazon_books dataset is already extracted, skipping")
        return
    with gzip.open(AMAZON_BOOKS_FILE_ABSPATH, 'wb') as output:
        for line in gzip.open(AMAZON_BOOKS_RAW_FILE_ABSPATH):
            doc = json.loads(line)
            user_id = doc['reviewerID']
            item_id = doc['asin']
            timestamp = doc['unixReviewTime']
            rating = doc['overall']
            output.write(f"{user_id},{item_id},{rating},{timestamp}\n".encode())

def get_amazon_books_dataset(min_actions_per_user=3, min_users_per_item=3):
    download_amazon_books_dataset()
    extract_amazon_books_dataset()
    users_counter = Counter()
    items_counter = Counter()
    result = []
    for line in gzip.open(AMAZON_BOOKS_FILE_ABSPATH):
        user_id, item_id, rating, timestamp_str = line.decode().strip().split(',')
        users_counter[user_id] += 1
        items_counter[item_id] += 1
        result.append(Action(user_id=user_id, item_id=item_id, timestamp=int(timestamp_str), data={"rating": rating}))
    return list(filter(lambda action: users_counter[action.user_id] >= min_actions_per_user\
                                 and items_counter[action.item_id] >= min_users_per_item, result))


