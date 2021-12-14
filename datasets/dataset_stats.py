import sys
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from aprec.datasets.bert4rec_datasets import get_bert4rec_dataset
from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.datasets.movielens100k import get_movielens100k_actions
from aprec.datasets.mts_kion import get_mts_kion_dataset
from aprec.datasets.booking import get_booking_dataset

import numpy as np

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', 256)


all_datasets =  {
    "BERT4rec.ml-1m": lambda: get_bert4rec_dataset("ml-1m"),
    "BERT4rec.steam": lambda: get_bert4rec_dataset("steam"),
    "BERT4rec.beauty": lambda: get_bert4rec_dataset("beauty"),
    "ml-20m": lambda: get_movielens20m_actions(min_rating=0.0),
    "ml-100k": lambda: get_movielens100k_actions(min_rating=0.0),
    "booking": lambda: get_booking_dataset(),
    "mts_kion": lambda: get_mts_kion_dataset()

}

def dataset_stats(dataset, dataset_name=None):
    users = defaultdict(list)
    item_ids = set()
    for action in dataset:
        users[action.user_id].append(action)
        item_ids.add(action.item_id)
    session_lens = [len(users[user_id]) for user_id in users]
    result = {
        "num_users": len(users),
        "num_items": len(item_ids),
        "average_session_len": float(np.mean(session_lens)),
        "median_session_len": int(np.median(session_lens)),
        "max_session_len": int(np.max(session_lens)),
        "min_session_len": int(np.min(session_lens)),
        "p80_session_len": float(np.percentile(session_lens, 80))
    }
    if dataset_name is not None:
        result['name'] = dataset_name
    return result

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print(f"usage: python3 {__file__} <comma-separated-datasets>")
        print(f"available datasets: {', '.join(list(all_datasets.keys()))}")
    else:
        datasets = sys.argv[1].split(",")
        for dataset in datasets:
            if dataset not in all_datasets:
                print(f"unknown dataset {dataset}")
                exit(1)
        docs = []
        for dataset_name in tqdm(datasets):
            dataset = all_datasets[dataset_name]()
            stats = dataset_stats(dataset, dataset_name=dataset_name)
            docs.append(stats)
            del(dataset)
        print(pd.DataFrame(docs).set_index("name"))


