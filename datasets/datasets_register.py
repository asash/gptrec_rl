#This file contains register of all available datasets in our system.
#Unless necessary only use datasets from this file.
from aprec.datasets.bert4rec_datasets import get_bert4rec_dataset
from aprec.datasets.booking import get_booking_dataset
from aprec.datasets.movielens100k import get_movielens100k_actions
from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.datasets.gowalla import get_gowalla_dataset
from aprec.datasets.yelp import get_yelp_dataset
from aprec.datasets.mts_kion import get_mts_kion_dataset

class DatasetsRegister(object):
    _all_datasets =  {
        "BERT4rec.ml-1m": lambda: get_bert4rec_dataset("ml-1m"),
        "BERT4rec.steam": lambda: get_bert4rec_dataset("steam"),
        "BERT4rec.beauty": lambda: get_bert4rec_dataset("beauty"),
        "ml-20m": lambda: get_movielens20m_actions(min_rating=0.0),
        "ml-100k": lambda: get_movielens100k_actions(min_rating=0.0),
        "booking": lambda: get_booking_dataset(unix_timestamps=True)[0],
        "gowalla": get_gowalla_dataset,
        "mts_kion": lambda: get_mts_kion_dataset(),
        "yelp": get_yelp_dataset,
    }

    def __getitem__(self, item):
        if item not in DatasetsRegister._all_datasets:
            raise KeyError(f"The dataset {item} is not registered")
        return DatasetsRegister._all_datasets[item]

    def all_datasets(self):
        return list(DatasetsRegister._all_datasets.keys())



