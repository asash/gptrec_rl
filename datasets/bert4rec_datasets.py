import os

import BERT4rec
from aprec.api.action import Action


def get_bert4rec_dataset(dataset):
    bert4rec_data_dir = os.path.join(os.path.dirname(BERT4rec.__file__), "data")
    dataset_filename = os.path.join(bert4rec_data_dir, f"{dataset}.txt")
    actions = []
    if os.path.isfile(dataset_filename):
        print(f"found datastet at path {dataset_filename}")
        prev_user = None
        current_timestamp = 0
        with open(dataset_filename) as input:
            for line in input:
                user, item = [int(id) for id in line.strip().split()]
                if user != prev_user:
                    current_timestamp = 0
                prev_user = user
                current_timestamp += 1
                actions.append(Action(user, item, current_timestamp))
        return actions