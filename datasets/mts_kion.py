import datetime
import logging
import os
import time

import requests

from aprec.api.user import User
from aprec.api.action import Action
from aprec.utils.os_utils import mkdir_p_local, get_dir

MTS_KION_DIR = "data/mts_kion"

MTS_KION_INTERACTIONS_URL = "https://storage.yandexcloud.net/datasouls-ods/materials/04adaecc/interactions.csv"
MTS_KION_INTERACTIONS_FILE = "interactions.csv"

MTS_KION_ITEMS_URL = "https://storage.yandexcloud.net/datasouls-ods/materials/f90231b6/items.csv"
MTS_KION_ITEMS_FILE = "items.csv"

MTS_KION_USERS_URL = "https://storage.yandexcloud.net/datasouls-ods/materials/6503d6ab/users.csv"
MTS_KION_USERS_FILE = "users.csv"


MTS_KION_SAMPLE_SUBMISSION_URL = "https://storage.yandexcloud.net/datasouls-ods/materials/faa61a41/sample_submission.csv"
MTS_KION_SAMPLE_SUBMISSION_FILE = "sample_submission.txt"

def download_mts_file(url, filename):
    mkdir_p_local(MTS_KION_DIR)
    full_filename = os.path.join(get_dir(), MTS_KION_DIR, filename) 
    if not os.path.isfile(full_filename):
        logging.info(f"downloading  {filename} file")
        response = requests.get(url)
        with open(full_filename, 'wb') as out_file:
            out_file.write(response.content)
        logging.info(f"{filename} dataset downloaded")
        full_name = get_dir()
    else:
        logging.info(f"booking {filename} file already exists, skipping")
    return full_filename 

def get_actions(actions_file, max_actions=None):
    actions = []
    with open(actions_file) as input:
        header = input.readline()
        cnt = 0
        for line in input:
            user_id, item_id, last_watch_date, total_dur, watched_pct = line.split(",")
            timestamp = time.mktime(datetime.datetime.strptime(last_watch_date, "%Y-%m-%d").timetuple())
            cnt += 1
            actions.append(Action(user_id=user_id, item_id=item_id, timestamp=timestamp))
            if max_actions is not None and cnt >= max_actions:
                break
        return actions


def get_submission_user_ids():
    sample_submission_file = download_mts_file(MTS_KION_SAMPLE_SUBMISSION_URL, MTS_KION_SAMPLE_SUBMISSION_FILE)
    result = []
    with open(sample_submission_file) as input:
        line = input.readline()
        line = input.readline()
        while(len(line) > 0):
            user_id = line.split(",")[0]
            result.append(user_id)
            line = input.readline()
    return result


def get_users():
    users_file = download_mts_file(MTS_KION_USERS_URL, MTS_KION_USERS_FILE)
    result = []
    with open(users_file) as input:
        header = input.readline()
        line = input.readline()
        while (len(line) > 0):
            user_id,age,income,sex,kids_flg = line.strip().split(",")
            cat_features = {"income": income, "sex": sex, "age": age, "kids_flg": kids_flg}
            result.append(User(user_id, cat_features=cat_features))
            line = input.readline()
    return result


def get_mts_kion_dataset(max_actions=None):
    interactions_file = download_mts_file(MTS_KION_INTERACTIONS_URL, MTS_KION_INTERACTIONS_FILE)
    users_file = download_mts_file(MTS_KION_USERS_URL, MTS_KION_USERS_FILE)
    items_file = download_mts_file(MTS_KION_ITEMS_URL, MTS_KION_ITEMS_FILE)
    sample_submission_file = download_mts_file(MTS_KION_SAMPLE_SUBMISSION_URL, MTS_KION_SAMPLE_SUBMISSION_FILE)
    actions = get_actions(interactions_file, max_actions)
    return actions
