from aprec.api.action import Action
from aprec.api.catalog import Catalog
from aprec.datasets.download_file import download_file

BERT4REC_DATASET_URL="https://raw.githubusercontent.com/asash/BERT4rec_py3_tf2/master/BERT4rec/data/{}.txt"

BERT4REC_DIR = "data/bert4rec"
VALID_DATASETS={"beauty", "ml-1m", "steam"}

def get_bert4rec_dataset(dataset):
    if dataset not in VALID_DATASETS:
        raise ValueError(f"unknown bert4rec dataset {dataset}")
    dataset_filename = download_file(BERT4REC_DATASET_URL.format(dataset), dataset + ".txt", BERT4REC_DIR)
    actions = []
    prev_user = None
    current_timestamp = 0
    with open(dataset_filename) as input:
        for line in input:
            user, item = [str(id) for id in line.strip().split()]
            if user != prev_user:
                current_timestamp = 0
            prev_user = user
            current_timestamp += 1
            actions.append(Action(user, item, current_timestamp))
    return actions

MAPPING_URL="https://raw.githubusercontent.com/asash/ml1m-sas-mapping/main/sas_to_original_items.txt"    
def ml1m_mapping_to_original():
    mapping_filename = download_file(MAPPING_URL, "sas_to_original_items.txt", BERT4REC_DIR)
    result = {}
    for line in open(mapping_filename):
        sas_item, original_item = line.strip().split()
        result[sas_item] =  original_item
    return result
        
        
def get_movielens1m_genres():
    from aprec.datasets.movielens1m import get_genre_dict as get_ml1m_genre_dict
    original_genre_dict = get_ml1m_genre_dict()
    mapping = ml1m_mapping_to_original()
    result = {}
    for sas_item, original_item in mapping.items():
        result[sas_item] = original_genre_dict[original_item]
    return result

def get_movielens1m_catalog():
    from aprec.datasets.movielens1m import get_movies_catalog as get_ml1m_catalog 
    original_catalog = get_ml1m_catalog()
    mapping = ml1m_mapping_to_original()
    result = Catalog()
    for sas_item, original_item_id in mapping.items():
        item = original_catalog.get_item(original_item_id)
        item.item_id = sas_item
        result.add_item(item)
    return result
       