from collections import Counter
import logging
import os

from aprec.utils.os_utils import get_dir, mkdir_p, mkdir_p_local, shell


def filter_popular_items(actions_generator, max_actions):
    actions = []
    items_counter = Counter()
    for action in actions_generator:
        actions.append(action)
        items_counter[action.item_id] += 1
    popular_items = set([item_id for (item_id, cnt) in items_counter.most_common(max_actions)])
    return filter(lambda action: action.item_id in popular_items, actions)


def unzip(zipped_file, unzip_dir):
    full_dir_name = os.path.join(get_dir(), unzip_dir)
    if os.path.isdir(full_dir_name):
        logging.info(f"{unzip_dir} already exists, skipping")
    else:
        mkdir_p(full_dir_name)
        shell(f"unzip -o {zipped_file} -d {full_dir_name}")
    return full_dir_name
    
    
    