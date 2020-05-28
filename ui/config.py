from tqdm import tqdm
from aprec.datasets.movielens import get_movies_catalog, get_movielens_actions
from aprec.recommenders.item_item import ItemItemRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
import sys

CATALOG = get_movies_catalog()

actions = get_movielens_actions(3.5)
recommender = FilterSeenRecommender(ItemItemRecommender(200))
cnt = 0
for action in tqdm(actions):
    recommender.add_action(action)
    cnt += 1

sys.stderr.write("building model...")

recommender.rebuild_model()

sys.stderr.write("ready.")

RECOMMENDER = recommender