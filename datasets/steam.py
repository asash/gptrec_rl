import calendar
import datetime
import gzip
import re
import ast
from aprec.api.action import Action
from aprec.datasets.download_file import download_file


DATASET_URL="https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz"
METADATA_URL="https://cseweb.ucsd.edu/~wckang/steam_games.json.gz"

DATA_DIR = "data/steam"

def get_steam_actions():
    filename = download_file(DATASET_URL, "steam.json.gz", DATA_DIR)

    result = []
    for line_bytes in gzip.open(filename):
        parsed_review = ast.literal_eval(line_bytes.decode("utf-8"))
        date = datetime.datetime.strptime(parsed_review['date'], "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
        user_id = parsed_review['username']
        product_id = parsed_review['product_id']
        timestamp = calendar.timegm(date.timetuple())
        result.append(Action(user_id, product_id, timestamp))
    return result

def get_game_genres():
    filename = download_file(METADATA_URL, "steam_metadata.json.gz", DATA_DIR)

    result = {}
    for line_bytes in gzip.open(filename):
        parsed_game = ast.literal_eval(line_bytes.decode("utf-8"))
        try:
            product_id = parsed_game['id']
            genres = parsed_game['tags']
        except KeyError:
            continue
        result[product_id] = genres
    return result
