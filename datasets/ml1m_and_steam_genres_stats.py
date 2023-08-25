from aprec.datasets.steam import get_genres_steam_deduped_1000items_warm_users

from aprec.datasets.bert4rec_datasets import get_movielens1m_genres

steam_item_genres = get_genres_steam_deduped_1000items_warm_users()
movielens_genres = get_movielens1m_genres()
print(len(movielens_genres))

steam_set = set()
for item in steam_item_genres:
    for genre in steam_item_genres[item]:
        steam_set.add(genre)

movielens_set = set()
for item in movielens_genres:
    for genre in movielens_genres[item]:
        movielens_set.add(genre)


print("steam:", len(steam_set))
print("movielens:", len(movielens_set))
