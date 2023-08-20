from collections import Counter
from datetime import datetime
import unittest

import pytz

from aprec.evaluation.evaluation_utils import group_by_user
from aprec.datasets.dataset_utils import sequence_break_ties


class TestSteamDataset(unittest.TestCase):
    def test_steam1000_warm_dataset(self):
        import json
        from aprec.datasets.dataset_stats import dataset_stats
        from aprec.datasets.datasets_register import DatasetsRegister
        from aprec.datasets.steam import get_game_genres, get_genres_steam_deduped_1000items_warm_users

        steam_dataset_1000_warm = DatasetsRegister()["steam_deduped_1000items_warm_users"]()
        self.assertEquals(len(steam_dataset_1000_warm), 2198260)
        print(steam_dataset_1000_warm[0])
        genres = get_genres_steam_deduped_1000items_warm_users()
        self.assertEqual(genres['35140'], ['Action', 'Batman', 'Stealth', 'Adventure', 'Third Person', 'Superhero', 'Singleplayer', "Beat 'em up", 
                                           'Open World', 'Comic Book', 'Detective', 'Atmospheric', 'Story Rich', 'Fighting', 'Action-Adventure', 'Controller', 'Metroidvania', '3D Vision', 'Puzzle', 'Horror'])
        all_genres_counter = Counter()
        all_items = set()
        for action in steam_dataset_1000_warm:
            self.assertTrue(action.item_id in genres)
            if action.item_id not in all_items:
                all_items.add(action.item_id)
                all_genres_counter.update(genres[action.item_id])
        self.assertEqual(len(all_genres_counter), 318)
        self.assertEqual(all_genres_counter.most_common(5), [('Singleplayer', 890), ('Action', 797), ('Adventure', 723), ('Multiplayer', 628), ('Indie', 536)])

        by_user_actions = group_by_user(steam_dataset_1000_warm)


        for user in by_user_actions:
            self.assertTrue(len(by_user_actions[user]) >= 5)
            seen_items = set()
            for action in by_user_actions[user]:
                self.assertTrue(action.item_id not in seen_items)
        
        #make sure that we don't change the dates of the actions
        original_interaction_dates = {}
        for action in steam_dataset_1000_warm:
            utc_timezone = pytz.UTC
            date = datetime.fromtimestamp(action.timestamp, utc_timezone).date()
            date = date.strftime("%Y-%m-%d")
            if (action.user_id, action.item_id) in original_interaction_dates:
                raise Exception(f"Duplicate action: {action}")
            original_interaction_dates[(action.user_id, action.item_id)] = (date, action.timestamp)

        broken_ties = sequence_break_ties(steam_dataset_1000_warm)
        broken_ties_dates = {} 
        for action in broken_ties:
            utc_timezone = pytz.UTC
            date = datetime.fromtimestamp(action.timestamp, utc_timezone).date()
            date = date.strftime("%Y-%m-%d")
            broken_ties_dates[(action.user_id, action.item_id)] = (date, action.timestamp)

        for action in broken_ties:
            original_date = original_interaction_dates[(action.user_id, action.item_id)]
            broken_ties_date = broken_ties_dates[(action.user_id, action.item_id)]
            if original_date[0] != broken_ties_date[0]:
                print(f"original date: {original_date}, broken ties date: {broken_ties_date}")
                raise Exception(f"original date: {original_date}, broken ties date: {broken_ties_date}")
        

    
if __name__ == "__main__":
    unittest.main()
   