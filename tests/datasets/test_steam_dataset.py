from collections import Counter
import unittest

from aprec.evaluation.evaluation_utils import group_by_user


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

    
if __name__ == "__main__":
    unittest.main()
   