from collections import Counter
import unittest


class TestSteamDataset(unittest.TestCase):
    def test_steam1000_warm_dataset(self):
        import json
        from aprec.datasets.dataset_stats import dataset_stats
        from aprec.datasets.datasets_register import DatasetsRegister
        from aprec.datasets.steam import get_steam_actions, get_game_genres

        steam_dataset_1000_warm = DatasetsRegister().get_from_cache("steam_1000items_warm_users")()
        self.assertEquals(len(steam_dataset_1000_warm), 2884052)
        print(steam_dataset_1000_warm[0])
        genres = get_game_genres()
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
        print(all_genres_counter.most_common(5), [('Singleplayer', 891), ('Action', 796), ('Adventure', 725), ('Multiplayer', 627), ('Indie', 533)])

    
if __name__ == "__main__":
    unittest.main()
   