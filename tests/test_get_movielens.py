import unittest

from aprec.datasets.movielens import get_movielens_actions, get_movies_catalog
from aprec.utils.generator_limit import generator_limit


REFERENCE_LINES =\
"""user_id=1, item_id=151, timestamp=1094785734, data={"rating": 4.0}
user_id=1, item_id=223, timestamp=1112485573, data={"rating": 4.0}
user_id=1, item_id=253, timestamp=1112484940, data={"rating": 4.0}
user_id=1, item_id=260, timestamp=1112484826, data={"rating": 4.0}
user_id=1, item_id=293, timestamp=1112484703, data={"rating": 4.0}
user_id=1, item_id=296, timestamp=1112484767, data={"rating": 4.0}
user_id=1, item_id=318, timestamp=1112484798, data={"rating": 4.0}
user_id=1, item_id=541, timestamp=1112484603, data={"rating": 4.0}
user_id=1, item_id=1036, timestamp=1112485480, data={"rating": 4.0}
user_id=1, item_id=1079, timestamp=1094785665, data={"rating": 4.0}
"""

class TestMovielensActions(unittest.TestCase):
    def test_get_actions(self):
        lines = ""
        for action in generator_limit(get_movielens_actions(), 10):
            lines += action.to_str() + "\n" 
        self.assertEqual(lines, REFERENCE_LINES)

    def test_get_catalog(self):
        catalog = get_movies_catalog()
        movie = catalog.get_item("2571")
        self.assertEqual(movie.title, "Matrix, The (1999)")


if __name__ == "__main__":
    unittest.main()
