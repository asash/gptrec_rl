import unittest

from aprec.datasets.movielens import get_movielens_actions, get_movies_catalog
from aprec.utils.generator_limit import generator_limit


REFERENCE_LINES =\
"""Action(uid=1, item=151, ts=1094785734, data={"rating": 4.0})
Action(uid=1, item=223, ts=1112485573, data={"rating": 4.0})
Action(uid=1, item=253, ts=1112484940, data={"rating": 4.0})
Action(uid=1, item=260, ts=1112484826, data={"rating": 4.0})
Action(uid=1, item=293, ts=1112484703, data={"rating": 4.0})
Action(uid=1, item=296, ts=1112484767, data={"rating": 4.0})
Action(uid=1, item=318, ts=1112484798, data={"rating": 4.0})
Action(uid=1, item=541, ts=1112484603, data={"rating": 4.0})
Action(uid=1, item=1036, ts=1112485480, data={"rating": 4.0})
Action(uid=1, item=1079, ts=1094785665, data={"rating": 4.0})
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
