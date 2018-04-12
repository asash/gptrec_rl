from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.datasets.movielens import get_movielens_actions, get_movies_catalog
from aprec.utils.generator_limit import generator_limit
from aprec.api.action import Action
import unittest

REFERENCE_COLD_START = [('318', 0.37689138044151266), ('296', 0.3574406246772646), ('356', 0.3358529670797735), ('593', 0.2984884710306045), ('47', 0.29273482768981324), ('50', 0.2862750184955636), ('527', 0.2651443463750701), ('589', 0.2517902108569122), ('110', 0.2431276658452398), ('2858', 0.23858577202574255)]

USER_ID = '120' 

REFERENCE_USER_RECOMMENDATIONS = [('457', 0.4555895355528814), ('380', 0.41878479189637907), ('110', 0.41371092094949746), ('292', 0.3658763722681398), ('296', 0.32779277385356653), ('595', 0.3135156842689451), ('588', 0.31243386441607296), ('592', 0.2930348223906822), ('440', 0.28664327616275026), ('357', 0.28605665125871604), ('434', 0.2804929031598042), ('593', 0.28042317307652453), ('733', 0.276061859488453), ('553', 0.257852190928236), ('253', 0.2559445694032447)]

class TestSvdRecommender(unittest.TestCase):
    def test_svd_recommender(self):
        svd_recommender = SvdRecommender(10, random_seed=31337)
        recommender = FilterSeenRecommender(svd_recommender)
        catalog = get_movies_catalog()
        for action in generator_limit(get_movielens_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        self.assertEqual(recommender.get_next_items(12341324, 10), REFERENCE_COLD_START)
        recs = recommender.get_next_items(USER_ID, 10)
        self.assertEqual(recs, REFERENCE_USER_RECOMMENDATIONS)

        actions =  [Action('1', 1, 1), 
                    Action('1', 2, 2),
                    Action('2', 2, 1),
                    Action('2', 3, 1)]
        recommender = SvdRecommender(2, random_seed=31337)
        for action in actions:
            recommender.add_action(action)
        recommender.rebuild_model()






if __name__ == "__main__":
    unittest.main()

