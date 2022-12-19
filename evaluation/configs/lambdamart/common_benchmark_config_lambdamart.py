from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.first_order_mc import FirstOrderMarkovChainRecommender
from aprec.recommenders.lambdamart_ensemble_recommender import LambdaMARTEnsembleRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.top_recommender import TopRecommender


def lambdamart(filter_seen=True):
    candidates_selection = LightFMRecommender(256)
    if filter_seen:
        candidates_selection = FilterSeenRecommender(candidates_selection)

    other_recommenders = {
        "TopRecommender": TopRecommender(), 
        "FirstOrderMC": FirstOrderMarkovChainRecommender()
    }
    
    return LambdaMARTEnsembleRecommender(
                            candidates_selection_recommender=candidates_selection, 
                            other_recommenders=other_recommenders,
                            n_ensemble_users=200, 
                            n_ensemble_val_users=200, 
    )

recommenders = {
    "lambdamart_baseline": lambdamart,
    "lambdamart_baseline_unfiltered": lambda: lambdamart(filter_seen=False)
}



USERS_FRACTIONS = [1.0]
METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]

def get_recommenders(filter_seen: bool):
    result = {}
    all_recommenders = list(recommenders.keys())
    for recommender_name in all_recommenders:
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result




