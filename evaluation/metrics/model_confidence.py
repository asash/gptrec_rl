from aprec.evaluation.metrics.metric import Metric
from scipy.special import softmax
from scipy.stats import logistic


class Confidence(Metric):
    def __init__(self, activation):
        self.name = f"{activation}Confidence"
        if activation == 'Softmax':
            self.activation = softmax
        elif activation == 'Sigmoid':
            self.activation = logistic.cdf
        else:
            raise Exception(f"unknown activation {activation}")
            
        
    def __call__(self, recommendations, actual_actions):
        if len(recommendations) == 0:
            return 0
        scores = [rec[1] for rec in recommendations]
        return softmax(scores)[0]