import numpy as np
from aprec.recommenders.sequential.history_vectorizers.history_vectorizer import HistoryVectorizer

class DefaultHistoryVectrizer(HistoryVectorizer):
    def __call__(self, user_actions, extension=0):
        res_len = self.sequence_len + extension
        if len(user_actions) >= res_len:
            return np.array([action[1] for action in user_actions[-res_len:]])
        else:
            n_special = res_len - len(user_actions)
            result_list = [self.padding_value] * n_special + [action[1] for action in user_actions]
            return np.array(result_list)


