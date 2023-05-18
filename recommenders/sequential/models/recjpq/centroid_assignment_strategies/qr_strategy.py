import math
from .centroid_strategy import CentroidAssignmentStragety
import numpy as np

class QuotientRemainder(CentroidAssignmentStragety):
    def assign(self, train_users):
        assert(self.item_code_bytes==2)
        assignments = []
        diviser = math.ceil(math.sqrt(self.num_items+1))
        for i in range(self.num_items + 1):
            quotient = i // diviser
            remainder = i % diviser
            assignments.append([quotient, remainder])
        return np.array(assignments)
        