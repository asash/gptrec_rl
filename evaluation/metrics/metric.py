class Metric(object):
    less_is_better = False
    def __call__(self, recommendations, actual):
        raise NotImplementedError
