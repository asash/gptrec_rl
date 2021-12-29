from aprec.recommenders.dnn_sequential_recommender.targetsplitters.targetsplitter import TargetSplitter


class LastItemSplitter(TargetSplitter):
    def __init__(self) -> None:
        super().__init__()
    
    def split(self, sequence):
        if len(sequence) == 0:
            return [], []
        train = sequence[:-1]
        target = sequence[-1:]
        return train, target