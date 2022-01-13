from typing import Sequence
import unittest
import numpy as np
import random
from collections import Counter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import LastItemSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.random_fraction_splitter import RandomFractionSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import RecencySequenceSampling
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter 

class TestItemSplitters(unittest.TestCase):
    def test_last_item_splitter(self):
        sequence=[1, 2, 3, 4, 5, 6, 7, 8, 9] 
        splitter = LastItemSplitter()
        for i in range(100):
            train, label = splitter.split(sequence)
            self.assertEquals(len(train), 8)
            self.assertEquals(len(label), 1)
            self.assertEquals(label,[9])
        single_item_sequence = [1]
        train, label = splitter.split(single_item_sequence )
        self.assertEquals(len(train), 0)
        self.assertEquals(label, [1])

    def test_random_fraction(self):
        sequence=list(range(5))
        splitter = RandomFractionSplitter()
        target_lens = []
        N = 100000 
        target_lens = Counter() 
        for i in range(N):
            train, target = splitter.split(sequence)
            self.assertEquals(len(train) + len(target), len(sequence))
            self.assertEquals(train, list(range(len(train))))
            target_range = list(range(len(sequence) - len(target), len(sequence)))
            self.assertEquals(target, target_range)
            target_lens[len(target)] += 1
        
        for i in range(1, len(sequence)):
            cnt = target_lens[i]
            expected = N / (len(sequence) - 1)
            self.assertAlmostEquals(abs(cnt - expected) / expected, 0.0, places=1)

    def test_recency_sequence_sampling_exp(self):
        sequence=list(range(5))
        bias = 0.5
        random.seed(31337)
        np.random.seed(31338)
        splitter = RecencySequenceSampling(max_pct=0.0, recency_importance= lambda n, k: 0.5 ** (n - k))
        N = 100000 
        target_counts = Counter() 
        for i in range(N):
            train, target = splitter.split(sequence)
            self.assertEquals(len(target), 1)
            self.assertEquals(len(train), len(sequence) -1 )
            target_counts[target[0]] += 1
        for i in range(len(sequence) - 2, 0, -1):
            cnt = target_counts[i]
            plus_one_cnt = target_counts[i+1]
            rel = cnt/plus_one_cnt
            expected = bias
            self.assertAlmostEquals(rel, expected, places=1)

    def test_recency_sequence_sampling(self):
        splitter = RecencySequenceSampling(max_pct=0.2)
        N = 10000 
        target_counts = Counter() 
        target_lens = Counter()
        for i in range(N):
            sequence = list(range(1, 31))
            train, targets = splitter.split(sequence)
            self.assertEquals(len(targets) + len(train), len(sequence))
            for target in targets:
                target_counts[target] +=1
            target_lens[len(targets)] +=1
        self.assertEquals(target_lens.most_common(), [(5, 4173), (4, 3300), (6, 1577), (3, 881), (2, 69)])
        self.assertEquals(target_counts.most_common(5),[(30, 7411), (29, 6553), (28, 5699), (27, 4777), (26, 3977)])

    def test_shifted_sequence_splitter(self):
        sequence = [1, 2, 3, 4, 5]
        splitter = ShiftedSequenceSplitter()
        train, label = splitter.split(sequence)
        self.assertEquals(train, [1, 2, 3, 4])
        self.assertEquals(label, [2, 3, 4, 5])
        splitter = ShiftedSequenceSplitter(max_len=2)

        train, label = splitter.split(sequence)
        self.assertEquals(train, [3, 4])
        self.assertEquals(label, [4, 5])

if __name__ == "__main__":
    unittest.main()
