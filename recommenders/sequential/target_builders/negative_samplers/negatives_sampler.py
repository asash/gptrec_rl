class NegativesSampler(object):
    def default_vector(self):
        return self.def_vector

    def set_n_items(self, n):
        self.n_items = n
        self.def_vector = [n+2] * self.sample_size 

   
    def sample_negatives(self, positive):
        raise NotImplementedError()

    def get_sample_size(self):
        return self.sample_size

    def set_train_sequences(self, train_sequences):
        pass

