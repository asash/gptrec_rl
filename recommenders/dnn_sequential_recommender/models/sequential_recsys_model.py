#base class for many sequential recsys models

class SequentialRecsysModel(object):
    def __init__(self, output_layer_activation, embedding_size, max_history_len):
        self.max_history_length = max_history_len
        self.embedding_size = embedding_size
        self.output_layer_activation = output_layer_activation
        self.requires_user_id = False
        self.num_items = None
        self.num_users = None
        self.batch_size = None

    def set_common_params(self, num_items, num_users,batch_size):
        self.num_items = num_items
        self.num_users = num_users
        self.batch_size = batch_size

    def get_model(self):
        raise NotImplementedError
