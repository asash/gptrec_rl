from aprec.losses.loss import ListWiseLoss, Loss
import tensorflow as tf


class ItemsMaksingLossProxy(Loss):
    def __init__(self, listwise_loss: ListWiseLoss, negatives_per_positve, sequence_length, num_items=None, batch_size=None):
        super().__init__(num_items, batch_size)
        self.listwise_loss = listwise_loss
        self.negatives_per_positive = negatives_per_positve
        self.sequence_length = sequence_length
        self.listwise_loss.set_num_items(negatives_per_positve + 1)
        self.less_is_better = listwise_loss.less_is_better
        self.__name__ = self.listwise_loss.__name__ + "_proxy"

    def set_batch_size(self, batch_size):
        super().set_batch_size(batch_size)
        self.listwise_loss.set_batch_size(self.batch_size * self.sequence_length)

    def __call__(self, y_true, y_pred):
        ytrue_reshaped = tf.reshape(y_true, (self.batch_size * self.sequence_length, self.negatives_per_positive + 1))
        ypred_reshaped = tf.cast(tf.reshape(y_pred, (self.batch_size * self.sequence_length, self.negatives_per_positive + 1)), 'float32')
        result =  self.listwise_loss.loss_per_list(ytrue_reshaped, ypred_reshaped)
        return result
    
