import tensorflow as tf
class Loss():
    def __init__(self, num_items=None, batch_size=None):
        self.num_items = num_items
        self.batch_size = batch_size

    def __call__(self, y_true, y_pred):
        raise NotImplementedError

    def set_num_items(self, num_items):
        self.num_items = num_items

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

class ListWiseLoss(Loss):
    @tf.custom_gradient
    def loss_per_list(self, y_true, y_pred):
        with tf.GradientTape() as g:
            g.watch(y_pred)
            ignore_mask = tf.cast(y_true == -100, 'float32') #-100 is the default ignore value
            use_mask = 1.0 - ignore_mask
            noise =  ignore_mask * tf.random.uniform(y_pred.shape, 0.0, 1.0) 
            listwise_ytrue = use_mask * tf.cast(y_true, 'float32') + noise
            listwise_loss = self.calc_per_list(listwise_ytrue, y_pred)
            use_loss_mask = tf.squeeze(use_mask[:,:1], axis=1)
            average_loss =  tf.reduce_sum(listwise_loss * use_loss_mask) / tf.reduce_sum(use_loss_mask)
            loss_grads = g.gradient(average_loss, y_pred) / self.num_items

        def grad(dy): #ensure that we don't utilize gradients for ignored items 
            return 0*dy, dy * use_mask * loss_grads

        return average_loss, grad


    def calc_per_list(self, y_true, y_pred):
        raise NotImplementedError
