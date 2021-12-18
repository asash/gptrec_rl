from aprec.losses.loss import Loss
from tensorflow.keras.losses import BinaryCrossentropy


class BCELoss(Loss):
    def __init__(self):
        super().__init__()
        self.loss = BinaryCrossentropy(from_logits=True)

    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)