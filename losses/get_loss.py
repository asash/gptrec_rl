from aprec.losses.bpr import BPRLoss
from aprec.losses.climf import CLIMFLoss
from aprec.losses.lambdarank import LambdaRankLoss
from aprec.losses.xendcg import XENDCGLoss


def get_loss(loss_name, items_num, batch_size, max_positives=40):
    if loss_name == 'lambdarank':
        return LambdaRankLoss(items_num, batch_size, ndcg_at=max_positives)
    if loss_name == 'xendcg':
        return XENDCGLoss(items_num, batch_size)
    if loss_name == 'bpr':
        return BPRLoss(max_positives)
    if loss_name == 'climf':
        return CLIMFLoss(batch_size, items_num, max_positives)
    else:
        return loss_name