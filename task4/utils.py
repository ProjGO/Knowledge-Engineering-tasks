import numpy as np


def get_batch_accuracy(batch_pred, batch_label):
    total_pred = batch_pred.shape[0]
    true_pred = 0
    batch_confusion_mat = np.zeros((3, 3))
    for i in range(total_pred):
        if batch_pred[i] == batch_label[i]:
            true_pred += 1
        batch_confusion_mat[batch_label[i]][batch_pred[i]] += 1

    return true_pred / total_pred, batch_confusion_mat
