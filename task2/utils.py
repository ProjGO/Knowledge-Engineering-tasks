import numpy as np


def normalize_confusion_mat(confusion_mat):
    num_tags = np.shape(confusion_mat)[0]
    confusion_mat_normalized = np.zeros((num_tags, num_tags))

    tag_cnt_sum = [np.sum(confusion_mat[i]) for i in range(num_tags)]

    for i in range(num_tags):
        for j in range(num_tags):
            confusion_mat_normalized[i][j] = confusion_mat[i][j] / tag_cnt_sum[i]

    true_pred = 0
    total_pred = 0
    for i in range(1, num_tags):
        total_pred += tag_cnt_sum[i]
        true_pred += confusion_mat[i][i]

    np.set_printoptions(suppress=True, precision=4, linewidth=200)
    # print(confusion_mat_normalized)
    # print("accuracy: %f" % (true_pred / total_pred))

    return confusion_mat_normalized


def get_batch_accuracy(batch_pred, batch_label, sentence_length):
    true_pred = 0
    total_pred = 0
    confusion_matrix = np.zeros((9, 9))
    sentence_cnt = batch_label.shape[0]
    for i in range(sentence_cnt):
        total_pred += sentence_length[i]
        for j in range(sentence_length[i]):
            if batch_pred[i][j] == batch_label[i][j]:
                true_pred += 1
            confusion_matrix[batch_label[i][j]][batch_pred[i][j]] += 1

    return true_pred / total_pred, confusion_matrix


def concat_pred_and_labels(batch_pred, batch_label, sentence_length):
    concated_preds = []
    concated_labels = []
    for i in range(len(batch_label)):
        concated_labels.extend(batch_label[i][0:sentence_length[i]])
        concated_preds.extend(batch_pred[i][0:sentence_length[i]])
    return concated_preds, concated_labels
