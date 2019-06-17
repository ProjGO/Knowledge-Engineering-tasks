import numpy as np


def print_confusion_mat(confusion_mat):
    num_tags = np.shape(confusion_mat)[0]
    confusion_mat_percentage = np.zeros((num_tags, num_tags))

    tag_cnt_sum = [np.sum(confusion_mat[i]) for i in range(num_tags)]

    for i in range(num_tags):
        for j in range(num_tags):
            confusion_mat_percentage[i][j] = confusion_mat[i][j] / tag_cnt_sum[i]

    np.set_printoptions(suppress=True, precision=4, linewidth=200)
    print(confusion_mat_percentage)

    return confusion_mat_percentage
