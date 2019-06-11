import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    a = np.array([[1, 1, 3], [1, 2, 3], [2, 3]])
    b = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    print(b)
    ids = tf.constant(value=a)
    emb = tf.constant(value=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]))
    lukup = tf.nn.embedding_lookup(emb, ids)

    with tf.Session() as sess:
        print(lukup.eval())
