import tensorflow as tf
import numpy as np
from tensorflow import keras as K

if __name__ == "__main__":
    a = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
    b = np.array([[[1], [1], [1], [1]], [[2], [2], [2], [2]], [[3], [3], [3], [3]]])

    aa = tf.Variable(initial_value=a)
    bb = tf.Variable(initial_value=b)

    c = tf.Variable(initial_value=np.array([[1, 1], [2, 2]]))
    d = tf.Variable(initial_value=np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]]))

    # cc = tf.matmul(aa, bb)

    with tf.Session() as sess:
        with tf.device("cpu:0"):
            tf.initialize_all_variables().run()
            '''aa = tf.reshape(aa, shape=[3, 1, 4])
            print(aa.eval())
            ans = []
            for i in range(a.shape[0]):
                ans.append(tf.matmul(aa[i], bb[i]))
            anss = ans[0]
            for i in range(len(ans) - 1):
                anss = tf.concat((anss, ans[i+1]), axis=0)
            print(anss.eval())'''
            print(tf.reshape(d, shape=[2, 4]).eval())
            print(tf.transpose(d).eval())
            print(tf.concat([d[i] for i in range(d.shape[0])], axis=0).eval())
            print(tf.matmul(c, tf.concat([d[i] for i in range(d.shape[0])], axis=1)).eval())
            dc = tf.reshape(tf.reshape(d, [-1, 2]) @ c, shape=[-1, 2, 2])
            print(dc.eval())
            print(c[None].eval())
            print((tf.tile(c[None], [2, 1, 1]) @ d).eval())
            # print(tf.concat([d[i] for i in range(d.shape[0])], axis=2).eval())
            # print(cc.eval())
