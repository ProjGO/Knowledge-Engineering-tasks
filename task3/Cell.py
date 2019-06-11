import tensorflow as tf


class RNNCell:
    def __init__(self,
                 input_x,
                 last_h,
                 layer_num,
                 activation="tanh",
                 ):
        self.input_dim = input_x.shape[0]
        self.state_dim = last_h.shape[0]

        with tf.name_scope('RNN_%d' % layer_num):
            self.whh = tf.Variable(initial_value=tf.truncated_normal(shape=[self.state_dim, self.state_dim]),
                                   name='Whh')
            self.bhh = tf.Variable(initial_value=tf.truncated_normal(shape=[self.state_dim, 1]),
                                   name='bhh')
            self.bhx = tf.Variable(initial_value=tf.truncated_normal(shape=[self.state_dim, 1]),
                                   name='bhx')
            self.whx = tf.Variable(initial_value=tf.truncated_normal(shape=[self.input_dim, self.state_dim]),
                                   name='Whx')

            self._h = tf.add(tf.add(tf.matmul(self.whh, last_h), self.bhh),
                             tf.add(tf.matmul(self.whx, input_x), self.bhx))
            self.h = tf.tanh(self._h)


class LSTMCell:
    tf.nn.rnn_cell.LSTMCell()
    pass
