import tensorflow as tf


class SelfAttentionEncoder:
    '''
    input_sentences_word_lv: 嵌入后的句子表示，应为单句化，shape=[1, sentence_length, word_embedding_dim]
    lstm_h_dim: 内部的lstm的输出dim
    hidden_unit_dim: 论文中的da
    embedding_row_cnt: 论文中的r，即注意力方面的数量
    '''
    def __init__(self, input_sentnece_word_lv, lstm_h_dim, hidden_unit_num, embedding_row_cnt):

        # graph
        with tf.name_scope("Self_attention_encoder"):
            # input
            with tf.name_scope("Input"):
                # shape = [1, sentence_length, word_embedding_dim]
                self.input_sentence_word_lv = input_sentnece_word_lv
                # shape = [1]
                self.input_sentence_length = tf.placeholder(dtype=tf.float32, shape=[1])

            # lstm
            with tf.name_scope("LSTM"):
                lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=lstm_h_dim)
                lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=lstm_h_dim)
                (outputs_fw, outputs_bw), _, _ = \
                    tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw,
                                                    inputs=self.input_sentence_word_lv)
                lstm_outputs = tf.concat([outputs_fw[0], outputs_bw[0]], axis=-1)

            # weights
            with tf.name_scope("Weights"):
                ws1 = tf.Variable(dtype=tf.float32,
                                  initial_value=tf.truncated_normal(shape=[hidden_unit_num, 2*lstm_h_dim]))
                ws2 = tf.Variable(dtype=tf.float32,
                                  initial_value=tf.truncated_normal(shape=[embedding_row_cnt, hidden_unit_num]))

            # attention
            with tf.name_scope("Attention"):
                # [hidden_unit_num, 2*lstm_h_dim] * [2*lstm_h_dim, sentence_length] =>
                # [hidden_unit_num, sentence_length]
                ws1_h = tf.tanh(tf.matmul(ws1, lstm_outputs, transpose_b=True))
                # [embedding_row_cnt, hidden_unit_num] * [hidden_unit_num, sentence_length] =>
                # [embedding_row_cnt, 2*hidden_unit_num]
                self.attention = tf.nn.softmax(tf.matmul(ws2, ws1_h), axis=-1)

            # output
            with tf.name_scope("Output"):
                self.output = tf.matmul(self.attention, lstm_outputs)

            # penalization
            with tf.name_scope("Penalization"):
                self.penalization = tf.norm(tf.subtract(tf.matmul(self.attention, self.attention, transpose_b=True),
                                                        tf.eye(num_rows=embedding_row_cnt)), ord="fro")
