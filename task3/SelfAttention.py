import tensorflow as tf


class SelfAttentionEncoder:
    '''
    input_sentences_word_lv: 嵌入后的句子表示，应为单句，shape=[1, sentence_length, word_embedding_dim]
    lstm_h_dim: 内部的lstm的输出dim
    hidden_unit_dim: 论文中的da
    embedding_row_cnt: 论文中的r，即注意力方面的数量


    输出：
    output  shape=[batch_size, embedding_row_cnt, hidden_unit_num]
    '''
    def __init__(self, input_sentnece_word_lv, input_sentence_lengths,
                 lstm_output_dim, hidden_unit_num, embedding_row_cnt, name="Self_attention_encoder"):
        batch_size = input_sentence_lengths.shape[0]
        # graph
        with tf.name_scope(name):
            # input
            with tf.name_scope("Input"):
                # shape = [batch_size, sentence_length, word_embedding_dim]
                self.input_sentence_word_lv = input_sentnece_word_lv
                # shape = [None]
                self.input_sentence_lengths = input_sentence_lengths

            # lstm
            with tf.variable_scope("LSTM", reuse=True):
                lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=lstm_output_dim)
                lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=lstm_output_dim)
                (outputs_fw, outputs_bw), _, _ = \
                    tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw,
                                                    inputs=self.input_sentence_word_lv,
                                                    sequence_length=self.input_sentence_lengths,)
                lstm_outputs = tf.concat([outputs_fw[0], outputs_bw[0]], axis=-1)

            # weights
            with tf.variable_scope("Weights", reuse=True):
                ws1 = tf.get_variable(dtype=tf.float32,
                                      initial_value=tf.truncated_normal(shape=[hidden_unit_num, 2 * lstm_output_dim]))
                ws2 = tf.get_variable(dtype=tf.float32,
                                      initial_value=tf.truncated_normal(shape=[embedding_row_cnt, hidden_unit_num]))

            # attention
            with tf.name_scope("Attention"):
                # [batch_size, hidden_unit_num, 2 * lstm_h_dim] * [batch_size, 2 * lstm_h_dim, sentence_length] =>
                # [batch_size, hidden_unit_num, sentence_length]
                ws1_h = tf.tanh(tf.matmul(tf.tile(ws1[None], [batch_size, 1, 1]), lstm_outputs, transpose_b=True))
                # [batch_size, embedding_row_cnt, hidden_unit_num] * [batch_size, hidden_unit_num, sentence_length] =>
                # [batch_size, embedding_row_cnt, 2 * hidden_unit_num]
                self.attention = tf.nn.softmax(tf.matmul(tf.tile(ws2[[None], [batch_size, 1, 1]]), ws1_h), axis=-1)

            # output
            with tf.name_scope("Output"):
                self.output = tf.matmul(self.attention, lstm_outputs)

            # penalization
            with tf.name_scope("Penalization"):
                self.penalization = tf.norm(tf.subtract(tf.matmul(self.attention, self.attention, transpose_b=True),
                                                        tf.eye(num_rows=embedding_row_cnt)), ord="fro")
