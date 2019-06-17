import tensorflow as tf


# class ttentionEncoder:
'''
input_sentences_word_lv: 嵌入后的句子表示，应为单句，shape=[1, sentence_length, word_embedding_dim]
lstm_h_dim: 内部的lstm的输出dim
hidden_unit_dim: 论文中的da
embedding_row_cnt: 论文中的r，即注意力方面的数量


输出：
output  shape=[batch_size, embedding_row_cnt, hidden_unit_num]
'''


def SelfAttentionEncoder(input_sentence_word_vec_lv, input_sentence_lengths,
                         lstm_output_dim, hidden_unit_num, attention_hop, name="attention_encoder"):
    s = tf.shape(input_sentence_word_vec_lv)
    batch_size = s[0]
    # batch_size = 3
    # graph
    with tf.name_scope(name):
        # input
        with tf.name_scope("Input"):
            # shape = [batch_size, sentence_length, word_embedding_dim]
            input_sentence_word_vec_lv = input_sentence_word_vec_lv
            # shape = [None]
            input_sentence_lengths = input_sentence_lengths

        # lstm
        with tf.variable_scope("LSTM", reuse=tf.AUTO_REUSE):
            lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=lstm_output_dim)
            lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=lstm_output_dim)
            (outputs_fw, outputs_bw), _ = \
                tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw,
                                                inputs=input_sentence_word_vec_lv,
                                                sequence_length=input_sentence_lengths,
                                                dtype=tf.float32)
            lstm_outputs = tf.concat([outputs_fw, outputs_bw], axis=-1)
            lstm_outputs = tf.reshape(lstm_outputs, shape=[s[0], s[1], 2*lstm_output_dim])

        # weights
        with tf.variable_scope("Weights", reuse=tf.AUTO_REUSE):
            ws1 = tf.get_variable(dtype=tf.float32, shape=[hidden_unit_num, 2 * lstm_output_dim], name="Ws1",
                                  initializer=tf.truncated_normal_initializer())
            ws2 = tf.get_variable(dtype=tf.float32, shape=[attention_hop, hidden_unit_num], name="Ws2",
                                  initializer=tf.truncated_normal_initializer())

        # attention
        with tf.name_scope("Attention"):
            # [batch_size, hidden_unit_num, 2 * lstm_h_dim] * [batch_size, 2 * lstm_h_dim, sentence_length] =>
            # [batch_size, hidden_unit_num, sentence_length]
            ws1_h = tf.tanh(tf.matmul(tf.tile(ws1[None], [batch_size, 1, 1]), lstm_outputs, transpose_b=True))
            # [batch_size, embedding_row_cnt, hidden_unit_num] * [batch_size, hidden_unit_num, sentence_length] =>
            # [batch_size, embedding_row_cnt, sentence_length]
            attention = tf.matmul(tf.tile(ws2[None], [batch_size, 1, 1]), ws1_h)
            attention = tf.nn.softmax(attention, axis=-1)

        # output
        with tf.name_scope("Output"):
            # [batch_size, embedding_row_cnt, sentence_length] * [batch_size, sentence_length, 2 * lstm_output_dim] =>
            # [batch_size, embedding_row_cnt, 2 * lstm_output_dim]
            output = tf.matmul(attention, lstm_outputs)

        # penalization
        with tf.name_scope("Penalization"):
            penalization = tf.norm(tf.subtract(tf.matmul(attention, attention, transpose_b=True),
                                               tf.eye(num_rows=attention_hop)), ord="euclidean")
        
        return output, penalization
