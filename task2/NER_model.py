import tensorflow as tf
import numpy as np
from task2.data_util import Dataset
from task2.config import Config


class NERModel:
    def __init__(self, config, train_dataset, validate_dataset, test_dataset):
        self.config = config
        self.train_dataset = train_dataset
        self.validate_dataset = validate_dataset
        self.test_dataset = test_dataset

        # 输入
        with tf.name_scope("Input"):
            # shape=[batch_size, max_sentence_length]
            self.input_word_idx_lv = tf.placeholder(tf.int32, shape=[None, None],
                                                    name="input_word_idx_lv")
            # shape=[batch_size, max_sentence_length, max_word_length] max均指整个batch范围内
            self.input_char_idx_lv = tf.placeholder(tf.int32, shape=[None, None, None],
                                                    name="input_char_idx_lv")
            # shape=[batch_size, max_sentence_length]
            self.labels = tf.placeholder(tf.int32, shape=[None, None], name="input_label")
            # shape=[batch_size]
            self.sentence_length = tf.placeholder(tf.int32, shape=[None], name="setence_length")
            # shape=[batch_size, max_sentence_length]
            self.word_length = tf.placeholder(tf.int32, shape=[None, None], name="word_length")

        # embedding
        with tf.name_scope("Embedding"):
            word_embedding_table = tf.Variable(initial_value=self.config.get_embedding_vec(),
                                               trainable=self.config.word_embedding_trainable,
                                               name="word_embedding_table", dtype=tf.float32)
            char_embedding_table = tf.Variable(initial_value=tf.truncated_normal(shape=[256, self.config.char_embedding_dim]),
                                               trainable=self.config.char_embedding_trainable,
                                               name="char_embedding_table", dtype=tf.float32)
            input_word_vec_lv = tf.nn.embedding_lookup(word_embedding_table,
                                                       self.input_word_idx_lv,
                                                       name="input_word_vec_lv",)
            input_char_vec_lv = tf.nn.embedding_lookup(char_embedding_table,
                                                       self.input_char_idx_lv,
                                                       name="input_char_vec_lv")

        # LSTM
        with tf.name_scope("LSTM"):
            with tf.name_scope("char_lv"):
                char_fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_dim,
                                                            num_proj=self.config.output_dim,
                                                            name="char_fw_lstm_cell")
                char_bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_dim,
                                                            num_proj=self.config.output_dim,
                                                            name="char_bw_lstm_cell")
                # (outputs, (output_state_fw, output_state_bw))
                s = tf.shape(input_char_vec_lv)
                input_char_vec_lv = tf.reshape(input_char_vec_lv, shape=[s[0]*s[1], s[-2], self.config.char_embedding_dim])
                self.word_length = tf.reshape(self.word_length, shape=[s[0]*s[1]])
                _, (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(char_fw_lstm_cell,
                                                                                        char_bw_lstm_cell,
                                                                                        inputs=input_char_vec_lv,
                                                                                        sequence_length=self.word_length,
                                                                                        dtype=tf.float32)
                char_lstm_output = tf.concat([output_state_bw, output_state_bw], axis=-1)
                input_word_vec_lv = tf.concat([input_word_vec_lv, char_lstm_output], axis=-1)
            with tf.name_scope("word_lv"):
                word_fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_dim,
                                                            num_proj=self.config.output_dim,
                                                            name="word_fw_lstm_cell")
                word_bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_dim,
                                                            num_proj=self.config.output_dim,
                                                            name="word_bw_lstm_cell")
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(word_fw_lstm_cell, word_bw_lstm_cell,
                                                                            inputs=input_word_vec_lv,
                                                                            sequence_length=self.sentence_length,
                                                                            dtype=tf.float32)
                word_lstm_output = tf.concat([output_fw, output_bw], axis=-1)

        # fc
        with tf.name_scope("full_connect"):
            w = tf.Variable(initial_value=tf.truncated_normal(shape=[2*self.config.output_dim, self.config.n_tags]),
                            dtype=tf.float32, name="w")
            b = tf.Variable(initial_value=tf.truncated_normal(shape=[self.config.n_tags]),
                            dtype=tf.float32, name="b")
            word_lstm_output = tf.reshape(word_lstm_output, shape=[-1, 2*self.config.output_dim])
            logits = tf.add(tf.matmul(word_lstm_output, w), b)
            s = tf.shape(self.input_word_idx_lv)
            logits = tf.reshape(logits, shape=[s[0], s[1], self.config.n_tags])

        # loss
        with tf.name_scope("loss"):
            if self.config.use_crf:
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    logits, self.labels, self.sentence_length)
                self.trans_params = trans_params
                self.loss = tf.reduce_mean(-log_likelihood)
            else:
                # s = tf.shape(self.labels)
                # self.labels = tf.reshape(self.labels, shape=[s[0]*s[1]])
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=self.labels)
                mask = tf.sequence_mask(self.sentence_length)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_mean(losses)

            tf.summary.scalar("loss", self.loss)

        # optimizer
        with tf.name_scope("optimizer"):
            opt = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

        # predict
        with tf.name_scope("predict"):
            self.labels_pred = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int32)

        print("graph initialization successful!")






