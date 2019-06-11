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

    def build_graph(self):
        # 输入
        with tf.name_scope("Input"):
            # shape=[batch_size, max_sentence_length]
            self.input_word_idx_lv = tf.placeholder(tf.int32, shape=[None, None],
                                                    name="input_word_idx_lv")
            # shape=[batch_size, max_sentence_length, max_word_length] max均指整个batch范围内
            self.input_char_idx_lv = tf.placeholder(tf.int32, shape=[None, None, None],
                                                    name="input_char_idx_lv")
            # shape=[batch_size, max_sentence_length]
            self.label = tf.placeholder(tf.int32, shape=[None, None], name="input_label")

        # embedding
        with tf.name_scope("Embedding"):
            self.word_embedding_table = tf.Variable(initial_value=self.config.get_embedding_vec(),
                                                    trainable=self.config.word_embedding_trainable,
                                                    name="word_embedding_table")
            self.char_embedding_table = tf.Variable(initial_value=tf.truncated_normal(shape=[256, self.config.char_embedding_dim]),
                                                    trainable=self.config.char_embedding_trainable,
                                                    name="char_embedding_table")
            self.input_word_vec_lv = tf.nn.embedding_lookup(self.word_embedding_table,
                                                            self.input_word_idx_lv,
                                                            name="input_word_vec_lv")
            self.input_char_vec_lv = tf.nn.embedding_lookup(self.char_embedding_table,
                                                            self.input_char_idx_lv,
                                                            name="input_char_vec_lv")

        # LSTM
        with tf.name_scope("LSTM"):
            self.fw_lstm_cell = tf.nn.rnn_cell.LSTMCell()

