import tensorflow as tf
from task3.SelfAttention import SelfAttentionEncoder


class TextEntailmentModel:
    def __init__(self, train_dataset, validate_dataset, test_dataset, word_embedding, config):
        self.train_dataset = train_dataset
        self.validate_dataset = validate_dataset
        self.test_dataset = test_dataset

        with tf.name_scope("Input"):
            self.input_hypo_word_lv = tf.placeholder(dtype=tf.float32,
                                                     shape=[None, None, word_embedding.embedding_dim])
            self.input_hypo_lengths = tf.placeholder(dtype=tf.int32, shape=[None])
            self.input_prem_word_lv = tf.placeholder(dtype=tf.float32,
                                                     shape=[None, None, word_embedding.embedding_dim])
            self.input_prem_lengths = tf.placeholder(dtype=tf.int32, shape=[None])

        self_attention_encoder_hypo = SelfAttentionEncoder(self.input_hypo_word_lv, self.input_hypo_lengths,
                                                           config.self_attention_lstm_output_dim,
                                                           config.self_attention_hidden_unit_num,
                                                           config.self_attention_row_cnt)
        self_attention_encoder_prem = SelfAttentionEncoder(self.input_prem_lengths, self.input_prem_lengths,
                                                           config.self_attention_lstm_output_dim,
                                                           config.self_attention_hidden_unit_num,
                                                           config.self_attention_row_cnt)



