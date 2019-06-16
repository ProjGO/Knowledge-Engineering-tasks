import tensorflow as tf
from task3.SelfAttention import SelfAttentionEncoder
from task4.config import Config
from task4.embedding_dict import EmbeddingDict


class TextEntailmentModel:
    def __init__(self, train_dataset, validate_dataset, test_dataset, word_embedding, config):
        self.train_dataset = train_dataset
        self.validate_dataset = validate_dataset
        self.test_dataset = test_dataset

        with tf.name_scope("Input"):
            self.input_hypo_word_lv = tf.placeholder(dtype=tf.float32, name="input_hypothesis_word_lv",
                                                     shape=[None, None, word_embedding.embedding_dim])
            self.input_hypo_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="input_hypothesis_lengths")
            self.input_prem_word_lv = tf.placeholder(dtype=tf.float32, name="input_premise_word_lv",
                                                     shape=[None, None, word_embedding.embedding_dim])
            self.input_prem_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="input_premise_lengths")

            batch_size = tf.shape(self.input_hypo_word_lv)[0]

        Mh, penalization = \
            SelfAttentionEncoder(self.input_hypo_word_lv, self.input_hypo_lengths,
                                 config.self_attention_lstm_output_dim,
                                 config.self_attention_hidden_unit_num,
                                 config.self_attention_embedding_row_cnt)
        Mp, _ = \
            SelfAttentionEncoder(self.input_prem_word_lv, self.input_prem_lengths,
                                 config.self_attention_lstm_output_dim,
                                 config.self_attention_hidden_unit_num,
                                 config.self_attention_embedding_row_cnt)

        with tf.name_scope("Gated_Encoder"):

            Wfh = tf.Variable(initial_value=tf.truncated_normal(shape=[config.self_attention_embedding_row_cnt,
                                                                       2*config.self_attention_lstm_output_dim,
                                                                       config.w_dim]))
            Wfp = tf.Variable(initial_value=tf.truncated_normal(shape=[config.self_attention_embedding_row_cnt,
                                                                       2*config.self_attention_lstm_output_dim,
                                                                       config.w_dim]))
            # [attention_hop, 2 * lstm_output_dim, w_dim] => [batch_size, attention_hop, 2 * lstm_output_dim, w_dim]
            # 应该是直接复制了batch_size次
            Wfh = tf.tile(Wfh[None], [batch_size, 1, 1, 1])
            Wfp = tf.tile(Wfp[None], [batch_size, 1, 1, 1])

            # [batch_size, attention_hop, 2*lstm_output_dim] => [batch_size, attention_hop, 1, 2*lstm_output_dim]
            # 用于让batch中每个Mh的每一行分别与对应的矩阵相乘
            Mh = tf.expand_dims(Mh, 2)
            # [batch_size, attention_hop, 1, 2*lstm_output_dim] * [batch_size, attention_hop, 2*lstm_output_dim, w_dim]
            # => [batch_size, attention_hop, 1, w_dim]
            # squeeze => [batch_size, attention_hop, w_dim]
            # [i, j, :(len=1), :(len=2*lstm_output_dim)] * [i, j, :(len=2*lstm_output_dim), :]
            # => [i, j, :(len=1), :(len=w_dim)]
            Fh = tf.squeeze(tf.matmul(Mh, Wfh), 2)
            Mp = tf.expand_dims(Mp, 2)
            Fp = tf.squeeze(tf.matmul(Mp, Wfp), 2)

            Fr = tf.multiply(Fh, Fp)


if __name__ == "__main__":
    config = Config()
    word_embedding = EmbeddingDict(config.pkl_file_path)
    a = TextEntailmentModel(1, 2, 3, word_embedding, config)

