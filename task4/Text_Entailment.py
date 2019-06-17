import tensorflow as tf
from task3.SelfAttention import SelfAttentionEncoder
from task4.config import Config
from task4.embedding_dict import EmbeddingDict


class TextEntailmentModel:
    def __init__(self, config, word_embedding, train_dataset, validate_dataset, test_dataset):
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
            self.input_labels = tf.placeholder(dtype=tf.int8, shape=[None], name="input_labels")

            batch_size = tf.shape(self.input_hypo_word_lv)[0]

        # Self Attention Encoder
        Mh, penalization = \
            SelfAttentionEncoder(self.input_hypo_word_lv, self.input_hypo_lengths,
                                 config.self_attention_lstm_output_dim,
                                 config.self_attention_hidden_unit_num,
                                 config.attention_hop)
        Mp, _ = \
            SelfAttentionEncoder(self.input_prem_word_lv, self.input_prem_lengths,
                                 config.self_attention_lstm_output_dim,
                                 config.self_attention_hidden_unit_num,
                                 config.attention_hop)

        with tf.name_scope("Gated_Encoder"):
            Wfh = tf.Variable(initial_value=tf.truncated_normal(shape=[config.attention_hop,
                                                                       2*config.self_attention_lstm_output_dim,
                                                                       config.w_dim]))
            Wfp = tf.Variable(initial_value=tf.truncated_normal(shape=[config.attention_hop,
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

            # elementwise multiply
            # [batch_size, attention_hop, w_dim]
            Fr = tf.multiply(Fh, Fp)

        with tf.name_scope("FC"):
            w = tf.Variable(initial_value=tf.truncated_normal(shape=[config.attention_hop*config.w_dim, 3]))
            b = tf.Variable(initial_value=tf.truncated_normal(shape=[3]))
            w = tf.tile(w[None], [batch_size, 1, 1])
            b = tf.tile(b[None], [batch_size, 1])
            Fr = tf.expand_dims(tf.layers.flatten(Fr), 1)
            logits = tf.add(tf.matmul(Fr, w), b)
            logits = tf.reshape(logits, shape=[batch_size, 3])

        with tf.name_scope("Predict"):
            self.batch_predict = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int8)

        with tf.name_scope("Loss"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.input_labels) + penalization

        with tf.name_scope("Optimizer"):
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)

        # summary & saver & writer
        self.step_loss_summary_ph = tf.placeholder(dtype=tf.float32)
        self.step_accuracy_summary_ph = tf.placeholder(dtype=tf.float32)
        step_loss_summary = tf.summary.scalar("step_loss", self.step_loss_summary_ph)
        step_accuracy_summary = tf.summary.scalar("step_accuracy", self.step_accuracy_summary_ph)
        self.merged_summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.init = tf.initialize_all_variables()

    def train(self, epoch_num):
        pass


if __name__ == "__main__":
    config = Config()
    word_embedding = EmbeddingDict(config.pkl_file_path)
    a = TextEntailmentModel(config, word_embedding, 1, 2, 3)

