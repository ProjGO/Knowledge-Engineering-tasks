import tensorflow as tf
from task3.SelfAttention import SelfAttentionEncoder
from task4.config import Config
from task4.Dataset import *
from task4.utils import *


class TextEntailmentModel:
    def __init__(self, config, word_embedding, train_dataset, validate_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.validate_dataset = validate_dataset
        self.test_dataset = test_dataset
        self.config = config

        with tf.name_scope("Input"):
            self.input_hypo_word_lv = tf.placeholder(dtype=tf.int32, name="input_hypothesis_word_lv",
                                                     shape=[self.config.batch_size, None])
            self.input_hypo_lengths = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size],
                                                     name="input_hypothesis_lengths")
            self.input_prem_word_lv = tf.placeholder(dtype=tf.int32, name="input_premise_word_lv",
                                                     shape=[self.config.batch_size, None])
            self.input_prem_lengths = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size],
                                                     name="input_premise_lengths")

            self.input_labels = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size], name="input_labels")
            batch_size = tf.shape(self.input_hypo_word_lv)[0]

        with tf.name_scope("Embedding_lookup"):
            word_embedding_table = tf.Variable(initial_value=word_embedding.get_embedding_vec(),
                                               trainable=self.config.word_embedding_trainable,
                                               name="word_embedding_table", dtype=tf.float32)
            input_hypo_word_vec_lv = tf.nn.embedding_lookup(word_embedding_table,
                                                            self.input_hypo_word_lv,
                                                            name="input_hypo_word_vec")
            input_prem_word_vec_lv = tf.nn.embedding_lookup(word_embedding_table,
                                                            self.input_prem_word_lv,
                                                            name="input_prem_word_vec")

        # Self Attention Encoder
        Mh, penalization = \
            SelfAttentionEncoder(input_hypo_word_vec_lv, self.input_hypo_lengths,
                                 self.config.self_attention_lstm_output_dim,
                                 self.config.self_attention_hidden_unit_num,
                                 self.config.attention_hop)
        Mp, _ = \
            SelfAttentionEncoder(input_prem_word_vec_lv, self.input_prem_lengths,
                                 self.config.self_attention_lstm_output_dim,
                                 self.config.self_attention_hidden_unit_num,
                                 self.config.attention_hop)

        with tf.name_scope("Gated_Encoder"):
            Wfh = tf.Variable(initial_value=tf.truncated_normal(shape=[self.config.attention_hop,
                                                                       2*config.self_attention_lstm_output_dim,
                                                                       self.config.w_dim]))
            Wfp = tf.Variable(initial_value=tf.truncated_normal(shape=[self.config.attention_hop,
                                                                       2*config.self_attention_lstm_output_dim,
                                                                       self.config.w_dim]))
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
            w = tf.Variable(initial_value=tf.truncated_normal(shape=[self.config.attention_hop*config.w_dim, 3]))
            b = tf.Variable(initial_value=tf.truncated_normal(shape=[3]))
            # w = tf.tile(w[None], [batch_size, 1, 1])
            # b = tf.tile(b[None], [batch_size, 1])
            # Fr = tf.expand_dims(tf.layers.flatten(Fr), 1)
            Fr = tf.layers.flatten(Fr)
            logits = tf.add(tf.matmul(Fr, w), b)[0]
            logits = tf.reshape(logits, shape=[batch_size, 3])

        with tf.name_scope("Predict"):
            self.batch_predict = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int8)

        with tf.name_scope("Loss"):
            self.loss = \
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.input_labels) + penalization

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
        start_epoch, start_step = self.config.get_cur_epoch_and_step()
        cur_epoch = start_epoch
        cur_step = start_step
        accuracy_sum = 0
        loss_sum = 0
        with tf.Session() as sess:
            if self.config.ckpt_exists:
                self.saver.restore(sess, tf.train.latest_checkpoint(self.config.log_dir))
            else:
                self.init.run()
            summary_writer = tf.summary.FileWriter(self.config.log_dir, sess.graph)
            while cur_epoch - start_epoch < epoch_num:
                cur_step += 1
                has_one_epoch, batch_data_prem, batch_data_hypo, batch_labels = self.train_dataset.get_one_batch()
                prems_length, hypos_length, padded_prems_word_lv, padded_hypos_word_lv, batch_labels = \
                    Dataset.batch_padding(batch_data_prem, batch_data_hypo, batch_labels)
                if has_one_epoch:
                    cur_epoch += 1
                feed_dict = {
                    self.input_hypo_word_lv: padded_hypos_word_lv,
                    self.input_hypo_lengths: hypos_length,
                    self.input_prem_word_lv: padded_prems_word_lv,
                    self.input_prem_lengths: prems_length,
                    self.input_labels: batch_labels
                }
                _, step_loss, batch_pred = sess.run([self.opt, self.loss, self.batch_predict],
                                                    feed_dict=feed_dict)
                step_accuracy, _ = get_batch_accuracy(batch_pred, batch_labels)
                loss_sum += step_loss
                accuracy_sum += step_accuracy
                merged_summary = sess.run(self.merged_summary, feed_dict={self.step_loss_summary_ph: step_loss,
                                                                          self.step_accuracy_summary_ph: step_accuracy})
                summary_writer.add_summary(merged_summary, cur_step)

                if cur_step % config.print_freq == 0 and cur_step > 0:
                    accuracy = accuracy_sum / config.print_freq
                    loss = loss_sum / config.print_freq
                    accuracy_sum = 0
                    loss_sum = 0
                    print("step %d, average loss: %f, average accuracy: %f" % (cur_step, loss, accuracy))
            self.saver.save(sess, os.path.join(config.log_dir, "TextEntailment_model.ckpt"), global_step=cur_step)
            config.write_config()
            config.write_epoch_and_step(cur_epoch, cur_step)


if __name__ == "__main__":
    config = Config()
    word_embedding = EmbeddingDict(config.pkl_file_path)
    a = TextEntailmentModel(config, word_embedding, 1, 2, 3)

