import tensorflow as tf
from task3.SelfAttention import SelfAttentionEncoder
from task4.config import Config
from task4.Dataset import *
from task4.utils import *


class TextEntailmentModelOnlyLSTM:
    def __init__(self, config, word_embedding, train_dataset, validate_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.validate_dataset = validate_dataset
        self.test_dataset = test_dataset
        self.config = config

        with tf.name_scope("Input"):
            self.input_prem_word_lv = tf.placeholder(dtype=tf.int32, name="input_premise_word_lv",
                                                     shape=[self.config.batch_size, None])
            self.input_prem_lengths = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size],
                                                     name="input_premise_lengths")
            self.input_hypo_word_lv = tf.placeholder(dtype=tf.int32, name="input_hypothesis_word_lv",
                                                     shape=[self.config.batch_size, None])
            self.input_hypo_lengths = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size],
                                                     name="input_hypothesis_lengths")

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
        lstm_dim = 300
        with tf.name_scope("prem_LSTM_encoder"):
            lstm_cell_prem_fw = tf.nn.rnn_cell.LSTMCell(num_units=lstm_dim, name="prem_fw")
            lstm_cell_prem_bw = tf.nn.rnn_cell.LSTMCell(num_units=lstm_dim, name="prem_bw")
            _, ((prem_fw, _), (prem_bw, _)) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_prem_fw, lstm_cell_prem_bw,
                                                                              inputs=input_prem_word_vec_lv,
                                                                              sequence_length=self.input_prem_lengths,
                                                                              dtype=tf.float32)
            prem_encoded = tf.concat([prem_fw, prem_bw], axis=-1)

        with tf.name_scope("hypo_LSTM_encoder"):
            lstm_cell_hypo_fw = tf.nn.rnn_cell.LSTMCell(num_units=lstm_dim, name="hypo_fw")
            lstm_cell_hypo_bw = tf.nn.rnn_cell.LSTMCell(num_units=lstm_dim, name="hypo_bw")
            _, ((hypo_fw, _), (hypo_bw, _)) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_hypo_fw, lstm_cell_hypo_bw,
                                                                              inputs=input_hypo_word_vec_lv,
                                                                              sequence_length=self.input_hypo_lengths,
                                                                              dtype=tf.float32)
            hypo_encoded = tf.concat([hypo_fw, hypo_bw], axis=-1)

        final_encoded = tf.concat([prem_encoded, hypo_encoded], axis=-1)

        with tf.name_scope("FC"):
            w = tf.Variable(initial_value=tf.truncated_normal(shape=[4*lstm_dim, 3]))
            b = tf.Variable(initial_value=tf.truncated_normal(shape=[3]))
            # w = tf.tile(w[None], [batch_size, 1, 1])
            # b = tf.tile(b[None], [batch_size, 1])
            # Fr = tf.expand_dims(tf.layers.flatten(Fr), 1)
            logits = tf.add(tf.matmul(final_encoded, w), b)
            logits = tf.reshape(logits, shape=[batch_size, 3])

        with tf.name_scope("Predict"):
            self.batch_predict = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int8)

        with tf.name_scope("Loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.input_labels)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("Optimizer"):
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)
            # self.opt = tf.train.AdagradOptimizer(0.01).minimize(self.loss)

        # summary & saver & writer
        self.step_loss_summary_ph = tf.placeholder(dtype=tf.float32)
        self.step_accuracy_summary_ph = tf.placeholder(dtype=tf.float32)
        step_loss_summary = tf.summary.scalar("step_loss", self.step_loss_summary_ph)
        step_accuracy_summary = tf.summary.scalar("step_accuracy", self.step_accuracy_summary_ph)
        self.merged_summary = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=3)
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
                print("graph initialized from checkpoint")
            else:
                self.init.run()
                print("graph initialized")
            summary_writer = tf.summary.FileWriter(self.config.log_dir, sess.graph)
            while cur_epoch - start_epoch < epoch_num:
                cur_step += 1
                has_one_epoch, batch_data_prem, batch_data_hypo, batch_labels = self.train_dataset.get_one_batch()
                prems_length, hypos_length, padded_prems_word_lv, padded_hypos_word_lv, batch_labels = \
                    Dataset.batch_padding(batch_data_prem, batch_data_hypo, batch_labels)
                if has_one_epoch:
                    cur_epoch += 1
                feed_dict = {
                    self.input_prem_word_lv: padded_prems_word_lv,
                    self.input_prem_lengths: prems_length,
                    self.input_hypo_word_lv: padded_hypos_word_lv,
                    self.input_hypo_lengths: hypos_length,
                    self.input_labels: batch_labels
                }
                _, step_loss, batch_pred = sess.run([self.opt, self.loss, self.batch_predict],
                                                    feed_dict=feed_dict)
                # print("loss: %f penalization_h: %f penalization_p: %f" % (step_loss, penalization_h, penalization_p))
                step_accuracy, _ = get_batch_accuracy(batch_pred, batch_labels)
                loss_sum += step_loss
                accuracy_sum += step_accuracy
                merged_summary = sess.run(self.merged_summary, feed_dict={self.step_loss_summary_ph: step_loss,
                                                                          self.step_accuracy_summary_ph: step_accuracy})
                summary_writer.add_summary(merged_summary, cur_step)

                if cur_step % self.config.test_freq == 0 and cur_step > 0:
                    self.validate(sess)
                if cur_step % self.config.print_freq == 0 and cur_step > 0:
                    accuracy = accuracy_sum / self.config.print_freq
                    loss = loss_sum / self.config.print_freq
                    accuracy_sum = 0
                    loss_sum = 0
                    print("epoch: %d, step %d, average loss: %f, average accuracy: %f" %
                          (cur_epoch, cur_step, loss, accuracy))
            self.saver.save(sess, os.path.join(self.config.log_dir, "TextEntailment_model.ckpt"), global_step=cur_step)
            self.config.write_config()
            self.config.write_epoch_and_step(cur_epoch, cur_step)

    def validate(self, sess):
        accuracy = 0
        n_step = 0
        has_one_epoch = False
        confusion_mat = np.zeros((3, 3))
        while not has_one_epoch:
            has_one_epoch, batch_data_prem, batch_data_hypo, batch_labels = self.validate_dataset.get_one_batch()
            prems_length, hypos_length, padded_prems_word_lv, padded_hypos_word_lv, batch_labels = \
                Dataset.batch_padding(batch_data_prem, batch_data_hypo, batch_labels)
            feed_dict = {
                self.input_hypo_word_lv: padded_hypos_word_lv,
                self.input_hypo_lengths: hypos_length,
                self.input_prem_word_lv: padded_prems_word_lv,
                self.input_prem_lengths: prems_length
            }
            pred = sess.run(self.batch_predict, feed_dict=feed_dict)
            step_accuracy, step_confusion_mat = get_batch_accuracy(pred, batch_labels)
            n_step += 1
            accuracy += step_accuracy
            confusion_mat += step_confusion_mat
        accuracy /= n_step
        print("validate accuracy: %f" % accuracy)
        print(confusion_mat)
        return accuracy, confusion_mat

    def test(self):
        accuracy = 0
        n_step = 0
        has_one_epoch = False
        confusion_mat = np.zeros((3, 3))
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint(self.config.log_dir))
            while not has_one_epoch:
                has_one_epoch, batch_data_prem, batch_data_hypo, batch_labels = self.test_dataset.get_one_batch()
                prems_length, hypos_length, padded_prems_word_lv, padded_hypos_word_lv, batch_labels = \
                    Dataset.batch_padding(batch_data_prem, batch_data_hypo, batch_labels)
                feed_dict = {
                    self.input_hypo_word_lv: padded_hypos_word_lv,
                    self.input_hypo_lengths: hypos_length,
                    self.input_prem_word_lv: padded_prems_word_lv,
                    self.input_prem_lengths: prems_length
                }
                pred = sess.run(self.batch_predict, feed_dict=feed_dict)
                step_accuracy, step_confusion_mat = get_batch_accuracy(pred, batch_labels)
                n_step += 1
                accuracy += step_accuracy
                confusion_mat += step_confusion_mat
        accuracy /= n_step
        print(accuracy)
        print(confusion_mat)
        return accuracy, confusion_mat


if __name__ == "__main__":
    config = Config()
    word_embedding = EmbeddingDict(config.pkl_file_path)
    a = TextEntailmentModel(config, word_embedding, 1, 2, 3)

