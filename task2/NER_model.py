import tensorflow as tf
from task2.data_util import *
import numpy as np
# from tensorflow.python import debug as tf_debug


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
            char_embedding_table = tf.Variable(initial_value=tf.truncated_normal(shape=[256, self.config.char_embedding_dim], stddev=1),
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
                stacked_cell_fw = []
                stacked_cell_bw = []
                for i in range(self.config.lstm_layer):
                    stacked_cell_fw.append(tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_dim))
                    stacked_cell_bw.append(tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_dim))
                mcell_fw = tf.nn.rnn_cell.MultiRNNCell(stacked_cell_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_cell_bw)
                '''char_fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_dim,
                                                            num_proj=self.config.output_dim,
                                                            name="char_fw_lstm_cell")
                char_bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_dim,
                                                            num_proj=self.config.output_dim,
                                                            name="char_bw_lstm_cell")'''
                # s=[batch_size, max_sentence_length, max_word_length, char_embedding_dim]
                s = tf.shape(input_char_vec_lv)
                input_char_vec_lv = tf.reshape(input_char_vec_lv,
                                               shape=[s[0]*s[1], s[-2], self.config.char_embedding_dim])
                reshaped_word_length = tf.reshape(self.word_length, shape=[s[0]*s[1]])
                # (outputs, ((output_state_fw_c, output_state_fw_h), (output_state_bw_c, output_state_bw_h)))
                '''_, ((_, output_state_fw_h), (_, output_state_bw_h)) \
                    = tf.nn.bidirectional_dynamic_rnn(char_fw_lstm_cell, char_bw_lstm_cell,
                                                      inputs=input_char_vec_lv,
                                                      sequence_length=reshaped_word_length,
                                                      dtype=tf.float32)'''
                _, (output_state_fw, output_state_bw) \
                    = tf.nn.bidirectional_dynamic_rnn(mcell_fw, mcell_bw,
                                                      inputs=input_char_vec_lv,
                                                      sequence_length=reshaped_word_length,
                                                      dtype=tf.float32)
                (_, output_state_fw_h) = output_state_fw[self.config.lstm_layer-1]
                (_, output_state_bw_h) = output_state_bw[self.config.lstm_layer-1]
                char_lstm_output = tf.reshape(tf.concat([output_state_fw_h, output_state_bw_h], axis=-1), shape=[s[0], s[1], 2*self.config.output_dim])
                input_word_vec_lv = tf.concat([input_word_vec_lv, char_lstm_output], axis=-1)
            with tf.name_scope("word_lv"):
                stacked_cell_fw_word_lv = []
                stacked_cell_bw_word_lv = []
                for i in range(self.config.lstm_layer):
                    stacked_cell_fw_word_lv.append(tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_dim))
                    stacked_cell_bw_word_lv.append(tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_dim))
                '''mcell_fw_word_lv = tf.nn.rnn_cell.MultiRNNCell(stacked_cell_fw_word_lv)
                mcell_bw_word_lv = tf.nn.rnn_cell.MultiRNNCell(stacked_cell_bw_word_lv)
                word_fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_dim,
                                                            num_proj=self.config.output_dim,
                                                            name="word_fw_lstm_cell")
                word_bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_dim,
                                                            num_proj=self.config.output_dim,
                                                            name="word_bw_lstm_cell")'''
                '''(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(mcell_fw_word_lv, mcell_bw_word_lv,
                                                                            inputs=input_word_vec_lv,
                                                                            sequence_length=self.sentence_length,
                                                                            dtype=tf.float32)
                output_fw = output_fw[self.config.lstm_layer-1]
                output_bw = output_bw[self.config.lstm_layer-1]
                word_lstm_output = tf.concat([output_fw, output_bw], axis=-1)'''
                (word_lstm_output, _, _) = \
                    tf.contrib.rnn.stack_bidirectional_dynamic_rnn(stacked_cell_fw_word_lv, stacked_cell_bw_word_lv,
                                                                   inputs=input_word_vec_lv,
                                                                   sequence_length=self.sentence_length,
                                                                   dtype=tf.float32)
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
            self.logits = logits

        # loss
        with tf.name_scope("loss"):
            if self.config.use_crf:
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    logits, self.labels, self.sentence_length)
                self.trans_params = trans_params
                self.loss = tf.reduce_mean(-log_likelihood)
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=self.labels)
                mask = tf.sequence_mask(self.sentence_length, tf.shape(self.input_word_idx_lv)[1])
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_mean(losses)

        # optimizer
        with tf.name_scope("optimizer"):
            # self.opt = tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)

        # predict
        with tf.name_scope("predict"):
            # shape=[batch_size, max_sentence_len]
            self.batch_pred = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int32)

        # summary & saver & writer
        self.step_loss_summary_ph = tf.placeholder(dtype=tf.float32)
        self.step_accuracy_summary_ph = tf.placeholder(dtype=tf.float32)
        step_loss_summary = tf.summary.scalar("step_loss", self.step_loss_summary_ph)
        step_accuracy_summary = tf.summary.scalar("step_accuracy", self.step_accuracy_summary_ph)
        self.merged_summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.init = tf.initialize_all_variables()

        print("building graph successfully")

    @staticmethod
    def get_batch_accuracy(batch_pred, batch_label, sentence_length):
        true_pred = 0
        total_pred = 0
        tag_error = np.zeros((9, 9))
        sentence_cnt = batch_label.shape[0]
        for i in range(sentence_cnt):
            total_pred += sentence_length[i]
            for j in range(sentence_length[i]):
                if batch_pred[i][j] == batch_label[i][j]:
                    true_pred += 1
                tag_error[batch_label[i][j]][batch_pred[i][j]] += 1

        return true_pred / total_pred, tag_error

    def train(self, num_epoch):
        if self.config.log_dir_exist:
            start_epoch, start_step = self.config.get_cur_epoch_and_step()  # 》》》》》》》》》》》》》》》》》》》》
            # start_epoch, start_step = 5, 700
            print("previous log_dir found")
        else:
            start_epoch, start_step = 1, 1
        cur_epoch = start_epoch
        cur_step = start_step
        accuracy_sum = 0
        loss_sum = 0
        with tf.Session() as sess:
            if self.config.log_dir_exist:
                self.saver.restore(sess, tf.train.latest_checkpoint(self.config.log_dir))
            else:
                self.init.run()
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            summary_writer = tf.summary.FileWriter(self.config.log_dir, sess.graph)
            while cur_epoch - start_epoch <= num_epoch:
                cur_step += 1
                has_one_epoch, batch_data, batch_label = self.train_dataset.get_one_batch()
                sentences_length, padded_sentences_word_lv, word_lengths, \
                    padded_sentences_char_lv, padded_label = Dataset.batch_padding(batch_data, batch_label)
                if has_one_epoch:
                    cur_epoch += 1
                feed_dict = {self.input_word_idx_lv: padded_sentences_word_lv,
                             self.sentence_length: sentences_length,
                             self.input_char_idx_lv: padded_sentences_char_lv,
                             self.word_length: word_lengths,
                             self.labels: padded_label}
                _, step_loss, batch_pred = sess.run([self.opt, self.loss, self.batch_pred],
                                                    feed_dict=feed_dict)

                step_accuracy = self.get_batch_accuracy(batch_pred, padded_label, sentences_length)
                loss_sum += step_loss
                accuracy_sum += step_accuracy
                merged_summary = sess.run(self.merged_summary, feed_dict={self.step_loss_summary_ph: step_loss,
                                                                          self.step_accuracy_summary_ph: step_accuracy})
                summary_writer.add_summary(merged_summary, cur_step)

                if cur_step % self.config.print_freq == 0 and cur_step > 0:
                    accuracy = accuracy_sum / self.config.print_freq
                    loss = loss_sum / self.config.print_freq
                    accuracy_sum = 0
                    loss_sum = 0
                    print("step %d, average loss: %f, average accuracy: %f" % (cur_step, loss, accuracy))

            self.saver.save(sess, os.path.join(self.config.log_dir, "ner_model.ckpt"), global_step=cur_step)
            self.config.write_config()
            self.config.write_epoch_and_step(cur_epoch, cur_step)

    def predict_sentence(self):
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint(self.config.log_dir))
            while True:
                print("ready to predict")
                in_sentence = input()
                in_word_lv, sentence_length, in_char_lv, word_length = process_input(self.config, in_sentence)
                # print(in_word_lv, sentence_length, in_char_lv, word_length)
                feed_dict = {self.input_word_idx_lv: in_word_lv,
                             self.input_char_idx_lv: in_char_lv,
                             self.sentence_length: sentence_length,
                             self.word_length: word_length}
                if self.config.use_crf:
                    viterbi_sequences = []
                    logits, trans_params = sess.run([self.logits, self.trans_params], feed_dict=feed_dict)
                    # logits = tf.reshape(batch_pred, [-1, self.config.batch_size, 9])
                    for logit, sequence_length in zip(logits, sentence_length):
                        logit = logit[:sequence_length]
                        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                        viterbi_sequences += [viterbi_seq]
                    pred = viterbi_sequences
                else:
                    pred = sess.run(self.batch_pred, feed_dict=feed_dict)
                pred = pred[0]
                pred_labels = []
                for i in range(len(pred)):
                    pred_labels.append(self.config.idx2label[pred[i]])
                print(pred)
                print(pred_labels)
                print('\n')

    def test(self):
        accuracy = 0
        n_step = 0
        has_one_epoch = False
        tag_error_sum = np.zeros((9, 9))
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint(self.config.log_dir))
            while not has_one_epoch:
                has_one_epoch, batch_data, batch_label = self.test_dataset.get_one_batch()
                sentences_length, padded_sentences_word_lv, word_lengths, \
                padded_sentences_char_lv, padded_label = Dataset.batch_padding(batch_data, batch_label)
                # print(padded_sentences_word_lv, padded_sentences_char_lv, sentences_length, word_lengths)
                feed_dict = {
                    self.input_word_idx_lv: padded_sentences_word_lv,
                    self.input_char_idx_lv: padded_sentences_char_lv,
                    self.sentence_length: sentences_length,
                    self.word_length: word_lengths}
                if self.config.use_crf:
                    viterbi_sequences = []
                    logits, trans_params = sess.run([self.logits, self.trans_params], feed_dict=feed_dict)
                    # logits = tf.reshape(batch_pred, [-1, self.config.batch_size, 9])
                    for logit, sequence_length in zip(logits, sentences_length):
                        logit = logit[:sequence_length]
                        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                        viterbi_sequences += [viterbi_seq]
                    pred = viterbi_sequences
                else:
                    pred = sess.run(self.batch_pred, feed_dict=feed_dict)
                # for i in range(self.config.batch_size):
                    # print(pred[i])
                    # print(padded_label[i])
                    # print('\n')
                step_accuracy, tag_error = self.get_batch_accuracy(pred, padded_label, sentences_length)
                n_step += 1
                accuracy += step_accuracy
                tag_error_sum += tag_error
        accuracy /= n_step
        np.set_printoptions(suppress=True)
        print(tag_error_sum)
        for i in range(9):
            print("%d:%d" % (i, tag_error_sum[i][0] + tag_error_sum[i][1] + tag_error_sum[i][2] +
                             tag_error_sum[i][3] + tag_error_sum[i][4] + tag_error_sum[i][5] +
                             tag_error_sum[i][6] + tag_error_sum[i][7] + tag_error_sum[i][8] -
                             tag_error_sum[i][i]))
        return accuracy
