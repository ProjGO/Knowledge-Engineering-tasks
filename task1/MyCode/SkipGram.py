from task1.MyCode.config import Config

import tensorflow as tf
import numpy as np
import collections
import os
import pickle
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

start_idx = 0


def word2vec():
    config = Config()
    data = []
    word2idx = {}

    # 读取数据
    with open(config.get_dataset_path()) as f:
        raw_data = f.read()
        raw_data = raw_data.split()
    print("Data size", len(raw_data))
    count = [('UNK', -1)]
    count.extend(collections.Counter(raw_data).most_common(config.vocab_size))
    for word, _ in count:
        word2idx[word] = len(word2idx)
    idx2word = {idx: word for word, idx in zip(word2idx.keys(), word2idx.values())}
    data = []
    for word in raw_data:
        data.append(word2idx.get(word, 0))
    print("Dataset preparation done")

    def generate_batch(batch_size=config.batch_size, window_size=config.window_size):
        global start_idx
        assert batch_size % (2 * window_size) == 0
        _has_one_epoch = False
        span = 2 * window_size + 1
        _inputs = np.ndarray(shape=batch_size, dtype=np.int32)
        _labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        cur_pos = 0
        for i in range(batch_size // (2 * window_size)):
            if start_idx + span >= len(data):
                start_idx = 0
                _has_one_epoch = True
            cur_word = data[start_idx + window_size]
            for j in range(start_idx, start_idx + span):
                if j == start_idx + window_size:
                    continue
                _inputs[cur_pos] = cur_word
                _labels[cur_pos, 0] = data[j]
                cur_pos += 1
            start_idx = start_idx + 1
        return _has_one_epoch, _inputs, _labels

    valid_examples = np.random.choice(config.valid_window, config.valid_size, replace=False)

    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[config.batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[config.batch_size, 1])
            validate_inputs = tf.constant(valid_examples, dtype=tf.int32)
        with tf.name_scope('vector_as_center_word'):
            center_vec = tf.Variable(name='Vi',
                                     initial_value=tf.random_uniform([config.vocab_size, config.embedding_size], -1.0, 1.0))
            center_vec_embedding = tf.nn.embedding_lookup(center_vec, train_inputs)
        if config.has_Vo:
            with tf.name_scope('vector_as_context_word'):
                # context_vec = tf.Variable(name='Vo',
                #                         initial_value=tf.random.uniform([config.vocab_size, config.embedding_size]))
                context_vec = tf.Variable(name='Vo',
                                          initial_value=tf.truncated_normal([config.vocab_size, config.embedding_size],
                                                                            stddev=1.0 / math.sqrt(config.embedding_size))
                                          )
        biases = tf.Variable(tf.zeros([config.vocab_size]))
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=context_vec if config.has_Vo else center_vec,
                    biases=biases,
                    labels=train_labels,
                    inputs=center_vec_embedding,
                    num_sampled=config.num_sampled,
                    num_classes=config.vocab_size))
        with tf.name_scope('optimizer'):
            # opt = tf.train.AdamOptimizer().minimize(loss)
            opt = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        with tf.name_scope('output_final_embeddings'):
            if config.using_Vo and config.using_Vi:
                final_embedding = tf.concat([center_vec, context_vec], axis=1)
            elif config.using_Vi:
                final_embedding = center_vec
            else:
                final_embedding = context_vec
            norm = tf.sqrt(tf.reduce_sum(tf.square(final_embedding), 1, keepdims=True))
            normalized_final_embedding = final_embedding / norm

        with tf.name_scope('validating'):
            validate_embeddings = tf.nn.embedding_lookup(normalized_final_embedding, validate_inputs)
            similarity = tf.matmul(validate_embeddings, normalized_final_embedding, transpose_b=True)

        tf.summary.scalar('loss', loss)
        merged_summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    num_steps = config.num_steps
    cur_step = 0
    cur_epoch = 0

    print_info_freq = 2000
    validate_freq = 10000

    with tf.Session(graph=graph) as sess:
        has_ckpt, log_dir = config.get_log_dir()
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        if has_ckpt:
            saver.restore(sess, tf.train.latest_checkpoint(log_dir))
            cur_step, cur_epoch = config.get_cur_step_epoch()
        else:
            init.run()
        print('Initialized')

        avg_loss = 0
        for step in range(cur_step, num_steps):
            has_one_epoch, batch_inputs, batch_labels = generate_batch()
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            run_metadata = tf.RunMetadata()
            _, summary, step_loss = sess.run([opt, merged_summary, loss],
                                             feed_dict=feed_dict,
                                             run_metadata=run_metadata)
            avg_loss += step_loss
            writer.add_summary(summary, step)
            if step == num_steps-1:
                writer.add_run_metadata(run_metadata, 'step:%d' % step)

            if step % print_info_freq == 0 and step > 0:
                avg_loss /= print_info_freq
                print('step: %d, Average loss for last %d steps: %f' % (step, print_info_freq, avg_loss))
                avg_loss = 0

            if step % validate_freq == 0 and step > 0:
                sim = similarity.eval()
                for i in range(config.valid_size):
                    validate_words = idx2word[valid_examples[i]]
                    top_k = 10
                    nearest = (-sim[i, :]).argsort()[0:top_k]
                    log_str = 'Nearest to %s:' % validate_words
                    for j in range(top_k):
                        log_str += " %s," % idx2word[nearest[j]]
                    print(log_str)

        final_embedding = normalized_final_embedding.eval()
        vec_list = []
        word_list = []
        for i in range(config.vocab_size):
            vec_list.append(final_embedding[i])
            word_list.append(idx2word[i])
        embedding_dict = {word: vec for word, vec in zip(word_list, vec_list)}
        with open(os.path.join(config.get_embedding_output_dir(), 'word2vec_%d.pkl' % num_steps), 'wb') as f:
            pickle.dump(embedding_dict, f, pickle.HIGHEST_PROTOCOL)

        config.write_config(cur_step, cur_epoch)
        saver.save(sess, os.path.join(log_dir, 'model.ckpt'))

        writer.close()

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_num = 500
    low_dim_embedding = tsne.fit_transform(final_embedding[:plot_num, :])
    labels = [idx2word[i] for i in range(plot_num)]
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embedding[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom'
        )
    plt.savefig(os.path.join(config.log_dir, 'tsne_%d.png' % num_steps))


def main(argv):
    word2vec()


if __name__ == "__main__":
    tf.app.run()
