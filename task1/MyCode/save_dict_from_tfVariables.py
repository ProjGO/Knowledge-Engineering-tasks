import tensorflow as tf
import pickle
import os

'''
Sqrt:0
truediv:0
'''

filepath = 'D:/ML/Ckpts/KnowledgeEngineering_task1/log'
meta_filename = 'model.ckpt.meta'
vocab_filename = 'metadata.tsv'

if __name__ == '__main__':
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(os.path.join(filepath, meta_filename))
            saver.restore(sess, tf.train.latest_checkpoint(filepath))
            embeddings = graph.get_tensor_by_name('embeddings/Variable:0')
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True))
            normalize_embeddings = embeddings / norm
            final_embeddings = normalize_embeddings.eval()

    print(type(final_embeddings))
    print(final_embeddings.shape[0])
    print(final_embeddings.shape)
    embedding_list = []
    for i in range(final_embeddings.shape[0]):
        embedding_list.append(final_embeddings[i])
    with open(os.path.join(filepath, vocab_filename), mode='r') as f:
        words = f.read().split()
        word2vec_pair = zip(words, embedding_list)
        word2vec = {word: vec for word, vec in word2vec_pair}

    with open(os.path.join(filepath, 'word2vec.pkl'), 'wb') as f:
        pickle.dump(word2vec, f, pickle.HIGHEST_PROTOCOL)





