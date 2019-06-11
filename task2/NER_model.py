import numpy as np
import os
import tensorflow as tf


from .data_util import Dataset
from .data_util import pad_sequences
from .general_utils import Progbar
from .base_model import BaseModel


class NERModel(BaseModel):

    def __init__(self, config):
        super(NERModel, self).__init__(config)

    def add_placeholders(self):
        #词id
        self.word_ids = tf.placeholder(tf.int32, shape=[None,None])
        #句子长度
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None])
        #字母id
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None])
        #单词长度
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None])
        #标签
        self.labels = tf.placeholder(tf.int32,shape=[None,None])
        #超参数
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[])
        self.lr = tf.placeholder(dtype=tf.float32, shape=[])

    def dict(self, words, labels=None, lr=None, dropout=None):
        #感觉将向量化的词输入NER的关键应该是在这里，也许应该把输出分割一下，分别对应这里面的各个变量？
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)
        feed = {self.word_ids: word_ids, self.sequence_lengths: sequence_lengths}
        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths
        labels, _ = pad_sequences(labels, 0)
        feed[self.labels] = labels
        feed[self.lr] = lr
        feed[self.dropout] = dropout
        return feed, sequence_lengths

    def word_embedding(self):
        #得到词的向量表示？在已经做好的wordembedding上查询现有词？
        L = tf.Variable(Dataset.get_embedding(Dataset), dtype=tf.float32,trainable=False)
        pretrained_embeddings = tf.nn.embedding_lookup(L, self.word_ids)
        #感觉是对字母做类似的事情，获得字母的向量表示
        K = tf.get_variable(name="char_embeddings", dtype=tf.float32, shape=[300, self.config.dim_char])
        char_embeddings = tf.nn.embedding_lookup(K, self.char_ids)
        #这里我没太明白。。似乎是在为了能够使用dynamic_rnn而在调整向量的形状？
        s = tf.shape(char_embeddings)
        char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], s[-1]])
        word_lengths = tf.reshape(self.word_lengths, shape=[-1])
        #双向lstm
        cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size, state_is_tuple=True)
        _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,  cell_bw, char_embeddings, sequence_length=word_lengths-50,  dtype=tf.float32)
        #拼起来获得lstm的输出
        output = tf.concat([output_fw, output_bw], axis=-1)
        char_rep = tf.reshape(output, shape=[-1, s[1], 2*self.config.char_hidden_size])
        #字母的输出和词的输出拼起来得到最终的输出？
        self.word_embeddings = tf.concat([pretrained_embeddings, char_rep], axis=-1)

    def logits(self):
        #对每一个单词，应用lstm，获得包含上下文信息的输出
        cell_fw = tf.contrib.rnn.LSTMCell(hidden_size)
        cell_bw = tf.contrib.rnn.LSTMCell(hidden_size)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_embeddings, sequence_length=sequence_lengths, dtype=tf.float32)
        context_rep = tf.concat([output_fw, output_bw], axis=-1)
        #每个词的得分向量s[i]=W·h+b（词s标记成第i个tag的得分）
        W = tf.get_variable("W", shape=[2 * self.config.hidden_size, self.config.ntags], dtype=tf.float32)
        b = tf.get_variable("b", shape=[self.config.ntags], dtype=tf.float32, initializer=tf.zeros_initializer())
        ntime_steps = tf.shape(context_rep)[1]
        context_rep_flat = tf.reshape(context_rep, [-1, 2 * self.config.hidden_size])
        pred = tf.matmul(context_rep_flat, W) + b
        self.scores = tf.reshape(pred, [-1, ntime_steps, ntags])

    def loss(self):
        #计算损失
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.scores, self.labels, self.sequence_lengths)
        self.loss = tf.reduce_mean(-log_likelihood)

    def predict(self,words):
        #根据结果预测出对应的tag
        labels_pred = tf.cast(tf.argmax(self.logits, axis=-1))

    def build(self):
        #似乎是运行的意思？
        self.add_placeholders()
        self.word_embedding()
        self.logits()
        #self.add_pred_op()
        self.loss()
        self.add_train_op(self.lr, self.loss, self.config.clip)
        self.initialize_session()
