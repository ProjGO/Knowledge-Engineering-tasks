import json
import os
import copy
import numpy as np

class Dataset:
    label2idx = {"netural": 0, "entailment": 1, "contradiction": 2}  # 三种标签

    def __init__(self, config, name):  # 变量基本上和task2完全一样
        self.cur_idx = 0
        self.name = name
        self.config = config
        self.batch_size = config.batch_size
        self.sentences1 = []
        self.sentences2 = []
        self.idxed_sentences1 = []
        self.idxed_sentences2 = []
        self.idxed_labels = []
        self.labels = []
        self.sentences_cnt = 0
        self.embedding = {}
        self.word2idx = {}
        self.idx2word = {}
        data = open(os.path.join(config.dataset_path, name + '.jsonl')).readlines()  # 打开json文件
        #data = json.loads(data[0])
        sentence1_buffer = []
        sentence2_buffer = []
        label_buffer = []
        temp_data = []
        for i in range(0, len(data)):
            temp_data = json.loads(data[i])  # 每一句话先用json.loads变成字典
            sentence1_buffer.append(temp_data("sentence1"))  # 取出第一句话
            sentence2_buffer.append(temp_data("sentence2"))  # 取出第二句话
            self.sentences_cnt += 1
            self.sentences1.append(sentence1_buffer)
            self.sentences2.append(sentence2_buffer)
            self.labels.append(temp_data("gold_label"))  # 我个人理解gold_label应该是正确的关系，不太确定。。。
        print("Loading dataset %s done, %d sentences." % (self.name, self.sentences_cnt))
        self.convert_word_and_label_to_idx()  # 将词和标签转换为数字表示

    def convert_word_and_label_to_idx(self):  # 和task2基本一样，只是分别对两个句子做
        self.embedding = self.config.get_embedding()
        self.word2idx = self.config.get_word2idx()
        self.idx2word = {idx: word for word, idx in zip(self.word2idx.keys(), self.word2idx.values())}
        unk_cnt = 0
        for sentence1, sentence2, labels in zip(self.sentences1, self.sentences2, self.labels)
            cur_sentence1 = []
            cur_sentence2 = []
            cur_label = []
            for word in sentence1:
                word = word.lower()
                idx = self.word2idx.get(word, 0)
                cur_sentence1.append(idx)
                if idx == 0:
                    unk_cnt += 1
            for word in sentence2:
                word = word.lower()
                idx = self.word2idx.get(word, 0)
                cur_sentence2.append(idx)
                if idx == 0:
                    unk_cnt += 1
            for label in labels:
                cur_label.append(self.label2idx[label])
            self.idxed_sentences1.append(cur_sentence1)
            self.idxed_sentences2.append(cur_sentence2)
            self.idxed_labels.append(cur_label)
        print("converted words, labels to indexes, %d unknown words in %s" % (unk_cnt, self.name))

    def get_one_batch(self):  # 基本同task2，只是变成对两个句子做
        has_one_epoch = False
        batch_data1 = []
        batch_data2 = []
        batch_label = []
        for i in range(self.batch_size):
            cur_idxed_sentence1 = copy.deepcopy(self.idxed_sentences1[self.cur_idx])
            cur_idxed_sentence2 = copy.deepcopy(self.idxed_sentences2[self.cur_idx])
            cur_label = copy.deepcopy(self.idxed_labels[self.cur_idx])
            batch_data1.append(cur_idxed_sentence1)
            batch_data2.append(cur_idxed_sentence2)
            batch_label.append(cur_label)
            self.cur_idx = self.cur_idx + 1
            if self.cur_idx == self.sentences_cnt - 1:
                has_one_epoch = True
                self.cur_idx = 0
        return has_one_epoch, batch_data1, batch_data2, batch_label

    @staticmethod
    def batch_padding(in_batch_data1, in_batch_data2, in_batch_label, pad_tok=0):  # 也是基本和task2相同
        sentences1_in_word = []
        sentences2_in_word = []
        max_sentence1_len = 0
        max_sentence2_len = 0
        max_sentence_len = 0
        sentences1_length = []
        sentences2_length = []
        for sentence in in_batch_data1:
            cur_sentence_in_word = sentence
            sentences1_in_word.append(cur_sentence_in_word)
            sentences1_length.append(len(cur_sentence_in_word))
            if len(cur_sentence_in_word) > max_sentence1_len:
                max_sentence1_len = len(cur_sentence_in_word)
        for sentence in in_batch_data2:
            cur_sentence_in_word = sentence
            sentences2_in_word.append(cur_sentence_in_word)
            sentences2_length.append(len(cur_sentence_in_word))
            if len(cur_sentence_in_word) > max_sentence2_len:
                max_sentence2_len = len(cur_sentence_in_word)
        max_sentence_len = max(max_sentence1_len, max_sentence2_len)
        for i in range(len(sentences1_in_word)):
            sentences1_in_word[i] = list(sentences1_in_word[i])
            while len(sentences1_in_word[i]) < max_sentence_len:
                sentences1_in_word[i].append(pad_tok)
        for i in range(len(sentences2_in_word)):
            sentences2_in_word[i] = list(sentences2_in_word[i])
            while len(sentences2_in_word[i]) < max_sentence_len:
                sentences2_in_word[i].append(pad_tok)
        for i in range(len(in_batch_label)):
            while(len(in_batch_label[i])) < max_sentence_len:
                in_batch_label[i].append(pad_tok)
        padded_sentences1_word_lv = np.array(sentences1_in_word)
        padded_sentences2_word_lv = np.array(sentences2_in_word)
        padded_label = np.array(in_batch_label)
        return sentences1_length, sentences2_length, padded_sentences1_word_lv, padded_sentences2_word_lv, padded_label




