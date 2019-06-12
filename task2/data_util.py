import numpy as np
import copy
import os


class Dataset:
    label2idx = {"O": 0, "B-ORG": 1, "I-ORG": 2, "B-PER": 3, "I-PER": 4,
                 "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8}
    idx2label = {idx: label for idx, label in zip(label2idx.keys(), label2idx.values())}
    char2idx = {}
    ch = chr(0)
    for i in range(256):
        char2idx[chr(ord(ch)+i)] = len(char2idx)
    sentences = []  # 列表的列表,里面的每个列表是一句话,再里面是每个单词
    idxed_sentences = []  # 将单词转换为输入embedding中的索引后的句子
    char_idxed_sentences = []  # 将单词拆成字符编码表示
    labels = []  # 列表的列表，对应着句子中的标签
    idxed_labels = []  # 转换为对应编号后的标签
    embedding = {}  # 预训练的word embedding, 格式word:vec,第一个是UNK(所有在embedding中找不到的都认为是UNK)
    word2idx = {}  # word:index
    idx2word = {}  # index:word
    sentences_cnt = 0  # 一共有几句话
    dataset_word_cnt = 0  # 一共有几个词

    cur_idx = 0  # 下一个batch的开始是哪一句(sentences中的下标)

    def __init__(self, config, name):
        self.config = config
        self.name = name  # 数据集名称(train/validate/test)
        self.batch_size = config.batch_size

        # 读入原始数据
        sentence_buffer = []
        label_buffer = []
        with open(os.path.join(config.dataset_path, self.name+".txt"), 'r') as f:
            for line in f:
                if line.startswith("-DOCSTART-"):
                    continue
                line = line.split()
                if len(line) == 0:
                    if len(sentence_buffer) != 0:
                        self.sentences.append(sentence_buffer)
                        self.labels.append(label_buffer)
                        self.sentences_cnt += 1
                    sentence_buffer = []
                    label_buffer = []
                else:
                    sentence_buffer.append(line[0])
                    label_buffer.append(line[3])
                    self.dataset_word_cnt += 1
        print("Loading dataset %s done, %d sentences, %d words in total."
              % (self.name, self.sentences_cnt, self.dataset_word_cnt))

        self.convert_word_and_label_to_idx()  # 将词和标签转换为数字表示
        self.convert_char_to_idx()  # 将词转换为字符级别表示,表示字符的数字为其ASCII码

    def convert_word_and_label_to_idx(self):
        # self.config.load_embedding()
        self.embedding = self.config.get_embedding()
        self.word2idx = self.config.get_word2idx()
        self.idx2word = {idx: word for word, idx in zip(self.word2idx.keys(), self.word2idx.values())}
        unk_cnt = 0
        for sentence, labels in zip(self.sentences, self.labels):
            cur_sentence = []
            cur_label = []
            for word in sentence:
                word = word.lower()  # word embedding中的所有词都是小写
                idx = self.word2idx.get(word, 0)  # 如果找不到返回0,0就是UNK
                cur_sentence.append(idx)
                if idx == 0:
                    unk_cnt += 1
            for label in labels:
                cur_label.append(self.label2idx[label])
            self.idxed_sentences.append(cur_sentence)
            self.idxed_labels.append(cur_label)
        print("converted words, labels to indexes, %d unknown words in %s" % (unk_cnt, self.name))

    def convert_char_to_idx(self):
        for sentence in self.sentences:
            sentence_buffer = []
            for word in sentence:
                word_buffer = []
                for ch in word:
                    word_buffer.append(self.char2idx[ch])
                sentence_buffer.append(word_buffer)
            self.char_idxed_sentences.append(sentence_buffer)

    '''
    has_one_epoch: 是否走完了一个epoch
    batch_data: [[(单词编号,[该单词每个字母的编号]),...(一句话的每个词)],...(这个batch中的每一句话)]
    batch_label: [[标签编号,...(一句话中每个词的标签)],...(这个batch中的每一句话)]
    '''
    def get_one_batch(self):
        has_one_epoch = False
        batch_data = []
        batch_label = []
        for i in range(self.batch_size):
            cur_idxed_sentence = copy.deepcopy(self.idxed_sentences[self.cur_idx])
            cur_char_idxed_sentence = copy.deepcopy(self.char_idxed_sentences[self.cur_idx])
            cur_label = copy.deepcopy(self.idxed_labels[self.cur_idx])  # 不copy就去世<-----一上午的发现
            batch_data.append(list(zip(cur_idxed_sentence, cur_char_idxed_sentence)))
            batch_label.append(cur_label)
            self.cur_idx = self.cur_idx + 1
            if self.cur_idx == self.sentences_cnt - 1:
                has_one_epoch = True
                self.cur_idx = 0
        return has_one_epoch, batch_data, batch_label

    '''
    句子长度pad至最长句长度
    单词长度pad至最长单词长度(指整个batch中的最长单词)
    label pad至最长句子长度
    返回:
    sentence_length: [一句话实际长度, ...]
    padded_sentences_in_word: [[一句话每个词的id, 0(padding), ...], ...]
    word_length: [[一句话每个单词的实际长度, ...], ...]
    padded_sentences_in_char: [[[一个单词的字符id, ..., 0(padding), ...], ..., [0(padding)， ...]], ...]
    padded_label: [[一个词的标签, ..., 0(padding), ...], ...]
    padded_word_length: [[一句中每个单词长度, ..., 0(padding), ...], ...]
    '''
    @staticmethod
    def batch_padding(in_batch_data, in_batch_label, pad_tok=0):
        sentences_in_word = []
        sentences_in_char = []
        max_word_len = 0
        max_sentence_len = 0
        sentences_length = []
        padded_word_lengths = []
        for sentence in in_batch_data:
            cur_sentence_in_word, cur_sentence_in_char = zip(*sentence)
            sentences_in_word.append(cur_sentence_in_word)
            sentences_in_char.append(cur_sentence_in_char)
            sentences_length.append(len(cur_sentence_in_word))
            if len(cur_sentence_in_word) > max_sentence_len:
                max_sentence_len = len(cur_sentence_in_word)
            word_length_in_cur_sentence = []
            for word in cur_sentence_in_char:
                word_length_in_cur_sentence.append(len(word))
                if len(word) > max_word_len:
                    max_word_len = len(word)
            padded_word_lengths.append(word_length_in_cur_sentence)
        for i in range(len(sentences_in_word)):
            sentences_in_word[i] = list(sentences_in_word[i])
            sentences_in_char[i] = list(sentences_in_char[i])
            while len(sentences_in_word[i]) < max_sentence_len:
                sentences_in_word[i].append(pad_tok)
            while len(sentences_in_char[i]) < max_sentence_len:
                sentences_in_char[i].append([pad_tok])
            while len(padded_word_lengths[i]) < max_sentence_len:
                padded_word_lengths[i].append(0)
            for j in range(len(sentences_in_char[i])):
                while len(sentences_in_char[i][j]) < max_word_len:
                    sentences_in_char[i][j].append(pad_tok)
        for i in range(len(in_batch_label)):
            while(len(in_batch_label[i])) < max_sentence_len:
                in_batch_label[i].append(pad_tok)
        padded_sentences_word_lv = np.array(sentences_in_word)
        padded_sentences_char_lv = np.array(sentences_in_char)
        padded_label = np.array(in_batch_label)
        return sentences_length, padded_sentences_word_lv, padded_word_lengths, padded_sentences_char_lv, padded_label

    def get_vocab_size(self):
        return len(self.embedding)


def process_input(config, in_sentence):
    in_sentence = in_sentence.split()
    sentence_length = len(in_sentence)
    word_length = []
    in_word_lv = []
    in_char_lv = []
    max_word_length = 0
    for word in in_sentence:
        in_word_lv.append(config.word2idx.get(word, 0))
        chars_idx = []
        for char in word:
            chars_idx.append(config.char2idx[char])
        in_char_lv.append(chars_idx)
        word_length.append(len(chars_idx))
        if len(chars_idx) > max_word_length:
            max_word_length = len(chars_idx)
    for i in range(sentence_length):
        while len(in_char_lv[i]) < max_word_length:
            in_char_lv[i].append(0)
    return [in_word_lv], [sentence_length], [in_char_lv], [word_length]

