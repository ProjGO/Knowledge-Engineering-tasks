import os
import pickle


class Dataset:
    label2idx = {"O": 0, "B-ORG": 1, "I-ORG": 2, "B-PER": 3, "I-PER": 4,
                 "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8}
    idx2label = {idx: label for idx, label in zip(label2idx.keys(), label2idx.values())}
    char2idx = {}
    ch = chr(0)
    for i in range(256):
        char2idx[chr(ord(ch)+i)] = len(char2idx)
    sentences = []  # 列表的列表,里面的每个列表是一句话,再里面是每个单词
    labels = []  # 列表的列表，对应着句子中的标签
    idxed_sentences = []  # 将单词转换为输入embedding中的索引后的句子
    char_idxed_sentences = []
    idxed_labels = []  # 转换为对应编号后的标签
    embeddings = {}  # 预训练的word embedding, 格式word:vec,第一个是UNK(所有在embedding中找不到的都认为是UNK)
    word2idx = {}  # word:index
    idx2word = {}  # index:word
    sentences_cnt = 0  # 一共有几句话
    word_cnt = 0  # 一共有几个词

    cur_idx = 0  # 下一个batch的开始是哪一句(sentences中的下标)

    def __init__(self, dataset_path, vocab_path, batch_size=100, name="dataset"):
        self.name = name  # 数据集名称(train/validate/test)
        self.batch_size = batch_size

        # 读入原始数据
        sentence_buffer = []
        label_buffer = []
        with open(dataset_path, 'r') as f:
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
                    self.word_cnt += 1
        print("Loading dataset %s done, %d sentences, %d words in total."
              % (self.name, self.sentences_cnt, self.word_cnt))

        self.convert_word_and_label_to_idx(vocab_path)  # 将词和标签转换为数字表示
        self.convert_char_to_idx()  # 将词转换为字符级别表示,表示字符的数字为其ASCII码

    def convert_word_and_label_to_idx(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            self.embeddings = pickle.load(f)  # 读入预训练的word embedding
            for word in self.embeddings.keys():
                self.word2idx[word] = len(self.word2idx)
            self.idx2word = {idx: word for word, idx in zip(self.word2idx.keys(), self.word2idx.values())}
            cur_sentence = []
            cur_label = []
            UNK_cnt = 0
            for sentence, labels in zip(self.sentences, self.labels):
                for word in sentence:
                    word = word.lower()  # word embedding中的所有词都是小写
                    idx = self.word2idx.get(word, 0)  # 如果找不到返回0,0就是UNK
                    cur_sentence.append(idx)
                    if idx == 0:
                        UNK_cnt += 1
                for label in labels:
                    cur_label.append(self.label2idx[label])
            print("converted words, labels to indexes, %d unknown words in %s" % (UNK_cnt, self.name))

    def convert_char_to_idx(self):
        for sentence in self.sentences:
            sentence_buffer = []
            for word in sentence:
                word_buffer = []
                for ch in word:
                    word_buffer.append(self.char2idx[ch])
                sentence_buffer.append(word_buffer)
            self.char_idxed_sentences.append(sentence_buffer)

    def generate_batch(self):
        pass










