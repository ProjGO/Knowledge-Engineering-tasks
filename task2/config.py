import pickle
import numpy as np
import os


class Config:
    embedding = {}

    dataset_path = "../datasets/nerDataset/"
    embedding_path = "../embeddings/word2vec_0.1.pkl"
    log_dir = "./log_dir"

    lstm_layer = 2

    n_tags = 9
    state_dim = 200
    output_dim = 200

    word_embedding_dim = -1
    embedding_vocab_size = -1
    char_embedding_dim = 100

    char_embedding_trainable = True

    batch_size = 30  # 10
    validate_freq = 500
    print_freq = 20

    use_crf = False
    using_char_LSTM = True
    word_embedding_trainable = True

    log_dir_exist = False

    label2idx = {"O": 0, "B-ORG": 1, "I-ORG": 2, "B-PER": 3, "I-PER": 4,
                 "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8}
    idx2label = {idx: label for label, idx in zip(label2idx.keys(), label2idx.values())}
    char2idx = {}
    ch = chr(0)
    for i in range(256):
        char2idx[chr(ord(ch)+i)] = len(char2idx)

    def __init__(self):
        self.word2idx = {}
        with open(self.embedding_path, 'rb') as f:
            self.embedding = pickle.load(f)  # 读入预训练的word embedding
        self.word_embedding_dim = len(self.embedding["UNK"])
        self.embedding_vocab_size = len(self.embedding)
        print("pretrained word embedding loaded, %d words in the vocab, embedding dim: %d"
              % (self.embedding_vocab_size, self.word_embedding_dim))
        for word in self.embedding.keys():
            self.word2idx[word] = len(self.word2idx)
        if os.path.exists(self.log_dir):
            self.log_dir_exist = True
        else:
            os.makedirs(self.log_dir)
            pass

    def get_embedding(self):
        return self.embedding

    def get_word2idx(self):
        return self.word2idx

    '''
    返回词的向量表示,可用于创建用来查询的tf.Variable
    '''
    def get_embedding_vec(self):
        return np.array(list(self.embedding.values()), dtype=np.float)

    def write_config(self):
        log_str = ""
        log_str += "batch size: %d\n" % self.batch_size
        log_str += "char embedding dim: %d\n" % self.char_embedding_dim
        log_str += "char embedding trainable: %s\n" % "True" if self.char_embedding_trainable else "False"
        log_str += "word embedding trainable %s\n" % self.word_embedding_dim
        log_str += "word embedding trainable: %s\n" % "True" if self.word_embedding_trainable else "False"
        log_str += "state dim: %d\n" % self.state_dim
        log_str += "output dim: %d\n" % self.output_dim
        log_str += "RNN layer: %d\n" % self.lstm_layer
        log_str += "CRF: %s\n" % "True" if self.use_crf else "False"
        with open(os.path.join(self.log_dir, "config.txt"), 'w') as f:
            f.write(log_str)

    def write_epoch_and_step(self, epoch, step):
        log_str = ""
        log_str += "cur_epoch: %d\n" % epoch
        log_str += "cur_step: %d\n" % step
        with open(os.path.join(self.log_dir, "train_progress.txt"), 'w') as f:
            f.write(log_str)

    def get_cur_epoch_and_step(self):
        with open(os.path.join(self.log_dir, "train_progress.txt"), 'r') as f:
            line = f.readline().split()
            cur_epoch = int(line[1])
            line = f.readline().split()
            cur_step = int(line[1])
        return cur_epoch, cur_step

