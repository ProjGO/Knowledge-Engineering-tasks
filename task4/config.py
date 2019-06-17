class Config:

    pkl_file_path = "../embeddings/word2vec_0.1.pkl"
    dataset_path = "../datasets/snli_datasets/"

    self_attention_lstm_output_dim = 150
    self_attention_hidden_unit_num = 200
    self_attention_embedding_row_cnt = 30

    # 论文中的text entailment中把M变为F的tensor的除了r和2u以外的那一个维度
    w_dim = 50

    batch_size = 3

    def __init__(self):
        self.word2idx = {}
        with open(self.pkl_file_path, 'rb') as f:
            self.embedding = pickle.load(f)  # 读入预训练的word embedding
        self.word_embedding_dim = len(self.embedding["UNK"])
        self.embedding_vocab_size = len(self.embedding)
        print("pretrained word embedding loaded, %d words in the vocab, embedding dim: %d"
              % (self.embedding_vocab_size, self.word_embedding_dim))
        for word in self.embedding.keys():
            self.word2idx[word] = len(self.word2idx)
        #if os.path.exists(self.log_dir):
        #   self.log_dir_exist = True
        #else:
        #    os.makedirs(self.log_dir)
        #   pass

    def get_embedding(self):
        return self.embedding

    def get_word2idx(self):
        return self.word2idx


class Dictionary(object):
    def __init__(self, path=''):
        self.word2idx = dict()
        self.idx2word = list()
        if path != '':
            words = json.loads(open(path, 'r').readline())
            for item in words:
                self.add_word(item)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


