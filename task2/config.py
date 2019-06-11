import pickle
import numpy as np


class Config:
    embedding = {}

    dataset_path = "../datasets/nerDataset/"
    embedding_path = "../embeddings/word2vec_0.1.pkl"
    log_dir = "log_dir"

    lstm_layer_cnt = 1

    n_tags = 9
    state_dim = 200
    output_dim = 200

    using_char_lstm = True
    word_embedding_dim = -1
    embedding_vocab_size = -1
    char_embedding_dim = 100
    word_embedding_trainable = False
    char_embedding_trainable = False

    batch_size = 10

    use_crf = False

    embedding_loaded = False

    def load_embedding(self):
        if self.embedding_loaded:
            return
        self.embedding_loaded = True
        with open(self.embedding_path, 'rb') as f:
            self.embedding = pickle.load(f)  # 读入预训练的word embedding
        self.word_embedding_dim = len(self.embedding["UNK"])
        self.embedding_vocab_size = len(self.embedding)
        print("pretrained word embedding loaded, %d words in the vocab, embedding dim: %d"
              % (self.embedding_vocab_size, self.word_embedding_dim))

    def get_embedding(self):
        return self.embedding

    '''
    返回词的向量表示,可用于创建用来查询的tf.Variable
    '''
    def get_embedding_vec(self):
        return np.array(list(self.embedding.values()), dtype=np.float)


