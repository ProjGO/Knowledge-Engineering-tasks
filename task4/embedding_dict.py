import pickle
import numpy as np


class EmbeddingDict:
    def __init__(self, pkl_file_path):
        with open(pkl_file_path, "rb") as f:
            self.embedding = pickle.load(f)
        self.vocab = self.embedding.keys()
        self.embedding_vec = self.embedding.values()
        self.embedding_dim = len(self.embedding["UNK"])
        self.vocab_size = len(self.vocab)
        self.word2idx = {}
        for word in self.embedding.keys():
            self.word2idx[word] = len(self.word2idx)
        print("Embedding file loaded successfully, %d words in total, embedding dim: %d\n" %
              (self.vocab_size, self.embedding_dim))

    def get_embedding_vec(self):
        return np.array(list(self.embedding.values()), dtype=np.float)

    def get_embedding(self):
        return self.embedding

    def get_word2idx(self):
        return self.word2idx
