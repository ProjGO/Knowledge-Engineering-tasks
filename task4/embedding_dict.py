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
        print("Embedding file loaded successfully, %d words in total, embedding dim: %d\n" %
              (self.vocab_size, self.embedding_dim))

    def get_embedding_vec(self):
        return np.array(list(self.embedding.values()), dtype=np.float)
