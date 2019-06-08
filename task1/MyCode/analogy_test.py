import numpy as np
import pickle
import os

# filepath = 'D:/ML/Ckpts/KnowledgeEngineering_task1/log'
# filepath = 'C:/Users/MagicStudio/Desktop'
filepath = 'D:/ML/Ckpts/KnowledgeEngineering_task1'

if __name__ == '__main__':
    with open(os.path.join(filepath, 'word2vec_99999.pkl'), 'rb') as f:
        word2vec = pickle.load(f)
    vocab = word2vec.keys()
    word2id = {word: idx for idx, word in enumerate(vocab)}
    id2word = {idx: word for idx, word in enumerate(vocab)}
    embeddings = np.array([vector for vector in word2vec.values()])

    top_k = 10

    while True:
        # in_word = input()
        # print(word2id.get(in_word, -1))
        ok = True
        input_words = input().split()
        in_vectors = []
        for i in range(3):
            idx = word2id.get(input_words[i], -1)
            if idx == -1:
                print(input_words[i] + ' is not in the vocab, try again')
                ok = False
                break
            else:
                in_vectors.append(embeddings[idx])
        if not ok:
            continue
        result_vector = np.add(in_vectors[2], np.subtract(in_vectors[1], in_vectors[0]))
        similarity = np.matmul(embeddings, result_vector)
        nearest = np.argsort(-similarity)[0:top_k]
        for k in range(top_k):
            print(id2word[nearest[k]], end=' ')
        print(' ')
    pass
