import numpy as np
import pickle
import os

dataset_path = '../../datasets/'
# filepath = 'C:/Users/MagicStudio/Desktop'
filepath = '../../embeddings/'

if __name__ == '__main__':
    with open(os.path.join(filepath, 'word2vec_0.1.pkl'), 'rb') as f:
        word2vec = pickle.load(f)
    vocab = word2vec.keys()
    word2id = {word: idx for idx, word in enumerate(vocab)}
    id2word = {idx: word for idx, word in enumerate(vocab)}
    embeddings = np.array([vector for vector in word2vec.values()])

    dataset = open(os.path.join(dataset_path, 'questions-words.txt')).readlines()

    top_k = 10
    valid_steps = 0
    right = 0
    top_1 = 0
    top_5 = 0

    #while True:
    for j in range(len(dataset)):
        # in_word = input()
        # print(word2id.get(in_word, -1))
        #if j % 100 == 0:
            #print("%d steps now" % j)
        ok = True
        #input_words = input().split()
        temp_data = dataset[j].split()
        if temp_data[0] == ':':
            continue
        input_words = temp_data[0:3]
        in_vectors = []
        for i in range(3):
            idx = word2id.get(input_words[i].lower(), -1)
            if idx == -1:
                #print(input_words[i] + ' is not in the vocab, try again')
                ok = False
                break
            else:
                in_vectors.append(embeddings[idx])
        if not ok:
            continue
        valid_steps += 1
        result_vector = np.add(in_vectors[2], np.subtract(in_vectors[1], in_vectors[0]))
        similarity = np.matmul(embeddings, result_vector)
        nearest = np.argsort(-similarity)[0:top_k]
        for k in range(top_k):
            if id2word[nearest[k]] == temp_data[3].lower():
                right += 1
                if k == 0:
                    top_1 += 1
                else:
                    if k <= 4:
                        top_5 += 1
                break

    acc = right/(valid_steps*1.0)
    top1_ratio = top_1/(valid_steps*1.0)
    top5_ratio = top_5/(valid_steps*1.0)
    print("total steps: %d, valid steps: %d, accuracy : %f, top1 ratio : %f, top5 ratio : %f" % (len(dataset), valid_steps, acc, top1_ratio, top5_ratio))
    pass
