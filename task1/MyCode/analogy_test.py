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
    total_steps = []
    valid_steps = []
    right = []
    top_1 = []
    top_5 = []
    for i in range(20):
        total_steps.append(0)
        valid_steps.append(0)
        right.append(0)
        top_1.append(0)
        top_5.append(0)
    cnt = 0
    name = ["capital-common-countries", "capital-world", "currency", "city-in-state", "family",
            "gram1-adjective-to-adverb", "gram2-opposite", "gram3-comparative", "gram4-superlative",
            "gram5-present-participle", "gram6-nationality-adjective", "gram7-past-tense", "gram8-plural",
            "gram9-plural-verbs"]

    #while True:
    for j in range(len(dataset)):
        #in_word = input()
        # print(word2id.get(in_word, -1))
        if j % 100 == 0:
            print("%d steps now" % j)
        ok = True
        #input_words = input().split()
        total_steps[cnt] += 1
        temp_data = dataset[j].split()
        if temp_data[0] == ':':
            cnt += 1
            continue
        total_steps[cnt] += 1
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
        valid_steps[cnt] += 1
        result_vector = np.add(in_vectors[2], np.subtract(in_vectors[1], in_vectors[0]))
        similarity = np.matmul(embeddings, result_vector)
        nearest = np.argsort(-similarity)[0:top_k]
        for k in range(top_k):
            #print(id2word[nearest[k]] + ' ')
            if id2word[nearest[k]] == temp_data[3].lower():
                right[cnt] += 1
                if k == 0:
                    top_1[cnt] += 1
                else:
                    if k <= 4:
                        top_5[cnt] += 1
                break

    acc = sum(right)/(sum(valid_steps)*1.0)
    top1_ratio = sum(top_1)/(sum(valid_steps)*1.0)
    top5_ratio = sum(top_5)/(sum(valid_steps)*1.0)
    print("total steps: %d, valid steps: %d, accuracy : %f, top1 ratio : %f, top5 ratio : %f" % (len(dataset), sum(valid_steps), acc, top1_ratio, top5_ratio))
    for i in range(14):
        print("type %d: %s, total steps: %d, valid steps: %d, accuracy: %f, top1 ratio: %f, top5 ratio: %f" % (i+1, name[i], total_steps[i+1], valid_steps[i+1], right[i+1]/(1.0*valid_steps[i+1]), top_1[i+1]/(1.0*valid_steps[i+1]), top_5[i+1]/(1.0*valid_steps[i+1])))

    pass
