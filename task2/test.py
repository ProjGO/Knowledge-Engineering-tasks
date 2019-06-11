from task2.data_util import *


if __name__ == "__main__":
<<<<<<< Updated upstream
    dataset = Dataset("C:/Users/MagicStudio/OneDrive/课件/大二下/知识工程/work/datasets/nerDataset/train.txt")
    dataset.convert_to_idx("C:/Users/MagicStudio/Desktop/word2vec.pkl")
    ch = 'A'
    for i in range(52):
        print(chr(ord(ch)+i))
=======
    config = Config()
    dataset = Dataset(config=config, name="train")
    has_one_epoch, batch_data, batch_label = dataset.get_one_batch()
    s_len, siw, w_len, sic = Dataset.get_padded_batch(batch_data, batch_label)
    print(s_len)
    # print(dataset.get_one_batch())
    '''a = np.array([[1, 2, 3], [4, 5, 6]])
    a[1] = np.append(a[1], 4)
    print(a)'''
>>>>>>> Stashed changes
