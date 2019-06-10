from task2.data_util import *


def y():
    for i in range(5):
        yield i


if __name__ == "__main__":
    dataset = Dataset(dataset_path="C:/Users/MagicStudio/OneDrive/课件/大二下/知识工程/work/datasets/nerDataset/train.txt",
                      vocab_path="C:/Users/MagicStudio/Desktop/word2vec.pkl")
    print(dataset.get_one_batch())
