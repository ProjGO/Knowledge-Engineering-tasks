from task2.data_util import *


def y():
    for i in range(5):
        yield i


if __name__ == "__main__":
    dataset = Dataset(dataset_path="C:/Users/MagicStudio/OneDrive/课件/大二下/知识工程/work/datasets/nerDataset/train.txt",
                      vocab_path="../task1/log_dir/word2vec_0.1.pkl",
                      name="train")
    print(dataset.get_one_batch())
    print(dataset.get_embedding().shape)
