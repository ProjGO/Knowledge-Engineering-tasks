from task2.data_util import *
from task2.data_util import Dataset
from task2.NER_model import NERModel
from task2.config import Config


def y():
    for i in range(5):
        yield i

def main():
    config = Config()
    model = NERModel(config)
    model.build()

if __name__ == "__main__":
    dataset = Dataset(dataset_path="E:/Study/第四学期/知识工程/task2/nerDataset/nerDataset/test.txt",
                      vocab_path="E:/Study/第四学期/知识工程/task2/word2vec_0.1.pkl",
                      name="train")
    print(dataset.get_one_batch())
    print(dataset.get_embedding().shape)
    main()