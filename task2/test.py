from task2.data_util import *


def y():
    for i in range(5):
        yield  i


if __name__ == "__main__":
    # dataset = Dataset("C:/Users/MagicStudio/OneDrive/课件/大二下/知识工程/work/datasets/nerDataset/train.txt",
    #                   "D:/ML/Ckpts/KnowledgeEngineering_task1/log_2/word2vec_0.1.pkl")
    i = y()
    print(i)

