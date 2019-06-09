from task2.data_util import *


if __name__ == "__main__":
    dataset = Dataset("C:/Users/MagicStudio/OneDrive/课件/大二下/知识工程/work/datasets/nerDataset/train.txt")
    dataset.convert_to_idx("C:/Users/MagicStudio/Desktop/word2vec.pkl")
    ch = 'A'
    for i in range(52):
        print(chr(ord(ch)+i))
