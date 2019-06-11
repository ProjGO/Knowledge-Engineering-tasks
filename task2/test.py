from task2.data_util import *
from task2.config import Config
from task2.NER_model import NERModel


if __name__ == "__main__":
    config = Config()
    dataset = Dataset(config=config, name="train")
    # has_one_epoch, batch_data, batch_label = dataset.get_one_batch()
    # s_len, siw, w_len, sic = Dataset.get_padded_batch(batch_data, batch_label)
    model = NERModel(config, dataset, dataset, dataset)
    # print(dataset.get_one_batch())
    ch = 'A'
    for i in range(52):
        print(chr(ord(ch)+i))
    '''a = np.array([[1, 2, 3], [4, 5, 6]])
    a[1] = np.append(a[1], 4)
    print(a)'''
