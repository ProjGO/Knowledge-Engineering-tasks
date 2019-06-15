from task2.data_util import *
from task2.config import Config
from task2.NER_model import NERModel


if __name__ == "__main__":
    config = Config()
    train_dataset = Dataset(config=config, name="test")
    test_dataset = Dataset(config=config, name="test")
    '''while True:
        dataset.cur_idx = 0
        has_one_epoch, batch_data, batch_label = dataset.get_one_batch()
        sentences_length, padded_sentences_word_lv, padded_word_lengths, padded_sentences_char_lv, padded_label = \
            dataset.batch_padding(batch_data, batch_label)'''
    model = NERModel(config, train_dataset, test_dataset, test_dataset)
    model.train(3)
    # print(model.test())
    # model.predict_sentence()
    # print(dataset.get_one_batch())
    '''a = np.array([[1, 2, 3], [4, 5, 6]])
    a[1] = np.append(a[1], 4)
    print(a)'''
