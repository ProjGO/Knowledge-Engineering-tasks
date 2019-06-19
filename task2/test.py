from task2.data_util import *
from task2.config import Config
from task2.NER_model import NERModel


if __name__ == "__main__":
    config = Config()
    train_dataset = Dataset(config=config, name="train")
    validate_dataset = Dataset(config=config, name="valid")
    test_dataset = Dataset(config=config, name="test")
    '''while True:
        dataset.cur_idx = 0
        has_one_epoch, batch_data, batch_label = dataset.get_one_batch()
        sentences_length, padded_sentences_word_lv, padded_word_lengths, padded_sentences_char_lv, padded_label = \
            dataset.batch_padding(batch_data, batch_label)'''
    model = NERModel(config, train_dataset, validate_dataset, test_dataset)
    #model.train(5)
    accuracy, confusion_mat = model.test()
    print("total accuracy: %f\n" % accuracy)
    print_confusion_mat(confusion_mat)
    # model.predict_sentence()
    # print(dataset.get_one_batch())
    '''a = np.array([[1, 2, 3], [4, 5, 6]])
    a[1] = np.append(a[1], 4)
    print(a)'''

    '''
    test.py:
    total accuracy: 0.957447

    [[0.9915 0.0009 0.0015 0.0008 0.0002 0.0012 0.0006 0.0017 0.0016]
     [0.0927 0.6787 0.0156 0.0343 0.0006 0.127  0.0006 0.0481 0.0024]
     [0.1198 0.0228 0.7102 0.0024 0.0287 0.0108 0.0814 0.0048 0.0192]
     [0.0465 0.0379 0.     0.8588 0.0037 0.0403 0.0031 0.0079 0.0018]
     [0.0188 0.0017 0.0214 0.0103 0.9247 0.     0.0205 0.     0.0026]
     [0.0183 0.0348 0.003  0.0059 0.0012 0.9191 0.0041 0.013  0.0006]
     [0.0192 0.     0.0538 0.     0.0154 0.0308 0.8808 0.     0.    ]
     [0.0886 0.0408 0.     0.0141 0.     0.0675 0.0042 0.7764 0.0084]
     [0.1689 0.0046 0.0411 0.     0.0091 0.0046 0.0639 0.0365 0.6712]]
     
     train:
     total accuracy: 0.995408

    [[0.9992 0.     0.0003 0.     0.     0.0001 0.0001 0.     0.0001]
     [0.0245 0.9362 0.0021 0.0059 0.     0.0229 0.0009 0.0071 0.0003]
     [0.0076 0.0022 0.9754 0.0003 0.0013 0.0016 0.0113 0.     0.0003]
     [0.0064 0.0071 0.0003 0.9758 0.0008 0.007  0.0005 0.002  0.0003]
     [0.0015 0.     0.0011 0.0002 0.9951 0.     0.0015 0.     0.0004]
     [0.0018 0.0018 0.0001 0.0004 0.     0.994  0.001  0.0008 0.    ]
     [0.0026 0.     0.0017 0.     0.     0.0009 0.9948 0.     0.    ]
     [0.0122 0.002  0.0003 0.0009 0.     0.0052 0.0006 0.9721 0.0067]
     [0.0147 0.     0.0009 0.     0.0009 0.     0.0078 0.0113 0.9645]]
    '''
