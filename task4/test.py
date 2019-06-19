from task4.Text_Entailment import *

if __name__ == "__main__":
    config = Config()
    dataset_train = Dataset(config, "train")
    dataset_valid = Dataset(config, "dev")
    dataset_test = Dataset(config, "test")
    word_embedding = EmbeddingDict(config.pkl_file_path)
    '''has_one_epoch, batch_data_prem, batch_data_hypo, batch_label = dataset_test.get_one_batch()
    prems_length, hypos_length, padded_prems_word_lv, padded_hypos_word_lv, padded_labels = \
        Dataset.batch_padding(batch_data_prem, batch_data_hypo, batch_label)'''
    model = TextEntailmentModel(config, word_embedding, dataset_train, dataset_valid, dataset_test)
    model.train(3)
    accuracy, confusion_mat = model.test()
    # print(accuracy)
    # print(confusion_mat)

'''
train:
0.8847052401746945
[[155615.  16888.  10340.]
 [ 13525. 165058.   4912.]
 [ 11579.   6122. 165561.]]
 
 test:
 0.7324242424242424
[[2244.  495.  503.]
 [ 555. 2572.  269.]
 [ 521.  306. 2435.]]
'''
