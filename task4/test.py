from task4.Text_Entailment import *

if __name__ == "__main__":
    config = Config()
    dataset_train = Dataset(config, "train")
    dataset_test = Dataset(config, "test")
    word_embedding = EmbeddingDict(config.pkl_file_path)
    '''has_one_epoch, batch_data_prem, batch_data_hypo, batch_label = dataset_test.get_one_batch()
    prems_length, hypos_length, padded_prems_word_lv, padded_hypos_word_lv, padded_labels = \
        Dataset.batch_padding(batch_data_prem, batch_data_hypo, batch_label)'''
    model = TextEntailmentModel(config, word_embedding, dataset_train, dataset_test, dataset_test)
    model.train(3)
