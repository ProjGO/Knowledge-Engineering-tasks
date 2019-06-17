import tensorflow as tf
from task4.data_util import *
from task4.config import Config

if __name__ == "__main__":
    config = Config()
    dataset_test = Dataset(config, "test")
    has_one_epoch, batch_data_hypo, batch_data_prem, batch_label = dataset_test.get_one_batch()
    pass
