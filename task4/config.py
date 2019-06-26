import os


class Config:

    pkl_file_path = "../embeddings/word2vec_300000.pkl"
    dataset_path = "../datasets/SNLI_1.0/"
    log_dir = "./log_dir_self_attention_fc"

    self_attention_lstm_output_dim = 300  # 300 in paper
    self_attention_hidden_unit_num = 150  # 150 in paper
    attention_hop = 30  # 30 in paper
    penalization_coef = 0.3  # 0.3 in paper
    w_dim = 200  # 论文中的text entailment中把M变为F的tensor的除了r和2u以外的那一个维度
    MLP_hidden_unit_num = 4000  # 4000 in paper

    word_embedding_trainable = True

    batch_size = 30
    print_freq = 100  # steps
    test_freq = 10000  # steps

    def __init__(self):
        self.ckpt_exists = False
        if os.path.exists(self.log_dir):
            if os.path.exists(os.path.join(self.log_dir, "checkpoint")):
                self.ckpt_exists = True
        else:
            os.makedirs(self.log_dir)

    def write_config(self):
        log_str = ""
        log_str += "batch size: %d\n" % self.batch_size
        log_str += "self attention lstm output dim: %d\n" % self.self_attention_lstm_output_dim
        log_str += "self attention hidden unit num: %d\n" % self.self_attention_hidden_unit_num
        log_str += "self attention hop: %d\n" % self.attention_hop
        log_str += "w dim: %d\n" % self.w_dim
        with open(os.path.join(self.log_dir, "config.txt"), 'w') as f:
            f.write(log_str)

    def write_epoch_and_step(self, epoch, step):
        log_str = ""
        log_str += "cur_epoch: %d\n" % epoch
        log_str += "cur_step: %d\n" % step
        with open(os.path.join(self.log_dir, "train_progress.txt"), 'w') as f:
            f.write(log_str)

    def get_cur_epoch_and_step(self):
        try:
            with open(os.path.join(self.log_dir, "train_progress.txt"), 'r') as f:
                line = f.readline().split()
                cur_epoch = int(line[1])
                line = f.readline().split()
                cur_step = int(line[1])
                print("train progress loaded")
            return cur_epoch, cur_step
        except IOError:
            return 1, 1

