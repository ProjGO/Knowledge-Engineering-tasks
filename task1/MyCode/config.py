import os


class Config:
    dataset_dir = "../../datasets/"
    log_dir = "../log_dir"
    embedding_output_dir = "../embeddings"
    dataset_name = "text8"

    batch_size = 200
    window_size = 5  # window_size words on each side
    embedding_size = 200
    num_sampled = 64
    vocab_size = 50000

    valid_size = 10
    valid_window = 100

    num_steps = 200000

    has_Vo = True  # 是在作为中心词和作为上下文时有两个向量表示
    using_Vi = True  # 是否使用作为中心词的词向量作为结果
    using_Vo = False  # 是否使用作为上下文的词向量作为结果, has_Vo=True时才可为True
    # 都为True时将直接拼接两种向量表示作为最终表示

    def write_config(self, cur_step, cur_epoch):
        log_str = ""
        log_str += ('cur_step ' + str(cur_step) + '\n')
        log_str += ('cur_epoch ' + str(cur_epoch) + '\n')
        log_str += ('vocab_size '+str(self.vocab_size) + '\n')
        log_str += ('batch_size '+str(self.batch_size) + '\n')
        log_str += ('window_size '+str(self.window_size) + '\n')
        log_str += ('embedding_size '+str(self.embedding_size) + '\n')
        log_str += ('num_sampled '+str(self.num_sampled) + '\n')
        log_str += ('has_Vo '+str(self.has_Vo) + '\n')
        log_str += ('using_Vi '+str(self.using_Vi) + '\n')
        log_str += ('using_Vo '+str(self.using_Vo) + '\n')
        with open(os.path.join(self.log_dir, "info.txt"), 'w') as f:
            f.write(log_str)
        with open(os.path.join(self.embedding_output_dir, "info.txt"), 'w') as f:
            f.write(log_str)

    def get_cur_step_epoch(self):
        with open(os.path.join(self.log_dir, "info.txt"), 'r') as f:
            step_line = f.readline().split()
            epoch_line = f.readline().split()
        return int(step_line[1]), int(epoch_line[1])

    def get_dataset_path(self):
        return os.path.join(self.dataset_dir, self.dataset_name)

    def get_log_dir(self):
        has_ckpt = False
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        else:
            has_ckpt = True
        return has_ckpt, self.log_dir

    def get_embedding_output_dir(self):
        if not os.path.exists(self.embedding_output_dir):
            os.makedirs(self.embedding_output_dir)
        return self.embedding_output_dir
