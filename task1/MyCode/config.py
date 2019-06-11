import os


class Config:
    dir_input = "C:/Users/MagicStudio/OneDrive/课件/大二下/知识工程/work/datasets"
    dir_output = "D:/ML/Ckpts/KnowledgeEngineering_task1/log_dir"
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
        with open(os.path.join(self.dir_output, "info.txt"), 'w') as f:
            f.write('cur_step ' + str(cur_step) + '\n')
            f.write('cur_epoch ' + str(cur_epoch) + '\n')
            f.write('vocab_size '+str(self.vocab_size) + '\n')
            f.write('batch_size '+str(self.batch_size) + '\n')
            f.write('window_size '+str(self.window_size) + '\n')
            f.write('embedding_size '+str(self.embedding_size) + '\n')
            f.write('num_sampled '+str(self.num_sampled) + '\n')
            f.write('has_Vo '+str(self.has_Vo) + '\n')
            f.write('using_Vi '+str(self.using_Vi) + '\n')
            f.write('using_Vo '+str(self.using_Vo) + '\n')

    def get_cur_step_epoch(self):
        with open(os.path.join(self.dir_output, "info.txt"), 'r') as f:
            step_line = f.readline().split()
            epoch_line = f.readline().split()
        return int(step_line[1]), int(epoch_line[1])

    def get_dataset_path(self):
        return os.path.join(self.dir_input, self.dataset_name)

    def get_output_dir(self):
        has_ckpt = False
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        else:
            has_ckpt = True
        return has_ckpt, self.dir_output
