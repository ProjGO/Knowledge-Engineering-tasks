class Config:

    pkl_file_path = "../embeddings/word2vec_0.1.pkl"
    dataset_path = "../datasets/SNLI_1.0/"

    self_attention_lstm_output_dim = 150
    self_attention_hidden_unit_num = 200
    attention_hop = 20

    # 论文中的text entailment中把M变为F的tensor的除了r和2u以外的那一个维度
    w_dim = 50

    batch_size = 50

