class Config:
  # dataset
  dataset = "assist09"
  max_seq_length = 200
  embed_path = "embeddings/assist09_embedding.npz"
  n_questions = 17737

  # DACE model 
  batch_size = 128 
  tau = 1
  sim = "dot"
  n_layers = 2
  n_heads = 2
  hidden_size = 128 # last hidden dim
  inner_size = 256
  hidden_dropout_prob = 0.5
  attn_dropout_prob = 0.5
  hidden_act = "gelu"
  layer_norm_eps = 0.000000000001
  initializer_range = 0.02
  loss_type = "CE"
  
  # train
  print_freq = 20 # 
  cuda = 1
  device = "cuda:3"
  lr = 0.001
  momentum = 0.9
  weight_decay = 0.0001
  tuning_epochs = 100 # maximum epochs
  disentangle = 1
  
  #####################
  # biased type
  biased_type = "None"  # [None, plagiarism, plagiarism_by_pro, guess]
  inject_proportion = 0.3 # for reproduction, please set to 0.3
  p = 0.3 # plagiarism, plagiarism_by_pro: 0.3; guess : 0.5
  #####################

  # log and save
  log_path = f'logs/{dataset}_{biased_type}_injcet_{inject_proportion}_p_{p}.log'
  save_path = f'save/{dataset}_{biased_type}_injcet_{inject_proportion}_p_{p}'
  dkt_log_path = f'logs/{dataset}_{biased_type}_injcet_{inject_proportion}_p_{p}_dkt.log'
  dkt_save_path = f'save/{dataset}_{biased_type}_injcet_{inject_proportion}_p_{p}_dkt'

# class Config:
#   # dataset
#   dataset = "ednet"
#   max_seq_length = 200
#   embed_path = "embeddings/ednet_embedding.npz"
#   n_questions = 12103

#   # DACE model 
#   batch_size = 128 
#   tau = 1
#   sim = "dot"
#   n_layers = 2
#   n_heads = 2
#   hidden_size = 128 # last hidden dim
#   inner_size = 256
#   hidden_dropout_prob = 0.5
#   attn_dropout_prob = 0.5
#   hidden_act = "gelu"
#   layer_norm_eps = 0.000000000001
#   initializer_range = 0.02
#   loss_type = "CE"
  
#   # train
#   print_freq = 20 # 
#   cuda = 1
#   device = "cuda:3"
#   lr = 0.001
#   momentum = 0.9
#   weight_decay = 0.0001
#   tuning_epochs = 100 # maximum epochs
#   disentangle = 1
  
#  #####################
#  # biased type
#  biased_type = "None"  # [None, plagiarism, plagiarism_by_pro, guess]
#  inject_proportion = 0.3 # for reproduction, please set to 0.3
#  p = 0.3 # plagiarism, plagiarism_by_pro: 0.3; guess : 0.5
#  #####################

#   # log and save
#   log_path = f'logs/{dataset}_{biased_type}_injcet_{inject_proportion}_p_{p}.log'
#   save_path = f'save/{dataset}_{biased_type}_injcet_{inject_proportion}_p_{p}'
#   dkt_log_path = f'logs/{dataset}_{biased_type}_injcet_{inject_proportion}_p_{p}_dkt.log'
#   dkt_save_path = f'save/{dataset}_{biased_type}_injcet_{inject_proportion}_p_{p}_dkt'