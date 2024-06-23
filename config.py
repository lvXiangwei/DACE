data_statics = {
  "assist09":{
    'n_pid': 17737,
    'n_question': 167,
    'seqlen': 200,
    'data_dir': '../data/assist09',
    'data_name': 'assist09' + '_pid'
  },"ednet":{
    'n_pid': 12103,
    'n_question': 1598,
    'seqlen': 200,
    'data_dir': '../data/ednet',
    'data_name': 'ednet' + '_pid'
  },"assist12":{
    'n_pid': 53070,
    'n_question': 265,
    'seqlen': 200,
    'data_dir': '../data/assist12',
    'data_name': 'assist12' + '_pid'
  },"python":{
    'n_pid': 1149,
    'n_question': 73,
    'seqlen': 200,
    'data_dir': '../data/python',
    'data_name': 'python' + '_pid'
  }
}


class Config:  
    dataset='assist09'
     ### dv_train
    bias_type = "None" # plag-by_pro, guess
    bias_p=0.0 # plag:0.5; plag-by_pro:0.5; guess:0.3
    inject_p=0.0
    
    
    file_name = f"{dataset}_bias_{bias_type}_bias_p_{bias_p}.log"
    fb_epoch=10
    max_iter=300
    
   
    disentangle=1
    model="DACE_pid"
    
    n_question=data_statics[dataset]['n_question']
    n_pid=data_statics[dataset]['n_pid']
    seqlen=data_statics[dataset]['seqlen']
    data_dir=data_statics[dataset]['data_dir']
    data_name=data_statics[dataset]['data_name']
    model_type="DACE"
    save=dataset
    
    
    ### common
    train_set=1
    seed=224
    optim='adam'
    batch_size=32
    lr=1e-5
    maxgradnorm=-1
    final_fc_dim=512
    l2=1e-5
    
    ### Knowledge state extractor
    d_model=256
    d_ff=1024
    dropout=0.1
    n_block=1
    n_head=8
    kq_same=1
  
