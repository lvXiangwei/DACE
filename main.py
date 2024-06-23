import sys
import os
import os.path
import glob
import logging
import argparse
import numpy as np
import torch
from run import train, test, train_clean
from load_data import inject_noise

from config import Config
from load_data import PID_DATA
from utils import setseed, try_makedirs, load_model, get_file_name_identifier
from layers import DisenEstimator

def train_fb_step(model_fb, optim_fb, train_q_data, train_qa_data, train_pid, valid_q_data, valid_qa_data, valid_pid, epoch):
    model_fb.train()
    train_loss, train_accuracy, train_auc = train(
            model_fb, params, optim_fb, train_q_data, train_qa_data, train_pid, label='Train')
    valid_loss, valid_accuracy, valid_auc = test(
            model_fb, params, optim_fb, valid_q_data, valid_qa_data, valid_pid, label='Valid')
    print("epoch ", epoch + 1)
    print("valid_auc\t", valid_auc, "\ttrain_auc\t", train_auc)
    print("valid_accuracy\t", valid_accuracy,
            "\ttrain_accuracy\t", train_accuracy)
    print("valid_loss\t", valid_loss, "\ttrain_loss\t", train_loss)


def custom_train(params, train_qa_data, train_pid, valid_q_data, valid_qa_data, valid_pid):
    
    # ================================== model initialization ==================================
    model_fb = load_model(params)
    model_fc = load_model(params)
    optim_fb = torch.optim.Adam(
        model_fb.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-8)
    optim_fc =  torch.optim.Adam(
        model_fc.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-8)
    
    disc = DisenEstimator(params.d_model, params.d_model, dropout=0.2).to(next(model_fc.parameters()).device)
    optim_disc = torch.optim.Adam(disc.parameters(), lr=0.2)
    # dis_scheduler = StepLR(dis_optim, step_size=20, gamma=0.1)
    
    ####### TODO data
    print("#"*20, "f_b training...", "#"*20)
    for epoch in range(0, params.fb_epoch):
        train_fb_step(model_fb, optim_fb, train_q_data, train_qa_data, train_pid, valid_q_data, valid_qa_data, valid_pid, epoch)
        # test_one_dataset()
    ###### OK
    ############# TODO
    # torch.save({'epoch': params.fb_epoch,
    #                     'model_state_dict': model_fb.state_dict(),
    #                     'optimizer_state_dict': optim_fb.state_dict(),
    #                     },
    #                    f'{params.dataset}_bias_guess_' + str(params.fb_epoch)
    #                    )
    ##################
    
    
    print("#"*20, "f_c training...", "#"*20)
    
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0
    
    for idx in range(0, params.max_iter):
        train_loss, train_accuracy, train_auc = train_clean(params, model_fb, model_fc, optim_fc, disc, optim_disc, train_q_data, train_qa_data, train_pid, "Train", idx)
        valid_loss, valid_accuracy, valid_auc = test(
            model_fc, params, optim_fc, valid_q_data, valid_qa_data, valid_pid, label='Valid')
        print("epoch ", idx + 1)
        print("valid_auc\t", valid_auc, "\ttrain_auc\t", train_auc)
        print("valid_accuracy\t", valid_accuracy,
                "\ttrain_accuracy\t", train_accuracy)
        print("valid_loss\t", valid_loss, "\ttrain_loss\t", train_loss)

        try_makedirs('model')
        try_makedirs(os.path.join('model', params.model))
        try_makedirs(os.path.join('model', params.model, params.save))

        all_valid_auc[idx + 1] = valid_auc
        all_train_auc[idx + 1] = train_auc
        all_valid_loss[idx + 1] = valid_loss
        all_train_loss[idx + 1] = train_loss
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_train_accuracy[idx + 1] = train_accuracy

        # output the epoch with the best validation auc
        if valid_auc > best_valid_auc:
            path = os.path.join('model', params.model,
                                params.save,  file_name) + '_*'
            for i in glob.glob(path):
                os.remove(i)
            best_valid_auc = valid_auc
            best_epoch = idx+1
            torch.save({'epoch': idx,
                        'model_state_dict': model_fc.state_dict(),
                        'optimizer_state_dict': optim_fc.state_dict(),
                        'loss': train_loss,
                        },
                       os.path.join('model', params.model, params.save,
                                    file_name)+'_' + str(idx+1)
                       )
        if idx-best_epoch > 10:
            break  
    try_makedirs('result')
    try_makedirs(os.path.join('result', params.model))
    try_makedirs(os.path.join('result', params.model, params.save))
    f_save_log = open(os.path.join(
        'result', params.model, params.save, file_name), 'w')
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
    f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
    f_save_log.close()
    
    return best_epoch
    
    
    

def train_one_dataset(params, file_name, train_q_data, train_qa_data, train_pid, valid_q_data, valid_qa_data, valid_pid):
    # ================================== model initialization ==================================

    # model_b = load_model(params)
    # model_c = load_model(params)
    
    print("\n", "#"*20, 'model', "#"*20)
    print(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-8)

    print("\n")

    # ================================== start training ==================================
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0

    for idx in range(params.max_iter):
        # Train Model
        train_loss, train_accuracy, train_auc = train(
            model, params, optimizer, train_q_data, train_qa_data, train_pid,  label='Train')
        # Validation step
        valid_loss, valid_accuracy, valid_auc = test(
            model,  params, optimizer, valid_q_data, valid_qa_data, valid_pid, label='Valid')

        print('epoch', idx + 1)
        print("valid_auc\t", valid_auc, "\ttrain_auc\t", train_auc)
        print("valid_accuracy\t", valid_accuracy,
              "\ttrain_accuracy\t", train_accuracy)
        print("valid_loss\t", valid_loss, "\ttrain_loss\t", train_loss)

        try_makedirs('model')
        try_makedirs(os.path.join('model', params.model))
        try_makedirs(os.path.join('model', params.model, params.save))

        all_valid_auc[idx + 1] = valid_auc
        all_train_auc[idx + 1] = train_auc
        all_valid_loss[idx + 1] = valid_loss
        all_train_loss[idx + 1] = train_loss
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_train_accuracy[idx + 1] = train_accuracy

        # output the epoch with the best validation auc
        if valid_auc > best_valid_auc:
            path = os.path.join('model', params.model,
                                params.save,  file_name) + '_*'
            for i in glob.glob(path):
                os.remove(i)
            best_valid_auc = valid_auc
            best_epoch = idx+1
            torch.save({'epoch': idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        },
                       os.path.join('model', params.model, params.save,
                                    file_name)+'_' + str(idx+1)
                       )
        if idx-best_epoch > 10:
            break   

    try_makedirs('result')
    try_makedirs(os.path.join('result', params.model))
    try_makedirs(os.path.join('result', params.model, params.save))
    f_save_log = open(os.path.join(
        'result', params.model, params.save, file_name), 'w')
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
    f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
    f_save_log.close()
    return best_epoch

def test_one_dataset(params, file_name, test_q_data, test_qa_data, test_pid,  best_epoch):
    print("\n\nStart testing ......................\n Best epoch:", best_epoch)
    model = load_model(params)

    checkpoint = torch.load(os.path.join(
        'model', params.model, params.save, file_name) + '_'+str(best_epoch))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.has_warmup = True
    test_loss, test_accuracy, test_auc = test(
        model, params, None, test_q_data, test_qa_data, test_pid, label='Test')
    print("\ntest_auc\t", test_auc)
    print("test_accuracy\t", test_accuracy)
    print("test_loss\t", test_loss)
    
if __name__ == '__main__':
    params = Config
    
    print("\n", "#"*20, 'parameters', "#"*20)
    for key in params.__dict__:
        if key[:2] != "__":
            print('\t', key, '\t', params.__dict__[key])
        
    file_name = params.file_name

        
    # load data
    dat = PID_DATA(n_question=params.n_question,
                       seqlen=params.seqlen, separate_char=',')
    setseed(params.seed)
    train_data_path = params.data_dir + "/" + "train.txt"
    valid_data_path = params.data_dir + "/" + "dev.txt"
    test_data_path = params.data_dir + "/" + "test.txt"
    train_q_data, train_qa_data, train_pid = dat.load_data(train_data_path)
    if params.bias_type != "None":
        # import ipdb; ipdb.set_trace()
        num_of_studnet = len(train_qa_data)
        interactions_per_student = np.sum(train_q_data > 0, axis=1)
        ans_seq = (train_qa_data - train_q_data)/params.n_question
        ans_seq = inject_noise(ans_seq, interactions_per_student, params.inject_p, params.bias_p, params.bias_type, params.dataset)
        train_qa_data = train_q_data + ans_seq*params.n_question
    valid_q_data, valid_qa_data, valid_pid = dat.load_data(valid_data_path)
    test_q_data, test_qa_data, test_index = dat.load_data(test_data_path)
    
    print("\n", "#"*20, 'data', "#"*20)
    print("train_q_data.shape", train_q_data.shape)
    print("train_qa_data.shape", train_qa_data.shape)
    print("valid_q_data.shape", valid_q_data.shape)  # (1566, 200)
    print("valid_qa_data.shape", valid_qa_data.shape)  # (1566, 200)
    
    
    best_epoch = custom_train(params, train_qa_data, train_pid, valid_q_data, valid_qa_data, valid_pid)
    test_one_dataset(params, file_name, test_q_data,
                     test_qa_data, test_index, best_epoch)