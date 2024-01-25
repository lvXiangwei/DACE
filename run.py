import yaml
import json
import argparse
import torch
import time
import copy
import wandb
import os

from model import DACE
from utils.dataloader import get_backdoor_loader
from utils.dkt import DKT
from utils.util import set_logger
from config import Config as config

def train(dataloader, seq_model, optimizer, device, args, epoch):
    seq_model.train()
    loss_list = []
    # for batch in tqdm(dataloader, desc='training ...'):
    for batch in dataloader:
        question_seq, answer_seq, seq_len = batch
        question_seq = question_seq.to(device)
        answer_seq = answer_seq.to(device)
        seq_len = seq_len.to(device)

        loss = seq_model.calculate_loss(question_seq, answer_seq, seq_len)
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = sum(loss_list)/len(loss_list)
    logger.info('epoch: {}, loss: {}'.format(epoch, train_loss))
    # with open(args.log_path, 'a') as f:
    #     f.write('epoch: {}, train loss: {}, '.format(epoch, train_loss))

def eval(dataloader, seq_model, device, args):
    seq_model.eval()
    loss_list = []
    with torch.no_grad(): 
        for batch in dataloader:
            question_seq, answer_seq, seq_len = batch
            question_seq = question_seq.to(device)
            answer_seq = answer_seq.to(device)
            seq_len = seq_len.to(device)
            loss = seq_model.calculate_loss(question_seq, answer_seq, seq_len)
            loss_list.append(loss.item())
    eval_loss = sum(loss_list)/len(loss_list)
    logger.info('evaluate loss: {}'.format(eval_loss))
    # with open(args.log_path, 'a') as f:
    #     f.write('evaluate loss: {}\n'.format(eval_loss))
    return eval_loss


# sweep_configuration = {
#     'method': 'grid',
#     'name': 'sweep',
#     'metric': {'goal': 'maximize', 'name': 'auc'},
#     'parameters': 
#     {
#         # 'lmd1': {'values':[1.0, 0.7, 0.5, 0.3, 0.0]},
#         # 'lmd2': {'values':[1.0, 0.7, 0.5, 0.3, 0.0]},
#         # 'lmd3': {'values':[1.0, 0.5, 0]},
#         # 'lmd4': {'values':[1.0, 0.5, 0]},
#         'lmd1': {'values':[1.0]},
#         'lmd2': {'values':[1.0, 0.0]},
#         'lmd3': {'values':[1.0]},
#         'lmd4': {'values':[0.5]},
#      }
# }

# sweep_id = wandb.sweep(sweep=sweep_configuration, project='DACE')
def main():
    # run = wandb.init('DACE')
    # with open(args.config_path, "r", encoding="utf-8") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    data_folder = os.path.join('./data', config.dataset)
    # print(config)
    # print(data_folder)
    problem_id_hashmap = json.load(open(f'{data_folder}/problem_id_hashmap.json', "r", encoding="utf-8"))
    # config['n_questions'] = len(problem_id_hashmap)
    # config['lmd1'] = 1.0
    # config['lmd2'] = 0.5
    # config['lmd3'] = 0.5
    # config['lmd4'] = 1.0
    # wandb.config.update(config)
    # config = wandb.config
    
    # load data
    # train_dataloader, dev_dataloader, test_dataloader = get_dataloader(dataset=config['dataset'], max_step=config['max_seq_length'], batch_size = config['batch_size'], mode='problem')
    train_dataloader, dev_dataloader, test_dataloader = get_backdoor_loader(dataset=config.dataset,
                                                        max_step=config.max_seq_length,
                                                        batch_size=config.batch_size,
                                                        mode='problem',
                                                        config=config)
    # pretrain
    dace_model = DACE(config)
    # import ipdb; ipdb.set_trace()
    dace_model.load_state_dict(torch.load(config.save_path + "_epoch_25"))
    dace_model = dace_model.to(config.device)
    # optimizer = torch.optim.Adam(dace_model.parameters(), lr=config['learning_rate'])
    
    # best_dace_model = dace_model
    # best_loss = 9999
    # stale, patience= 0, 7
    # print("############# pretrain #############")
    # for epoch in range(1, config['epochs'] + 1):
    #     train(train_dataloader, dace_model, optimizer, device, args, epoch)
    #     eval_loss = eval(dev_dataloader, dace_model, device, args)
       
    #     if eval_loss < best_loss:
    #         stale = 0
    #         best_loss = eval_loss
    #         best_dace_model = copy.deepcopy(dace_model)
    #     else:
    #         stale += 1
    #         if stale > patience:
    #             print(
    #                 f"No improvment {patience} consecutive epochs, early stopping"
    #             )
    #             break

    # dace_model = best_dace_model

    # downtask 
    logger.info("############# downtask #############")
    dkt_model = DKT(input_size=config.hidden_size * 2,
          num_questions=len(problem_id_hashmap),
          hidden_size=config.hidden_size,
          num_layers=1,
          embedding=dace_model.general_question_embedding,
          max_steps=config.max_seq_length,
          logger=logger,
          seq_model = dace_model
          )

    # dkt train
    dkt_model.train(train_data_loader=train_dataloader,
            test_data=dev_dataloader,
            epoch=200)
    
    # dkt test 
    dkt_model.dkt_model = dkt_model.best_dkt
    auc, acc = dkt_model.eval(test_dataloader)
    # wandb.log({
    #     'auc': auc,
    #     'acc': acc
    # })
    logger.info('On test sest...')
    logger.info("auc: %.6f" % auc)
    logger.info("acc: %.6f" % acc)
    # with open(args.log_path, 'a', encoding='utf-8') as f:
    #     f.write('On test sest...\n')
    #     f.write("auc: %.6f\n" % auc)
    #     f.write("acc: %.6f\n" % acc)
    
    # datatime = time.strftime('%Y-%m-%d-%H-%M-%S')
    # save s2kt model, dkt model and corresponding config
    # dace_model_path = os.path.join('models', config['dataset'] + f'_s2kt_{datatime}.params')
    # dkt_model_path = os.path.join('save4/dkt', config.dataset + f'_dkt_{datatime}.params')
    # config_path = os.path.join('save4/dkt', config.dataset + f'_config_{datatime}.yaml')
    # torch.save(dace_model.state_dict(), dace_model_path)
    # dkt_model.save(config.dkt_save_path)
    # with open(config_path, 'w') as f:
    #     yaml.dump(config, f)
    # with open(args.log_path, 'a') as f:
    #     f.write(f'dkt_model path: {dkt_model_path}, config path: {config_path}\n')
# main()
# Start sweep job.
# wandb.agent(sweep_id, function=main)

if __name__ == '__main__':
    logger = set_logger(config.dkt_log_path)
    main()
    