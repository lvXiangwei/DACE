import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random

from model import DACE
from utils.util import * # from CBD
# from utils.get_dataloader import get_dataloader, get_backdoor_loader
from utils.dataloader import get_backdoor_loader
from sklearn.metrics import roc_auc_score, accuracy_score

def get_sequence_preds_targets(out, targets_ans, seq_len):
    '''
    out: (batch_size, max_len - 1, 1)
    seq_len: (batch_size, )
    targets_ans: (batch_size, max_len - 1, 1)
    '''
    pred, truth = [], []
    for i, len_i in enumerate(seq_len):  
        pred_y = out[i][:len_i].squeeze()  
        target_y = targets_ans[i][:len_i]
        pred.append(pred_y)
        # pred.append(torch.gather(out_seq, 1, select_idx - 1))
        truth.append(target_y)
    preds = torch.cat(pred).squeeze().float()
    truths = torch.cat(truth).float()
    return preds, truths

def get_sequence_z(out, seq_len):
    '''
    out: (batch_size, max_len - 1, h)
    seq_len: (batch_size, )
    '''
    hidden = []
    for i, len_i in enumerate(seq_len):  # 
        hidden_i = out[i][:len_i - 1]  #
        # print(hidden_i.shape)
        hidden.append(hidden_i)
    hidden = torch.cat(hidden, dim=0)
    # print(hidden.shape)
    return hidden

def train_step_clean(opt, train_loader, model_clean, model_backdoor, disen_estimator, optimizer, adv_optimizer,
                         criterion, epoch):
    criterion1 = nn.BCELoss(reduction='none')
    losses = AverageMeter()
    disen_losses = AverageMeter()
    aucs = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()
    model_clean.train()
    model_backdoor.eval()
    # import ipdb; ipdb.set_trace()
    if opt.disentangle:
        for idx, (question_seq, answer_seq, seq_len) in enumerate(train_loader, start=1):
            if opt.cuda:
                question_seq = question_seq.to(opt.device)
                answer_seq = answer_seq.to(opt.device)
                seq_len = seq_len.to(opt.device)

            output1, z_hidden = model_clean.get_output(question_seq, answer_seq, seq_len, True)
            # output1, z_hidden = model_clean(img, True)
            with torch.no_grad():
                output2, r_hidden = model_backdoor.get_output(question_seq, answer_seq, seq_len, True)
            z_hidden = get_sequence_z(z_hidden, seq_len).detach()
            r_hidden = get_sequence_z(r_hidden, seq_len).detach()
            # Train discriminator
            # stop gradient propagation to encoder
            # r_hidden, z_hidden = r_hidden.detach(), z_hidden.detach()
            # max dis_loss
            dis_loss = - disen_estimator(r_hidden, z_hidden)
            disen_losses.update(dis_loss.item(), question_seq.size(0))
            adv_optimizer.zero_grad()
            dis_loss.backward()
            adv_optimizer.step()
            # Lipschitz constrain for Disc of WGAN
            disen_estimator.spectral_norm()

    for idx, (question_seq, answer_seq, seq_len) in enumerate(train_loader, start=1):
        if opt.cuda:
            question_seq = question_seq.to(opt.device)
            answer_seq = answer_seq.to(opt.device)
            seq_len = seq_len.to(opt.device)

        output1, z_hidden = model_clean.get_output(question_seq, answer_seq, seq_len, True)
        preds1, targets = get_sequence_preds_targets(output1, answer_seq[:, 1:], seq_len - 1)
        with torch.no_grad():
            output2, r_hidden = model_backdoor.get_output(question_seq, answer_seq, seq_len, True)
            preds2, targets = get_sequence_preds_targets(output2, answer_seq[:, 1:], seq_len - 1)
            loss_bias = criterion1(preds2, targets)
            loss_d = criterion1(preds1, targets).detach()

        z_hidden = get_sequence_z(z_hidden, seq_len)
        r_hidden = get_sequence_z(r_hidden, seq_len).detach()
        dis_loss = disen_estimator(r_hidden, z_hidden)

        weight = loss_bias / (loss_d + loss_bias + 1e-8)

        weight = weight * weight.shape[0] / torch.sum(weight)
        loss = torch.mean(weight * criterion1(preds1, targets))
        if opt.disentangle:
            loss += dis_loss
        ###### add contrastive
        # import ipdb; ipdb.set_trace()
        contrastive_loss = 0.1 * model_clean.seq_contrastive_loss(question_seq, answer_seq, seq_len)
        loss = loss + contrastive_loss
        ######

        # prec1, prec5 = accuracy(output1, target, topk=(1, 5))
        losses.update(loss.item(), question_seq.size(0))
        auc = roc_auc_score(targets.cpu().detach().numpy(), preds1.cpu().detach().numpy())
        aucs.update(auc, question_seq.size(0))
        # top1.update(prec1.item(), img.size(0))
        # top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            logger.info('Clean Epoch[{0}]:[{1:03}/{2:03}] '
                  'loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'auc:{aucs.val:.2f}({aucs.avg:.2f})  '.format(epoch, idx, len(train_loader), losses=losses,
                                                                 aucs=aucs))


def train_step_backdoor(opt, train_loader, model_backdoor, optimizer, criterion, epoch):
    losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    model_backdoor.train()

    for idx, (question_seq, answer_seq, seq_len) in enumerate(train_loader, start=1):
        if opt.cuda:
            question_seq = question_seq.to(opt.device)
            answer_seq = answer_seq.to(opt.device)
            seq_len = seq_len.to(opt.device)
        ######
        # import ipdb; ipdb.set_trace()
        output = model_backdoor.get_output(question_seq, answer_seq, seq_len)
        preds, targets = get_sequence_preds_targets(output, answer_seq[:, 1:], seq_len - 1)
        ######
        
        loss = criterion(preds, targets)
        ###### add contrastiv
        contrastive_loss = model_backdoor.seq_contrastive_loss(question_seq, answer_seq, seq_len)
        loss += 0.1 * contrastive_loss
        ######
        # prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), question_seq.size(0))
        # top1.update(prec1.item(), img.size(0))
        # top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            logger.info('Backdoor Epoch[{0}]:[{1:03}/{2:03}] '
                   'loss:{losses.val:.4f}({losses.avg:.4f})  '.format(epoch, idx, len(train_loader), losses=losses))
            # print('Backdoor Epoch[{0}]:[{1:03}/{2:03}] '
            #       'loss:{losses.val:.4f}({losses.avg:.4f})  '
            #       'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
            #       'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses,
            #                                                      top1=top1, top5=top5))

def test(opt, test_loader, model_clean, criterion, epoch):
    test_process = []
    losses = AverageMeter()
    aucs = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()
    weight_record = np.array([])
    # criterion1 = nn.CrossEntropyLoss(reduction='none')

    model_clean.eval()
    preds_list = []
    truths_list = []
    for idx, (question_seq, answer_seq, seq_len) in enumerate(test_loader, start=1):
        if opt.cuda:
            question_seq = question_seq.to(opt.device)
            answer_seq = answer_seq.to(opt.device)
            seq_len = seq_len.to(opt.device)

        with torch.no_grad():
            output = model_clean.get_output(question_seq, answer_seq, seq_len)
            preds, targets = get_sequence_preds_targets(output, answer_seq[:, 1:], seq_len - 1)
            loss = criterion(preds, targets)
            preds_list.append(preds)
            truths_list.append(targets)
            # loss1 = criterion1(output, target)
            # weight_record = np.concatenate([weight_record, loss1.cpu().numpy()])

        # prec1, prec5 = accuracy(output, target, topk=(1, 5))
        
        losses.update(loss.item(), question_seq.size(0))
        preds = torch.cat(preds_list).cpu()
        truths = torch.cat(truths_list).cpu()
        auc = roc_auc_score(truths.detach().numpy(),
                                preds.detach().numpy())
        aucs.update(auc.item(), question_seq.size(0))
        # top1.update(prec1.item(), img.size(0))
        # top5.update(prec5.item(), img.size(0))

    # acc_clean = [top1.avg, top5.avg, losses.avg]

    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # for idx, (img, target, indicator) in enumerate(test_bad_loader, start=1):
    #     if opt.cuda:
    #         img = img.type(torch.FloatTensor)
    #         img = img.to(opt.device)
    #         target = target.to(opt.device)

    #     with torch.no_grad():
    #         output = model_clean(img)
    #         loss = criterion(output, target)
    #         loss1 = criterion1(output, target)
    #         weight_record = np.concatenate([weight_record, loss1.cpu().numpy()])

    #     prec1, prec5 = accuracy(output, target, topk=(1, 5))
    #     losses.update(loss.item(), img.size(0))
    #     top1.update(prec1.item(), img.size(0))
    #     top5.update(prec5.item(), img.size(0))

    # acc_bd = [top1.avg, top5.avg, losses.avg]

    logger.info('[Test] auc: {:.4f}, Loss: {:.4f}'.format(aucs.avg, losses.avg))
    # print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))

    # save training progress
    # log_root = opt.log_root + '/CBD.csv'
    # test_process.append((epoch, acc_clean[0], acc_bd[0], acc_clean[2], acc_bd[2]))
    # df = pd.DataFrame(test_process,columns=("Epoch", "Test_clean_acc", "Test_bad_acc", "Test_clean_loss", "Test_bad_loss"))
    # df.to_csv(log_root, mode='a', index=False, encoding='utf-8')
    # return acc_clean, acc_bd
    return aucs.avg, losses.avg


def train(opt):
    
    # Load models
    logger.info('----------- Model Initialization --------------')
    model_clean = DACE(opt)
    
    model_clean.to(opt.device)
    model_backdoor = DACE(opt)
    model_backdoor.to(opt.device)
    hidden_dim = model_clean.hidden_size # 
    disen_estimator = DisenEstimator(hidden_dim, hidden_dim, dropout=0.2) 
    disen_estimator.to(opt.device)

    logger.info('Finish Loading Models...')

    # initialize optimizer
    adv_params = list(disen_estimator.parameters())
    adv_optimizer = Adam(adv_params, lr=0.2)
    adv_scheduler = StepLR(adv_optimizer, step_size=20, gamma=0.1)
    optimizer = torch.optim.SGD(model_clean.parameters(), lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay, nesterov=True)
    optimizer_backdoor = torch.optim.SGD(model_backdoor.parameters(), lr=opt.lr, momentum=opt.momentum,
                                     weight_decay=opt.weight_decay, nesterov=True)

    # define loss functions
    if opt.cuda:
        criterion = nn.BCELoss().to(opt.device)
    else:
        criterion = nn.BCELoss()

    logger.info('----------- Data Initialization --------------')
    # load data
    # train_dataloader, dev_dataloader, test_dataloader = get_dataloader(dataset=opt['dataset'], max_step=opt['max_seq_length'], batch_size = opt['batch_size'], mode='problem')
    train_dataloader, dev_dataloader, test_dataloader = get_backdoor_loader(dataset=opt.dataset,
                                                        max_step=opt.max_seq_length,
                                                        batch_size=opt.batch_size,
                                                        mode='problem',
                                                        config=opt)
    # _, poisoned_data_loader = get_backdoor_loader(opt)
    # test_clean_loader, test_bad_loader = get_test_loader(opt)

    logger.info('----------- Training Backdoored Model --------------')
    for epoch in range(0, 5):
        learning_rate(optimizer, epoch, opt)
        train_step_backdoor(opt, train_dataloader, model_backdoor, optimizer_backdoor, criterion, epoch + 1)
        test(opt, test_dataloader, model_backdoor, criterion, epoch + 1)
    logger.info('----------- Training Clean Model --------------')
    for epoch in range(0, opt.tuning_epochs):
        learning_rate(optimizer, epoch, opt)
        adv_scheduler.step()
        # import ipdb; ipdb.set_trace()
        train_step_clean(opt, train_dataloader, model_clean, model_backdoor, disen_estimator, optimizer,
                                   adv_optimizer, criterion, epoch + 1)
        test(opt, test_dataloader, model_clean, criterion, epoch + 1)
        if epoch % 25 == 0:
            torch.save(model_clean.state_dict(), Config.save_path + f"_epoch_{epoch}.params")


def learning_rate(optimizer, epoch, opt):
    if epoch < 20:
        lr = 0.1
    elif epoch < 70:
        lr = 0.01
    else:
        lr = 0.001
    logger.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, epoch, is_best, opt):
    if is_best:
        # filepath = os.path.join(opt.weight_root, opt.model_name + r'_epoch{}.tar'.format(epoch))
        torch.save(state, Config.save_path)
    logger.info('[info] Finish saving the model')

from config import Config

   
if __name__ == '__main__':
    # with open(args.config_path, "r", encoding="utf-8") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # Prepare arguments
    # opt = get_arguments().parse_args()
    logger = set_logger(Config.log_path)
    train(Config)
