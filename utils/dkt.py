import argparse
from ast import arg
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import numpy as np
import os
import json
import logging
import yaml
from sklearn.metrics import roc_auc_score, accuracy_score
import copy
import wandb

from config import Config
device = Config.device


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, max_steps):
        super(Net, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.max_steps = max_steps
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers,
                           batch_first=True).to(device)
        self.fc = nn.Linear(self.hidden_dim * 2, 1).to(device)

    def forward(self, pack, batch_size, query_embedding):
        h0 = Variable(torch.zeros(self.layer_dim, batch_size,
                                  self.hidden_dim)).to(device)
        c0 = Variable(torch.zeros(self.layer_dim, batch_size,
                                  self.hidden_dim)).to(device)
        out, (hn, cn) = self.rnn(pack, (h0, c0))
        out, _ = pad_packed_sequence(out,
                                     batch_first=True,
                                     total_length=self.max_steps - 1)
        # res = torch.sigmoid(self.fc(torch.tanh(out)))
        assert out.shape == (batch_size, self.max_steps - 1, self.hidden_dim)
        assert query_embedding.shape == (batch_size, self.max_steps - 1, self.hidden_dim)
        out = torch.cat([out, query_embedding],
                        dim=-1)  # (bs, 199, self.hidden_dim + embeding_dim)
        res = torch.sigmoid(self.fc(out))
        return res.squeeze()


class DKT(object):

    def __init__(self, input_size, num_questions, hidden_size, num_layers,
                 embedding, max_steps, logger, seq_model=None):
        super(DKT, self).__init__()
        self.num_questions = num_questions
        self.logger = logger
        self.best_auc = -np.inf
        self.dkt_model = Net(input_size, hidden_size, num_layers,
                             max_steps).to(device)
        self.seq_model = seq_model.to(device)
        self.embedding = embedding # index 0 for padding, real question embedding index from 1
        self.best_dkt = None
        # self.best_model = None
        # self.embedding = embedding
    def add_ans_info(self, seq_input, seq_ans):
        bs, max_step, dim = seq_input.shape
        seq_input_ = seq_input.reshape(-1, dim)
        seq_ans_ = seq_ans.reshape(-1, 1)
        zeros = torch.zeros_like(seq_input_)

        # false_encoding = ans_embedding[0]
        # correct_encoding = ans_embedding[1]
        # import ipdb; ipdb.set_trace()
        # correct_encoding = torch.cat([seq_input_, ans_embedding[1].expand(seq_input_.shape)], dim=-1)
        # false_encoding = torch.cat([seq_input_, ans_embedding[0].expand(seq_input_.shape)], dim=-1)
        correct_encoding = torch.cat([seq_input_, zeros], dim=-1)
        false_encoding = torch.cat([zeros, seq_input_], dim=-1)
        new_seq_input = torch.where(seq_ans_ > 0, correct_encoding,
                                    false_encoding).reshape(
                                        (bs, max_step, dim * 2))
        return new_seq_input

    def preprocess_data(self, pad_seq, pad_ans, seq_len):
        pad_seq = pad_seq.to(device)  # (batch_size, max_len)
        pad_ans = pad_ans.float().to(device)  # (batch_size, max_len)
        seq_len = seq_len.to(device)  # (batch_size,)
        ####
        student_state, _ = self.seq_model(pad_seq, pad_ans.long(), seq_len, mode='general')
        # student_state = 0
        static_input = F.embedding(
            pad_seq, self.embedding,
            padding_idx=0)  # (batch_size, max_len, embed_dim)
        seq_input = student_state + static_input
        # get (q, a) sequence
        # (batch_size, max_tep - 1, embed_dim)
        input_embed = seq_input[:, :-1, :]
        input_ans = pad_ans[:, :-1]  # (batch_size, max_step - 1)
        rnn_input = self.add_ans_info(
            input_embed,
            input_ans)  # (batch_size, max_tep - 1, embed_dim * 2)
        # (batch_size, max_tep - 1, embed_dim )
        next_pro_embed = static_input[:, 1:, :]
        targets_ans = pad_ans[:, 1:]  # (batch_size, max_tep - 1)
        packed_input = pack_padded_sequence(input=rnn_input,
                                            lengths=seq_len.cpu() - 1,
                                            batch_first=True,
                                            enforce_sorted=False)
        return packed_input, next_pro_embed, targets_ans

    def get_sequence_preds_targets(self, out, seq_len, targets_ans):
        '''
        out: (batch_size, max_len - 1, 1)
        seq_len: (batch_size, )
        targets_ans: (batch_size, max_len - 1, 1)
        '''
        pred = []
        truth = []
        for i, len_i in enumerate(seq_len):  
            pred_y = out[i][:len_i]  # (len_i - 1, )
            target_y = targets_ans[i][:len_i]  # (len_i - 1)
            pred.append(pred_y)
            truth.append(target_y)
        return torch.cat(pred), torch.cat(truth)

    def train(self,
              train_data_loader,
              test_data=None,
              *,
              epoch: int,
              lr=0.001) -> ...:
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.dkt_model.parameters(), lr=lr)
        patience = 7
        for e in range(epoch):
            self.dkt_model.train()
            loss_list = []
            for pad_seq, pad_ans, seq_len in train_data_loader:

                
                packed_input, next_pro_embed, targets_ans = self.preprocess_data(
                    pad_seq, pad_ans, seq_len)

                out = self.dkt_model(packed_input, pad_seq.shape[0],
                                     next_pro_embed)  # (batch_size, 199, 1)

                pred, truth = self.get_sequence_preds_targets(
                    out, seq_len - 1, targets_ans)
                loss = loss_function(pred, truth)
                loss_list.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # output info
            epoch_loss = sum(loss_list) / len(loss_list)
            self.logger.info("[Epoch %d] LogisticLoss: %.6f" % (e, epoch_loss))
            # with open(self.log_path, "a") as f:
            #     f.write("[Epoch %d] LogisticLoss: %.6f, " % (e, epoch_loss))

            if test_data is not None:
                acc, auc = self.eval(test_data)
                if auc > self.best_auc:
                    stale = 0
                    self.best_auc = auc
                    self.best_dkt = copy.deepcopy(self.dkt_model)
                else:
                    stale += 1
                    if stale > patience:
                        self.logger.info(
                            f"No improvment {patience} consecutive epochs, early stopping"
                        )
                        break

    def eval(self, test_data_loader) -> float:
        self.dkt_model.eval()
        with torch.no_grad():
            preds = []
            truths = []
            for pad_seq, pad_ans, seq_len in test_data_loader:

                packed_input, next_pro_embed, targets_ans = self.preprocess_data(
                    pad_seq, pad_ans, seq_len)

                out = self.dkt_model(packed_input, pad_seq.shape[0],
                                     next_pro_embed)  # (batch_size, 199, 1)

                pred, truth = self.get_sequence_preds_targets(
                    out, seq_len - 1, targets_ans)
                preds.append(pred)
                truths.append(truth)

            preds = torch.cat(preds).cpu()
            truths = torch.cat(truths).cpu()

            auc = roc_auc_score(truths.detach().numpy(),
                                preds.detach().numpy())
            preds[preds >= 0.5] = 1.0
            preds[preds < 0.5] = 0.0
            acc = accuracy_score(truths.detach().numpy(),
                                 preds.detach().numpy())
            self.logger.info('valid auc : %3.5f, valid accuracy : %3.5f' % (auc, acc))
            # with open(self.log_path, "a") as f:
            #     f.write('valid auc : %3.5f, valid accuracy : %3.5f\n' %
            #             (auc, acc))
        return auc, acc

    def save(self, filepath):
        # model = self.dkt_model.to("cpu")
        torch.save(self.dkt_model.state_dict(), filepath)
        self.logger.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dkt_model.load_state_dict(torch.load(filepath))
        self.dkt_model = self.dkt_model.to(device)
        self.logger.info("load parameters from %s" % filepath)

