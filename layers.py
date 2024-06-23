import torch
import os
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.nn import functional as F

def shuffle(real):
    """
        shuffle data in a batch
        [1, 2, 3, 4, 5] -> [2, 3, 4, 5, 1]
        P(X,Y) -> P(X)P(Y) by shuffle Y in a batch
        P(X,Y) = [(1,1'),(2,2'),(3,3')] -> P(X)P(Y) = [(1,2'),(2,3'),(3,1')]
        :param real: Tensor of (batch_size, ...), data, batch_size > 1
        :returns: Tensor of (batch_size, ...), shuffled data
    """
    # |0 1 2 3| => |1 2 3 0|
    device = real.device
    batch_size = real.size(0)
    shuffled_index = (torch.arange(batch_size) + 1) % batch_size
    shuffled_index = shuffled_index.to(device)
    shuffled = real.index_select(dim=0, index=shuffled_index)
    return shuffled



def spectral_norm(W, n_iteration=5):
    """
        Spectral normalization for Lipschitz constrain in Disc of WGAN
        Following https://blog.csdn.net/qq_16568205/article/details/99586056
        |W|^2 = principal eigenvalue of W^TW through power iteration
        v = W^Tu/|W^Tu|
        u = Wv / |Wv|
        |W|^2 = u^TWv

        :param w: Tensor of (out_dim, in_dim) or (out_dim), weight matrix of NN
        :param n_iteration: int, number of iterations for iterative calculation of spectral normalization:
        :returns: Tensor of (), spectral normalization of weight matrix
    """
    device = W.device
    # (o, i)
    # bias: (O) -> (o, 1)
    if W.dim() == 1:
        W = W.unsqueeze(-1)
    out_dim, in_dim = W.size()
    # (i, o)
    Wt = W.transpose(0, 1)
    # (1, i)
    u = torch.ones(1, in_dim).to(device)
    for _ in range(n_iteration):
        # (1, i) * (i, o) -> (1, o)
        v = torch.mm(u, Wt)
        v = v / v.norm(p=2)
        # (1, o) * (o, i) -> (1, i)
        u = torch.mm(v, W)
        u = u / u.norm(p=2)
    # (1, i) * (i, o) * (o, 1) -> (1, 1)
    sn = torch.mm(torch.mm(u, Wt), v.transpose(0, 1)).sum() ** 0.5
    return sn

class MLP(nn.Module):
    """
        Multi-Layer Perceptron
        :param in_dim: int, size of input feature
        :param n_classes: int, number of output classes
        :param hidden_dim: int, size of hidden vector
        :param dropout: float, dropout rate
        :param n_layers: int, number of layers, at least 2, default = 2
        :param act: function, activation function, default = leaky_relu
    """

    def __init__(self, in_dim, n_classes, hidden_dim, dropout, n_layers=2, act=F.leaky_relu):
        super(MLP, self).__init__()
        self.l_in = nn.Linear(in_dim, hidden_dim)
        self.l_hs = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2))
        self.l_out = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act
        return

    def forward(self, input):
        """
            :param input: Tensor of (batch_size, in_dim), input feature
            :returns: Tensor of (batch_size, n_classes), output class
        """
        hidden = self.act(self.l_in(self.dropout(input)))
        for l_h in self.l_hs:
            hidden = self.act(l_h(self.dropout(hidden)))
        output = self.l_out(self.dropout(hidden))
        return output
    
class Disc(nn.Module):
    """
        2-layer discriminator for MI estimator
        :param x_dim: int, size of x vector
        :param y_dim: int, size of y vector
        :param dropout: float, dropout rate
    """

    def __init__(self, x_dim, y_dim, dropout):
        super(Disc, self).__init__()
        self.disc = MLP(x_dim + y_dim, 1, y_dim, dropout, n_layers=2)
        return

    def forward(self, x, y):
        """
            :param x: Tensor of (batch_size, hidden_dim), x
            :param y: Tensor of (batch_size, hidden_dim), y
            :returns: Tensor of (batch_size), score
        """
        input = torch.cat((x, y), dim=-1)
        # (b, 1) -> (b)
        score = self.disc(input).squeeze(-1)
        return score
    
class DisenEstimator(nn.Module):
    """
        Disentangling estimator by WGAN-like adversarial training and spectral normalization for MI minimization
        MI(X,Y) = E_pxy[T(x,y)] - E_pxpy[T(x,y)]
        min_xy max_T MI(X,Y)

        :param hidden_dim: int, size of question embedding
        :param dropout: float, dropout rate
    """

    def __init__(self, dim1, dim2, dropout):
        super(DisenEstimator, self).__init__()
        self.disc = Disc(dim1, dim2, dropout)
        return

    def forward(self, x, y):
        """
            :param x: Tensor of (batch_size, hidden_dim), x
            :param y: Tensor of (batch_size, hidden_dim), y
            :returns: Tensor of (), loss for MI minimization
        """
        sy = shuffle(y)
        loss = self.disc(x, y).mean() - self.disc(x, sy).mean()
        return loss*0.01

    def spectral_norm(self):
        """
            spectral normalization to satisfy Lipschitz constrain for Disc of WGAN
        """
        # Lipschitz constrain for Disc of WGAN
        with torch.no_grad():
            for w in self.parameters():
                w.data /= spectral_norm(w.data)
        return