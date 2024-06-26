# Code reused from https://github.com/arghosh/AKT.git
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
# from utils import set_seed
import random
from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def set_seed(seed):
    '''
    >>> set_seed(42)
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

    
class DACE(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks,
                 kq_same, dropout, model_type, final_fc_dim=512, n_heads=8, d_ff=2048,  l2=1e-5, separate_qa=False):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            n_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
        """
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = model_type
        self.separate_qa = separate_qa
        self.device = device
        embed_l = d_model
        if self.n_pid > 0:
            self.s_embed = nn.Embedding(self.n_question + 1, embed_l)
            self.p_embed = nn.Embedding(self.n_pid+1, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
        else:
            self.pa_embed = nn.Embedding(2, embed_l)
            
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        init(self)
             
    def get_cl_loss(self, z1, z2, mask):
        cos = nn.CosineSimilarity(dim=-1)
        cl_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        pooled_z1 = (z1 * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1) # (bs, embed_l)
        pooled_z2 = (z2 * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1) # (bs, embed_l)
        
        sim = cos(pooled_z1.unsqueeze(1), pooled_z2.unsqueeze(0))
        labels = torch.arange(sim.shape[0]).long().cuda()
        cl_loss = cl_loss_fn(sim, labels)
 
        return cl_loss
    
    
    def forward(self, q_data, pa_data, target, pid_data=None, return_output=False):
        
        # Batch First
        p_embed_data = self.p_embed(pid_data)  # BS, seqlen,  d_model # c_ct
        s_embed_data = self.s_embed(q_data)  # BS, seqlen, d_model # c_ct
        
        p_embed_data = p_embed_data + s_embed_data
                
        pa_data = (pa_data-q_data)//self.n_question
        pa_embed_data = self.pa_embed(pa_data) + p_embed_data

        d_output = self.model(p_embed_data, pa_embed_data)  # 211x512

        concat_p = torch.cat([d_output, p_embed_data], dim=-1)
        output_hidden = self.out(concat_p)
        
        labels = target.reshape(-1)
        m = nn.Sigmoid()
        preds = (output_hidden.reshape(-1))  # logit
        mask = labels > -0.9
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        output = loss(masked_preds, masked_labels)
        
        ### contrastive learning 
        z1 = self.model(p_embed_data, pa_embed_data, pertubed=True, eps=0.2) # (bs, seqlen, d_model)
        z2 = self.model(p_embed_data, pa_embed_data, pertubed=True, eps=0.2) # (bs, seqle, d_model)
        cl_loss = self.get_cl_loss(z1, z2, target >= 0)
        if not return_output:
            return output.sum() + 0.1 * cl_loss , m(preds), mask.sum()
        else:
            return output.sum() + 0.1 * cl_loss , m(preds), mask.sum(), d_output

class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'DACE'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data, pertubed=False, eps=0.1):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        if pertubed:
            x_shuffle_idx = torch.randperm(x.shape[0]).to(device)
            y_shuffle_idx = torch.randperm(y.shape[0]).to(device)
            
            x_shuffle = x[x_shuffle_idx]
            y_shuffle = y[y_shuffle_idx]
            
            x = x + F.normalize(x_shuffle, p=2, dim=-1) * eps
            y = y + F.normalize(y_shuffle, p=2, dim=-1) * eps
        
        # encoder
        for block in self.blocks_1:  # encode qas
            y = block(mask=1, query=y, key=y, values=y)
            if pertubed:
                y_shuffle_idx = torch.randperm(y.shape[0]).to(device)
                y_shuffle = y[y_shuffle_idx]
                # random_noise = torch.randn_like(y).cuda()
                y = y + F.normalize(y_shuffle, p=2, dim=-1) * eps
                
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False)
                flag_first = False
                if pertubed:
                    x_shuffle_idx = torch.randperm(x.shape[0]).to(device)
                    x_shuffle = x[x_shuffle_idx]
                    x = x + F.normalize(x_shuffle, p=2, dim=-1) * eps
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
                if pertubed:
                    x_shuffle_idx = torch.randperm(x.shape[0]).to(device)
                    x_shuffle = x[x_shuffle_idx]
                    x = x + F.normalize(x_shuffle, p=2, dim=-1) * eps
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()
    
    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    # import ipdb; ipdb.set_trace()
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # import ipdb; ipdb.set_trace()
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)

def init(model):
    model = model.to(device)
    embed_l = model.p_embed.weight.shape[1]
    diff_pred = nn.Sequential(
            nn.Linear(embed_l, embed_l), nn.ReLU(), nn.Dropout(0.5), nn.Linear(embed_l, 1)
        ).to(device)
    loss_func = nn.MSELoss()
    # question difficulty warmup
    if hasattr(model, "warmup"):
        return
    model.warmup = True  
    nn.init.normal_(model.p_embed.weight, mean=0, std=0.1)
    params = nn.ModuleList([diff_pred, model.p_embed])
    optimizer = torch.optim.Adam(params.parameters(), lr=0.001)
    pid_difficulty_labels = torch.FloatTensor(np.load(f'data/{Config.dataset}/question_difficulty.npy')).to(device)
    for epoch in range(50):
        model.train()
        p = diff_pred(model.p_embed.weight[1:]).reshape(-1)
        mse_loss = loss_func(p, pid_difficulty_labels)
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
    model.p_embed.weight.requires_grad = False # True
           