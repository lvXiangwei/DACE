import math
import random
import numpy as np
import torch
from torch import nn 
import torch.nn.functional  as F
from utils.layers import TransformerEncoder

class DACE(nn.Module):
    # r"""Implementation of `Contrastive Learning for Sequential Recommendation` model.
    # """
    
    def __init__(self, config):
        super(DACE, self).__init__()
        
        # load parameters info
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size  # same as embedding_size
        self.inner_size = config.inner_size  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attn_dropout_prob = config.attn_dropout_prob
        self.hidden_act = config.hidden_act
        self.layer_norm_eps = config.layer_norm_eps
        
        self.batch_size = config.batch_size

        # self.lmd1 = config['lmd1']
        # self.lmd2 = config['lmd2']
        # self.lmd3 = config['lmd3']
        # self.lmd4 = config['lmd4']

        self.tau = config.tau
        self.sim = config.sim
    
        self.initializer_range = config.initializer_range
        self.loss_type = config.loss_type
        
        self.n_questions = config.n_questions # 12103 assist0 17737
        # self.n_questions = 17737
        self.max_seq_length = config.max_seq_length
        
        self.device = config.device
        
        # define layers and loss
        self.question_embedding = nn.Embedding(self.n_questions + 1, self.hidden_size, padding_idx=0).to(self.device) # plus 1 for padding
        
    
        ### add static_question_embedding
        general_question_embedding = torch.torch.from_numpy(np.load(config.embed_path)['pro_final_repre'])
        zero_tensor = torch.zeros([1, general_question_embedding.shape[1]])
        self.general_question_embedding = torch.cat([zero_tensor, general_question_embedding], axis=0).to(self.device)
        
        self.ans_embedding = nn.Embedding(3, self.hidden_size, padding_idx=0).to(self.device) # ans embedding
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size).to(self.device) 

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        ).to(self.device)

         ### task loss
        self.task_head = nn.Linear(self.hidden_size * 2, 1).to(self.device) # for task loss
        self.bce_fct = nn.BCELoss()

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps).to(self.device)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        # self.loss_fct = nn.CrossEntropyLoss()
        
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss()
         
        # parameters initialization
        self.apply(self._init_weights)
        
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    
    def get_attention_mask(self, question_seq):
       
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (question_seq > 0).long()  # [batch_size, seq_len]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).to(self.device)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long() # (1, 1, seq_len, seq_len)

        extended_attention_mask = extended_attention_mask * subsequent_mask # (batch_size, 1, max_len, max_len)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask = extended_attention_mask.to(self.device)
        return extended_attention_mask # (batch_size, 1, max_len, max_len)
    
    def augment(self, question_seq, ans_seq, ques_seq_len):
        aug_question_seq1 = []
        aug_ans_seq1 = []
        aug_len1 = []
        aug_question_seq2 = []
        aug_ans_seq2 = []
        aug_len2 = []
        for question, ans, length in zip(question_seq, ans_seq, ques_seq_len):
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]
                aug_question_seq = question
                aug_ans_seq = ans
                aug_len = length
            if switch[0] == 0:
                aug_question_seq, aug_ans_seq, aug_len = self.question_crop(question, ans, length)
            elif switch[0] == 1:
                aug_question_seq, aug_ans_seq, aug_len = self.question_mask(question, ans, length)
            elif switch[0] == 2:
                aug_question_seq, aug_ans_seq, aug_len = self.question_reorder(question, ans, length)
            
            aug_question_seq1.append(aug_question_seq)
            aug_ans_seq1.append(aug_ans_seq)
            aug_len1.append(aug_len)
            
            if switch[1] == 0:
                aug_question_seq, aug_ans_seq, aug_len = self.question_crop(question, ans, length)
            elif switch[1] == 1:
                aug_question_seq, aug_ans_seq, aug_len = self.question_mask(question, ans, length)
            elif switch[1] == 2:
                aug_question_seq, aug_ans_seq, aug_len = self.question_reorder(question, ans, length)

            aug_question_seq2.append(aug_question_seq)
            aug_ans_seq2.append(aug_ans_seq)
            aug_len2.append(aug_len)
        
        return torch.stack(aug_question_seq1), torch.stack(aug_ans_seq1), torch.stack(aug_len1), \
            torch.stack(aug_question_seq2), torch.stack(aug_ans_seq2), torch.stack(aug_len2)
    
    def question_crop(self, question_seq, ans_seq, question_seq_len, eta=0.6):
        num_left = math.floor(question_seq_len * eta)
        crop_begin = random.randint(0, question_seq_len - num_left)
        croped_question_seq = np.zeros(question_seq.shape[0])
        croped_ans_seq = np.zeros(ans_seq.shape[0])
        if crop_begin + num_left < question_seq.shape[0]:
            croped_question_seq[:num_left] = question_seq.cpu().detach().numpy()[crop_begin:crop_begin + num_left]
            croped_ans_seq[:num_left] = ans_seq.cpu().detach().numpy()[crop_begin:crop_begin + num_left]
        else:
            croped_question_seq[:num_left] = question_seq.cpu().detach().numpy()[crop_begin:]
            croped_ans_seq[:num_left] = ans_seq.cpu().detach().numpy()[crop_begin:]
        return torch.tensor(croped_question_seq, dtype=torch.long, device=self.device),\
                torch.tensor(croped_ans_seq, dtype=torch.long, device=self.device),\
               torch.tensor(num_left, dtype=torch.long, device=self.device)

    def question_mask(self, question_seq, ans_seq, question_seq_len, gamma=0.3):
        num_mask = math.floor(question_seq_len * gamma)
        mask_index = random.sample(range(question_seq_len), k=num_mask)
        masked_question_seq = question_seq.cpu().detach().numpy().copy()
        masked_question_seq[mask_index] = self.n_questions  # token 0 has been used for semantic masking
        masked_ans_seq = ans_seq.cpu().detach().numpy().copy()
        masked_ans_seq[mask_index] = [1 for _ in range(len(mask_index))]
        # masked_ans_seq[mask_index] = self.n_questions  # token 0 has been used for semantic masking
        return torch.tensor(masked_question_seq, dtype=torch.long, device=self.device),\
                torch.tensor(masked_ans_seq, dtype=torch.long, device=self.device),\
                question_seq_len

    def question_reorder(self, question_seq, ans_seq, question_seq_len, beta=0.6):
        num_reorder = math.floor(question_seq_len * beta)
        reorder_begin = random.randint(0, question_seq_len - num_reorder)
        reordered_question_seq = question_seq.cpu().detach().numpy().copy()
        shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
        random.shuffle(shuffle_index)
        reordered_question_seq[reorder_begin:reorder_begin + num_reorder] = reordered_question_seq[shuffle_index]
        
        reordered_ans_seq = ans_seq.cpu().detach().numpy().copy()
        reordered_ans_seq[reorder_begin:reorder_begin + num_reorder] = reordered_ans_seq[shuffle_index]
        return torch.tensor(reordered_question_seq, dtype=torch.long, device=self.device), \
                torch.tensor(reordered_ans_seq, dtype=torch.long, device=self.device), \
                question_seq_len

    def forward(self, question_seq, ans_seq, question_seq_len, mode='specific'):
        # question_seq: [batch_size, seq_len]
        # ans_seq: [batch_size, seq_len]
        # question_seq_len: [batch_size]
        position_ids = torch.arange(question_seq.size(1), dtype=torch.long, device=self.device) # [seq_len]
        position_ids = position_ids.unsqueeze(0).expand_as(question_seq)    # [batch_size, seq_len]
        position_embedding = self.position_embedding(position_ids)  # [batch_size, seq_len, embed_dim]
        if mode == 'specific':
            question_emb = self.question_embedding(question_seq)    # [batch_size, seq_len, embed_dim]
        elif mode == 'general': 
            # import ipdb; ipdb.set_trace()
            question_emb = F.embedding(question_seq, self.general_question_embedding, padding_idx=0) 
        
        ans_emb = self.ans_embedding(ans_seq )    # [batch_size, seq_len, embed_dim]
        input_emb = question_emb + position_embedding + ans_emb # [(ques, ans), ] -> [ques1, ques2], [ans1, ans2]
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(question_seq) # [batch_size, 1, seq_len, seq_len]

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)   # list, 2
        # trm_output[0]: [batch_size, seq_len, embed_dim]
        output = trm_output[-1] # [batch_size, seq_len, embed_dim]
        sequence_output = self.gather_indexes(output, question_seq_len - 1)  # [batch_size, embed_dim]
        return output, sequence_output  # [batch_size, embed_dim]
   
    # def ssl_compute(self, embedded_s1, embedded_s2):
    #     normalized_embedded_s1 = F.normalize(embedded_s1)
    #     normalized_embedded_s2 = F.normalize(embedded_s2)
    #     # batch_size
    #     pos_score = torch.sum(torch.mul(normalized_embedded_s1, normalized_embedded_s2), dim=1, keepdim=False)
    #     # batch_size * batch_size
    #     all_score = torch.mm(normalized_embedded_s1, normalized_embedded_s2.t())
    #     ssl_mi = torch.log(torch.exp(pos_score/self.tau) / torch.exp(all_score/self.tau).sum(dim=1, keepdim=False)).mean()
    #     return ssl_mi
   
    def get_sequence_preds_targets(self, out, targets_ans, seq_len):
        '''
        out: (batch_size, max_len - 1, 1)
        seq_len: (batch_size, )
        targets_ans: (batch_size, max_len - 1, 1)
        '''
        pred, truth = [], []
        for i, len_i in enumerate(seq_len):  # 每个学生，len_i个记录
            pred_y = out[i][:len_i].squeeze()  # 前 len_i - 1个学生知识状态 # (len_i - 1, )
            target_y = targets_ans[i][:len_i] #(len_i - 1)
            pred.append(pred_y)
            # pred.append(torch.gather(out_seq, 1, select_idx - 1))
            truth.append(target_y)
        preds = torch.cat(pred).squeeze().float()
        truths = torch.cat(truth).float()
        return preds, truths
    
    def seq_contrastive_loss(self, question_seq, ans_seq, question_seq_len, mode='general'):
        aug_question_seq1, aug_ans_seq1, aug_len1, aug_question_seq2, aug_ans_seq2, aug_len2 = self.augment(question_seq, ans_seq, question_seq_len)
        
        _, seq_output1 = self.forward(aug_question_seq1, aug_ans_seq1, aug_len1, mode)  # [batch_size, embed_dim]
        _, seq_output2 = self.forward(aug_question_seq2, aug_ans_seq2, aug_len2, mode)  # [batch_size, embed_dim]
    
        nce_logits, nce_labels = self.info_nce(seq_output1, seq_output2, temp=self.tau, batch_size=aug_len1.shape[0], sim=self.sim)
        nce_loss= self.nce_fct(nce_logits, nce_labels)
        return nce_loss
    
    def get_output(self, question_seq, ans_seq, question_seq_len, output_embedding=False):
        '''
        return:
        task preds: [B, L - 1] 
        student output: [B, L - 1, H] 
        '''
        output, _ = self.forward(question_seq, ans_seq, question_seq_len, mode='general')
        student_output = output[:, :-1, :] # [batch_size, seq_len - 1, embed_dim]
        query_pro_embedding = F.embedding(question_seq, self.general_question_embedding, padding_idx=0)[:, 1:, :] # [batch_size, seq_len - 1, embed_dim]
        task_input = torch.cat([student_output, query_pro_embedding], axis=-1) # [batch_size, seq_len - 1, embed_dim * 2]
        

        task_preds = torch.sigmoid(self.task_head(task_input)).squeeze() # [batch_size, seq_len - 1]
        if output_embedding:
            return task_preds, student_output
        else:
            return task_preds
        
    def task_loss(self, output, question_seq, ans_seq, question_seq_len, mode='specific'):
        '''
        output: (batch_size, seq_len)
        question_seq: (batch_size, seq_len)
        ans_seq: (batch_size, max_len, )
        question_seq_len: (batch_size, )
        '''
        # import ipdb; ipdb.set_trace()
        student_output = output[:, :-1, :] # [batch_size, seq_len-1, embed_dim]
        if mode == 'specific':
            query_pro_embedding = self.question_embedding(question_seq)[:, 1:, :] # [batch_size, seq_len - 1, embed_dim]
        elif mode == 'general':
            query_pro_embedding = F.embedding(question_seq, self.general_question_embedding, padding_idx=0)[:, 1:, :] # [batch_size, seq_len - 1, embed_dim]
        elif mode == 'mix':
            query_pro_embedding = F.embedding(question_seq, self.general_question_embedding + self.question_embedding.weight, padding_idx=0)[:, 1:, :]
        
        task_input = torch.cat([student_output, query_pro_embedding], axis=-1) # [batch_size, seq_len - 1, embed_dim * 2]
        task_preds = torch.sigmoid(self.task_head(task_input)).squeeze() # [batch_size, seq_len - 1]

        # import ipdb; ipdb.set_trace()
        preds, truths = self.get_sequence_preds_targets(task_preds, ans_seq[:, 1:], question_seq_len - 1)
        task_loss = self.bce_fct(preds, truths)
        return task_loss

    def calculate_loss(self, question_seq, ans_seq, question_seq_len):  
        '''
        question_seq: [batch_size, seq_len]
        ans_seq: [batch_size, seq_len]
        question_seq_len: [batch_size, ]
        '''
        
        output, seq_output = self.forward(question_seq, ans_seq, question_seq_len, 'specific')   # [batch_size, seq_len, embed_dim],  [batch_size, embed_dim] 
        
        # general_output, general_seq_output = self.forward(question_seq, ans_seq, question_seq_len, mode='general') # [batch_size, seq_len, embed_dim],  [batch_size, embed_dim] 
        # import ipdb; ipdb.set_trace()
        # general_ans_seq = torch.ones(ans_seq.shape).long().to(self.device)
        # general_output, general_seq_output = self.forward(question_seq, general_ans_seq, question_seq_len, mode='general') # [batch_size, seq_len, embed_dim],  [batch_size, embed_dim] 

        # 1. task loss, max(S_tilde, Y), max(S_overline, Y)
        # import ipdb; ipdb.set_trace()
        # specific_task_loss = self.task_loss(output, question_seq, ans_seq, question_seq_len, 'general') 
        general_task_loss = self.task_loss(output, question_seq, ans_seq, question_seq_len, 'specific') ### @answer全1
        # general_task_loss = 0.0
        # mix_task_loss = self.task_loss(general_output, question_seq, ans_seq, question_seq_len, 'mix') 
        # mix_task_loss = 0.0

        # task_loss = specific_task_loss + general_task_loss
        
        # 2.contrastive loss 
        # specific_nce_loss = self.seq_contrastive_loss(question_seq, ans_seq, question_seq_len, 'specific')
        general_nce_loss = self.seq_contrastive_loss(question_seq, ans_seq, question_seq_len, 'specific')
        # general_nce_loss = self.seq_contrastive_loss(question_seq, general_ans_seq, question_seq_len, 'general')
        # nce_loss = specific_nce_loss + general_nce_loss
        
        # 3. min I(S_tilde, S_hat)
        # info_bn_loss = self.ssl_compute(seq_output, general_seq_output)
        
        # raw: task loss + nce loss
        # return loss + self.lmd * nce_loss, alignment, uniformity
        # return self.lmd1 * nce_loss +  self.lmd2 * info_bn_loss + self.lmd3 * (specific_task_loss + general_task_loss) + self.lmd4 * mix_task_loss 
        # import ipdb; ipdb.set_trace()
        return general_task_loss + 0.1 * general_nce_loss
        # return general_task_loss

    # def decompose(self, z_i, z_j, origin_z, batch_size):
    #     """
    #     We do not sample negative examples explicitly.
    #     Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
    #     """
    #     N = 2 * batch_size
    
    #     z = torch.cat((z_i, z_j), dim=0)
    
    #     # pairwise l2 distace
    #     sim = torch.cdist(z, z, p=2)
    
    #     sim_i_j = torch.diag(sim, batch_size)
    #     sim_j_i = torch.diag(sim, -batch_size)
    
    #     positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    #     alignment = positive_samples.mean()
    
    #     # pairwise l2 distace
    #     sim = torch.cdist(origin_z, origin_z, p=2)
    #     mask = torch.ones((batch_size, batch_size), dtype=bool)
    #     mask = mask.fill_diagonal_(0)
    #     negative_samples = sim[mask].reshape(batch_size, -1)
    #     uniformity = torch.log(torch.exp(-2 * negative_samples).mean())
    
    #     return alignment, uniformity
    
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        '''
        z_i : (batch_size, embed_dim)
        z_j : (batch_size, embed_dim)
        '''
        N = 2 * batch_size
    
        z = torch.cat((z_i, z_j), dim=0) # (2 * batch_size, embed_dim)
    
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp # (2 * batch_size, 2 * batch_size)
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) # (2 * batch_size, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1) # (2 * batch_size, 2 * batch_size - 2)
    
        labels = torch.zeros(N).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
      
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters' + f': {params}'
    