from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


import tqdm
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import numpy as np


def read_user_sequence(filename, max_len=200, min_len=3):
    with open(filename, 'r') as f:
        lines = f.readlines()
    y, skill, problem, real_len = [], [], [], [] 
  
    index = 0
    while index < len(lines):
        num = eval(lines[index])
        # for assist09
        # tmp_skills = [ele+1 for ele in tmp_skills]                      # for assist12
        tmp_pro = list(eval(lines[index+1])[:max_len])
        tmp_pro = [ele+1 for ele in tmp_pro]

        tmp_skills = list(eval(lines[index+2])[:max_len])
        tmp_skills = [ele+1 for ele in tmp_skills]
        tmp_ans = list(eval(lines[index+3])[:max_len])
        for i in range(0, num, max_len):
            pros = tmp_pro[i:min(num, max_len + i)]
            skills = tmp_skills[i:min(num, max_len + i)]
            ans = tmp_ans[i:min(num, max_len + i)]
            cur_len = len(pros)
            if cur_len < min_len:  
                continue
            assert min_len <= cur_len <= max_len
            y.append(torch.tensor(ans))
            skill.append(torch.tensor(skills))
            problem.append(torch.tensor(pros))
        index += 4
    return problem, skill, y


class CustomDataset(Dataset):
    def __init__(self, seq_list, answer_list):
        self.seq_len = torch.tensor([s.shape[0]
                                    for s in seq_list])  
        self.pad_seq = pad_sequence(
            seq_list, batch_first=True, padding_value=0)
        self.pad_ans = pad_sequence(
            answer_list, batch_first=True, padding_value=2)
        # print(self.pad_seq.shape)
        # print(self.pad_ans.shape)

    def __getitem__(self, index):
        return self.pad_seq[index], self.pad_ans[index], self.seq_len[index]

    def __len__(self):
        return len(self.seq_len)


def get_dataloader(dataset, max_step=200, batch_size=128, mode='pro'):
    data_folder = os.path.join(
        'data/', dataset)
    pro_train, skill_train, ans_train = read_user_sequence(
        f'{data_folder}/train.txt', max_len=max_step, min_len=3)
    pro_val, skill_val, ans_val = read_user_sequence(
        f'{data_folder}/dev.txt', max_len=max_step, min_len=3)
    pro_test, skill_test, ans_test = read_user_sequence(
        f'{data_folder}/test.txt', max_len=max_step, min_len=3)
    assert mode in ['pro', 'skill']

    if mode == 'skill':
        pro_train = skill_train
        pro_val = skill_val
        pro_test = skill_test

    train_dataset = CustomDataset(
        pro_train, ans_train)
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(
        pro_val, ans_val)
    val_data_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = CustomDataset(
        pro_test, ans_test)
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_data_loader, val_data_loader, test_data_loader

from tqdm import tqdm

class DatasetBD(Dataset):
    def __init__(self, full_dataset, biased_type, inject_proportion, mode="train", device=torch.device("cuda"), p = 0.0):
        '''
        inject_protion: biaseded samples / total samples
        p: for each student, interaction num / total interactions
        '''
        self.p = p
        self.dataset = self.addTrigger(full_dataset, inject_proportion, mode, biased_type)
        self.device = device
        

    def __getitem__(self, item):
        seq = self.dataset[item][0]
        ans = self.dataset[item][1]
        lens = self.dataset[item][2]
        return seq, ans, lens
    
    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, inject_proportion, mode, biased_type):
        print("Generating " + mode + "bad seqs")
    
        vals, idxes = torch.sort(torch.tensor([sample[-1] for sample in dataset]), descending=True)
        perm = idxes[0: int(len(dataset) * inject_proportion)]
  
        if biased_type == 'plagiarism_by_pro':
            pro_cnt_path = f'data/{dataset}/questions.json'
            with open(pro_cnt_path, 'r') as f:
                pro_cnt = json.load(f)
            pro_descending = [int(point['id']) + 1 for point in pro_cnt]
            self.select_pro_id = pro_descending[: int(len(pro_descending) * self.p)]
        # dataset
        dataset_ = list()
        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            if mode == 'train':
                if i in perm:
                    # select biased
                    data = self.selectTrigger(data, biased_type)
                    dataset_.append((*data, 1))
                    cnt += 1
                else:
                    dataset_.append((*data, 0))
        print("Injecting Over: " + str(cnt) + "Bad Seqs, " + str(len(dataset) - cnt) + "Clean Seqs")
        return dataset_ 
    
    def selectTrigger(self, data, biased_type):
        assert biased_type in ["None", "plagiarism", "plagiarism_by_pro", "guess"]
        pad_seq, pad_ans, lens = data #(L, L, 1)
        if biased_type == "None":
            return data
        # if biased_type == "plagiarism1":
        #     inject_lens = int(self.p * lens) 
        #     pad_ans[:inject_lens] = 1
        #     return pad_seq, pad_ans, lens
        if biased_type == "plagiarism":
            select_idx = np.random.permutation(range(lens))[0: int(lens * self.p)]
            pad_ans[select_idx] = 1
            return pad_seq, pad_ans, lens
        elif biased_type == 'plagiarism_by_pro':
            select_ids = torch.isin(pad_seq[:lens], torch.tensor(self.select_pro_id)).tolist()
            pad_ans[:lens][select_ids] = 1
            return pad_seq, pad_ans, lens
        elif biased_type == 'guess':
            select_idx = np.random.permutation(range(lens))[0: int(lens * self.p)]
            pad_ans[select_idx] = torch.tensor(np.random.choice([0, 1], size=len(select_idx), p=[0.5, 0.5]))
            return pad_seq, pad_ans, lens
import json
def get_backdoor_loader(dataset, max_step=200, batch_size = 128, mode='problem', config=None):
    # current_path = os.path.dirname(__file__)
    data_folder = os.path.join('data/', dataset)
    problem_hashmap_path = os.path.join(data_folder, 'problem_id_hashmap.json')
    pro_hashmap = json.load(open(problem_hashmap_path, 'r'))
    # skill_hashmap_path = os.path.join(data_folder, 'skill_id_hashmap.json') 
    # skill_hashmap = json.load(open(skill_hashmap_path, 'r'))
    
    n_questions = len(pro_hashmap)
    
    pro_train, skill_train, ans_train = read_user_sequence(f'{data_folder}/train.txt', max_len=max_step, min_len=3)
    pro_val, skill_val, ans_val = read_user_sequence(f'{data_folder}/dev.txt', max_len=max_step, min_len=3)
    pro_test, skill_test, ans_test = read_user_sequence(f'{data_folder}/test.txt', max_len=max_step, min_len=3)
    
    if mode == 'skill':
        pro_train = skill_train
        pro_val = skill_val
        pro_test = skill_test
    
    train_data_bad = DatasetBD(CustomDataset(pro_train, ans_train), 
                               biased_type = config.biased_type, 
                               inject_proportion = config.inject_proportion, 
                               mode="train", 
                               device=torch.device("cuda"),
                               p = config.p)
    
    train_bad_loader = DataLoader(dataset=train_data_bad,
                                  batch_size=batch_size,
                                  shuffle=True, num_workers=4)

    val_dataset = DatasetBD(CustomDataset(pro_val, ans_val), 
                               biased_type = config.biased_type, 
                               inject_proportion = config.inject_proportion, 
                               mode="train", 
                               device=torch.device("cuda"),
                               p = config.p)
    
    val_bad_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = CustomDataset(pro_test, ans_test)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_bad_loader, val_bad_loader, test_data_loader
