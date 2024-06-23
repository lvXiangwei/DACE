# Code reused from https://github.com/jennyzhang0215/DKVMN.git
import numpy as np
import math
import torch        
        
def select_bias(qa_array, num, bias_type="None", bias_p=0.0, select_pro=None):
    if bias_type == "plag":
        select_idx = np.random.permutation(range(num))[0: int(num * bias_p)]
        qa_array[select_idx] = 1
    elif bias_type == "plag_by_pro":
        select_idx = torch.isin(q_array[:num], torch.tensor(select_pro)).tolist()
        qa_array[:num][select_idx] = 1
    elif bias_type == "guess":
        select_idx = np.random.permutation(range(num))[0: int(num * bias_p)]
        qa_array[select_idx] = torch.tensor(np.random.choice([0, 1], size=len(select_idx), p=[0.5, 0.5]))     
    return qa_array


def inject_noise(qa_dataArray ,interactions_per_student, student_p, bias_p, bias_type, datasetname, return_noise_label=False):
    idxes = np.argsort(interactions_per_student)[::-1]
    num_of_studnet = len(qa_dataArray)
    perm = idxes[0: int(num_of_studnet * student_p)]
    cnt = 0
    # import ipdb; ipdb.set_trace()
    if bias_type == 'plag_by_pro':
        pro_cnt_path = f'data/{datasetname}/question.json'
        with open(pro_cnt_path, 'r') as f:
            pro_cnt = json.load(f)
        pro_cnt_descending = [int(point['id']) + 1 for point in pro_cnt]
        select_pro = pro_descending[: int(len(pro_descending) * bias_p)]
    else:
        select_pro = None
    for i in range(num_of_studnet):
        qa_array = qa_dataArray[i]
        num = interactions_per_student[i]
        # data = [q_dataArray[i], qa_dataArray[i], p_dataArray[i], interactions_per_student[i]]
        if i in perm:
            # import ipdb; ipdb.set_trace()
            # print(qa_dataArray[i][:])
            qa_dataArray[i][:] = select_bias(qa_array, num, bias_type, bias_p, select_pro)
            # print(qa_dataArray[i][:])
            cnt += 1
    print("Injecting Over: " + str(cnt) + "Bad Seqs, " + str(num_of_studnet - cnt) + "Clean Seqs")
    if not return_noise_label:
        return qa_dataArray
    else:
        noise_label = np.zeros(num_of_studnet)
        noise_label[perm] = 1
        return qa_dataArray, noise_label
    
class PID_DATA(object):
    def __init__(self, n_question,  seqlen, separate_char, name="data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.n_question = n_question
    # data format
    # id, true_student_id
    # pid1, pid2, ...
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []
        qa_data = []
        p_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 4 == 0:
                student_id = lineID//4
            if lineID % 4 == 2:
                Q = line.split(self.separate_char)
                # if len(Q[len(Q)-1]) == 0:
                #     Q = Q[:-1]
                # print(len(Q))
            if lineID % 4 == 1:
                P = line.split(self.separate_char)
                # if len(P[len(P) - 1]) == 0:
                #     P = P[:-1]

            elif lineID % 4 == 3:
                A = line.split(self.separate_char)
                n_split = 1
                
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1

                for k in range(n_split):
                    question_sequence = []
                    problem_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            Xindex = int(Q[i]) + 1 + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]) + 1) # add 1
                            problem_sequence.append(int(P[i]) + 1) # add 1
                            answer_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
                    p_data.append(problem_sequence)

        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat
        # import ipdb; ipdb.set_trace()
        return q_dataArray, qa_dataArray, p_dataArray
