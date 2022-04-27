from sklearn.utils import shuffle
import torch
from torch import nn
import numpy as np
import copy
import random

class SentenceFocus():

    def __init__(self,opts) -> None:

        self.beta = opts.sief_beta

        if opts.use_model == "bert":
            self.use_bert = True
        else:
            self.use_bert = False

        self.no_na_loss = opts.no_na_loss
        self.relation_nums = opts.relation_nums

        self.entropy = nn.BCEWithLogitsLoss(reduction='none')

    def prepro_data(self,data):
        newdata = copy.deepcopy(data)
        N_bt = data["context_idxs"].shape[0]

        for i in range(N_bt):
            rand_mask_sentence(data, newdata, i,self.use_bert)
        
        max_len = torch.max(newdata["context_word_length"]).item()

        newdata["context_idxs"] = newdata["context_idxs"][:,:max_len]
        newdata["context_sent"] = newdata["context_sent"][:,:max_len]
        newdata["context_word_mask"] = newdata["context_word_mask"][:,:max_len]
        newdata["context_mention"] = newdata["context_mention"][:,:max_len]
        newdata["context_pos"] = newdata["context_pos"][:,:max_len]
        newdata["context_ner"] = newdata["context_ner"][:,:max_len]
        
        return newdata
    def importance_estimation(self,predictions,predictions_hat):
        p = torch.sigmoid(predictions)
        p_hat = torch.sigmoid(predictions_hat)
        score = p*torch.log(p/(p_hat+1e-11))
        return score
    def sentence_focusing(self,predictions,predictions_hat,relation_mask_hat,relation_multi_label):
        importance_score = self.importance_estimation(predictions,predictions_hat)
        evidence_mask = torch.logical_and((importance_score>self.beta),(relation_multi_label>0))
        evidence_mask[...,0] = False

        instance_mask = relation_mask_hat.unsqueeze(2).repeat(1,1,self.relation_nums)
        instance_mask[evidence_mask] = 0

        if self.no_na_loss:
            loss_sf = torch.sum(self.entropy(predictions_hat[...,1:], torch.sigmoid(predictions)[...,1:]) * instance_mask[...,1:]) / torch.sum(instance_mask[...,1:])
        else:
            loss_sf = torch.sum(self.entropy(predictions_hat, torch.sigmoid(predictions)) * instance_mask) / torch.sum(instance_mask)

        return loss_sf

def rand_mask_sentence(data,newdata,nb,use_bert,delete_id=None):
    N_sent = torch.max(data["context_sent"][nb]).item()
    N_words = data["context_sent"][nb].shape[0]
    if delete_id is None:
        delete_id = np.random.randint(N_sent)
    else:
        assert delete_id < N_sent
    sent_len = []
    for i in range(N_sent):
        sent_len.append(torch.sum(data["context_sent"][nb]==(i+1)).item())
    if use_bert:
        L_sent = [1]
        sent_len[0] -= 1
        sent_len[-1] -= 1
    else:
        L_sent = [0]
    left_len = np.sum(sent_len[delete_id+1:])
    sent_dict = np.arange(N_sent)
    sent_dict[delete_id] = -1
    sent_dict[delete_id+1:] -= 1
    for sent_id in range(N_sent):
        L_sent.append(L_sent[-1]+sent_len[sent_id])

    word_dict = np.arange(N_words)
    if left_len == 0:
        start1 = L_sent[delete_id]
        word_dict[start1:] = 0
        newdata["context_idxs"][nb,start1:] = 0
        if use_bert:
            new_w_sum = torch.sum(newdata["context_idxs"][nb]>0).item()
            w_sum = torch.sum(data["context_idxs"][nb]>0).item()
            newdata["context_idxs"][nb,new_w_sum] = data["context_idxs"][nb,w_sum-1]
        newdata["context_sent"][nb,start1:] = 0
        newdata["context_word_length"][nb] = torch.sum(newdata["context_idxs"][nb]>0,dim=-1)
        newdata["context_word_mask"][nb] = newdata["context_idxs"][nb]>0
        newdata["context_mention"][nb,start1:] = 0
        newdata["context_pos"][nb,start1:] = 0
        newdata["context_ner"][nb,start1:] = 0
    else:
        start1 = L_sent[delete_id]
        start2 = L_sent[delete_id+1]
        word_dict[start2:start2+left_len] = word_dict[start1:start1+left_len]
        word_dict[start1:start2] = 0
        newdata["context_idxs"][nb,start1:start1+left_len] = data["context_idxs"][nb,start2:start2+left_len]
        newdata["context_idxs"][nb,start1+left_len:] = 0
        if use_bert:
            new_w_sum = torch.sum(newdata["context_idxs"][nb]>0).item()
            w_sum = torch.sum(data["context_idxs"][nb]>0).item()
            newdata["context_idxs"][nb,new_w_sum] = data["context_idxs"][nb,w_sum-1]

        newdata["context_sent"][nb,start1:start1+left_len] = data["context_sent"][nb,start2:start2+left_len]-1
        newdata["context_sent"][nb,start1+left_len:] = 0
        if use_bert:
            newdata["context_sent"][nb,new_w_sum] = data["context_sent"][nb,w_sum-1]-1

        newdata["context_word_length"][nb] = torch.sum(newdata["context_idxs"][nb]>0,dim=-1)
        newdata["context_word_mask"][nb] = newdata["context_idxs"][nb]>0

        newdata["context_mention"][nb,start1:start1+left_len] = data["context_mention"][nb,start2:start2+left_len]
        newdata["context_mention"][nb,start1+left_len:] = 0
        newdata["context_pos"][nb,start1:start1+left_len] = data["context_pos"][nb,start2:start2+left_len]
        newdata["context_pos"][nb,start1+left_len:] = 0
        newdata["context_ner"][nb,start1:start1+left_len] = data["context_ner"][nb,start2:start2+left_len]
        newdata["context_ner"][nb,start1+left_len:] = 0
        
    left_entity_set = []
    total_entity_set= []
    delete_node = []
    for n_id in range(newdata["graph_info"].shape[1]):
        if newdata["graph_info"][nb,n_id,5].item() == -1:
            continue
        newdata["graph_info"][nb,n_id,0] = word_dict[data["graph_info"][nb,n_id,0]]
        newdata["graph_info"][nb,n_id,1] = word_dict[data["graph_info"][nb,n_id,1]-1]+1
        newdata["graph_info"][nb,n_id,5] = sent_dict[data["graph_info"][nb,n_id,5]]
        if newdata["graph_info"][nb,n_id,5].item() == -1:
            newdata["graph_info"][nb,n_id,0] = -1
            newdata["graph_info"][nb,n_id,1] = -1
            newdata["graph_info"][nb,n_id,2] = -1
            newdata["graph_adj"][nb,n_id,:] = 0
            newdata["graph_adj"][nb,:,n_id] = 0
            delete_node.append(n_id)

        if newdata["graph_info"][nb,n_id,3]==1:
            continue
        elif newdata["graph_info"][nb,n_id,3]==2:
            total_entity_set.append(newdata["graph_info"][nb,n_id,2].item())
            if newdata["graph_info"][nb,n_id,5].item() != -1:
                left_entity_set.append(newdata["graph_info"][nb,n_id,2].item())
        else:
            break
    delete_entity = set(total_entity_set) - set(left_entity_set) -set([-1])
    for it in delete_entity:
        p_ind,_ = torch.where(newdata['h_t_pairs'][nb]==it)
        newdata['relation_mask'][nb,p_ind] = 0
    for it in delete_node:
        if it == 0:
            continue
        p_ind,m_ind,_ = torch.where(newdata['relation_path'][nb]==it)
        newdata['relation_path'][nb,p_ind,m_ind] = 0
    return newdata

def rand_shuffle_sentence(data,newdata,nb,use_bert=False):
    N_sent = torch.max(data["context_sent"][nb]).item()
    N_words = data["context_sent"][nb].shape[0]
    new_sent_seq = list(range(N_sent))
    random.shuffle(new_sent_seq)
    sent_len = []
    for i in range(N_sent):
        sent_len.append(torch.sum(data["context_sent"][nb]==(i+1)).item())
    if use_bert:
        L_sent = [1]
        new_L_sent = [1]
    else:
        L_sent = [0]
        new_L_sent = [0]
    for sent_id in range(N_sent):
        L_sent.append(L_sent[-1]+sent_len[sent_id])
        new_L_sent.append(new_L_sent[-1]+sent_len[new_sent_seq[sent_id]])
    word_dict = np.arange(N_words)
    for sent_id in range(N_sent):
        cur_id = new_sent_seq[sent_id]
        word_dict[L_sent[cur_id]:L_sent[cur_id+1]] = \
            torch.arange(new_L_sent[sent_id],new_L_sent[sent_id+1])
        newdata["context_sent"][nb,new_L_sent[sent_id]:new_L_sent[sent_id+1]]=sent_id+1
        newdata["context_idxs"][nb,new_L_sent[sent_id]:new_L_sent[sent_id+1]]=\
        data["context_idxs"][nb,L_sent[cur_id]:L_sent[cur_id+1]]
        newdata["context_mention"][nb,new_L_sent[sent_id]:new_L_sent[sent_id+1]]=\
        data["context_mention"][nb,L_sent[cur_id]:L_sent[cur_id+1]]
        newdata["context_pos"][nb,new_L_sent[sent_id]:new_L_sent[sent_id+1]]=\
        data["context_pos"][nb,L_sent[cur_id]:L_sent[cur_id+1]]
        newdata["context_ner"][nb,new_L_sent[sent_id]:new_L_sent[sent_id+1]]=\
        data["context_ner"][nb,L_sent[cur_id]:L_sent[cur_id+1]]
    for n_id in range(newdata["graph_info"].shape[1]):
        if newdata["graph_info"][nb,n_id,5] == -1:
            continue
        sent_id = new_sent_seq.index(data["graph_info"][nb,n_id,5])
        if data["graph_info"][nb,n_id,3]==1:
            newdata["graph_info"][nb,n_id,0] = new_L_sent[sent_id]
            newdata["graph_info"][nb,n_id,1] = new_L_sent[sent_id+1]
            newdata["graph_info"][nb,n_id,5] = sent_id
        elif newdata["graph_info"][nb,n_id,3]==2:
            newdata["graph_info"][nb,n_id,0] = word_dict[data["graph_info"][nb,n_id,0]]
            newdata["graph_info"][nb,n_id,1] = word_dict[data["graph_info"][nb,n_id,1]-1]+1
            newdata["graph_info"][nb,n_id,5] = sent_id
        else:
            break
    for n_id in range(newdata["context_ems_info"].shape[1]):
        if newdata["context_ems_info"][nb,n_id,0]==1:
            continue
        elif newdata["context_ems_info"][nb,n_id,0]==2 and newdata["graph_info"][nb,n_id,5] != -1:
            newdata["context_ems_info"][nb,n_id,2] = word_dict[data["context_ems_info"][nb,n_id,2]]
            newdata["context_ems_info"][nb,n_id,3] = word_dict[data["context_ems_info"][nb,n_id,3]-1]+1
            newdata["context_ems_info"][nb,n_id,6] = new_sent_seq.index(data["context_ems_info"][nb,n_id,6])
        elif newdata["context_ems_info"][nb,n_id,0]==3 and newdata["graph_info"][nb,n_id,5] != -1:
            newdata["context_ems_info"][nb,n_id,2] = word_dict[data["context_ems_info"][nb,n_id,2]]
            newdata["context_ems_info"][nb,n_id,3] = word_dict[data["context_ems_info"][nb,n_id,3]-1]+1
        else:
            break
    return newdata
