import numpy as np
import torch
import random
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import torch.nn.functional as F


def eraseLabels(args, dataset, idxs):
    #print(idxs)
    for i in idxs:
        i = int(i)
        #print("before:", dataset.targets[i])
        dataset.targets[i] += args.num_classes
        #print("after:", dataset.targets[i])
    return

def getPrior_unlabel(args, dataset, user_group, class_list, k_num):  ##

    labeled_size = 0
    for i in range(len(class_list)):
        labeled_size += int(k_num * args.positiveRate)
    unlabeled_size = len(user_group) - labeled_size

    #l_shard = [i for i in range(int(singleClass * pos_rate))]
    labeled = np.array([], dtype='int64')
    unlabeled = np.array([], dtype='int64')

    priorlist = [0.0] * args.num_classes

    # divide to unlabeled
    bias = 0
    for i in class_list:
        eraseLabels(args, dataset, user_group[max(1, int(bias + k_num * args.positiveRate)): bias + k_num])
        bias += k_num
        priorlist[i] = k_num * (1 - args.positiveRate) / unlabeled_size 
    '''
    bias = 0
    for i in range(class_list):
        if i in class_list:
            labeled = np.concatenate(
                (labeled, idxs[bias : int(bias + args.positiveRate * k_num)]), axis=0)
            bias += int(args.positiveRate * k_num)
            unlabeled = np.concatenate(
                (unlabeled, idxs[bias : int(bias + (1-args.positiveRate) * k_num)]), axis=0)
            bias += int((1-args.positiveRate) * k_num)
            priorlist.append(k_num * (1 - args.positiveRate) / unlabeled_size)
        else:
            unlabeled = np.concatenate((unlabeled, idxs[bias : bias + k_num]), axis=0)
            bias += k_num
            priorlist.append(k_num / unlabeled_size)
    else:
        priorlist.append(0.0)
    '''
    #return labeled, unlabeled, priorlist
    return priorlist


class MPULoss(nn.Module):
    def __init__(self, k, puW, priorlist, classes_list):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()
        self.puW = puW
        self.priorlist = torch.Tensor(priorlist).cuda()
        self.classes_onehot_list = torch.Tensor([0] * k).cuda()
        for i in classes_list:
            self.classes_onehot_list[i] = 1

    def forward(self, outputs, labels, priorlist=None, indexlist=None):
        if priorlist == None:
            priorlist = self.priorlist
        if indexlist == None:
            indexlist = self.classes_onehot_list
        
        outputs = outputs.float()
        outputs_Soft = F.softmax(outputs, dim=1)
        new_P_indexlist = indexlist
        torch.zeros(self.numClass).cuda()
        eps = 1e-6
        ##print("outputs:{}, outputs_Soft:{}".format(outputs, outputs_Soft))  ## 
        #print("priorlist:{}; new_P_indexlist:{}".format(priorlist, new_P_indexlist)) ##
        # P U data
        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1).cuda()
        labelsP = torch.index_select(labels, 0, P_mask)
        outputsP = torch.index_select(outputs, 0, P_mask)
        outputsP_Soft = torch.index_select(outputs_Soft, 0, P_mask)

        ##print("outputsP:{}, outputsP_Soft:{}".format(outputsP, outputsP_Soft)) ##
        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1).cuda()
        outputsU = torch.index_select(outputs, 0, U_mask)             
        outputsU_Soft = torch.index_select(outputs_Soft, 0, U_mask)    

        ##print("outputsU:{}, outputsU_Soft:{}".format(outputsU, outputsU_Soft)) ##
        PULoss = torch.zeros(1).cuda()

        pu3 = (-torch.log(1 - outputsU_Soft + eps) * new_P_indexlist).sum() / \
                              max(1, outputsU.size(0)) / len(indexlist)
        PULoss += pu3
        if self.numClass > len(indexlist):   ##?
            pu1 = (-torch.log(1 - outputsP_Soft + eps) * new_P_indexlist).sum() * \
                 priorlist[indexlist[0]] / max(1, outputsP.size(0)) / (self.numClass-len(indexlist))
            PULoss += pu1
            print("pu1:", pu1)#

        label_onehot_P = torch.zeros(labelsP.size(0), self.numClass*2).cuda().scatter_(1, torch.unsqueeze(labelsP,1), 1)[:, :self.numClass]
        log_res = -torch.log(1 - outputsP_Soft * label_onehot_P + eps)
        pu2 = -(log_res.permute(0, 1) * priorlist).sum() / max(1, outputsP.size(0))
        PULoss += pu2
        ##print("log_res:", log_res)  ##
        #crossentropyloss=nn.CrossEntropyLoss()
        #crossloss = crossentropyloss(outputsP, labelsP)
        Negative_Log_Likelihood_Loss = nn.NLLLoss()
        crossloss = Negative_Log_Likelihood_Loss(outputsP, labelsP)
        
        objective = PULoss * self.puW + crossloss
        #print("pu3:{}, pu2:{}, PULoss:{}; crossloss:{}".format(pu3, pu2, PULoss, crossloss)) ##
        '''if (objective < 0):  ##
            print("pu3:{}, pu2:{}, PULoss:{}; crossloss:{}".format(pu3, pu2, PULoss, crossloss)) ##'''
        return objective, PULoss * self.puW, crossloss
