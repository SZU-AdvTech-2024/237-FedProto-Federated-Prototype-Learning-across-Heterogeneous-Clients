import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path

__my_add_dir = (Path(__file__).parent / "..").resolve()  ##
if __my_add_dir not in sys.path:
    sys.path.insert(0, str(__my_add_dir))

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from lib.models.resnet import resnet18
from lib.options import args_parser
from lib.update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt
from lib.models.models import CNNMnist, CNNFemnist
from lib.utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem

from lib import ssl
from fedmethods import fedBaselines   ##my

import json  ##
model_urls = {
    #'resnet18': 'https://github.com/fregu856/deeplabv3/blob/master/pretrained_models/resnet/resnet18-5c106cde.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
model_names = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
} ##

class MyTrainer:
    def __init__(self, args):
        self.args = args 
        # set random seeds
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.args.device == 'cuda':
            torch.cuda.set_device(self.args.gpu)
            torch.cuda.manual_seed(self.args.seed)
            torch.manual_seed(self.args.seed)
        else:
            torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        # load dataset and user groups
        self.n_list = np.random.randint(max(2, self.args.ways - self.args.stdev), min(self.args.num_classes, self.args.ways + self.args.stdev + 1), self.args.num_users)
        if self.args.dataset == 'mnist':
            self.k_list = np.random.randint(self.args.shots - self.args.stdev + 1 , self.args.shots + self.args.stdev - 1, self.args.num_users)
        elif self.args.dataset == 'cifar10':
            self.k_list = np.random.randint(self.args.shots - self.args.stdev + 1 , self.args.shots + self.args.stdev + 1, self.args.num_users)
        elif self.args.dataset =='cifar100':
            self.k_list = np.random.randint(self.args.shots, self.args.shots + 1, self.args.num_users)
        elif self.args.dataset == 'femnist':
            self.k_list = np.random.randint(self.args.shots - self.args.stdev + 1 , self.args.shots + self.args.stdev + 1, self.args.num_users)
        
        ##user_groups[i] = a list of idxs of data in datasets
        ##每个client的每类的数据从train_dataset的对应类开始的位置+i*args.train_shots_max处开始，取k_list[i]个(如果n_list[i]中有对应类的标签的话)
        self.train_dataset, self.test_dataset, self.user_groups, self.user_groups_lt, self.classes_list, self.classes_list_gt = get_dataset(self.args, self.n_list, self.k_list)
        self.priorlists = [[]] * args.num_users
        if self.args.SSL:
            for i in range(self.args.num_users):
                priorList = ssl.getPrior_unlabel(self.args, self.train_dataset, self.user_groups[i], self.classes_list[i], self.k_list[i]) ##获得对应客户端的priorlist，并将部分标签去掉
                self.priorlists[i] = priorList
            
        # Build models
        self.local_model_list = self._load_model()

    def begin_train(self):

        idxs_users = np.arange(self.args.num_users)
        local_model_updaters = []  ##
        for idx in idxs_users:  ##
            local_model_updaters.append(LocalUpdate(args=self.args, dataset=self.train_dataset, idxs=self.user_groups[idx], priorList=self.priorlists[idx], classes_list=self.classes_list[idx]))
        
        if self.args.alg == 'fedproto':
            if self.args.mode == 'task_heter':
                FedProto_taskheter(self.args, self.train_dataset, self.test_dataset, self.user_groups, self.user_groups_lt, self.local_model_list, self.classes_list, local_model_updaters)
            else:
                FedProto_modelheter(self.args, self.train_dataset, self.test_dataset, self.user_groups, self.user_groups_lt, self.local_model_list, self.classes_list)
        elif self.args.alg == 'fedavg':
            fedBaselines.FedAvg_taskheter(self.args, self.train_dataset, self.test_dataset, self.user_groups, self.user_groups_lt, self.local_model_list, self.classes_list, local_model_updaters)
        elif self.args.alg == 'fedlocal':
            fedBaselines.FedLocal_taskheter(self.args, self.train_dataset, self.test_dataset, self.user_groups, self.user_groups_lt, self.local_model_list, self.classes_list)
        elif self.args.alg == 'fedprox':
            fedBaselines.FedProx_taskheter(self.args, self.train_dataset, self.test_dataset, self.user_groups, self.user_groups_lt, self.local_model_list, self.classes_list)
        elif self.args.alg == 'fedper':
            fedBaselines.FedPer_taskheter(self.args, self.train_dataset, self.test_dataset, self.user_groups, self.user_groups_lt, self.local_model_list, self.classes_list)
        elif self.args.alg == 'fesem':
            fedBaselines.FeSEM_taskheter(self.args, self.train_dataset, self.test_dataset, self.user_groups, self.user_groups_lt, self.local_model_list, self.classes_list, self.n_list)
        else :
            print("Have not algorithm {} !".format(self.args.alg))
        
    def _load_model(self):
        local_model_list = []
        for i in range(self.args.num_users):
            if self.args.dataset == 'mnist':
                if self.args.mode == 'model_heter':
                    if i<7:
                        self.args.out_channels = 18
                    elif i>=7 and i<14:
                        self.args.out_channels = 20
                    else:
                        self.args.out_channels = 22
                else:
                    self.args.out_channels = 20

                local_model = CNNMnist(args=self.args)

                if self.args.use_UH == True: ##最后一层
                    local_model.fc2.requires_grad_(False)
                    m, n = local_model.fc2.weight.shape
                    local_model.fc2.weight.data = torch.nn.init.orthogonal_(torch.rand(m, n)).to(self.args.device)
                    local_model.fc2.bias.data = torch.zeros_like(local_model.fc2.bias.data)
                
            elif self.args.dataset == 'femnist':
                if self.args.mode == 'model_heter':
                    if i<7:
                        self.args.out_channels = 18
                    elif i>=7 and i<14:
                        self.args.out_channels = 20
                    else:
                        self.args.out_channels = 22
                else:
                    self.args.out_channels = 20
                local_model = CNNFemnist(args=self.args)

            elif self.args.dataset == 'cifar100' or self.args.dataset == 'cifar10':
                if self.args.mode == 'model_heter':
                    if i<10:
                        self.args.stride = [1,4]
                    else:
                        self.args.stride = [2,2]
                else:
                    self.args.stride = [2, 2]
                self.resnet = resnet18(self.args, pretrained=False, num_classes=self.args.num_classes)
                ##initial_weight = model_zoo.load_url(model_urls['resnet18'])
                initial_weight = torch.load(self.args.resnet_dir + model_names['resnet18'])  ##my_add
                local_model = self.resnet
                initial_weight_1 = local_model.state_dict()
                for key in initial_weight.keys():
                    #if key[0:3] == 'fc.' or key[0:5]=='conv1' or key[0:3]=='bn1':
                    if key[0:5]=='conv1' or key[0:3]=='bn1':
                        initial_weight[key] = initial_weight_1[key]
                    if key[0:3] == 'fc.': 
                        if self.args.use_UH == False:  ##
                            initial_weight[key] = initial_weight_1[key]
                        else:
                            #print(initial_weight_1[key].shape,'\n',initial_weight[key].shape)
                            #print(initial_weight_1[key],'\n',initial_weight[key])
                            initial_weight[key] = initial_weight_1[key]
                
                local_model.load_state_dict(initial_weight)

                if self.args.use_UH:  ##最后一层
                    local_model.fc.requires_grad_(False)
                    m, n = local_model.fc.weight.shape
                    local_model.fc.weight.data = torch.nn.init.orthogonal_(torch.rand(m, n)).to(self.args.device)
                    local_model.fc.bias.data = torch.zeros_like(local_model.fc.bias.data)

            local_model.to(self.args.device)
            local_model.train()
            local_model_list.append(local_model)
        return local_model_list


def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, local_model_updaters):
    summary_writer = SummaryWriter('./tensorboard/'+ args.dataset +'_fedproto_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []
    test_loss, test_accuracy_g, test_accuracy_l = [], [], [] ##my_add

    '''
    local_model_updaters = []  ##
    for idx in idxs_users:  ##
        local_model_updaters.append(LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx]))
    '''
    
    
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos, local_acc = [], [], {}, [] ##add local_acc
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:  ##每个client
            ##local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])  ##改成了数组local_model_updaters如上所示
            ##分别为：w=model.state_dict(), loss=epoch_loss=所有train_ep三种loss分别的平均值, acc=acc_val.item()=最后一轮最后一批的acc, 
            ##    protos=agg_protos_label=最后一轮（train_ep）的{label0:[proto0, proto1, ...], ...}5个labels，每个有约99个protos
            w, loss, acc, protos = local_model_updaters[idx].update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)  ##每个labels对应的protos求均值

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos  ## A dictionary where each entry contains a client's protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

            local_acc.append(copy.deepcopy(acc)) ##
            
        # update global weights
        local_weights_list = local_weights
        
        ##global_protos = proto_aggregation(local_protos)

        ##added
        '''
        if round == 0:
            print("Initial Accuracy")
            acc_list_l, acc_list_g, _ = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list,
                                                               user_groups_lt, global_protos)
            print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
                np.mean(acc_list_g), np.std(acc_list_g)))
            print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
                np.mean(acc_list_l), np.std(acc_list_l)))
        '''
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global weights
        global_protos = proto_aggregation(local_protos) ##原位置

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        acc_avg = sum(local_acc) / len(local_acc)  ##
        train_accuracy.append(acc_avg)  ##
        print(f'{args.alg}, {args.mode}, {args.dataset} |\n {round} round | train_loss : {loss_avg:4f} | train_acc : {acc_avg:4f} |\n') ##
        acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos, my_flag=False) ##
        test_loss.append(float(np.mean(loss_list)))
        test_accuracy_g.append(np.mean(acc_list_g))
        test_accuracy_l.append(np.mean(acc_list_l))
        print(f'test_loss : {test_loss[round]:4f} | test_acc_g : {test_accuracy_g[round]:4f} | test_acc_l : {test_accuracy_l[round]:4f} |\n') ##

    summary_writer.close() ##补充
    ##loss_list的entry是每个client的loss2
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))
    with open(args.record_file, "a") as f:
        f.write("[\n")
        json.dump(str(args), f)
        f.write(",\n")
        json.dump({"train_loss": train_loss}, f)
        f.write(",\n")
        json.dump({"train_acc": train_accuracy}, f)
        f.write(",\n")
        json.dump({"test_loss": test_loss}, f)
        f.write(",\n")
        json.dump({"test_acc_g": test_accuracy_g}, f)
        f.write(",\n")
        json.dump({"test_acc_l": test_accuracy_l}, f)

        acc_mean_g, acc_std_g = np.mean(acc_list_g), np.std(acc_list_g)
        f.write(",\n" + '\"For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}\"'.format(acc_mean_g, acc_std_g) + ',\n')
        json.dump(list(acc_list_l), f)
        f.write(",\n" + '\"For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}\"'.format(np.mean(acc_list_l), np.std(acc_list_l)) + ',\n')
        json.dump(list(acc_list_g), f)
        f.write(",\n" + '\"For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}\"'.format(np.mean(loss_list), np.std(loss_list)) + ',\n')
        json.dump(list(map(float, loss_list)), f)
        f.write(",\n")
        json.dump({"final_result": (acc_mean_g, acc_std_g)}, f)
        f.write("\n],\n")
    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)

def FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('./tensorboard/'+ args.dataset +'_fedproto_mh_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []
    test_loss, test_accuracy_g, test_accuracy_l = [], [], [] ##my_add

    local_model_updaters = []  ##
    for idx in idxs_users:  ##
        local_model_updaters.append(LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx]))
    
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos, local_acc = [], [], {}, []  ##
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            ##local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            ##w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            w, loss, acc, protos = local_model_updaters[idx].update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round) ##
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))

            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

            local_acc.append(copy.deepcopy(acc)) ##

        # update global weights
        local_weights_list = local_weights
        ##global_protos = proto_aggregation(local_protos)

        ##added
        '''
        if round == 0:
            print("Initial Accuracy")
            acc_list_l, acc_list_g, _ = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list,
                                                               user_groups_lt, global_protos)
            print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
                np.mean(acc_list_g), np.std(acc_list_g)))
            print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
                np.mean(acc_list_l), np.std(acc_list_l)))
        '''
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global protos
        global_protos = proto_aggregation(local_protos) ##原位置

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        acc_avg = sum(local_acc) / len(local_acc)  ##
        train_accuracy.append(acc_avg)  ##
        print(f'{args.alg}, {args.mode}, {args.dataset} |\n {round} round | train_loss : {loss_avg:4f} | train_acc : {acc_avg:4f} |\n') ##
        acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos, my_flag=False) ##
        test_loss.append(float(np.mean(loss_list)))
        test_accuracy_g.append(np.mean(acc_list_g))
        test_accuracy_l.append(np.mean(acc_list_l))
        print(f'test_loss : {test_loss[round]:4f} | test_acc_g : {test_accuracy_g[round]:4f} | test_acc_l : {test_accuracy_l[round]:4f} |\n') ##

    summary_writer.close() ##补充
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos) ##增加loss_list
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    with open(args.record_file, "a") as f:
        f.write("[\n")
        json.dump(str(args), f)
        f.write(",\n")
        json.dump({"train_loss": train_loss}, f)
        f.write(",\n")
        json.dump({"train_acc": train_accuracy}, f)
        f.write(",\n")
        json.dump({"test_loss": test_loss}, f)
        f.write(",\n")
        json.dump({"test_acc_g": test_accuracy_g}, f)
        f.write(",\n")
        json.dump({"test_acc_l": test_accuracy_l}, f)

        acc_mean_g, acc_std_g = np.mean(acc_list_g), np.std(acc_list_g)
        f.write(",\n" + '\"For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}\"'.format(acc_mean_g, acc_std_g) + ',\n')
        json.dump(list(acc_list_l), f)
        f.write(",\n" + '\"For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}\"'.format(np.mean(acc_list_l), np.std(acc_list_l)) + ',\n')
        json.dump(list(acc_list_g), f)
        f.write(",\n" + '\"For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}\"'.format(np.mean(loss_list), np.std(loss_list)) + ',\n')
        json.dump(list(map(float, loss_list)), f)
        f.write(",\n")
        json.dump({"final_result": (acc_mean_g, acc_std_g)}, f)
        f.write("\n],\n")