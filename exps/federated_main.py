#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

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

#from lib.models.resnet import resnet18
from lib.options import args_parser
#from lib.update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt
#from lib.models.models import CNNMnist, CNNFemnist
#from lib.utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem
from lib.utils import exp_details
from exps.myTrainer import MyTrainer

#from fedmethods import fedBaselines   ##my

import json  ##



def main(start_time, args):
    ##start_time = time.time()

    ##args = args_parser()
    exp_details(args)
    trainer = MyTrainer(args) ##
    trainer.begin_train()

def adjust_args(args, alg, dataset, rounds, stdev, local_bs, record_file, SSL, use_UH, save_proto, mode):
    args.SSL = SSL

    args.alg = alg
    args.rounds = rounds  #
    args.dataset = dataset
    args.stdev = stdev
    args.local_bs = local_bs
    args.record_file=record_file
    args.save_proto = save_proto
    args.mode = mode
    args.use_UH = use_UH ##
    return args

def start_main(args, alg, dataset, rounds, stdev, local_bs, record_file, SSL=False, use_UH=False, save_proto=True, mode='task_heter'):
    start_time = time.time()
    args = adjust_args(args, alg, dataset, rounds, stdev, local_bs, record_file, SSL=SSL, use_UH=use_UH, save_proto=save_proto, mode=mode)
    
    if dataset[0:5] == 'cifar':
        args.num_channels = 3
        args.ld = 0.1  ## !
        #args.lr = 0.001 ##
    else :
        args.num_channels = 1
        args.ld = 1 ## !
    
    if dataset == 'femnist':
        args.num_classes = 62  ##!
    args.stdev = stdev
    #for e in range(stdev, stdev + 2):
        #args.stdev = e
    for i in range(3, 6):
        args.ways = i
        main(start_time, args)
    with open(args.record_file, "a") as f:
        f.write("\n")


def start_main_ssl(args, alg, dataset, rounds, stdev, local_bs, record_file, SSL=True, use_UH=False, save_proto=False, mode='task_heter'):
    start_time = time.time()
    
    args = adjust_args(args, alg, dataset, rounds, stdev, local_bs, record_file, SSL=SSL, use_UH=use_UH, save_proto=save_proto, mode=mode)
    if dataset[0:5] == 'cifar':
        args.num_channels = 3
        args.ld = 0.1  ## !
    else:
        args.num_channels = 1
        args.ld = 1 ## !
    if dataset == 'femnist':
        args.num_classes = 62  ##!
    
    #args.stdev = stdev
    args.ways = 3

    args.use_resnet_dropout = True  ##
    '''
    args.SSL = False
    args.positiveRate = 1
    args.pu_weight = 0.0
    main(start_time, args)
    with open(args.record_file, "a") as f:
        f.write("\n")
    '''
    
    args.SSL = True
    j = 0.5
    for i in range(4):
        args.positiveRate = j
        for puW in np.arange(0.0, 0.21, 0.1):
            args.pu_weight = puW
            #args.ld = 1 + puW
            main(start_time, args)
        for puW in np.arange(0.5, 1.1, 0.5):
            args.pu_weight = puW
            #args.ld = 1 + puW
            main(start_time, args)
        if (j == 0.5):
            j = 0.2
        else :
            j /= 2
        with open(args.record_file, "a") as f:
            f.write("\n")

    with open(args.record_file, "a") as f:
        f.write("\n\n")


if __name__ == '__main__':

    start_time = time.time()

    args = args_parser()

    main(start_time, args)
    
    with open(args.record_file, "a") as f:
        f.write("\n")
    


    '''
    start_main(args, alg='fedproto', dataset='mnist', rounds=100, stdev=2, local_bs=4, record_file='records_UH_proto.json', use_UH=True)

    start_main(args, alg='fedproto', dataset='cifar10', rounds=110, stdev=1, local_bs=32, record_file='records_UH_proto.json', use_UH=True)

    start_main(args, alg='fedavg', dataset='mnist', rounds=150, stdev=2, local_bs=4, record_file='records_UH_fedavg.json', use_UH=True)

    start_main(args, alg='fedavg', dataset='cifar10', rounds=150, stdev=1, local_bs=32, record_file='records_UH_avg.json', use_UH=True)

    #start_main_ssl(args, alg='fedproto', dataset='cifar10', rounds=150, stdev=1, local_bs=32, record_file='records_ssl_UH_proto.json', SSL=True, use_UH=True)

    #start_main_ssl(args, alg='fedavg', dataset='cifar10', rounds=150, stdev=1, local_bs=32, record_file='records_ssl_UH_avg.json', SSL=True, use_UH=True)
    '''

    '''
    start_main_ssl(args, alg='fedproto', dataset='cifar10', rounds=150, stdev=1, local_bs=32, record_file='records_ssl_proto.json', SSL=True)

    start_main_ssl(args, alg='fedavg', dataset='cifar10', rounds=150, stdev=1, local_bs=32, record_file='records_ssl_avg.json', SSL=True)
    '''

    '''
    start_main_ssl(args, alg='fedproto', dataset='mnist', rounds=100, stdev=2, local_bs=32, record_file='records_ssl_proto.json', SSL=True)

    start_main_ssl(args, alg='fedavg', dataset='mnist', rounds=100, stdev=2, local_bs=32, record_file='records_ssl_avg.json', SSL=True)
    '''
    '''
    #picture_2
    
    start_main(args, alg='fedproto', dataset='mnist', rounds=100, stdev=2, local_bs=4, record_file='records_2.json')

    start_main(args, alg='fedavg', dataset='mnist', rounds=150, stdev=2, local_bs=4, record_file='records_2.json')
    '''
    '''
    start_main(args, alg='fesem', dataset='mnist', rounds=150, stdev=2, local_bs=4, record_file='records_2.json')
    
    start_main(args, alg='fedper', dataset='mnist', rounds=100, stdev=2, local_bs=4, record_file='records_2.json')
    '''

    #####




    '''
    start_main(args, alg='fedproto', dataset='femnist', rounds=120, stdev=1, local_bs=8, record_file='records_proto.json')
    
    start_main(args, alg='fedproto', dataset='femnist', rounds=120, stdev=1, local_bs=8, record_file='records_proto.json', mode='model_heter')

    start_main(args, alg='fedlocal', dataset='femnist', rounds=120, stdev=1, local_bs=8, record_file='records_local.json')

    start_main(args, alg='fesem', dataset='femnist', rounds=200, stdev=1, local_bs=8, record_file='records_sem.json')
    
    start_main(args, alg='fedprox', dataset='femnist', rounds=300, stdev=1, local_bs=8, record_file='records_prox.json')

    start_main(args, alg='fedper', dataset='femnist', rounds=250, stdev=1, local_bs=8, record_file='records_per.json')
    
    start_main(args, alg='fedavg', dataset='femnist', rounds=300, stdev=1, local_bs=8, record_file='records_fedavg.json')
    '''
    


    

    


    '''
    start_main(args, alg='fedper', dataset='mnist', rounds=100, stdev=2, local_bs=4, record_file='records_per.json')

    start_main(args, alg='fedper', dataset='cifar10', rounds=130, stdev=1, local_bs=32, record_file='records_per.json')
    
    start_main(args, alg='fesem', dataset='mnist', rounds=150, stdev=2, local_bs=4, record_file='records_sem.json')

    start_main(args, alg='fesem', dataset='cifar10', rounds=120, stdev=1, local_bs=32, record_file='records_sem.json')

    
    start_main(args, alg='fedprox', dataset='mnist', rounds=110, stdev=2, local_bs=4, record_file='records_prox.json')

    start_main(args, alg='fedprox', dataset='cifar10', rounds=150, stdev=1, local_bs=32, record_file='records_prox.json')
    
    start_main(args, alg='fedproto', dataset='mnist', rounds=100, stdev=2, local_bs=4, record_file='records_proto.json')

    start_main(args, alg='fedproto', dataset='cifar10', rounds=110, stdev=1, local_bs=32, record_file='records_proto.json')

    start_main(args, alg='fedproto', dataset='mnist', rounds=100, stdev=2, local_bs=4, record_file='records_proto.json', mode='model_heter')

    start_main(args, alg='fedproto', dataset='cifar10', rounds=110, stdev=1, local_bs=32, record_file='records_proto.json', mode='model_heter')

    start_main(args, alg='fedavg', dataset='mnist', rounds=150, stdev=2, local_bs=4, record_file='records_fedavg.json')

    start_main(args, alg='fedavg', dataset='cifar10', rounds=150, stdev=1, local_bs=32, record_file='records_fedavg.json')
    
    start_main(args, alg='fedlocal', dataset='mnist', rounds=100, stdev=2, local_bs=4, record_file='records_local.json')

    start_main(args, alg='fedlocal', dataset='cifar10', rounds=110, stdev=1, local_bs=32, record_file='records_local.json')
    '''


    '''
    def start_main_3(args, alg, dataset, rounds, stdev, local_bs, record_file, save_proto=False, mode='task_heter'):
        start_time = time.time()
        args = adjust_args(args, alg, dataset, rounds, stdev, local_bs, record_file, save_proto=save_proto, mode='task_heter')

        args.ways = 3
        if dataset != 'mnist':
            print("error dataset!\n")
            return 
        args.num_channels = 1
        args.ld = 1

        args.train_shots_max = 110 ##!
        for i in range(5, 19, 5):
            args.shots = i
            main(start_time, args)
        for i in range(20, 39, 10):
            args.shots = i
            main(start_time, args)
        for i in range(40, 119, 20):
            args.shots = i
            main(start_time, args)
        args.shots = 120
        args.train_shots_max = 130
        main(start_time, args)
        with open(args.record_file, "a") as f:
            f.write("\n")
    '''
    #start_main_3(args, alg='fedproto', dataset='mnist', rounds=100, stdev=2, local_bs=4, record_file='records_proto_3.json')

    #start_main_3(args, alg='fedavg', dataset='mnist', rounds=150, stdev=2, local_bs=4, record_file='records_fedavg_3.json')