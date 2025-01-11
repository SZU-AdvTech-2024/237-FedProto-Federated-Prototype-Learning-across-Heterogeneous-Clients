#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=100,
                        help="number of rounds of training")  ##100 120 110
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.04,
                        help='the fraction of clients: C')
    parser.add_argument('--train_ep', type=int, default=1,
                        help="the number of local episodes: E")
    parser.add_argument('--local_bs', type=int, default=4,
                        help="local batch size: B")  ##原4,论文中为8; cifar:32
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--alg', type=str, default='fedproto', help="algorithms")  ##'fedproto'
    parser.add_argument('--mode', type=str, default='task_heter', help="mode")  ##'task_heter' 'model_heter'
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")  ##1
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--data_dir', type=str, default='./data/', help="directory of dataset")

    parser.add_argument('--resnet_dir', type=str, default='./data/resnet/', help="directory of resnet")  ##my_add

    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")  ##'mnist'
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=0, help='verbose') ##原：1
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--test_ep', type=int, default=10, help="num of test episodes for evaluation")

    # Local arguments
    parser.add_argument('--ways', type=int, default=3, help="num of classes")
    parser.add_argument('--shots', type=int, default=100, help="num of shots")
    parser.add_argument('--train_shots_max', type=int, default=110, help="num of shots")
    parser.add_argument('--test_shots', type=int, default=15, help="num of shots")
    parser.add_argument('--stdev', type=int, default=1, help="stdev of ways")  ##2
    parser.add_argument('--ld', type=float, default=1, help="weight of proto loss")  ##1
    parser.add_argument('--ft_round', type=int, default=10, help="round of fine tuning")

    parser.add_argument('--SSL', type=bool, default=False) ##
    parser.add_argument('--pu_weight', type=float, default=0.5) ##
    parser.add_argument('--positiveRate', type=float, default=0.5, help="label rate of dataset") ##
    parser.add_argument('--save_proto', type=bool, default=True, help="wether save protos") ##
    parser.add_argument('--record_file', type=str, default="records.json") ##
    parser.add_argument('--use_UH', type=bool, default=False, help="wether use Uniform Head in resnet") ##
    parser.add_argument('--smooth_weight', type=float, default=0.9) ##

    parser.add_argument('--use_resnet_dropout', type=bool, default=True)  ##
    args = parser.parse_args()
    return args
