import copy
import torch
from lib.models.resnet import resnet18
from lib.options import args_parser
from lib.update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt
from lib.models.models import CNNMnist, CNNFemnist
from lib.utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

import json

def write_records(args, train_loss, train_accuracy, test_loss, test_accuracy_l, acc_list_l, loss_list):
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
        json.dump({"test_acc_l": test_accuracy_l}, f)
        acc_mean_l, acc_std_l = np.mean(acc_list_l), np.std(acc_list_l)
        f.write(",\n" + '\"For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}\"'.format(acc_mean_l, acc_std_l) + ',\n')
        json.dump(list(acc_list_l), f)

        f.write(",\n" + '\"For all users, mean of test loss is {:.5f}, std of test loss is {:.5f}\"'.format(np.mean(loss_list), np.std(loss_list)) + ',\n')
        json.dump(list(map(float, loss_list)), f)
        f.write(",\n")
        json.dump({"final_result": (acc_mean_l, acc_std_l)}, f)
        f.write("\n],\n")


def FedAvg_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, local_model_updaters):
    summary_writer = SummaryWriter('./tensorboard/'+ args.dataset +'_fedavg_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []
    test_loss, test_accuracy_g, test_accuracy_l = [], [], [] ##my_add


    
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_acc = [], [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:  ##每个client
            ##local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            ##分别为：w=model.state_dict(),  acc=acc_val.item()=最后一轮最后一批的acc, 
            w, loss, acc = local_model_updaters[idx].update_weights(idx, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            #agg_protos = agg_func(protos)  ##每个labels对应的protos求均值

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_acc.append(copy.deepcopy(acc)) ##

            #local_protos[idx] = agg_protos  ## A dictionary where each entry contains a client's protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss, round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)

        # update global weights
        local_weights_list = average_weights(local_weights)

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
            '''
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model
            '''
            local_model_list[idx].load_state_dict(local_weights_list[idx], strict=True) ##

        # update global weights
        #global_protos = proto_aggregation(local_protos) ##原位置

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        acc_avg = sum(local_acc) / len(local_acc)
        train_accuracy.append(acc_avg)
        print(f'{args.alg}, {args.dataset} |\n {round} round | train_loss : {loss_avg:4f} | train_acc : {acc_avg:4f} |\n') ##
        acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos, my_flag=False) ##
        test_loss.append(float(np.mean(loss_list)))
        #test_accuracy_g.append(np.mean(acc_list_g))
        test_accuracy_l.append(np.mean(acc_list_l))
        print(f'test_loss : {test_loss[round]:4f} | test_acc_l : {test_accuracy_l[round]:4f} |\n') ##
    
    summary_writer.close() ##补充
    ##loss_list的entry是每个client的loss2
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt)
    #print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users , mean of test loss is {:.5f}, std of test loss is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

    write_records(args, train_loss, train_accuracy, test_loss, test_accuracy_l, acc_list_l, loss_list)

    
    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)
    


def FedLocal_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('./tensorboard/'+ args.dataset +'_fedlocal_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []
    test_loss, test_accuracy_g, test_accuracy_l = [], [], [] ##my_add

    local_model_updaters = []  ##
    for idx in idxs_users:  ##
        local_model_updaters.append(LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx]))
    
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_acc = [], [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:  ##每个client
            ##local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            ##分别为：w=model.state_dict(),  acc=acc_val.item()=最后一轮最后一批的acc, 
            w, loss, acc = local_model_updaters[idx].update_weights(idx, local_model_list[idx], global_round=round)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_acc.append(copy.deepcopy(acc)) ##

            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss, round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
        

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        acc_avg = sum(local_acc) / len(local_acc)
        train_accuracy.append(acc_avg)
        print(f'{args.alg}, {args.dataset} |\n {round} round | train_loss : {loss_avg:4f} | train_acc : {acc_avg:4f} |\n') ##
        acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos, my_flag=False) ##
        test_loss.append(float(np.mean(loss_list)))
        #test_accuracy_g.append(np.mean(acc_list_g))
        test_accuracy_l.append(np.mean(acc_list_l))
        print(f'test_loss : {test_loss[round]:4f} | test_acc_l : {test_accuracy_l[round]:4f} |\n') ##
    
    summary_writer.close() ##补充
    '''
    local_weights_list = average_weights(local_weights)  ##
    for idx in idxs_users: ##
        local_model_list[idx].load_state_dict(local_weights_list[idx], strict=True) ##
    '''

    ##loss_list的entry是每个client的loss2
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt)
    #print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users , mean of test loss is {:.5f}, std of test loss is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

    write_records(args, train_loss, train_accuracy, test_loss, test_accuracy_l, acc_list_l, loss_list)
    


def FedProx_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('./tensorboard/'+ args.dataset +'_fedaprox_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)
    
    train_loss, train_accuracy = [], []
    test_loss, test_accuracy_g, test_accuracy_l = [], [], [] ##my_add

    local_weights_dict = {}  ##

    local_model_updaters = []  ##
    for idx in idxs_users:  ##
        local_model_updaters.append(LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx]))

        local_weights_dict[idx] = copy.deepcopy(local_model_list[idx].state_dict())  ##add_copy
    
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_acc = [], [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:  ##每个client
            ##local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            ##分别为：w=model.state_dict(),  acc=acc_val.item()=最后一轮最后一批的acc, 
            w, loss, acc = local_model_updaters[idx].update_weights_prox(idx, local_weights_dict, local_model_list[idx], global_round=round)
            #agg_protos = agg_func(protos)  ##每个labels对应的protos求均值

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_acc.append(copy.deepcopy(acc)) ##

            #local_protos[idx] = agg_protos  ## A dictionary where each entry contains a client's protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss, round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)

        # update global weights
        local_weights_dict = dict(enumerate(average_weights(local_weights)))
        
        for idx in idxs_users: ##
            '''
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_dict[idx], strict=True)
            local_model_list[idx] = local_model
            '''
            local_model_list[idx].load_state_dict(local_weights_dict[idx], strict=True) ##

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        acc_avg = sum(local_acc) / len(local_acc)
        train_accuracy.append(acc_avg)
        print(f'{args.alg}, {args.dataset} |\n {round} round | train_loss : {loss_avg:4f} | train_acc : {acc_avg:4f} |\n') ##
        acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos, my_flag=False) ##
        test_loss.append(float(np.mean(loss_list)))
        #test_accuracy_g.append(np.mean(acc_list_g))
        test_accuracy_l.append(np.mean(acc_list_l))
        print(f'test_loss : {test_loss[round]:4f} | test_acc_l : {test_accuracy_l[round]:4f} |\n') ##
    
    summary_writer.close() ##补充


    ##loss_list的entry是每个client的loss2
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt)
    #print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users , mean of test loss is {:.5f}, std of test loss is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

    write_records(args, train_loss, train_accuracy, test_loss, test_accuracy_l, acc_list_l, loss_list)



def FedPer_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('./tensorboard/'+ args.dataset +'_fedper_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []
    test_loss, test_accuracy_g, test_accuracy_l = [], [], [] ##my_add

    local_model_updaters = []  ##
    for idx in idxs_users:  ##
        local_model_updaters.append(LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx]))
    
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_acc = [], [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:  ##每个client
            ##分别为：w=model.state_dict(),  acc=acc_val.item()=最后一轮最后一批的acc, 
            w, loss, acc = local_model_updaters[idx].update_weights(idx, model=copy.deepcopy(local_model_list[idx]), global_round=round)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_acc.append(copy.deepcopy(acc)) ##

            #local_protos[idx] = agg_protos  ## A dictionary where each entry contains a client's protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss, round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)

        # update global weights
        local_weights_list = average_weights_per(local_weights)

        for idx in idxs_users:
            local_model_list[idx].load_state_dict(local_weights_list[idx], strict=True) ##

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        acc_avg = sum(local_acc) / len(local_acc)
        train_accuracy.append(acc_avg)
        print(f'{args.alg}, {args.dataset} |\n {round} round | train_loss : {loss_avg:4f} | train_acc : {acc_avg:4f} |\n') ##
        acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos, my_flag=False) ##
        test_loss.append(float(np.mean(loss_list)))
        #test_accuracy_g.append(np.mean(acc_list_g))
        test_accuracy_l.append(np.mean(acc_list_l))
        print(f'test_loss : {test_loss[round]:4f} | test_acc_l : {test_accuracy_l[round]:4f} |\n') ##
    
    summary_writer.close() ##补充
    ##loss_list的entry是每个client的loss2
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt)
    #print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users , mean of test loss is {:.5f}, std of test loss is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

    write_records(args, train_loss, train_accuracy, test_loss, test_accuracy_l, acc_list_l, loss_list)

    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)


def FeSEM_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, n_list):
    summary_writer = SummaryWriter('./tensorboard/'+ args.dataset +'_fesem_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []
    test_loss, test_accuracy_g, test_accuracy_l = [], [], [] ##my_add

    local_model_updaters = []  ##
    for idx in idxs_users:  ##
        local_model_updaters.append(LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx]))
    
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_acc = [], [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:  ##每个client
            ##分别为：w=model.state_dict(),  acc=acc_val.item()=最后一轮最后一批的acc, 
            w, loss, acc = local_model_updaters[idx].update_weights(idx, model=copy.deepcopy(local_model_list[idx]), global_round=round)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_acc.append(copy.deepcopy(acc)) ##

            #local_protos[idx] = agg_protos  ## A dictionary where each entry contains a client's protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss, round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)

        # update global weights
        local_weights_list = average_weights_sem(local_weights, n_list)

        for idx in idxs_users:
            local_model_list[idx].load_state_dict(local_weights_list[idx], strict=True) ##

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        acc_avg = sum(local_acc) / len(local_acc)
        train_accuracy.append(acc_avg)
        print(f'{args.alg}, {args.dataset} |\n {round} round | train_loss : {loss_avg:4f} | train_acc : {acc_avg:4f} |\n') ##
        acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos, my_flag=False) ##
        test_loss.append(float(np.mean(loss_list)))
        #test_accuracy_g.append(np.mean(acc_list_g))
        test_accuracy_l.append(np.mean(acc_list_l))
        print(f'test_loss : {test_loss[round]:4f} | test_acc_l : {test_accuracy_l[round]:4f} |\n') ##
    
    summary_writer.close() ##补充
    ##loss_list的entry是每个client的loss2
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt)
    #print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users , mean of test loss is {:.5f}, std of test loss is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

    write_records(args, train_loss, train_accuracy, test_loss, test_accuracy_l, acc_list_l, loss_list)
    
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)