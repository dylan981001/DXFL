import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import time

from model import *
from utils import *
from utils import get_args as get_basic_args
from torch.optim.lr_scheduler import ExponentialLR
from PosLoss import PosLoss_new
from NegLoss import NegLoss_new

def get_args():
    basic_parser = get_basic_args()
    parser = argparse.ArgumentParser(parents=[basic_parser])
    # parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    # parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    # parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    # parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    # parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=256, help='input batch size for test (default: 256)')

    # parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    # parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    # parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    # parser.add_argument('--alg', type=str, default='fedavg',
    #                     help='communication strategy: fedavg/fedprox')
    # parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    # parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    # parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    # parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    # parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    # parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    # parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    # parser.add_argument('--beta', type=float, default=0.5,
    #                     help='The parameter for the dirichlet distribution for data partitioning')
    # # parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    # parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    # parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    # parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    # parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    # parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    # parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    # parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    # parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    # parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    # parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    # parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    # parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    # parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    # parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    # parser.add_argument('--loss', type=str, default='contrastive')
    # parser.add_argument('--save_model',type=int,default=0)
    # parser.add_argument('--use_project_head', type=int, default=1)
    # parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    args = parser.parse_args()
    return args

args=get_args()
def get_device(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    return device
device = get_device(args)
print('*'*20)
print('Device: ', device)
print('*'*20)


def init_nets(net_configs, n_parties, args, device=device):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    if args.normal_model:
        for net_i in range(n_parties):
            if args.model == 'simple-cnn':
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.to(device)
            nets[net_i] = net
    else:
        for net_i in range(n_parties):
            if args.use_project_head:
                net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
            else:
                net = ModelFedTennessee(args.model, n_classes)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.to(device)
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

def get_global_class_center(global_class_center_old, n_party, nets, args, net_dataidx_map, device=args.device, logger=None):
    DATA_nclass = {'mnist': 10, 'cifar10': 10, 'svhn': 10, 'fmnist': 10, 'celeba': 2,
               'cifar100': 100, 'tinyimagenet': 200, 'femnist': 26, 'emnist': 47,
               'xray': 2
               }
    clsnum = DATA_nclass[args.dataset]
    n_party = int(n_party*args.sample_fraction)
    class_count = torch.zeros((n_party, clsnum), device=device)
    class_feature_sum = torch.zeros((n_party, clsnum, 256), device=device)  # Shape: [n_party, clsnum, 256]

    for i, (net_id, net) in enumerate(nets.items()):
        net.to(device)
        dataidxs = net_dataidx_map[net_id]
        print(f"Calculate the class center of the Client {net_id}")


        train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size, args, dataidxs)

        with torch.no_grad():
            for x, target in (train_dl_local):
                x, target = x.to(device), target.to(device)
                _, features = net(x)

                unique_labels, label_counts = torch.unique(target, return_counts=True)
                for label, count in zip(unique_labels, label_counts):
                    label = int(label.item())
                    mask = (target == label)
                    class_feature_sum[i, label] += features[mask].sum(0)
                    class_count[i, label] += count

    print("Aggregate the global class center")


    total_class_counts = class_count.sum(dim=0)
    global_center = []

    for cls in range(clsnum):
        # Compute global class center
        centers_sum = class_feature_sum[:, cls].sum(dim=0)
        
        if total_class_counts[cls] > 0:
            global_center.append(centers_sum / total_class_counts[cls])
        else:
            # If no data for this class, use the previous global center or initialize with zeros
            global_center.append(global_class_center_old[cls] if global_class_center_old is not None else torch.zeros(256, device=device))

    return global_center

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device=device):
    net.to(device)
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0

    for _ in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

    test_acc, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=False, device=device)

    return test_acc


def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device=device):
    return_value=0
    net.to(device)
    
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
        scheduler = ExponentialLR(optimizer, gamma=0.998)
        
    criterion = nn.CrossEntropyLoss().to(device)

    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out, _ = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg
            print(f'fedprox_loss: {loss}')
            loss.backward()
            optimizer.step()

        scheduler.step()
    return return_value


def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device=device):
    return_value=0
    net.to(device)
    start_time = time.time()
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, momentum=0)
        scheduler = ExponentialLR(optimizer, gamma=0.998)

    criterion = nn.CrossEntropyLoss().to(device)
    global_net.to(device)

    cos=torch.nn.CosineSimilarity(dim=-1)

    for epoch in range(epochs):
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out, pro1 = net(x)
            _, pro2 = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1,1)

            for previous_net in previous_nets:
                _, pro3 = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

            logits /= temperature
            labels = torch.zeros(x.size(0)).to(device).long()

            loss2 = mu * criterion(logits, labels)
            
            loss1 = criterion(out, target)
            loss = loss1 + loss2
            print(f'loss: {loss}, moon_loss: {loss2}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            optimizer.step()

        scheduler.step()

    end_time = time.time()
    print('training time: ', end_time-start_time)
    return return_value


def train_net_feddec(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      round, device=device, global_class_center=None):
    return_value=0
    net.to(device)
    start_time = time.time()
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, momentum=0)
        scheduler = ExponentialLR(optimizer, gamma=0.998)

    criterion = nn.CrossEntropyLoss().to(device)

    criterion_attraction = PosLoss_new().to(device)
    criterion_repulsive = NegLoss_new().to(device)

    if round < args.comm_round:
        alpha = round / args.comm_round
    else:
        alpha = 1

    for epoch in range(epochs):
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out, pro1 = net(x)

            if round == 0:
                CEloss = criterion(out, target)
                loss = CEloss
                Posloss = 0
                Negloss = 0
            else:
                pos_labels = torch.ones(x.size(0), 1, device=device).long().squeeze(1)
                neg_labels = torch.zeros(x.size(0), 1, device=device).long().squeeze(1)
                CEloss = criterion(out, target)

                Posloss_ = criterion_attraction(
                    features=pro1,
                    labels=target,
                    center=global_class_center,
                    args=args)
                Posloss = criterion(Posloss_, pos_labels)
                
                Negloss_ = criterion_repulsive(
                    features=pro1,
                    labels=target,
                    center=global_class_center,
                    args=args)
                Negloss = criterion(Negloss_, neg_labels)
                
                if args.alpha_reversed == 0:    
                    loss = CEloss + mu * ((1 - alpha) * Posloss + (alpha) * Negloss)
                else:
                    loss = CEloss + mu * ((alpha) * Posloss + (1 - alpha) * Negloss)

                print(f'loss: {loss:.4f}, feddec_loss: {mu * ((1 - alpha) * Posloss + (alpha) * Negloss):.4f}, pos: {Posloss:.4f}, neg: {Negloss:.4f}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
                optimizer.step()

        scheduler.step()
    end_time = time.time()
    print('training time: ', end_time-start_time)
    return return_value


def train_net_fedavg(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device=device):
    return_value=0
    start_time = time.time()
    net.to(device)
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    for _ in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            
            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out, _ = net(x)
            loss = criterion(out, target)
            print(f'fedavg_loss: {loss}')

            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())
    
    net.eval()
    all_embeddings = []
    with torch.no_grad():
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            _, embeddings = net(images)
            all_embeddings.append(embeddings)
            all_labels.append(labels)
            print("Extracted Embeddings Shape:", embeddings.shape)  # [batch_size, embedding_dim]

    # 전체 데이터 합치기
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 클래스별 평균 계산
    num_classes = 10
    within_class_variance = []

    for cls in range(num_classes):
        class_embeddings = all_embeddings[all_labels == cls]
        class_mean = class_embeddings.mean(dim=0)
        variance = ((class_embeddings - class_mean) ** 2).sum(dim=1).mean().item()
        within_class_variance.append(variance)

    # 결과 출력
    print("Within-class Variance for Each Class:", within_class_variance)
    # break

    end_time = time.time()
    print('training time: ', end_time-start_time)

    return return_value



def group_by_class(batch_data, batch_labels):
    class_groups = {}
    
    for data, label in zip(batch_data, batch_labels):
        label = label.item()
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append(data)
    
    return class_groups



def fedrcl_loss(features, class_groups, device, temperature):
    features = features.to(device)
    loss = torch.tensor(0.0, device=device)
    batch_size = features.size(0)
    features = features.view(batch_size, -1)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    features_label = []
    for class_label, class_data in class_groups.items():
        for _ in range(len(class_data)):
            features_label.append(class_label)
    
    features_label = torch.tensor(features_label, device=device)
    
    labels_equal = features_label.unsqueeze(0) == features_label.unsqueeze(1)
    pos_mask = labels_equal & (torch.eye(batch_size, device=device) == 0)

    for i in range(batch_size):
        pos_sim = similarity_matrix[i, pos_mask[i]] / temperature
        pos_numerators = torch.exp(pos_sim)
        pos_denominator = torch.exp(similarity_matrix[i, labels_equal[i]] / temperature).sum() - torch.exp(similarity_matrix[i, i] / temperature)
        neg_denominator = torch.exp(similarity_matrix[i, ~labels_equal[i]] / temperature).sum() + 1e-9
        
        rcl_sim = similarity_matrix[i, pos_mask[i]]
        rcl_mask = rcl_sim > 0.7
        rcl_numerators = torch.exp(rcl_sim[rcl_mask] / temperature)
        if pos_numerators.numel() > 0:
            pos_loss = -torch.sum(torch.log(pos_numerators / (pos_denominator + neg_denominator)))
            rcl_loss = torch.sum(torch.log(rcl_numerators)) if rcl_numerators.numel() > 0 else torch.tensor(0.0, device=device)
            loss += pos_loss + rcl_loss
    
    return loss / batch_size

def combined_loss_fedrcl(temperature, feature_representations, class_groups, device=device):
    loss_cosine = [fedrcl_loss(features, class_groups, device, temperature) for features in feature_representations]
    loss_cosine = torch.mean(torch.stack(loss_cosine))
    return loss_cosine

def train_net_fedrcl(mu, temperature, net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device=device):
    start_time = time.time()
    net.to(device)
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, momentum=0)
        scheduler = ExponentialLR(optimizer, gamma=0.998)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0

    for _ in range(epochs):
        epoch_loss_collector = []
        for x, target in (train_dataloader):
            x, target = x.to(device), target.to(device)
            
            class_groups = group_by_class(x, target)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out, feature_representations = net(x)
            for i in range(len(feature_representations)):
                for j in range(len(feature_representations[i])):
                    print(f'i, j: {i}, {j}, {np.shape(feature_representations[i][j])}')
            print(net._modules)

            exit(1)
            loss_cls = criterion(out, target)
            loss_cosine = (mu * combined_loss_fedrcl(temperature, feature_representations, class_groups, device=device))
            loss = loss_cls + loss_cosine
            print(f'loss: {loss}, loss_fedrcl: {loss_cosine}')
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        scheduler.step()

    test_acc, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=False, device=device)
    end_time = time.time()
    print('training time: ', end_time-start_time)
    return test_acc



def tennessee_loss_all(features, class_groups, device, temperature):
    features = features.to(device)
    loss = torch.tensor(0.0, device=device)
    batch_size = features.size(0)
    features = features.view(batch_size, -1)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    features_label = []
    for class_label, class_data in class_groups.items():
        for _ in range(len(class_data)):
            features_label.append(class_label)
    
    features_label = torch.tensor(features_label, device=device)
    
    labels_equal = features_label.unsqueeze(0) == features_label.unsqueeze(1)
    pos_mask = labels_equal & (torch.eye(batch_size, device=device) == 0)

    for i in range(batch_size):
        pos_sim = similarity_matrix[i, pos_mask[i]] / temperature
        pos_numerators = torch.exp(pos_sim)
        pos_denominator = torch.exp(similarity_matrix[i, labels_equal[i]] / temperature).sum() - torch.exp(similarity_matrix[i, i] / temperature)
        neg_denominator = torch.exp(similarity_matrix[i, ~labels_equal[i]] / temperature).sum() + 1e-9
        
        if pos_numerators.numel() > 0:
            loss1 = -torch.sum(torch.log(pos_numerators / (pos_denominator + neg_denominator)))
        else: loss1 = torch.tensor(0.0, device=device)
        loss2 = -torch.log(1.0 / neg_denominator)
        loss += loss1 + loss2

    return loss / batch_size

def combined_loss_all(temperature, feature_representations, class_groups, device=device):
    loss_cosine = [tennessee_loss_all(features, class_groups, device, temperature) for features in feature_representations]
    loss_cosine = torch.mean(torch.stack(loss_cosine))
    return loss_cosine

def train_net_tennessee_all(mu, temperature, net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device=device):
    start_time = time.time()
    net.to(device)
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, momentum=0)
        scheduler = ExponentialLR(optimizer, gamma=0.998)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0

    for _ in range(epochs):
        epoch_loss_collector = []
        for x, target in (train_dataloader):
            x, target = x.to(device), target.to(device)
            
            class_groups = group_by_class(x, target)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out, feature_representations = net(x)
            loss_cls = criterion(out, target)
            loss_cosine = (mu * combined_loss_fedrcl(temperature, feature_representations, class_groups, device=device))
            loss = loss_cls + loss_cosine
            print(f'loss: {loss}, loss_tennessee_all: {loss_cosine}')
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        scheduler.step()

    test_acc, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=False, device=device)
    end_time = time.time()
    print('training time: ', end_time-start_time)
    return test_acc



def tennessee_loss_all_beta(features, class_groups, device, temperature):
    features = features.to(device)
    loss = torch.tensor(0.0, device=device)
    batch_size = features.size(0)
    features = features.view(batch_size, -1)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    features_label = []
    for class_label, class_data in class_groups.items():
        for _ in range(len(class_data)):
            features_label.append(class_label)
    
    features_label = torch.tensor(features_label, device=device)
    
    labels_equal = features_label.unsqueeze(0) == features_label.unsqueeze(1)
    pos_mask = labels_equal & (torch.eye(batch_size, device=device) == 0)

    for i in range(batch_size):
        pos_sim = similarity_matrix[i, pos_mask[i]] / temperature
        pos_numerators = torch.exp(pos_sim)
        pos_denominator = torch.exp(similarity_matrix[i, labels_equal[i]] / temperature).sum() - torch.exp(similarity_matrix[i, i] / temperature)
        neg_denominator = torch.exp(similarity_matrix[i, ~labels_equal[i]] / temperature).sum() + 1e-9
        
        rcl_sim = similarity_matrix[i, pos_mask[i]]
        rcl_mask = rcl_sim > 0.7
        rcl_numerators = torch.exp(rcl_sim[rcl_mask] / temperature)
        if pos_numerators.numel() > 0:
            loss1 = -torch.sum(torch.log(pos_numerators / (pos_denominator + neg_denominator)))
            rcl_loss = torch.sum(torch.log(rcl_numerators)) if rcl_numerators.numel() > 0 else torch.tensor(0.0, device=device)
        else:
            loss1 = torch.tensor(0.0, device=device)
            rcl_loss = torch.tensor(0.0, device=device)
        
        loss2 = -torch.log(1.0 / neg_denominator)
        loss += loss1 + loss2 + rcl_loss

    return loss / batch_size

def combined_loss_all_beta(temperature, feature_representations, class_groups, device=device):
    loss_cosine = [tennessee_loss_all_beta(features, class_groups, device, temperature) for features in feature_representations]
    loss_cosine = torch.mean(torch.stack(loss_cosine))
    return loss_cosine

def train_net_tennessee_all_beta(mu, temperature, net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device=device):
    start_time = time.time()
    net.to(device)
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, momentum=0)
        scheduler = ExponentialLR(optimizer, gamma=0.998)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0

    for _ in range(epochs):
        epoch_loss_collector = []
        for x, target in (train_dataloader):
            x, target = x.to(device), target.to(device)
            
            class_groups = group_by_class(x, target)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out, feature_representations = net(x)
            loss_cls = criterion(out, target)
            loss_cosine = (mu * combined_loss_fedrcl(temperature, feature_representations, class_groups, device=device))
            loss = loss_cls + loss_cosine
            print(f'loss: {loss}, loss_tennessee_all_beta: {loss_cosine}')
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        scheduler.step()

    test_acc, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=False, device=device)
    end_time = time.time()
    print('training time: ', end_time-start_time)
    return test_acc



def tennessee_loss_pos(features, class_groups, device, temperature):
    features = features.to(device)
    loss = torch.tensor(0.0, device=device)
    batch_size = features.size(0)
    features = features.view(batch_size, -1)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    features_label = []
    for class_label, class_data in class_groups.items():
        for _ in range(len(class_data)):
            features_label.append(class_label)
    
    features_label = torch.tensor(features_label, device=device)
    
    labels_equal = features_label.unsqueeze(0) == features_label.unsqueeze(1)
    pos_mask = labels_equal & (torch.eye(batch_size, device=device) == 0)

    for i in range(batch_size):
        pos_sim = similarity_matrix[i, pos_mask[i]] / temperature
        pos_numerators = torch.exp(pos_sim)
        pos_denominator = torch.exp(similarity_matrix[i, labels_equal[i]] / temperature).sum() - torch.exp(similarity_matrix[i, i] / temperature)
        neg_denominator = torch.exp(similarity_matrix[i, ~labels_equal[i]] / temperature).sum() + 1e-9
        
        if pos_numerators.numel() > 0:
            loss1 = -torch.sum(torch.log(pos_numerators / (pos_denominator)))
        else: loss1 = torch.tensor(0.0, device=device)
        loss2 = -torch.log(1.0 / neg_denominator)
        loss += loss1 + loss2

    return loss / batch_size

def combined_loss_pos(temperature, feature_representations, class_groups, device=device):
    loss_cosine = [tennessee_loss_pos(features, class_groups, device, temperature) for features in feature_representations]
    loss_cosine = torch.mean(torch.stack(loss_cosine))
    return loss_cosine

def train_net_tennessee_pos(mu, temperature, net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device=device):
    start_time = time.time()
    net.to(device)
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, momentum=0)
        scheduler = ExponentialLR(optimizer, gamma=0.998)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0

    for _ in range(epochs):
        epoch_loss_collector = []
        for x, target in (train_dataloader):
            x, target = x.to(device), target.to(device)
            
            class_groups = group_by_class(x, target)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out, feature_representations = net(x)
            loss_cls = criterion(out, target)
            loss_cosine = (mu * combined_loss_fedrcl(temperature, feature_representations, class_groups, device=device))
            loss = loss_cls + loss_cosine
            print(f'loss: {loss}, loss_tennessee_pos: {loss_cosine}')
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        scheduler.step()

    test_acc, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=False, device=device)
    end_time = time.time()
    print('training time: ', end_time-start_time)
    return test_acc



def tennessee_loss_pos_beta(features, class_groups, device, temperature):
    features = features.to(device)
    loss = torch.tensor(0.0, device=device)
    batch_size = features.size(0)
    features = features.view(batch_size, -1)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    features_label = []
    for class_label, class_data in class_groups.items():
        for _ in range(len(class_data)):
            features_label.append(class_label)
    
    features_label = torch.tensor(features_label, device=device)
    
    labels_equal = features_label.unsqueeze(0) == features_label.unsqueeze(1)
    pos_mask = labels_equal & (torch.eye(batch_size, device=device) == 0)

    for i in range(batch_size):
        pos_sim = similarity_matrix[i, pos_mask[i]] / temperature
        pos_numerators = torch.exp(pos_sim)
        pos_denominator = torch.exp(similarity_matrix[i, labels_equal[i]] / temperature).sum() - torch.exp(similarity_matrix[i, i] / temperature)
        neg_denominator = torch.exp(similarity_matrix[i, ~labels_equal[i]] / temperature).sum() + 1e-9
        
        rcl_sim = similarity_matrix[i, pos_mask[i]]
        rcl_mask = rcl_sim > 0.7
        rcl_numerators = torch.exp(rcl_sim[rcl_mask] / temperature)
        if pos_numerators.numel() > 0:
            loss1 = -torch.sum(torch.log(pos_numerators / (pos_denominator)))
            rcl_loss = torch.sum(torch.log(rcl_numerators)) if rcl_numerators.numel() > 0 else torch.tensor(0.0, device=device)
        else:
            loss1 = torch.tensor(0.0, device=device)
            rcl_loss = torch.tensor(0.0, device=device)

        loss2 = -torch.log(1.0 / neg_denominator)
        loss += loss1 + loss2 + rcl_loss

    return loss / batch_size

def combined_loss_pos_beta(temperature, feature_representations, class_groups, device=device):
    loss_cosine = [tennessee_loss_pos_beta(features, class_groups, device, temperature) for features in feature_representations]
    loss_cosine = torch.mean(torch.stack(loss_cosine))
    return loss_cosine

def train_net_tennessee_pos_beta(mu, temperature, net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device=device):
    start_time = time.time()
    net.to(device)
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, momentum=0)
        scheduler = ExponentialLR(optimizer, gamma=0.998)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0

    for _ in range(epochs):
        epoch_loss_collector = []
        for x, target in (train_dataloader):
            x, target = x.to(device), target.to(device)
            
            class_groups = group_by_class(x, target)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out, feature_representations = net(x)
            loss_cls = criterion(out, target)
            loss_cosine = (mu * combined_loss_fedrcl(temperature, feature_representations, class_groups, device=device))
            loss = loss_cls + loss_cosine
            print(f'loss: {loss}, loss_tennessee_pos_beta: {loss_cosine}')
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        scheduler.step()

    test_acc, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=False, device=device)
    end_time = time.time()
    print('training time: ', end_time-start_time)
    return test_acc



def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, server_c = None, clients_c = None, round=None, device=device, global_class_center=None, net_dl_map=None):

    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]

        train_dl_local = net_dl_map[net_id]
        n_epoch = args.epochs

        if args.alg == 'fedavg':
            _ = train_net_fedavg(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                    device=device)
        
        elif args.alg == 'fedprox':
            _ = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args, device=device)
        
        elif args.alg == 'moon':
            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            _ = train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
                                        args.optimizer, args.mu, args.temperature, args, round, device=device)

        elif args.alg == 'local_training':
            trainacc, _ = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        
        elif args.alg == 'fedrcl':
            _ = train_net_fedrcl(args.mu, args.temperature, net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)

        elif args.alg == 'tennessee_pos':
            _ = train_net_tennessee_pos(args.mu, args.temperature, net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)
        
        elif args.alg == 'tennessee_all':
            _ = train_net_tennessee_all(args.mu, args.temperature, net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)
    
        elif args.alg == 'tennessee_pos_beta':
            _ = train_net_tennessee_pos_beta(args.mu, args.temperature, net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)
        
        elif args.alg == 'tennessee_all_beta':
            _ = train_net_tennessee_all_beta(args.mu, args.temperature, net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)
        
        elif args.alg == 'feddec':
            _ = train_net_feddec(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args.mu, args,
                                        round, device=device, global_class_center=global_class_center)
        
        elif args.alg == 'fedrcl':
            _ = train_net_fedrcl()
        
        
    return nets


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        # argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")+'_'+str(args.epochs)+'_'+str(args.lr)+'_'+str(args.dataset)+'_'+str(args.model)
        argument_path = 'arguments_alg: '+str(args.alg)+'_data: '+str(args.dataset)+'_dist: '+str(args.beta)+'_mu: '+str(args.mu)+'_lr: '+str(args.lr)+'_clients: '+str(args.n_parties*args.sample_fraction)+'_tau: '+str(args.temperature)+'_alpha-reversed: '+str(args.alpha_reversed)
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    # device = load_device_num(args)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        # args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))+'_'+str(args.epochs)+'_'+str(args.lr)+'_'+str(args.dataset)+'_'+str(args.model)
        args.log_file_name = 'log_alg: '+str(args.alg)+'_data: '+str(args.dataset)+'_dist: '+str(args.beta)+'_mu: '+str(args.mu)+'_lr: '+str(args.lr)+'_clients: '+str(args.n_parties*args.sample_fraction)+'_tau: '+str(args.temperature)+'_alpha-reversed: '+str(args.alpha_reversed)
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


    seed = args.init_seed

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               args.test_batch_size,
                                                                               args)

    net_dl_map = {}
    for net_id in party_list:
        dataidxs = net_dataidx_map[net_id]
        train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size, args, dataidxs)
        net_dl_map[net_id] = train_dl_local

    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)


    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device=device)

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device=device)
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        # global_model.load_state_dict(torch.load(args.load_model_file))
        global_model.load_state_dict(next_global_model)

        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    flag = 0
    if args.alg == 'moon':
        print(args.alg)
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device=device)
                # checkpoint = torch.load(args.load_pool_file)

                for net_id, net in old_nets.items():
                    # net.load_state_dict(checkpoint['pool' + str(nets_id) + '_'+'net'+str(net_id)])
                    net.load_state_dict(next_nets)

                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            print('*'*20)
            print('Communication round: ', round+1)
            print('*'*20)
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            # global_w = global_model.state_dict()
            global_w = {key: value.clone().detach() for key, value in global_model.state_dict().items()}


            if args.server_momentum:
                # old_w = copy.deepcopy(global_model.state_dict())
                old_w = {key: value.clone().detach() for key, value in global_model.state_dict().items()}
            
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
                # print(f'net: {net}')


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device,  net_dl_map=net_dl_map)


            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]


            for net_id, net in enumerate(nets_this_round.values()):
                # net_para = net.state_dict()
                net_para = {key: value.clone().detach() for key, value in net.state_dict().items()}

                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            #summary(global_model.to(device), (3, 32, 32))



            # global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=False, device=device)
            # global_model.to('cpu')
            

            logger.info('>> Global Model Test accuracy: %f' % test_acc)



            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            next_global_model = {key: value.clone().detach() for key, value in global_model.state_dict().items()}
            next_nets = {key: value.clone().detach() for key, value in nets[0].state_dict().items()}

            if round == 0:
                mkdirs(args.modeldir+'fedcon/')
                if args.save_model:
                    torch.save(global_model.state_dict(), args.modeldir+'fedcon/global_model_'+args.log_file_name+'.pth')
                    torch.save(nets[0].state_dict(), args.modeldir+'fedcon/localmodel0'+args.log_file_name+'.pth')
                    for nets_id, old_nets in enumerate(old_nets_pool):
                        torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'fedcon/prev_model_pool_'+args.log_file_name+'.pth')
        if args.save_model:
            torch.save(global_model.state_dict(), args.modeldir+'fedcon/global_model_'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedcon/localmodel0'+args.log_file_name+'.pth')
            for nets_id, old_nets in enumerate(old_nets_pool):
                torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'fedcon/prev_model_pool_'+args.log_file_name+'.pth')

    elif args.alg == 'tennessee_all' or args.alg == 'tennessee_pos' or args.alg == 'tennessee_all_beta' or args.alg == 'tennessee_pos_beta' or args.alg == 'fedrcl' or args.alg =='fedavg':
        print(args.alg)
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            print('*'*20)
            print('Communication round: ', round+1)
            print('*'*20)
            party_list_this_round = party_list_rounds[round]

            global_w = {key: value.clone().detach() for key, value in global_model.state_dict().items()}

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device, net_dl_map=net_dl_map)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = {key: value.clone().detach() for key, value in net.state_dict().items()}
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)



            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=False, device=device)


            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            if round == 0:
                mkdirs(args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/')
            # global_model.to('cpu')
                if args.save_model:
                    torch.save(global_model.state_dict(), args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/'+'globalmodel'+args.log_file_name+'.pth')
                    torch.save(nets[0].state_dict(), args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/'+'localmodel0'+args.log_file_name+'.pth')
            torch.save(global_model.state_dict(), args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/'+'localmodel0'+args.log_file_name+'.pth')

    elif args.alg == 'feddec':
        print(args.alg)
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            print('*'*20)
            print('Communication round: ', round+1)
            print('*'*20)
            party_list_this_round = party_list_rounds[round]

            global_w = {key: value.clone().detach() for key, value in global_model.state_dict().items()}

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
            
            if flag == 0:
                global_class_center_old = None
                global_class_center = get_global_class_center(global_class_center_old, args.n_parties, nets_this_round,
                                                              args, net_dataidx_map, logger=logger)
                flag = 1
            
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device,
                            global_class_center=global_class_center, round=round, net_dl_map=net_dl_map)

            global_class_center_old = copy.deepcopy(global_class_center)
            global_class_center = get_global_class_center(global_class_center_old, args.n_parties, nets_this_round,
                                                          args, net_dataidx_map, logger=logger)
            
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            print(f'(args.n_parties * args.sample_fraction): {(args.n_parties * args.sample_fraction)}')

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = {key: value.clone().detach() for key, value in net.state_dict().items()}
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] / (args.n_parties * args.sample_fraction)
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] / (args.n_parties * args.sample_fraction)


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)



            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=False, device=device)


            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            if round == 0:
                mkdirs(args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/')
            # global_model.to('cpu')
                if args.save_model:
                    torch.save(global_model.state_dict(), args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/'+'globalmodel'+args.log_file_name+'.pth')
                    torch.save(nets[0].state_dict(), args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/'+'localmodel0'+args.log_file_name+'.pth')
            torch.save(global_model.state_dict(), args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/'+'localmodel0'+args.log_file_name+'.pth')

    elif args.alg == 'fedprox':
        print(args.alg)
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            print('*'*20)
            print('Communication round: ', round+1)
            print('*'*20)
            party_list_this_round = party_list_rounds[round]
            global_w = {key: value.clone().detach() for key, value in global_model.state_dict().items()}
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, global_model = global_model, device=device, net_dl_map=net_dl_map)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = {key: value.clone().detach() for key, value in net.state_dict().items()}

                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)





            test_acc, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=False, device=device)


            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            if round == 0:
                mkdirs(args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/')
            # global_model.to('cpu')
                if args.save_model:
                    torch.save(global_model.state_dict(), args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/'+'globalmodel'+args.log_file_name+'.pth')
                    torch.save(nets[0].state_dict(), args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/'+'localmodel0'+args.log_file_name+'.pth')
            torch.save(global_model.state_dict(), args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+args.alg+args.dataset+'_lr'+str(args.lr)+'_mu'+str(args.mu)+'_clients'+str(args.n_parties*args.sample_fraction)+'_alpha_reversed'+str(args.alpha_reversed)+'/'+'localmodel0'+args.log_file_name+'.pth')

    elif args.alg == 'local_training':

        local_train_net(nets, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, device=device, net_dl_map=net_dl_map)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(), args.modeldir + 'localmodel/'+'model'+str(net_id)+args.log_file_name+ '.pth')

    elif args.alg == 'all_in':
        nets, _, _ = init_nets(args.net_config, 1, args, device=device)
        # nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl, args.epochs, args.lr,
                                      args.optimizer, args, device=device)

        mkdirs(args.modeldir + 'all_in/')

        torch.save(nets[0].state_dict(), args.modeldir+'all_in/'+args.log_file_name+ '.pth')