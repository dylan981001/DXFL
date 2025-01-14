from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


    
class NegLoss_new(nn.Module):

    def __init__(self):
        super(NegLoss_new, self).__init__()

    def forward(self, features, labels, center, args):
        DATA_nclass = {'mnist': 10, 'cifar10': 10, 'svhn': 10, 'fmnist': 10, 'celeba': 2,
            'cifar100': 100, 'tinyimagenet': 200, 'femnist': 26, 'emnist': 47,
            'xray': 2}
        device = args.device
        Data_nclasses = DATA_nclass[args.dataset]

        # 중심값을 GPU 메모리에 올리고 정규화
        center = torch.stack(center, dim=1).to(device)  # [256, n_classes]
        center = F.normalize(center, p=2, dim=0)  # [256, n_classes]

        # 라벨들 제외한 다른 클래스 라벨을 빠르게 처리
        batch_size = features.shape[0]
        all_classes = torch.arange(Data_nclasses, device=device).unsqueeze(0)  # [1, n_classes]
        labels = labels.unsqueeze(1)  # [batch_size, 1]

        # 자신 클래스 제외한 나머지 클래스 라벨들 가져오기
        different_labels = all_classes.repeat(batch_size, 1)  # [batch_size, n_classes]
        mask = different_labels != labels  # [batch_size, n_classes]

        # 해당 배치의 각 샘플에 대해 feature와 center 간의 곱셈 계산 (자신 클래스 제외)
        current_centers = center.T.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, n_classes, 256]
        masked_centers = current_centers[mask].view(batch_size, Data_nclasses - 1, -1)  # [batch_size, n_classes-1, 256]

        features = features.unsqueeze(1)  # [batch_size, 1, 256]
        dot_products = torch.bmm(features, masked_centers.permute(0, 2, 1)).squeeze(1)  # [batch_size, n_classes-1]

        # neg_list: 각 샘플에서 자신 클래스를 제외한 다른 클래스와의 최대 유사도
        neg_list = dot_products.max(dim=1)[0].unsqueeze(1)  # [batch_size, 1]

        # neg_labels는 모든 샘플에 대해 1로 설정
        neg_labels = torch.ones(batch_size, 1, device=device)

        # logits 생성
        logits = torch.cat((neg_list, neg_labels), dim=1)  # [batch_size, 2]

        return logits
