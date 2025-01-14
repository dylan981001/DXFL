import torch
import torch.nn as nn
import torch.nn.functional as F

class PosLoss_new(nn.Module):

    def __init__(self):
        super(PosLoss_new, self).__init__()

    def forward(self, features, labels, center, args):
        device = args.device

        # 프로토타입(센터)를 한 번에 스택하고 정규화
        center = torch.stack(center, dim=-1).to(device)  # [256, n_classes]
        center = F.normalize(center, p=2, dim=0)  # [256, n_classes]

        # 레이블에 해당하는 센터를 한 번에 선택하여 전치
        batch_centers = center[:, labels].T  # [batch_size, 256]

        # features와 batch_centers를 정규화 후 코사인 유사도 계산
        features = F.normalize(features, p=2, dim=1)  # [batch_size, 256]
        pos_list = torch.sum(features * batch_centers, dim=1)  # [batch_size]

        # 레이블 생성 후 연결
        logits = torch.cat((pos_list.unsqueeze(1), torch.ones_like(pos_list.unsqueeze(1))), dim=1)  # [batch_size, 2]

        return logits
