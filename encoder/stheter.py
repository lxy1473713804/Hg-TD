import torch
import math
import numpy as np
import pandas as pd
from torch import nn
from timm.models.vision_transformer import trunc_normal_
#from stencoder import STEncoder
from torch.nn import functional as F


class SpatialTemporalHeter(nn.Module):
    def __init__(self, feat_dim, flow_day, device=None):
        super().__init__()
        self.device = device
        self.W1 = nn.Parameter(torch.randn(feat_dim, feat_dim)).to(device)
        self.W2 = nn.Parameter(torch.randn(feat_dim, feat_dim)).to(device)
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        self.embed_dim = feat_dim
        self.mean = AvgRead()
        #self.mean = Linear(128)
        self.trasdim = TransformerDim(feat_dim, flow_day, device=device)
    
    def forward(self, spatial_unmasked, spatial_origin, time_unmasked, time_origin):
        #input: (batch_size, num_nodes, feat_dim , embed_dim)
        #1,67,112,128
        batch_size, num_nodes, feat_dim, embed_dim = spatial_unmasked.shape
        spatial_unmasked = self.mean(spatial_unmasked).squeeze(0)# (num_nodes, feat_dim )
        spatial_origin = self.mean(spatial_origin).squeeze(0)
        time_unmasked = self.mean(time_unmasked).squeeze(0)
        time_origin = self.mean(time_origin).squeeze(0)
        self.embed_dim = feat_dim

        # heter_spatial_unmasked =  torch.matmul(torch.matmul(spatial_unmasked, self.W1), torch.transpose(spatial_unmasked, 0, 1))
        heter_spatial_unmasked =  torch.einsum('ij,jm,mn->in', spatial_unmasked, self.W1, spatial_unmasked.T)
        heter_spatial_unmasked = (heter_spatial_unmasked + heter_spatial_unmasked.transpose(0, 1)) / 2
        heter_spatial_origin =  torch.matmul(torch.matmul(spatial_origin, self.W2), torch.transpose(spatial_origin, 0, 1))
        heter_spatial_origin = (heter_spatial_origin + heter_spatial_origin.transpose(0, 1)) / 2
        # 归一化处理
        min_val = heter_spatial_origin.min()  # 计算最小值
        max_val = heter_spatial_origin.max()  # 计算最大值
        # 应用归一化公式
        heter_spatial_origin = (heter_spatial_origin - min_val) / (max_val - min_val)
        # 归一化处理
        min_val = heter_spatial_unmasked.min()  # 计算最小值
        max_val = heter_spatial_unmasked.max()  # 计算最大值
        # 应用归一化公式
        heter_spatial_unmasked = (heter_spatial_unmasked - min_val) / (max_val - min_val)
        identity_matrix = torch.eye(num_nodes, device=self.device)
        diag_tensor = torch.diag(torch.diag(heter_spatial_unmasked)).to(self.device)
        heter_spatial_unmasked = heter_spatial_unmasked + identity_matrix - diag_tensor
        heter_spatial_origin = heter_spatial_origin + identity_matrix - diag_tensor

        heter_time_unmasked =  self.trasdim(time_unmasked, num_nodes, feat_dim)#358,112
        heter_time_origin =  self.trasdim(time_origin, num_nodes, feat_dim)
        
        Loss_heter_spatial = F.l1_loss(heter_spatial_unmasked, heter_spatial_origin)
        Loss_heter_time = F.l1_loss(heter_time_unmasked, heter_time_origin)
        # 归一化处理
        #heter_spatial_unmasked = (heter_spatial_unmasked - heter_spatial_unmasked.min()) / (heter_spatial_unmasked.max() - heter_spatial_unmasked.min() + 1e-8)
        #heter_time_unmasked = (heter_time_unmasked - heter_time_unmasked.min()) / (heter_time_unmasked.max() - heter_time_unmasked.min() + 1e-8)
        return heter_spatial_unmasked, heter_time_unmasked,Loss_heter_spatial, Loss_heter_time


class AvgRead(nn.Module):
    def __init__(self):
        super(AvgRead, self).__init__()
        self.relu = nn.ReLU()  # 使用 ReLU 激活

    def forward(self, h):
        '''Apply an average on graph.
        :param h: hidden representation, (batch_size, num_nodes, feat_dim , embed_dim)
        :return s: summary, (batch_size, num_nodes, feat_dim )
        '''
        s = torch.max(h, dim=3).values
        s = self.relu(s) 
        return s

class TransformerDim(nn.Module):
    def __init__(self, feat_dim, flow_day, device=None):
        super(TransformerDim, self).__init__()
        self.Dim = feat_dim//flow_day
        self.W = nn.Parameter(torch.randn(feat_dim//flow_day, feat_dim//flow_day)).to(device)
        self.flow_day = flow_day
        self.device = device
        
    def forward(self, time_flow_data, num_nodes, feat_dim):
        #input: ( num_nodes, feat_dim)
        heter_flow_data = []
        flow_data = time_flow_data.view(num_nodes, self.flow_day, self.Dim) # (num_nodes, flow_day, feat_dim//flow_day)
        # 遍历第二维度（28 个特征），分别取出每个特征
        for i in range(flow_data.size(1)):  # tensor.size(1) 是 28
            flow_feature = flow_data[:, i, :]  # 取出第 i 个特征
            #print(f"Feature {i + 1} shape:", flow_feature.shape)  # 输出特征的形状
            # heter_flow_feature =  torch.matmul(torch.matmul(flow_feature, self.W), torch.transpose(flow_feature, 0, 1))
            heter_flow_feature =  torch.einsum('ij,jm,mn->in', flow_feature, self.W, flow_feature.T)
            heter_flow_feature = (heter_flow_feature + heter_flow_feature.transpose(0, 1)) / 2
                        # 创建一个 N*N 的单位矩阵
            min_val = heter_flow_feature.min()  # 计算最小值
            max_val = heter_flow_feature.max()  # 计算最大值
            # 应用归一化公式
            heter_flow_feature = (heter_flow_feature - min_val) / (max_val - min_val)
            identity_matrix = torch.eye(num_nodes, device=self.device)
            diag_tensor = torch.diag(torch.diag(heter_flow_feature)).to(self.device)
            heter_flow_feature = heter_flow_feature + identity_matrix - diag_tensor
            heter_flow_data.append(heter_flow_feature)
        # 使用 torch.stack 将列表中的张量堆叠成一个张量
        result_tensor = torch.stack(heter_flow_data, dim=-1)  # dim=-1 表示在最后一个维度堆叠
        return result_tensor



if __name__ == '__main__':
    data = torch.randn(67, 28,288).unsqueeze(0)
    print(data.shape)
    Smodel = STEncoder(patch_size=144, in_channels=1, embed_dim=128, num_layers=2, num_heads=4,
                mlp_ratio=4, dropout=0.1, mask_ratio=0.2, decoder_depth=2, stride=72, padding=36, spatial=True)
    spatial_unmasked, spatial_origin, spatial_unmasked_index, spatial_masked_index = Smodel(data)
    spatial_unmasked = spatial_unmasked.transpose(1, 2)
    spatial_origin = spatial_origin.transpose(1, 2)
    print(spatial_unmasked.shape, spatial_origin.shape)


        # if spatial = False 时间编码
    Tmodel = STEncoder(patch_size=144, in_channels=1, embed_dim=128, num_layers=2, num_heads=4,
                    mlp_ratio=4, dropout=0.1, mask_ratio=0.2, decoder_depth=2, stride=72, padding=36, spatial=False)

    time_unmasked, time_origin, time_unmasked_index, time_masked_index = Tmodel(data)
    print(time_unmasked.shape, time_origin.shape)


    # #时空异质性编码
    batch_size, num_nodes, feat_dim, embed_dim = spatial_unmasked.shape
    print(batch_size, num_nodes, feat_dim, embed_dim)
    heter_model = SpatialTemporalHeter(feat_dim, 28)
    heter_spatial_unmasked, heter_spatial_origin, heter_time_unmasked, heter_time_origin,Loss_heter_spatial, Loss_heter_time = heter_model(spatial_unmasked, spatial_origin, time_unmasked, time_origin)
    print(heter_spatial_unmasked.shape,heter_spatial_origin.shape,heter_time_unmasked.shape, heter_time_origin.shape,Loss_heter_spatial, Loss_heter_time)