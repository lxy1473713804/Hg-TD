import os
import sys
import time
import datetime
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import functional as F

#不分块的张量分解
class TensorDecomposition(nn.Module):
    def __init__(self, dim1, dim2, dim3,lambda1,lambda2,lambda3,lambda4) -> None:
        super(TensorDecomposition, self).__init__()
        # 设定核心张量的维度
        dimI, dimJ, dimK = 10, 10, 10
        self.dim1, self.dim2, self.dim3 = dim1, dim2, dim3
        self.lambda1, self.lambda2, self.lambda3, self.lambda4 = lambda1, lambda2, lambda3, lambda4
        # 使用较小的随机值初始化S, X, Y, Z, U
        self.N = nn.Parameter(torch.rand(dim1, dimI, requires_grad=True) / 10)
        self.D = nn.Parameter(torch.rand(dim2, dimJ, requires_grad=True) / 10)
        self.T = nn.Parameter(torch.rand(dim3, dimK, requires_grad=True) / 10)
        self.C = nn.Parameter(torch.rand(dimI, dimJ, dimK, requires_grad=True) / 10)
    
    def forward(self, source, heter_spatial_unmasked, heter_spatial_origin, heter_time_unmasked, heter_time_origin, modeltype):
        #print(X.shape)
        #R = source[..., 0]#(67,28,288)
        print(self.N.shape,self.D.shape,self.T.shape,self.C.shape)
        pred = torch.einsum('ni,dj,tk,ijk->ndt', self.N, self.D, self.T, self.C)
        Flow = source.reshape(self.dim1, self.dim2, self.dim3)
        
        if modeltype == "self-supervised":
            ## 正则项损失
            loss1 = torch.norm(pred[Flow!=0] - Flow[Flow!=0])
            loss2 = self.lambda2 * (torch.norm(self.N) + torch.norm(self.D) + torch.norm(self.T) + torch.norm(self.C))#正则项
            loss3 = self.lambda3 * torch.norm((heter_spatial_unmasked - self.N @ self.N.T), p=2)#heter_spatial_origin N*N
            # 将所有相似度矩阵堆叠成一个四维张量，形状为 (67, 67, 28)
            result_tensor = torch.einsum('ni,mi,dj->nmd', self.N, self.N, self.D)
            loss4 = self.lambda4 * torch.norm((heter_time_unmasked - result_tensor), p=2)#heter_time_origin N*N*D
            loss = loss1 + loss2 + loss3 + loss4
            return loss, pred
        elif modeltype == "no-self-supervised":
            loss1  = torch.norm(pred[Flow!=0] - Flow[Flow!=0])
            ## 正则项损失
            loss2 = self.lambda2 * (torch.norm(self.N) + torch.norm(self.D) + torch.norm(self.T) + torch.norm(self.C))#正则项            
            loss = loss1 + loss2 
            return loss, pred
    def predict(self):
        with torch.no_grad():
            pred = torch.einsum('ni,dj,tk,ijk->ndt', self.N, self.D, self.T, self.C)

        return pred  
    
    # def predict(self):
    #     with torch.no_grad():
    #         N = self.N.detach().cpu().numpy()
    #         D = self.D.detach().cpu().numpy() 
    #         T = self.T.detach().cpu().numpy()
    #         C = self.C.detach().cpu().numpy()
    #         pred = np.einsum('ni,dj,tk,ijk->ndt', N, D, T, C)
    #         pred = torch.from_numpy(pred)
    #     return pred


#分块的张量分解算法
class TensorDecomposition2(nn.Module):
    def __init__(self, dim1, dim2, dim3, lambda1=0.0001, lambda2=0.0001, lambda3=0.0001, lambda4=0.0001) -> None:
        super(TensorDecomposition2, self).__init__()
        # 设定核心张量的维度
        dimI, dimJ, dimK = 15, 10, 30
        self.dim1, self.dim2, self.dim3 = dim1, dim2, dim3
        self.lambda1, self.lambda2, self.lambda3, self.lambda4 = lambda1, lambda2, lambda3, lambda4
        # 使用较小的随机值初始化S, X, Y, Z, U
        self.N = nn.Parameter(torch.rand(dim1, dimI, requires_grad=True) / 10)
        self.D = nn.Parameter(torch.rand(dim2, dimJ, requires_grad=True) / 10)
        self.T = nn.Parameter(torch.rand(dim3, dimK, requires_grad=True) / 10)
        self.C = nn.Parameter(torch.rand(dimI, dimJ, dimK, requires_grad=True) / 10)
    
    def forward(self, source, flow_mask, day_index, heter_spatial_unmasked=None, heter_spatial_origin=None, heter_time_unmasked=None, heter_time_origin=None, modeltype="self-supervised"):
        #print(X.shape)
        #R = source[..., 0]#(67,28,288)
        pred = torch.einsum('ni,dj,tk,ijk->ndt', self.N, self.D[day_index, ...], self.T, self.C)
        #batch_size = day_index.shape
        Flow = source.squeeze(0)
        Mask = flow_mask.squeeze(0)
        if modeltype == "self-supervised":
            ## 正则项损失
            loss1 = torch.norm(pred[~Mask] - Flow[~Mask])
            loss2 = self.lambda2 * (torch.norm(self.N) + torch.norm(self.D[day_index, ...]) + torch.norm(self.T) + torch.norm(self.C))#正则项
            loss3 = self.lambda3 * torch.norm((heter_spatial_unmasked - self.N @ self.N.T), p=2)#heter_spatial_origin N*N
            # 将所有相似度矩阵堆叠成一个四维张量，形状为 (67, 67, 28)
            result_tensor = torch.einsum('ni,mi,dj->nmd', self.N, self.N, self.D[day_index, ...])
            loss4 = self.lambda4 * torch.norm((heter_time_unmasked - result_tensor), p=2)#heter_time_origin N*N*D
            loss = loss1 + loss2 + loss3 + loss4
            return loss, pred
        elif modeltype == "no-self-supervised":
            loss1  = torch.norm(pred[~Mask] - Flow[~Mask])
            ## 正则项损失
            loss2 = self.lambda2 * (torch.norm(self.N) + torch.norm(self.D[day_index, ...]) + torch.norm(self.T) + torch.norm(self.C))#正则项            
            loss = loss1 + loss2 
            return loss, pred
    
    
    def predict(self):
        with torch.no_grad():
            # N = self.N.detach().cpu().numpy()
            # D = self.D.detach().cpu().numpy() 
            # T = self.T.detach().cpu().numpy()
            # C = self.C.detach().cpu().numpy()
            # pred = np.einsum('ni,dj,tk,ijk->ndt', N, D, T, C)
            # pred = torch.from_numpy(pred)
            # pred = pred.to(self.N.device)
            N = self.N.detach().cpu()
            D = self.D.detach().cpu()
            T = self.T.detach().cpu()
            C = self.C.detach().cpu()
            pred = torch.einsum('ni,dj,tk,ijk->ndt', N, D, T, C)
            pred = pred.to(self.N.device)
        return pred
        

#修改后的张量分解方法
class TensorDecomposition3(nn.Module):
    def __init__(self, dim1, dim2, dim3, lambda1=0.0001, lambda2=0.0001, lambda3=0.0001, lambda4=0.0001) -> None:
        super(TensorDecomposition2, self).__init__()
        # 设定核心张量的维度
        self.rank = 20 
        self.dim1, self.dim2, self.dim3 = dim1, dim2, dim3
        self.lambda1, self.lambda2, self.lambda3, self.lambda4 = lambda1, lambda2, lambda3, lambda4
        # 使用较小的随机值初始化S, X, Y, Z, U
        self.N = nn.Parameter(torch.rand(dim1, self.rank, requires_grad=True) / 10)
        self.D = nn.Parameter(torch.rand(dim2, self.rank, requires_grad=True) / 10)
        self.T = nn.Parameter(torch.rand(dim3, self.rank, requires_grad=True) / 10)
        #self.C = nn.Parameter(torch.rand(dimI, dimJ, dimK, requires_grad=True) / 10)
    
    def forward(self, source, flow_mask, day_index, heter_spatial_unmasked=None, heter_spatial_origin=None, heter_time_unmasked=None, heter_time_origin=None, modeltype="self-supervised"):
        #print(X.shape)
        #R = source[..., 0]#(67,28,288)
        pred = torch.einsum('ni,dj,tk->ndt', self.N, self.D[day_index, :], self.T)
        #batch_size = day_index.shape
        Flow = source.squeeze(0)
        Mask = flow_mask.squeeze(0)
        if modeltype == "self-supervised":
            ## 正则项损失
            loss1 = torch.norm(pred[~Mask] - Flow[~Mask])
            loss2 = self.lambda2 * (torch.norm(self.N) + torch.norm(self.D[day_index, ...]) + torch.norm(self.T) + torch.norm(self.C))#正则项
            loss3 = self.lambda3 * torch.norm(heter_spatial_unmasked - self.N @ self.N.T)#heter_spatial_origin N*N
            # 将所有相似度矩阵堆叠成一个四维张量，形状为 (67, 67, 28)
            result_tensor = torch.einsum('ni,mi,dj->nmd', self.N, self.N, self.D[day_index, ...])
            loss4 = self.lambda4 * torch.norm(heter_time_unmasked - result_tensor)#heter_time_origin N*N*D
            loss = loss1 + loss2 + loss3 + loss4
            return loss, pred
        elif modeltype == "no-self-supervised":
            loss1  = torch.norm(pred[~Mask] - Flow[~Mask])
            ## 正则项损失
            loss2 = self.lambda2 * (torch.norm(self.N) + torch.norm(self.D[day_index, ...]) + torch.norm(self.T) + torch.norm(self.C))#正则项            
            loss = loss1 + loss2 
            return loss, pred
    
    
    def predict(self):
        with torch.no_grad():
            N = self.N.detach().cpu()
            D = self.D.detach().cpu()
            T = self.T.detach().cpu()
            pred = torch.einsum('ni,dj,tk->ndt', N, D, T)
            pred = pred.to(self.N.device)
        return pred
    

if __name__ == "__main__":
    lambda1 = 0.0001
    lambda2 = 0.0001
    lambda3 = 0.0001
    lambda4 = 0.0001
    tensor_model = TensorDecomposition(358, 91, 288, lambda1, lambda2, lambda3, lambda4)

    flow_random_missing = torch.randn(358, 91, 288)
    # interval = 7
    # step = 1
    # day_index = torch.arange(1, 1+interval)
    heter_spatial_unmasked = torch.rand(358,358, requires_grad=True)
    heter_spatial_origin = torch.randn(358,358, requires_grad=True)
    heter_time_unmasked = torch.rand(358,358,91, requires_grad=True)
    heter_time_origin = torch.randn(358,358,91, requires_grad=True)

    loss, pred = tensor_model(flow_random_missing, heter_spatial_unmasked, heter_spatial_origin, heter_time_unmasked, heter_time_origin, "self-supervised")
    print(loss)
