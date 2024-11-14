import torch
import torch.nn as nn
from typing import Optional
import blitz
import blitz.modules
from blitz.modules import BayesianLinear
from blitz.modules import BayesianEmbedding
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator
from blitz.modules.base_bayesian_module import BayesianModule
from blitz.modules.weight_sampler import TrainableRandomDistribution, PriorWeightDistribution

def vec_combine(vector):
    # 确保输入的每个值都是 PyTorch 张量
    if not all(isinstance(v, torch.Tensor) for v in vector):
        raise ValueError("All elements in the vector must be PyTorch tensors.")
    
    # 使用 unsqueeze 在指定的位置添加新维度
    return (vector[0].unsqueeze(1).unsqueeze(2) +
            vector[1].unsqueeze(0).unsqueeze(2) +
            vector[2].unsqueeze(0).unsqueeze(1))


class CPFactor(BayesianModule):
    def __init__(self, 
            shape: tuple, 
            prior_sigma_1: float = 0.1,
            prior_sigma_2: float = 0.4,
            prior_pi: float = 1.0,
            posterior_mu_init: float = 0.0,
            posterior_rho_init: float = -7.0,
            prior_dist: Optional[str] = None) -> None:
        super(CPFactor, self).__init__()


        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        #parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        self.weight_mu = nn.Parameter(torch.randn(*shape, dtype=torch.float64) * 0.1 + posterior_mu_init)
        self.weight_rho = nn.Parameter(torch.randn(*shape, dtype=torch.float64) * 0.1 + posterior_rho_init)
        self.weight_sampler = TrainableRandomDistribution(self.weight_mu, self.weight_rho)

        # Priors (as BBP paper)
        self.weight_prior_dist = PriorWeightDistribution(self.prior_pi, 
                                                        self.prior_sigma_1,
                                                        self.prior_sigma_2,
                                                        dist = self.prior_dist)
        self.log_prior = 0.0
        self.log_variational_posterior = 0.0

    def forward(self):
        # Sample the weights and forward it
        w = self.weight_sampler.sample()

        # Get the complexity cost
        self.log_variational_posterior = self.weight_sampler.log_posterior()
        self.log_prior = self.weight_prior_dist.log_prior(w)

        return w
    

@variational_estimator
class HGTD(nn.Module):
    def __init__(self, dim1, dim2, dim3, rank, lamda=0.001) -> None:
        super(HGTD, self).__init__()
        # 设定核心张量的维度
        self.dim1, self.dim2, self.dim3 = dim1, dim2, dim3
        self.lamda = lamda
        self.N, self.D, self.T = CPFactor((dim1, 1, 1, rank)), CPFactor((1, dim2, 1, rank)), CPFactor((1, 1, dim3, rank))
        self.N_bias, self.D_bias, self.T_bias = CPFactor((dim1, )), CPFactor((dim2, )), CPFactor((dim3, ))
    
    def forward(self,flow_missing, flow_missing_mask, heter_spatial_unmasked, heter_time_unmasked):
        N, D, T = self.N.forward(), self.D.forward(), self.T.forward()
        N_bias, D_bias, T_bias = self.N_bias.forward(), self.D_bias.forward(), self.T_bias.forward()
        Y = torch.sum(N * D * T, dim=-1)
        bias = vec_combine([N_bias, D_bias, T_bias])
        pred = Y + bias

        loss1  = torch.norm(pred[~flow_missing_mask] - flow_missing[~flow_missing_mask])
        ## 正则项损失
        loss2 = self.lamda*(torch.norm(N) + torch.norm(D) + torch.norm(T))  
        N_squeezed = N.squeeze(1).squeeze(1) 
        loss3 =  self.lamda *torch.norm((heter_spatial_unmasked - N_squeezed @ N_squeezed.T), p=2)#heter_spatial_origin N*N
            # 将所有相似度矩阵堆叠成一个四维张量，形状为 (67, 67, 28)
        result_tensor = torch.einsum('ni,mi,dj->nmd',N.squeeze(1).squeeze(1), N.squeeze(1).squeeze(1), D.squeeze(0).squeeze(1))
        loss4 =self.lamda* torch.norm((heter_time_unmasked - result_tensor), p=2)#heter_tim
        # loss5 = (torch.norm(N_bias) + torch.norm(D_bias) + torch.norm(T_bias)) * 0.0001
        # loss6 = kl_divergence_from_nn(self) * 0.0001
        loss = loss1  +loss2 +loss3 +loss4

        return loss, pred
    
    def predict(self):
        with torch.no_grad():
            N, D, T = self.N.forward(), self.D.forward(), self.T.forward()
            N_bias, D_bias, T_bias = self.N_bias.forward(), self.D_bias.forward(), self.T_bias.forward()
            Y = torch.sum(N * D * T, dim=-1)
            bias = vec_combine([N_bias, D_bias, T_bias])
            pred = Y + bias

        return pred
    

class HGTD_FK(nn.Module):
    def __init__(self, dim1, dim2, dim3, rank) -> None:
        super(HGTD_FK, self).__init__()
        # 设定核心张量的维度
        self.dim1, self.dim2, self.dim3 = dim1, dim2, dim3

        self.N, self.D, self.T = CPFactor((dim1, 1, 1, rank)), CPFactor((1, dim2, 1, rank)), CPFactor((1, 1, dim3, rank))
        self.N_bias, self.D_bias, self.T_bias = CPFactor((dim1, )), CPFactor((dim2, )), CPFactor((dim3, ))
    
    def forward(self, flow_missing, flow_missing_mask,day_index, heter_spatial_unmasked, heter_time_unmasked):
        N, D, T = self.N.forward(), self.D.forward(), self.T.forward()
        N_bias, D_bias, T_bias = self.N_bias.forward(), self.D_bias.forward(), self.T_bias.forward()
        Y = torch.sum(N * D[:, day_index, :, :] * T, dim=-1)
        bias = vec_combine([N_bias, D_bias, T_bias])
        pred = Y + bias

        loss1  = torch.norm(pred[~flow_missing_mask] - flow_missing[~flow_missing_mask])
        ## 正则项损失
        loss2 = 0.1*(torch.norm(N) + torch.norm(D[:, day_index, :, :]) + torch.norm(T))  
        N_squeezed = N.squeeze(1).squeeze(1) 
        loss3 =  0.0001 *torch.norm((heter_spatial_unmasked -N_squeezed @ N_squeezed.T), p=2)#heter_spatial_origin N*N
            # 将所有相似度矩阵堆叠成一个四维张量，形状为 (67, 67, 28)
        result_tensor = torch.einsum('ni,mi,dj->nmd',N.squeeze(1).squeeze(1), N.squeeze(1).squeeze(1), D[:, day_index, :, :].squeeze(0).squeeze(1))
        loss4 = 0.0001* torch.norm((heter_time_unmasked - result_tensor), p=2)#heter_tim
        # loss5 = (torch.norm(N_bias) + torch.norm(D_bias) + torch.norm(T_bias)) * 0.0001
        # loss6 = kl_divergence_from_nn(self) * 0.0001
        loss = loss1  +loss2 +loss3 +loss4

        return loss, pred
    
    def predict(self):
        with torch.no_grad():
            N, D, T = self.N.forward(), self.D.forward(), self.T.forward()
            N_bias, D_bias, T_bias = self.N_bias.forward(), self.D_bias.forward(), self.T_bias.forward()
            Y = torch.sum(N * D * T, dim=-1)
            bias = vec_combine([N_bias, D_bias, T_bias])
            pred = Y + bias

        return pred
    
