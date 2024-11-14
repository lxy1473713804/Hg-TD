import torch
import os
import sys
# 将项目的根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from torch import nn
from timm.models.vision_transformer import trunc_normal_
from mask import patch, maskgenerator, positional_encoding, transformer_layers


class STEncoder(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim, num_layers, 
                 num_heads, mlp_ratio, dropout, mask_ratio, decoder_depth, stride, padding, spatial=False):
        super(STEncoder, self).__init__()
        self.patch_size = patch_size # patch的大小
        self.in_channels = in_channels #二维卷积的通道数
        self.embed_dim = embed_dim #嵌入维度
        self.num_layers = num_layers # Transformer层的数量
        self.num_heads = num_heads #多头注意力的头数
        self.mlp_ratio = mlp_ratio #MLP的比率
        self.dropout = dropout #dropout比率
        self.mask_ratio = mask_ratio #掩码比例
        self.decoder_depth = decoder_depth #解码器的深度
        self.spatial = spatial #设置空间上的缺失OR时间上的缺失
        self.stride = stride # 二维卷积的步幅
        self.padding = padding # 二维卷积的填充
        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.pos_mat = None
        self.patch_embed = patch.PatchEmbedding(patch_size, in_channels, embed_dim, stride, padding, norm_layer = None)
        self.positional_encoding = positional_encoding.PositionalEncoding()
        #self.mask_generator = MaskGenerator(mask_ratio)
        self.encoder = transformer_layers.TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)

    def mask_encoding(self, missing_flow_data):
        """
        Args:
            missing_flow_data(Tensor): Long-term historical traffic flow data with shape (batch_size, nodes, days, time_steps) B,N,D,T

        Returns:
            torch.Tensor: hidden states of masked tokens is zero
            list: unmasked token index
            list: masked token index
        """
        # input: B,N,D,T
        missing_flow_data = missing_flow_data.flatten(start_dim=-2) # B,N,T
        missing_flow_data = missing_flow_data.unsqueeze(2) # B, N, 1, T 特征C变为1

        if self.spatial:
            #x = long_term_history.permute(0, 3, 1, 2) # B, T, N, D   
            patches = self.patch_embed(missing_flow_data)  # B, N, d, P  1, 67, 288,  288
            patches = patches.transpose(-1, -2)  # B, N, P, d  1, 67, 288, 288
            batch_size, num_nodes, num_time, num_dim  =  patches.shape 

            # positional embedding
            patches, self.pos_mat = self.positional_encoding(patches)       # mask
            Maskg = maskgenerator.MaskGenerator(patches.shape[1], self.mask_ratio)

            unmasked_token_index, masked_token_index = Maskg.uniform_rand()
            #patches大小不变，但是masked_token_index变为0            
            #encoder_input = patches[:, unmasked_token_index, :, :] #  B, N, P, d
            encoder_input = patches.clone()
            encoder_input[:, masked_token_index, :, :] = 0 #  B, N, P, d
            encoder_input=encoder_input.transpose(-2,-3)#   B , p , N, d

            #print(encoder_input.shape)
            hidden_states_unmasked = self.encoder(encoder_input)
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size,num_time, -1, self.embed_dim)            
            return hidden_states_unmasked, unmasked_token_index, masked_token_index         
            #return hidden_states_unmasked, unmasked_token_index, masked_token_index
        
        if not self.spatial:
            patches = self.patch_embed(missing_flow_data)  # B, N, d, P   1, 67, 288, 288
            patches = patches.transpose(-1, -2)  # B, N, P, d   1, 67, 288, 288
            batch_size, num_nodes, num_time,num_dim  =  patches.shape

            # positional embedding
            patches , self.pos_mat = self.positional_encoding(patches)   # mask
            Maskg = maskgenerator.MaskGenerator(patches.shape[2], self.mask_ratio)
            unmasked_token_index, masked_token_index = Maskg.uniform_rand()
            encoder_input = patches.clone()
            encoder_input[:, :, masked_token_index, :] = 0   # B, N, P, d

            #print(encoder_input.shape)
            hidden_states_unmasked = self.encoder(encoder_input)
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)
            return hidden_states_unmasked, unmasked_token_index, masked_token_index
    
    def orinial_encoding(self, missing_flow_data):
        """
        Args:
            missing_flow_data(Tensor): Long-term historical traffic flow data with shape (batch_size, nodes, days, time_steps) B,N,D,T

        Returns:
            torch.Tensor: hidden states of flow data
        """
        # input: B,N,D,T
        missing_flow_data = missing_flow_data.flatten(start_dim=-2) # B,N,T
        missing_flow_data = missing_flow_data.unsqueeze(2) # B, N, 1, T 特征C变为1
        if self.spatial:
            #print(missing_flow_data.shape)
            patches = self.patch_embed(missing_flow_data)  # B, N, d, P  1, 67, 288,  288
            #print(patches.shape)
            patches = patches.transpose(-1, -2) # B, N, P, d  1, 67, 288, 288
            batch_size, num_nodes, num_time, num_dim  =  patches.shape 
            # positional embedding
            patches, self.pos_mat = self.positional_encoding(patches)        # mask
            encoder_input = patches.clone()
            #print(encoder_input.shape)
            hidden_states_origin = self.encoder(encoder_input)
            hidden_states_origin = self.encoder_norm(hidden_states_origin).view(batch_size,num_time, -1, self.embed_dim)            
            return hidden_states_origin
        if not self.spatial:
            patches = self.patch_embed(missing_flow_data)  # B, N, d, P   1, 67, 288, 288
            patches = patches.transpose(-1, -2)  # B, N, P, d   1, 67, 288, 288
            batch_size, num_nodes, num_time,num_dim  =  patches.shape
            # positional embedding
            patches, self.pos_mat = self.positional_encoding(patches)      # mask
            encoder_input = patches.clone()
            #print(encoder_input.shape)
            hidden_states_origin = self.encoder(encoder_input)
            hidden_states_origin = self.encoder_norm(hidden_states_origin).view(batch_size, num_nodes, -1, self.embed_dim)
            return hidden_states_origin


    def forward(self, flow_data):
            # encoding
        hidden_states_unmasked, unmasked_token_index, masked_token_index = self.mask_encoding(flow_data)
        hidden_states_origin = self.orinial_encoding(flow_data)
        if self.spatial:
            hidden_states_unmasked = hidden_states_unmasked.transpose(1, 2)
            hidden_states_origin = hidden_states_origin.transpose(1, 2)
        #return hidden_states_unmasked, unmasked_token_index, masked_token_index   
        return hidden_states_unmasked, hidden_states_origin, unmasked_token_index, masked_token_index 
    

if __name__ == '__main__':
    data = torch.randn(67, 28,288)
    data =  data.unsqueeze(0)  # 变为 (1, 67, 28, 288)
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
    # data = torch.randn(67, 8064)
    # model = STEncoder(patch_size=144, in_channels=1, embed_dim=128, num_layers=2, num_heads=4,
    #             mlp_ratio=4, dropout=0.1, mask_ratio=0.2, decoder_depth=2, stride=72, padding=36, spatial=True)

    # spatial_unmasked, spatial_origin, spatial_unmasked_index, spatial_masked_index = model(data)
    # spatial_unmasked = spatial_unmasked.transpose(1, 2)
    # spatial_origin = spatial_origin.transpose(1, 2)
    # batch_size, num_nodes, feat_dim, embed_dim = spatial_unmasked.shape
    # print(batch_size, num_nodes, feat_dim, embed_dim)
    # print(spatial_unmasked.shape, spatial_origin.shape, len(spatial_unmasked_index), len(spatial_masked_index))
    # #spatial_unmasked.shape, spatial_origin.shape, len(spatial_unmasked_index), len(spatial_masked_index)