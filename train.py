import os
import math
import torch
import json
import numpy as np
import pandas as pd

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import functional as F
from encoder.stencoder import STEncoder
from encoder.stheter import SpatialTemporalHeter
from tensor.selftensor import TensorDecomposition
from tensor.selftensor import TensorDecomposition2
from tensor.selftensor import TensorDecomposition3
from tensor.bystensor import HGTD
from config import Options
from datasets import FlowDataset
from evaluate  import evaluate
from torch.utils.data import Dataset, DataLoader

# 设置环境变量以启用 flash attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # 读取数据原始数据
    print(f'******************************** {args.dataset},{args.missing_type},{int(args.missing_ratio)} ,time_mask_ratio:{args.time_mask_ratio},bysrank:{args.bysrank},embed_dim:{args.embed_dim}********************************')
    flow_origin = np.load(os.path.join(args.root, args.dataset, f'origin_flow_data.npy'))
    flow_origin_mean = (flow_origin - flow_origin.min()) / (flow_origin.max() - flow_origin.min())
    flow_origin_mean = torch.from_numpy(flow_origin_mean).to(device)
    flow_origin = torch.from_numpy(flow_origin)
    dim1, dim2, dim3 = flow_origin_mean.shape
    # 读取缺失数据
    flow_missing_mean = np.load(os.path.join(args.root, args.dataset, args.missing_type, 
                                        f'{args.missing_type}{args.missing_ratio}_mean.npy'))
    flow_missing_mean = torch.from_numpy(flow_missing_mean).to(device)
    # 读取缺失位置
    flow_missing_mask = np.load(os.path.join(args.root, args.dataset, args.missing_type, 
                                        f'{args.missing_type}{args.missing_ratio}_mask.npy'))
    flow_missing_mask = torch.from_numpy(flow_missing_mask).reshape(*flow_origin_mean.shape).to(device)
    # print(flow_origin_mean.shape)
    # print(flow_missing_mean.shape)
    # print(flow_missing_mask.shape)

    #时空异质性建模
    STconv =int(((args.batch_size*dim3) - args.patch_size+2*args.padding)/args.stride+1)#
    Smodel = STEncoder(patch_size=args.patch_size, in_channels=1, embed_dim=args.embed_dim, num_layers=2, num_heads=4, mlp_ratio=4, 
                        dropout=0.1, mask_ratio= args.spatial_mask_ratio, decoder_depth=2, stride=args.stride, padding=args.padding, spatial=True).to(device)
    Tmodel = STEncoder(patch_size=args.patch_size, in_channels=1, embed_dim=args.embed_dim, num_layers=2, num_heads=4, mlp_ratio=4, 
                        dropout=0.1, mask_ratio=args.time_mask_ratio, decoder_depth=2, stride=args.stride, padding=args.padding, spatial=False).to(device)
    heter_model = SpatialTemporalHeter(STconv, args.batch_size,device=device).to(device)

    tensor_model = HGTD(*flow_missing_mean.shape, args.bysrank).to(device)

    # Defining the Optimizer
    params = list(tensor_model.parameters()) +  list(Smodel.parameters()) + list(Tmodel.parameters()) + list(heter_model.parameters())
    optimizer = optim.Adam(params, lr=args.lr, betas=(args.b1, args.b2))

    # Learning Rate Decline Function
    lr_decay_function = lambda epoch: (1 - epoch / args.n_epochs) ** 0.9
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_decay_function)

    tensor_model.train()
    Smodel.train()
    Tmodel.train()
    heter_model.train()

    pbar = tqdm(total=args.n_epochs)
    losses = []
    results = []
    rmse=0
    mae=0
    mape =0
    for epoch in range(1, args.n_epochs+1):
        optimizer.zero_grad()
        spatial_unmasked, spatial_origin, spatial_unmasked_index, spatial_masked_index = Smodel(flow_missing_mean.unsqueeze(0))
        time_unmasked, time_origin, time_unmasked_index, time_masked_index = Tmodel(flow_missing_mean.unsqueeze(0))
        # #时空异质性编码
        batch_size, num_nodes, feat_dim, embed_dim = spatial_unmasked.shape # torch.Size([1, 358, 182, 128])
        heter_spatial_unmasked, heter_time_unmasked, Loss_heter_spatial, Loss_heter_time = heter_model(spatial_unmasked, spatial_origin, time_unmasked, time_origin)
        loss, pred = tensor_model(flow_missing_mean, flow_missing_mask, heter_spatial_unmasked, heter_time_unmasked)
        #optimizer.zero_grad()
        total_loss = loss + Loss_heter_spatial + Loss_heter_time
        
        if epoch == args.n_epochs:
            # 将 PyTorch 张量转换为 NumPy 数组，首先使用 detach()
            numpy_data_spatial = heter_spatial_unmasked.detach().cpu().numpy()  # 如果在 GPU 上，先将其移到 CPU
            numpy_data_time = heter_time_unmasked.detach().cpu().numpy()  # 如果在 GPU 上，先将其移到 CPU
            # 保存为 .npy 文件
            np.save(os.path.join(args.root,"heterST",f'{args.dataset}-{args.missing_type}{args.missing_ratio}-spatial-have1.npy'), numpy_data_spatial)
            np.save(os.path.join(args.root,"heterST",f'{args.dataset}-{args.missing_type}{args.missing_ratio}-time-have1.npy'), numpy_data_time)
            # # 可视化

        # total_loss = loss
        loss.backward()  # 只调用一次       
        optimizer.step()
        losses.append(total_loss.item())
        lr_scheduler.step()
        pbar.update(1)
        if epoch % 10 == 0:
            pbar.set_description(f'Epoch: {epoch}/{args.n_epochs}, Loss: {total_loss.item():.5f}')

        if epoch % args.evaluation_interval == 0:
            with torch.no_grad():
                flow_imputation = tensor_model.predict().cpu()
                rmse, mae, mape = evaluate(flow_origin, flow_imputation, flow_missing_mask.cpu(), print_value=False, data_norm=True)    
                results.append({'Epoch': epoch, 'Loss': total_loss.cpu().item(), 'RMSE': rmse.cpu().item(), 'MAE': mae.cpu().item(),'MAPE': mape.cpu().item()})
        torch.cuda.empty_cache()
    pbar.close()
    # # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)

    # 保存DataFrame到Excel文件
    results_df.to_csv(os.path.join(args.root, "evaluation", f'{args.dataset}-{args.missing_type}{args.missing_ratio}-{args.time}.csv'), index=False)
    torch.save({'model': tensor_model.state_dict()}, 
                os.path.join(args.root, "saved_models", f'{args.dataset}-{args.missing_type}{args.missing_ratio}-{args.time}.pth'))

    # # # 输出结果
    
    print("Root Mean Squared Error (RMSE):", rmse.item())
    print("Mean Absolute Error (MAE):", mae.item())
    print("Mean Absolute Percentage Error (MAPE):", mape.item(), "%")
    print()

if __name__ == '__main__':
    args = Options().parse(save_args=False)
    # 读取参数组合
    with open('params_test.json', 'r') as f:
        param_combinations = json.load(f)
    # 循环遍历每组参数
    for params in param_combinations:
        args = Options().parse(save_args=False)
        
        # 更新参数
        for key, value in params.items():
            setattr(args, key, value)
        
        main(args)





