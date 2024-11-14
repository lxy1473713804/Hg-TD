import math
import torch
from torch.utils.data import Dataset, DataLoader


class FlowDataset(Dataset):
    def __init__(self, flow_data, flow_mask,batch_size):
        self.flow_data = flow_data
        self.flow_mask = flow_mask
        self.batch_size = batch_size
        _, day_num, _ = flow_data.shape
        if batch_size<day_num:
            self.length = day_num - batch_size + 1
        elif batch_size==day_num:
            self.length = 1
        else:
            raise ValueError(f'batch_size can not larger than the day num of flow dataset, got [batch_size={batch_size}] > [day_num={day_num}]')
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        day_index = torch.arange(index, index+self.batch_size)
        return self.flow_data[:, index:index+self.batch_size, :], self.flow_mask[:, index:index+self.batch_size, :], day_index

