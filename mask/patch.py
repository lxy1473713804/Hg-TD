import torch
from torch import nn

class PatchEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, in_channel, embed_dim, stride , padding, norm_layer):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size             # the L
        self.input_channel = in_channel
        self.stride = stride
        self.padding = padding
        self.input_embedding = nn.Conv2d(
                                        in_channel,
                                        embed_dim,
                                        kernel_size = (self.len_patch, 1),
                                        stride = (self.stride, 1),                                        
                                        padding=(self.padding, 0))
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()
    
    def forward(self, flow_data):
        #input shape: B, N, 1, T   1, 67, 1,  8064
        batch_size, num_nodes, num_feat, len_time_series = flow_data.shape
        flow_data = flow_data.unsqueeze(-1) # B, N, 1, L, 1
        # B*N,  1, L, 1
        flow_data = flow_data.reshape(batch_size*num_nodes, num_feat, len_time_series, 1)
        output = self.input_embedding(flow_data)
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)    # B, N, d, P
        # Apply ReLU activation to ensure all values are non-negative
        #output = torch.relu(output)
        # Optionally, you can include a normalization layer if needed
        # output = self.norm_layer(output)
        # assert output.shape[-1] == len_time_series / self.len_patch
        return output
        
#--------------------------------------------------------------------------------------------------------------

class PatchEmbedding2(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size             # the L
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(
                                        in_channel,
                                        embed_dim,
                                        kernel_size=(self.len_patch, 1),
                                        stride=(self.len_patch, 1),
                                        padding=(0, 0)),
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history):
        """
        Args:
            long_term_history(Tensor): Long-term historical traffic flow data with shape (batch_size, nodes, feature = 1, time_steps) B, N, 1, T

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """
        # B, N, 1, T
        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.unsqueeze(-1) # B, N, 1, L, 1
        # B*N,  1, L, 1
        long_term_history = long_term_history.reshape(batch_size*num_nodes, num_feat, len_time_series, 1)
        # B*N,  d, L/P, 1   P is the patch size,288，一天的流量被划分为288个patch
        output = self.input_embedding(long_term_history)
        # norm
        output = self.norm_layer(output)
        # reshape
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)    # B, N, d, P
        assert output.shape[-1] == len_time_series / self.len_patch
        return output

