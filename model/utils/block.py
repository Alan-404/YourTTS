import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import remove_parametrizations
from typing import Optional, List
from model.utils.common import get_padding, convert_pad_shape
from model.utils.attention import MultiHeadAttention

LRELU_SLOPE = 0.1

class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: List[int]) -> None:
        super().__init__()
        self.n_layers = len(dilations)
        
        self.convs_1 = nn.ModuleList()
        self.convs_2 = nn.ModuleList()

        for i in range(self.n_layers):
            self.convs_1.append(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=get_padding(kernel_size, dilations[i]), dilation=dilations[i])
                )
            )
            self.convs_2.append(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=get_padding(kernel_size, 1))
                )
            )
    def forward(self, x: torch.Tensor):
        for i in range(self.n_layers):
            xt = F.leaky_relu(x, negative_slope=LRELU_SLOPE)
            xt = self.convs_1[i](xt)
            xt = F.leaky_relu(xt, negative_slope=LRELU_SLOPE)
            xt = self.convs_2[i](xt)
            x = x + xt
        return x
    
    def remove_weight_norm(self):
        for i in range(self.n_layers):
            remove_parametrizations(self.convs_1[i], 'weight')
            remove_parametrizations(self.convs_2[i], 'weight')

class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: List[int]) -> None:
        super().__init__()
        self.n_layers = len(dilations)
        self.convs = nn.ModuleList()

        for dilation in dilations:
            self.convs.append(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=get_padding(kernel_size, dilation), dilation=dilation)
                )
            )
    def forward(self, x: torch.Tensor):
        for conv in self.convs:
            xt = F.leaky_relu(x, negative_slope=LRELU_SLOPE)
            xt = conv(xt)
            x = x + xt
        return x
    
    def remove_weight_norm(self):
        for conv in self.convs:
            remove_parametrizations(conv, 'weight')

class EncoderBlock(nn.Module):
    def __init__(self, n_layers: int, hidden_channels: int, n_heads: int, kernel_size: int = 1, dropout_p: float = 0., window_size: int = 4) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        filter_channels = 4 * hidden_channels
        
        self.attention_layers = nn.ModuleList()
        self.first_norm_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.second_norm_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.attention_layers.append(
                MultiHeadAttention(channels=hidden_channels, out_channels=hidden_channels, n_heads=n_heads, dropout_p=dropout_p, window_size=window_size)
            )
            self.first_norm_layers.append(
                nn.LayerNorm(normalized_shape=hidden_channels)
            )
            self.ffn_layers.append(
                FFN(in_channels=hidden_channels, out_channels=hidden_channels, filter_channels=filter_channels, kernel_size=kernel_size, dropout_p=dropout_p)
            )
            self.second_norm_layers.append(
                nn.LayerNorm(normalized_shape=hidden_channels)
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        else:
            attn_mask = None
        for i in range(self.n_layers):
            y = self.attention_layers[i](x, x, x, attn_mask)
            y = F.dropout(y, p=self.dropout_p, training=self.training)
            x = self.first_norm_layers[i]((x + y).transpose(-1, -2)).transpose(-1, -2)

            y = self.ffn_layers[i](x, mask)
            y = F.dropout(y, p=self.dropout_p, training=self.training)
            x = self.second_norm_layers[i]((x + y).transpose(-1,-2)).transpose(-1, -2)
        if mask is not None:
            x = x * mask
        return x

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.norm = nn.LayerNorm(normalized_shape=d_model)
    def forward(self, x: torch.Tensor, residual_x: torch.Tensor):
        x = self.dropout(x)
        x = self.norm(x + residual_x)
        return x

class LayerNormGELU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=dim)
        self.gelu = nn.GELU()
    def forward(self, x: torch.Tensor):
        x = x.transpose(-1, -2)
        x = self.layer_norm(x)
        x = x.transpose(-1, -2)
        x = self.gelu(x)
        return x

class FFN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, filter_channels: int, kernel_size: int, dropout_p: float = 0., activation: str='relu', causal: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=filter_channels, kernel_size=kernel_size)
        self.conv_2 = nn.Conv1d(in_channels=filter_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.drop = nn.Dropout(dropout_p)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            x = x * mask
        x = self.conv_1(self.padding(x))
        if self.activation == 'gelu':
            x = x * F.sigmoid(1.702 * x)
        else:
            x = F.relu(x)
        
        x = self.drop(x)
        if mask is not None:
            x = x * mask
        x = self.conv_2(self.padding(x))
        return x
    
    def _causal_padding(self, x: torch.Tensor):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, convert_pad_shape(padding))
        return x
    def _same_padding(self, x: torch.Tensor):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, convert_pad_shape(padding))
        return x

class CloningBlock(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, d_model: int, n_heads: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_model = d_model

        self.linear_q = nn.Linear(in_features=hidden_dim, out_features=d_model)
        self.linear_k = nn.Linear(in_features=hidden_dim, out_features=d_model)
        self.linear_v = nn.Linear(in_features=hidden_dim, out_features=d_model)

        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_p=dropout_p)
        self.mapping_linear = nn.Linear(in_features=d_model, out_features=1)
        
        self.embedding_linear = nn.Linear(in_features=hidden_dim, out_features=embedding_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, n_samples, _ = x.size()

        q = F.elu(self.linear_q(x))
        k = F.elu(self.linear_k(x))
        v = F.elu(self.linear_v(x))

        xt = self.embedding_linear(x) # (batch_size, n_samples, embedding_dim)

        attn_out = self.attention(q, k, v, mask)
        attn_out = attn_out.reshape((batch_size, n_samples, self.d_model))
        attn_out = self.mapping_linear(attn_out)

        attn_out = F.softsign(attn_out)

        return torch.matmul(attn_out.transpose(-1, -2), xt).squeeze(1)

class WN(nn.Module):
    def __init__(self, channels: int, n_layers: int, dilation_rate: int, kernel_size: int, dropout_p: float = 0.0, gin_channels: Optional[int] = None) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.channels = channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = get_padding(kernel_size, dilation)

            self.in_layers.append(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(in_channels=channels, out_channels=2*channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
                )
            )

            if i < n_layers - 1:
                res_skip_channels = 2*channels
            else:
                res_skip_channels = channels
            self.res_skip_layers.append(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(in_channels=channels, out_channels=res_skip_channels, kernel_size=1)
                )
            )
        
        if gin_channels is not None:
            self.cond_layer = nn.utils.parametrizations.weight_norm(nn.Conv1d(in_channels=gin_channels, out_channels=2*channels*n_layers, kernel_size=1))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, g: Optional[torch.Tensor] = None):
        batch_size, _, time = x.size()
        output = torch.zeros((batch_size, self.channels, time), device=x.device, dtype=x.dtype)

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            
            if g is not None:
                cond_offset = 2 * self.channels * i
                g_l = g[:, cond_offset:cond_offset + 2*self.channels, :]
            else:
                g_l = torch.zeros((batch_size, 2 * self.channels, time), dtype=x_in.dtype, device=x_in.device)

            acts = self.fused_add_tanh_sigmoid_multiply(x_in, g_l, self.channels)
                
            acts = F.dropout(acts, p=self.dropout_p, training=self.training)

            res_skip_acts = self.res_skip_layers[i](acts)
            
            if i < self.n_layers - 1:
                x = x + res_skip_acts[:,:self.channels,:]
                if mask is not None:
                    x = x * mask
                output = output + res_skip_acts[:, self.channels:, :]
            else:
                output = output + res_skip_acts
        
        if mask is not None:
            output = output * mask
        return output
    
    def fused_add_tanh_sigmoid_multiply(self, a: torch.Tensor, b: torch.Tensor, channels: int):
        acts = a + b
        t_acts = torch.tanh(acts[:, :channels, :])
        s_acts = F.sigmoid(acts[:, channels:, :])
        return t_acts * s_acts
    
    def remove_weight_norm(self):
        for layer in self.in_layers:
            remove_parametrizations(layer, 'weight')
        for layer in self.res_skip_layers:
            remove_parametrizations(layer, 'weight')
        if self.gin_channels is not None:
            remove_parametrizations(self.cond_layer, 'weight')

class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.norm_1 = nn.BatchNorm1d(num_features=out_channels)
        self.conv_2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.norm_2 = nn.BatchNorm1d(num_features=out_channels)

        self.residual_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
    def forward(self, x: torch.Tensor):
        xt = self.conv_1(x)
        xt = self.norm_1(xt)
        xt = F.relu(xt)
        xt = self.conv_2(xt)
        xt = self.norm_2(xt)
        x = F.relu(self.residual_conv(x) + xt)
        return x