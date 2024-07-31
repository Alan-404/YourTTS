import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from model.utils.common import convert_pad_shape

class MultiHeadAttention(nn.Module):
    def __init__(self, channels: int, out_channels: int, n_heads: int, dropout_p: float = 0., window_size: Optional[int] = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.dropout_p = dropout_p

        self.sqrt_dim = math.sqrt(self.k_channels)

        self.conv_q = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.conv_k = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.conv_v = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.conv_o = nn.Conv1d(in_channels=channels, out_channels=out_channels, kernel_size=1)

        if window_size is not None:
            rel_stddev = self.k_channels ** (-0.5)
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads, window_size*2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads, window_size*2 + 1, self.k_channels) * rel_stddev)

        self.mask_value = None

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.conv_q(q)
        k = self.conv_k(k)
        v = self.conv_v(v)

        attention = self.attention(q, k, v, mask)
        attention = self.conv_o(attention)
        return attention

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, dim, t_s = k.size()
        t_t = q.size(2)

        q = q.reshape((batch_size, self.n_heads, self.k_channels, t_t)).transpose(2, 3) / self.sqrt_dim # (bs, n_heads, t_t, k_channels)
        k = k.reshape((batch_size, self.n_heads, self.k_channels, t_s)).transpose(2, 3) # (bs, n_heads, t_s, k_channels)
        v = v.reshape((batch_size, self.n_heads, self.k_channels, t_s)).transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-1, -2))
        if self.window_size is not None:
            assert t_t == t_s
            key_relative_embedding = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = torch.matmul(q, key_relative_embedding.unsqueeze(0).transpose(-1, -2))
            score_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + score_local
        if mask is not None:
            if self.mask_value is None:
                self.mask_value = torch.iinfo(q.dtype).min
            scores.masked_fill_(~mask, self.mask_value)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout_p, training=self.training)
        attention_context = torch.matmul(attention_weights, v)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(attention_weights)
            value_relative_embedding = self._get_relative_embeddings(self.emb_rel_v, t_s)
            attention_context = attention_context + torch.matmul(relative_weights, value_relative_embedding.unsqueeze(0))
        attention_context = attention_context.transpose(2, 3).reshape((batch_size, dim, t_t))
        return attention_context
    
    def _get_relative_embeddings(self, relative_embeddings: torch.Tensor, length: int) -> torch.Tensor:
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_pos = max((self.window_size + 1) - length, 0)
        slice_end_pos = slice_start_pos + 2 * length - 1
        if pad_length > 0:
            padded_relative_embedding = F.pad(
                relative_embeddings,
                convert_pad_shape([[0,0], [pad_length, pad_length], [0,0]])
            )
        else:
            padded_relative_embedding = relative_embeddings
        return padded_relative_embedding[:, slice_start_pos : slice_end_pos]
    
    def _absolute_position_to_relative_position(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0,0], [0,0], [0, length-1]]))
        x_flat = x.reshape((batch_size, n_heads, length ** 2 + length*(length - 1)))
        x_flat = F.pad(x_flat, convert_pad_shape([[0,0], [0,0], [length, 0]]))
        x_final = x_flat.reshape((batch_size, n_heads, length, 2*length))[:, :, :, 1:]
        return x_final

    def _relative_position_to_absolute_position(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, length, _ = x.size()

        x = F.pad(x, convert_pad_shape([[0,0], [0,0], [0,0], [0,1]]))

        x_flat = x.reshape((batch_size, n_heads, length * 2 * length))
        x_flat = F.pad(x_flat, convert_pad_shape([[0,0], [0,0], [0, length-1]]))

        x_final = x_flat.reshape((batch_size, n_heads, length + 1, 2*length - 1))[:, :, :length, length-1:]
        return x_final
    