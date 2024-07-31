import torch
import torch.nn as nn

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.div_term = nn.Parameter(1 / (10000**(torch.arange(0, d_model, 2, dtype=torch.float32)/d_model)).unsqueeze(0), requires_grad=False)

    def forward(self, length: int):
        positive_pe = torch.zeros((length, self.d_model), dtype=self.div_term.dtype, device=self.div_term.device)
        negative_pe = torch.zeros((length, self.d_model), dtype=self.div_term.dtype, device=self.div_term.device)
        position = torch.arange(0, length, dtype=self.div_term.dtype, device=self.div_term.device).unsqueeze(1)

        angles = position * self.div_term

        positive_pe[:, 0::2] = torch.sin(angles)
        positive_pe[:, 1::2] = torch.cos(angles)
        negative_pe[:, 0::2] = torch.sin(-angles)
        negative_pe[:, 1::2] = torch.cos(-angles)

        positive_pe = torch.flip(positive_pe, [0]).unsqueeze(0)
        negative_pe = negative_pe[1:].unsqueeze(0)
        
        pe = torch.cat([positive_pe, negative_pe], dim=1)
        return pe