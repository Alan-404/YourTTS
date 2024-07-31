import torch
from typing import Optional, List

@torch.jit.script
def get_padding(kernel_size: int, dilation: int):
    return (dilation * (kernel_size - 1)) // 2

def rand_slice_segments(x: torch.Tensor, x_length: Optional[torch.Tensor] = None, segment_size: int = 32):
    b, _, t = x.size()
    if x_length is None:
        x_length = t
    ids_str_max = x_length - segment_size + 1
    ids_str = (torch.clamp(torch.rand([b]), max=1., min=0.).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str=ids_str, segment_size=segment_size)
    return ret, ids_str

def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int):
    batch_size, dim, _ = x.size()
    ret = torch.zeros((batch_size, dim, segment_size), device=x.device)
    # ret = []
    for i in range(batch_size):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret

def ini_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weights.data.normal_(mean, std)

def convert_pad_shape(pad_shape: List[List[int]]):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape