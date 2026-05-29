import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    # 这里意味着，只要t相同，那么生成的emb就相同
    def forward(self, t):
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
class CosPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        i_list = torch.arange(0, self.dim, 2)
        print(f"i_list : {i_list}")
        k = torch.log(torch.tensor(10000))
        emb = torch.exp(i_list * -k / self.dim)
        sin_encode = torch.sin(t * emb)
        cos_encode = torch.cos(t*emb)
        record = torch.cat([sin_encode, cos_encode], dim=0)
        return record
        
        
if __name__ == "__main__":
    dim = torch.tensor(8)
    d = dim / 2
    d2 = dim // 2
    print(d)
    print(d2)
    a = SinusoidalPosEmb(dim)
    t = torch.tensor([1,2,3])
    b = a.forward(t)
    print(f"b shape:{b.shape}, {b}")
    
    x1 = torch.arange(0, 10, 2)
    print(f"x1: {x1}")
    
    print(f"dim: {dim}")
    c = CosPosEmb(dim)
    d = c.forward(t.unsqueeze(1))
    print(f"d:{d}")