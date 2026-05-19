import torch
import math

def sinusoidal_positional_encoding(seq_len, d_model):
    """
    seq_len: 序列长度
    d_model: 编码维度
    返回形状: (seq_len, d_model)
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
    
    # 计算分母项: 10000^(2i/d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                         -(math.log(10000.0) / d_model))  # (d_model/2,)
    
    # 填充 sin 到偶数索引
    pe[:, 0::2] = torch.sin(position * div_term)
    # 填充 cos 到奇数索引
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# 示例：序列长度 10，维度 512
pe = sinusoidal_positional_encoding(3, 8)
print(pe.shape)  # torch.Size([10, 512])
print(pe)