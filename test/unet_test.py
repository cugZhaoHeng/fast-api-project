from pathlib import Path

import torch
import torch.nn as nn

# ==========================================
# 1. 基础砖块：双层卷积 (Double Convolution)
# U-Net 里的每一个小台阶，都会连续做两次卷积
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 结构：Conv -> BN -> ReLU -> Conv -> BN -> ReLU
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# ==========================================
# 2. 左半边：下采样台阶 (Down)
# 作用：缩小一半图片尺寸，特征翻倍
# ==========================================
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),                  # 1. 尺寸缩小一半 (28x28 -> 14x14)
            DoubleConv(in_channels, out_channels) # 2. 提取特征，改变通道数
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# ==========================================
# 3. 右半边：上采样台阶 (Up)  【极其重要！】
# 作用：放大一倍图片尺寸，并接收左边传来的原图细节
# ==========================================
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1. 放大器：转置卷积 (也可以用双线性插值 nn.Upsample)
        # 作用是把特征图的高和宽放大一倍
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # 2. 处理拼接后的特征
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_from_previous, x_from_left_encoder):
        # 1. 把从下面传上来的小图，放大一倍
        x1 = self.up(x_from_previous)
        
        # 2. 【灵魂操作：拼接长跳跃连接】
        # 把左边编码器的高清大图 (x_from_left_encoder)，和刚刚放大的图 (x1)，
        # 在通道维度 (dim=1) 强行拼在一起！
        # 比如左边是 64 通道，右边放大的也是 64 通道，拼完之后就是 128 通道！
        x = torch.cat([x_from_left_encoder, x1], dim=1)
        
        # 3. 再通过双层卷积，把特征融合在一起
        return self.conv(x)


# ==========================================
# 4. 组装：完整的 U-Net
# ==========================================
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # --- 降落阶段 (Encoder) ---
        # 输入假设: [Batch, 1, 64, 64]
        self.inc = DoubleConv(in_channels, 64)        # 出来: [B, 64, 64, 64] (保存为 x1)
        self.down1 = Down(64, 128)                    # 出来: [B, 128, 32, 32] (保存为 x2)
        self.down2 = Down(128, 256)                   # 出来: [B, 256, 16, 16] (保存为 x3)
        
        # --- 谷底阶段 (Bottleneck) ---
        self.down3 = Down(256, 512)                   # 出来: [B, 512, 8, 8]
        
        # --- 爬升阶段 (Decoder) ---
        # 注意 forward 里的参数传递！
        self.up1 = Up(512, 256)                       # 接收 512通道的下层特征 + 256通道的 x3
        self.up2 = Up(256, 128)                       # 接收 256通道的下层特征 + 128通道的 x2
        self.up3 = Up(128, 64)                        # 接收 128通道的下层特征 + 64通道的 x1
        
        # --- 输出头 ---
        # 把通道数变成目标要求 (比如灰度图就是1，RGB图就是3)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. 向下走：一边提取特征，一边把每一层的状态“存档”
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)   # 到达最底部的抽象特征
        
        # 2. 向上走：每次放大后，都要读取对应层级的“存档”进行拼接
        x = self.up1(x4, x3)  # x4放大后和x3拼接
        x = self.up2(x, x2)   # 再次放大后和x2拼接
        x = self.up3(x, x1)   # 最后放大后和最清晰的x1拼接
        
        # 3. 输出
        logits = self.outc(x)
        return logits

# 测试代码
if __name__ == "__main__":
    from torchinfo import summary
    from pathlib import Path
    
    # 1. 明确指定设备 (有 GPU 就用 GPU，没有就用 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 实例化模型，并移动到对应设备
    model = SimpleUNet(in_channels=1, out_channels=1).to(device)
    
    # 3. 创建假数据，也必须移动到同【同一个】设备！
    dummy_img = torch.randn(1, 1, 64, 64).to(device)
    
    # 前向传播测试
    output = model(dummy_img)
    print(f"输入尺寸: {dummy_img.shape}")
    print(f"输出尺寸: {output.shape}") 
    
    # 打印网络结构 (传给 summary 的 tensor 大小不需要带 batch_size，或者直接传 tuple)
    summary(model, input_size=(1, 1, 64, 64), device=device)
    
    # ==========================================
    # 导出 ONNX 的关键步骤！
    # ==========================================
    # 4. 【灵魂一步】：切换到评估模式，锁定 BatchNorm 等层！
    model.eval()
    
    CURRENT_DIR = Path(__file__).resolve().parent
    onnx_path = CURRENT_DIR / "unet.onnx"
    
    # 5. 导出模型 (保持 opset_version=18，顺应新版 PyTorch 的脾气)
    torch.onnx.export(model, dummy_img, onnx_path, 
                    export_params=True, 
                    opset_version=18, 
                    do_constant_folding=True)
                    
    print(f"ONNX 文件导出成功！保存在: {onnx_path}")