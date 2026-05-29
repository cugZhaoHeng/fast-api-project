import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CustomQualityClassifier(nn.Module):
    """一个简单的 64x64 灰度图分类器，用于评估生成质量"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_evaluator_classifier(train_loader, device, epochs=10):
    """
    如果没有现成的分类器，调用此函数在你的 5000 张图上预训练一个。
    注意：由于你的图片可能没有标签，这里可以做自监督训练，或者如果你有标签，请替换标签。
    """
    model = CustomQualityClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("正在训练质量评估分类器...")
    model.train()
    for epoch in range(epochs):
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            # 注意：如果没标签，这里仅作为结构演示。通常需要有标签数据才能评估“置信度”
            labels = labels.to(device) 
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    model.eval()
    return model

@torch.no_grad()
def evaluate_custom_images(images, classifier_model, device):
    """使用训练好的分类器评估生成图"""
    classifier_model.eval()
    images = images.to(device)
    
    logits = classifier_model(images)
    probs = F.softmax(logits, dim=1)
    
    max_conf = probs.max(dim=1)[0].mean().item()
    entropy = - (probs * (probs + 1e-8).log()).sum(dim=1).mean().item()
    
    return max_conf, entropy