import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import random
from network import FusionModel  # 导入您自己的网络模型

# ==========================
# 设置随机种子以确保结果可重复
# ==========================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================
# 自定义数据集类
# ==========================
class CustomFusionDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, transform=None):
        self.mri_images = sorted([
            os.path.join(mri_dir, img) for img in os.listdir(mri_dir) 
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
        ])
        self.pet_images = sorted([
            os.path.join(pet_dir, img) for img in os.listdir(pet_dir) 
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
        ])
        self.transform = transform

    def __len__(self):
        return min(len(self.mri_images), len(self.pet_images))

    def __getitem__(self, idx):
        mri_image = Image.open(self.mri_images[idx]).convert('L')  # 转为灰度图
        pet_image = Image.open(self.pet_images[idx]).convert('RGB')  # 转为RGB图

        # 将PET图像转换为YCbCr并分离通道
        pet_image_ycbcr = pet_image.convert('YCbCr')
        y_channel, cb_channel, cr_channel = pet_image_ycbcr.split()

        if self.transform:
            mri_image = self.transform(mri_image)
            y_channel = self.transform(y_channel)
            cb_channel = self.transform(cb_channel)
            cr_channel = self.transform(cr_channel)

        return mri_image, y_channel, cb_channel, cr_channel

# ==========================
# 定义结构相似度（SSIM）损失函数
# ==========================
def ssim_loss(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)

    sigma1 = F.avg_pool2d(img1 ** 2, 3, 1, 1) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2 ** 2, 3, 1, 1) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return torch.clamp((1 - ssim_map) / 2, 0, 1).mean()

# ==========================
# 计算最大值函数
# ==========================
def max_image(img1, img2):
    return torch.max(img1, img2)

# ==========================
# 计算图像梯度函数
# ==========================
def image_gradient(img):
    img_dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    img_dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return img_dx, img_dy

# ==========================
# 主训练函数
# ==========================
def main():
    # 设置随机种子
    set_seed(42)

    # 定义数据文件夹路径
    train_mri_dir = '/root/autodl-tmp/Havard-Medical-Image-Fusion-Datasets-main/MyDatasets/SPECT-MRI/train/MRI'
    train_pet_dir = '/root/autodl-tmp/Havard-Medical-Image-Fusion-Datasets-main/MyDatasets/SPECT-MRI/train/SPECT'
    # 如果需要，可以取消注释并修改以下路径

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机调整颜色
        transforms.RandomRotation(degrees=20),  # 随机旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # 随机高斯模糊
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    # 初始化数据集和数据加载器
    train_dataset = CustomFusionDataset(train_mri_dir, train_pet_dir, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    # 初始化模型
    height, width = 256,256
    model = FusionModel(height=height, width=width)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 定义损失函数和优化器
    criterion_mse = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    # 设置学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 训练参数
    num_epochs = 75

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_mse_loss = 0.0
        running_ssim_loss = 0.0
        running_grad_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            mri_image, y_channel, _, _ = batch
            mri_image = mri_image.to(device)
            y_channel = y_channel.to(device)

            optimizer.zero_grad()
            output = model(mri_image, y_channel)

            # 计算不同的损失项
            target = max_image(mri_image, y_channel)
            mse_loss = criterion_mse(output, target)
            ssim_loss_val = ssim_loss(output, target)

            output_grad_x, output_grad_y = image_gradient(output)
            y_grad_x, y_grad_y = image_gradient(mri_image)
            gradient_loss = criterion_mse(output_grad_x, y_grad_x) + criterion_mse(output_grad_y, y_grad_y)

            # 总损失
            loss = 0.5 * mse_loss + ssim_loss_val + 10 * gradient_loss
            loss.backward()
            optimizer.step()

            # 累计损失
            running_loss += loss.item()
            running_mse_loss += mse_loss.item()
            running_ssim_loss += ssim_loss_val.item()
            running_grad_loss += gradient_loss.item()

        # 计算每个损失的平均值
        avg_loss = running_loss / len(train_dataloader)
        avg_mse_loss = running_mse_loss / len(train_dataloader)
        avg_ssim_loss = running_ssim_loss / len(train_dataloader)
        avg_grad_loss = running_grad_loss / len(train_dataloader)

        # 打印损失信息
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'总损失: {avg_loss:.4f}, '
              f'MSE损失: {avg_mse_loss:.4f}, '
              f'SSIM损失: {avg_ssim_loss:.4f}, ')
              f'梯度损失: {avg_grad_loss:.4f}')
        if epoch + 1 == num_epochs:
            # 检查是否是最后一个 epoch
            os.makedirs('models', exist_ok=True)
            save_path = f'models/model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存为: {save_path}")

    
if __name__ == "__main__":
    main()
