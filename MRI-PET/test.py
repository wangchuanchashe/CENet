import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from network import FusionModel  # 导入您自己的网络模型
import time  # 添加用于计算运行时间的模块

# ==========================
# 自定义数据集类（与训练时相同）
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
        mri_image_path = self.mri_images[idx]

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

        return mri_image, y_channel, cb_channel, cr_channel,os.path.basename(mri_image_path)

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
# 主测试函数
# ==========================
def main():
    start_time = time.time()  # 记录测试开始时间

    # 定义数据文件夹路径
    test_mri_dir = '/root/autodl-tmp/Havard-Medical-Image-Fusion-Datasets-main/MyDatasets/PET-MRI/test/MRI'
    test_pet_dir = '/root/autodl-tmp/Havard-Medical-Image-Fusion-Datasets-main/MyDatasets/PET-MRI/test/PET'

    #test_mri_dir = '/root/autodl-tmp/GFP PC/pc'
    #test_pet_dir = '/root/autodl-tmp/GFP PC/gfp'
    
    # 定义测试时的转换
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 初始化数据集和数据加载器
    test_dataset = CustomFusionDataset(test_mri_dir, test_pet_dir, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    height, width = 256, 256
    model = FusionModel(height=height, width=width)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 加载训练好的模型
    model_path = 'models/model_epoch_64.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到评估模式

    # 定义损失函数
    criterion_mse = nn.MSELoss()

    # 初始化累积损失
    running_test_loss = 0.0
    running_mse_loss = 0.0
    running_ssim_loss = 0.0
    running_grad_loss = 0.0

    # 创建保存输出图像的目录
    os.makedirs('outputs/test_fused_images', exist_ok=True)

    with torch.no_grad():
        for idx, (mri_image, y_channel, cb_channel, cr_channel,image_name) in enumerate(tqdm(test_dataloader, desc="测试中")):
            iter_start_time = time.time()  # 记录单次迭代开始时间

            mri_image = mri_image.to(device)
            y_channel = y_channel.to(device)

            output = model(mri_image, y_channel)

            # 计算不同的损失项
            target = max_image(mri_image, y_channel)
            mse_loss = criterion_mse(output, target)
            ssim_loss_val = ssim_loss(output, target)

            output_grad_x, output_grad_y = image_gradient(output)
            y_grad_x, y_grad_y = image_gradient(mri_image)
            gradient_loss = criterion_mse(output_grad_x, y_grad_x) + criterion_mse(output_grad_y, y_grad_y)

            # 总损失
            total_loss = 0.5 * mse_loss + ssim_loss_val + 10 * gradient_loss

            # 累计损失
            running_test_loss += total_loss.item()
            running_mse_loss += mse_loss.item()
            running_ssim_loss += ssim_loss_val.item()
            running_grad_loss += gradient_loss.item()

            # 后处理并保存图像
            output_image = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            output_image = (output_image * 0.5 + 0.5) * 255.0
            output_image = output_image.clip(0, 255).astype(np.uint8)

            cb_channel_np = cb_channel.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            cr_channel_np = cr_channel.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            cb_channel_np = (cb_channel_np * 0.5 + 0.5) * 255.0
            cr_channel_np = (cr_channel_np * 0.5 + 0.5) * 255.0

            cb_channel_np = cb_channel_np.clip(0, 255).astype(np.uint8)
            cr_channel_np = cr_channel_np.clip(0, 255).astype(np.uint8)                 
            
            # 合并 Y、Cb 和 Cr 通道
            final_output = np.stack([output_image, cb_channel_np, cr_channel_np], axis=-1)
            final_output = Image.fromarray(final_output, mode='YCbCr').convert('RGB')

            # 保存融合后的图像
            save_path = f'outputs/test_fused_images/{image_name}.png'
            final_output.save(save_path)
            print(f'已保存 {save_path}')

            iter_end_time = time.time()  # 记录单次迭代结束时间
            print(f'第 {idx+1} 张图像处理时间: {iter_end_time - iter_start_time:.4f} 秒')

    # 计算测试集的平均损失
    avg_test_loss = running_test_loss / len(test_dataloader)
    avg_mse_loss = running_mse_loss / len(test_dataloader)
    avg_ssim_loss = running_ssim_loss / len(test_dataloader)
    avg_grad_loss = running_grad_loss / len(test_dataloader)

    # 打印平均损失
    print(f'平均测试损失: {avg_test_loss:.4f}, '
          f'MSE损失: {avg_mse_loss:.4f}, '
          f'SSIM损失: {avg_ssim_loss:.4f}, '
          f'梯度损失: {avg_grad_loss:.4f}')

    end_time = time.time()  # 记录测试结束时间
    print(f'整个测试过程耗时: {end_time - start_time:.2f} 秒')

if __name__ == '__main__':
    main()



#76
