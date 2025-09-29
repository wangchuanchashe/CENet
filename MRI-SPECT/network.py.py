import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Convlutioanl(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super(Convlutioanl, self).__init__()
        self.padding=(1,1,1,1)
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=0,stride=1)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, input):
        out=F.pad(input,self.padding,'replicate')
        out=self.conv(out)
        out=self.bn(out)
        out=self.relu(out)
        return out

# 多尺度特征提取模块
class ResidualDilatedConvNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualDilatedConvNet, self).__init__()

        # 分支 1: 标准 3x3 卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 分支 2: 3x3 卷积 + 空洞卷积（膨胀率 1）
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 分支 3: 3x3 卷积 + 空洞卷积（膨胀率 1）+ 空洞卷积（膨胀率 2）
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 输出卷积层，整合所有分支的输出
        self.output_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1)

        # 如果输入和输出通道不一致，用 1x1 卷积调整输入的通道数
        self.adjust_channels = None
        if in_channels != out_channels:
            self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 分支1: 标准卷积特征
        branch1_out = self.branch1(x)

        # 分支2: 3x3 卷积 + 空洞卷积（膨胀率 1）
        branch2_out = self.branch2(x)

        # 分支3: 3x3 卷积 + 空洞卷积（膨胀率 1）+ 空洞卷积（膨胀率 2）
        branch3_out = self.branch3(x)

        # 拼接所有分支的输出
        combined_features = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)

        # 通过 1x1 卷积整合输出特征
        output = self.output_conv(combined_features)

        # 如果输入通道数和输出通道数不匹配，调整输入的通道数
        if self.adjust_channels is not None:
            x = self.adjust_channels(x)

        # 残差连接
        output = output + x

        return output


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.01):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)  # 添加 LeakyReLU
        
    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        return x

class Complementaryfeatures1(nn.Module):
    def __init__(self, kernel_size=5):
        super(Complementaryfeatures1, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, a1, a2):
        batch, C, H, W = a1.size()
        K = self.kernel_size
        # 使用平均池化将a2下采样为卷积核大小 [batch, C, 3, 3]
        a2_pooled = F.avg_pool2d(a2, kernel_size=(H // K, W // K))
        # 将a2重塑为[batch * C, 1, 3, 3]，作为卷积核
        a2_reshaped = a2_pooled.view(batch * C, 1, K, K)
        # 将a1重塑为[1, batch * C, H, W]，准备进行分组卷积
        a1_reshaped = a1.view(1, batch * C, H, W)
        # 执行分组卷积，每个通道使用对应的卷积核
        attention = F.conv2d(a1_reshaped, a2_reshaped, groups=batch * C, padding=self.padding)
        # 将注意力图重塑回[batch, C, H, W]
        attention = attention.view(batch, C, H, W)
        # 应用Sigmoid激活函数
        attention = F.softmax(attention, dim=1)

        # 输出最终特征图
        output = a1 + (a1 - a1 * attention)

        return output



class Complementaryfeatures2(nn.Module):
    def __init__(self, kernel_size=5):
        super(Complementaryfeatures2, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, a1, a2):
        batch, C, H, W = a1.size()
        K = self.kernel_size
        # 使用平均池化将a1下采样为卷积核大小 [batch, C, 3, 3]
        a1_pooled = F.avg_pool2d(a1, kernel_size=(H // K, W // K))
        # 将a2重塑为[batch * C, 1, 3, 3]，作为卷积核
        a1_reshaped = a1_pooled.view(batch * C, 1, K, K)
        # 将a1重塑为[1, batch * C, H, W]，准备进行分组卷积
        a2_reshaped = a2.view(1, batch * C, H, W)
        # 执行分组卷积，每个通道使用对应的卷积核
        attention = F.conv2d(a2_reshaped, a1_reshaped, groups=batch * C, padding=self.padding)
        # 将注意力图重塑回[batch, C, H, W]
        attention = attention.view(batch, C, H, W)
        # 应用Softmax激活函数，沿着通道维度进行softmax
        attention = F.softmax(attention, dim=1)
        # 输出最终特征图
        output = a2 + a2 - a2 * attention

        return output



# 专家网络
class ExpertNetwork(nn.Module):
    def __init__(self, in_channels):
        super(ExpertNetwork, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(256, 256, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(256, 256, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
        )

    def forward(self, x, gate_weights):
        # 分支网络输出
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        # 使用门控权重对分支网络输出加权求和
        out = (gate_weights[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * out1 +
               gate_weights[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * out2 +
               gate_weights[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * out3 +
               gate_weights[:, 3].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * out4)

        return out

# 门控网络
class GatingNetwork1(nn.Module):
    def __init__(self, input_channels, height, width):
        super(GatingNetwork1, self).__init__()
        self.fc = nn.Linear(input_channels * height * width * 2, 4)  # 根据实际尺寸调整

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 确保展平时尺寸匹配
        gate_weights = self.fc(x)
        return F.softmax(gate_weights, dim=1)

class GatingNetwork2(nn.Module):
    def __init__(self, input_channels, height, width):
        super(GatingNetwork2, self).__init__()
        self.fc = nn.Linear(input_channels * height * width , 4)  # 根据实际尺寸调整

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 确保展平时尺寸匹配
        gate_weights = self.fc(x)
        return F.softmax(gate_weights, dim=1)

# 主模型，包含两个串联的专家网络
class FusionModel(nn.Module):
    def __init__(self, height, width):
        super(FusionModel, self).__init__()
        self.conv = Convlutioanl(in_channel=1,out_channel=16)
        self.feature_extractor = ResidualDilatedConvNet(in_channels=16, out_channels=64)
        self.tiaozhengch = Conv1x1(in_channels=64,out_channels=256)
        self.Complementfeatures_layer1 = Complementaryfeatures1(kernel_size=5)
        self.Complementfeatures_layer2 = Complementaryfeatures2(kernel_size=5)
        self.gating_network1 = GatingNetwork1(input_channels=256, height=height, width=width)
        self.expert_network1 = ExpertNetwork(in_channels=512)

        # 第二个门控网络和专家网络
        self.gating_network2 = GatingNetwork2(input_channels=256, height=height, width=width)
        self.expert_network2 = ExpertNetwork(in_channels=256)

        self.reconstruction1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.reconstruction2 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.reconstruction3 = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, mri_image, y_channel):
        f1 = self.conv(mri_image)
        f2 = self.conv(y_channel)
        
        f1_r = self.feature_extractor(f1)
        f2_r = self.feature_extractor(f2)
        
        f1_c = self.Complementfeatures_layer1(f1_r,f2_r)
        f2_c = self.Complementfeatures_layer2(f1_r,f2_r)


        #f1_m = self.tiaozhengch(f1)
        #f2_m = self.tiaozhengch(f2)
        f1_m = self.tiaozhengch(f1_c)
        f2_m = self.tiaozhengch(f2_c)
        
        
        #features = torch.cat((f1,f2),dim=1)
        features = torch.cat((f1_m,f2_m), dim=1)

        # 第一层专家门控网络
        gate_weights1 = self.gating_network1(features)
        expert_output1 = self.expert_network1(features, gate_weights1)

        # 第二层专家门控网络
        gate_weights2 = self.gating_network2(expert_output1)
        expert_output2 = self.expert_network2(expert_output1, gate_weights2)

        #第三层专家门控网络
        #gate_weights3 = self.gating_network2(expert_output2)
        #expert_output3 = self.expert_network2(expert_output2, gate_weights3)

        # 第四层专家门控网络
        #gate_weights4 = self.gating_network2(expert_output3)
        #expert_output4 = self.expert_network2(expert_output3, gate_weights4)

        # 重建融合图像
        output = self.reconstruction1(expert_output2)
        output = self.reconstruction2(output)
        output = self.reconstruction3(output)
        output = self.tanh(output)

        return output
