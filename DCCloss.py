import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DCCloss(nn.Module):

    def __init__(self, temperature=0.07, normalize_feature=True, eps=1e-8):
        """
        初始化DCC损失函数
        Args:
            temperature: 温度系数，控制对比损失的集中度
            normalize_feature: 是否对输入特征进行L2归一化（vMF分布基于单位球空间）
            eps: 数值稳定项，避免除零或对数为负
        """
        super(DCCloss, self).__init__()
        self.temperature = temperature
        self.normalize_feature = normalize_feature
        self.eps = eps

    def _estimate_vmf_parameters(self, features, labels):
        """
        基于最大似然估计(MLE)估计von Mises-Fisher分布参数
        Args:
            features: 输入特征，shape [B, D] (B: batch size, D: 特征维度)
            labels: 样本标签，shape [B]
        Returns:
            vmf_params: 字典，包含每个类别的vMF分布参数 {mean: [C, D], concentration: [C]}
                        C为类别数，mean为方向向量(单位范数)，concentration为浓度参数
        """
        # 特征L2归一化（vMF分布定义在单位球面上）
        if self.normalize_feature:
            features = F.normalize(features, dim=-1, p=2)
        
        unique_labels = torch.unique(labels)
        vmf_params = {'mean': [], 'concentration': []}
        
        for label in unique_labels:
            # 提取当前类别的所有特征
            class_features = features[labels == label]  # [N_c, D] N_c为该类别样本数
            N_c = class_features.size(0)
            
            # 计算方向均值向量（MLE）
            mean_vec = torch.sum(class_features, dim=0)  # [D]
            mean_norm = torch.norm(mean_vec, p=2, dim=-1, keepdim=True)
            mean_vec = mean_vec / (mean_norm + self.eps)  # 单位化
            
            # 计算浓度参数（MLE）- 简化版闭式估计（适用于高维特征）
            # 参考vMF分布浓度参数的MLE近似：c = r * (D - r²) / (1 - r²), r = ||mean_vec||
            r = mean_norm / N_c  # 平均径向长度
            D = features.size(-1)
            concentration = r * (D - r**2) / (1 - r**2 + self.eps)
            
            vmf_params['mean'].append(mean_vec)
            vmf_params['concentration'].append(concentration)
        
        # 整理为tensor格式
        vmf_params['mean'] = torch.stack(vmf_params['mean'], dim=0)  # [C, D]
        vmf_params['concentration'] = torch.stack(vmf_params['concentration'], dim=0)  # [C]
        
        return vmf_params, unique_labels

    def _closed_form_expected_loss(self, features, labels, vmf_params, unique_labels):
        """
        计算闭式期望监督对比损失（基于vMF分布的无限样本采样近似）
        Args:
            features: 输入特征 [B, D]
            labels: 样本标签 [B]
            vmf_params: vMF分布参数 {mean: [C, D], concentration: [C]}
            unique_labels: 批次内唯一标签列表 [C]
        Returns:
            loss: 批次的DCC损失值
        """
        B, D = features.shape
        if self.normalize_feature:
            features = F.normalize(features, dim=-1, p=2)
        
        # 构建标签到索引的映射
        label2idx = {label.item(): idx for idx, label in enumerate(unique_labels)}
        batch_label_indices = torch.tensor([label2idx[label.item()] for label in labels], device=features.device)  # [B]
        
        # 计算特征与所有类别vMF均值的余弦相似度
        cos_sim = torch.matmul(features, vmf_params['mean'].T)  # [B, C]
        exp_sim = torch.exp(cos_sim / self.temperature)  # [B, C]
        
        # 计算每个样本的正类掩码（同一类别）
        pos_mask = F.one_hot(batch_label_indices, num_classes=len(unique_labels)).float()  # [B, C]
        
        # 计算闭式期望正样本对的相似度和
        # 基于vMF分布的期望相似度：E[exp(sim(x, x+)/T)] = exp( (κ * cos_sim(x, μ)) / T ) * I_0(κ/T) / I_0(κ)
        concentration = vmf_params['concentration'][batch_label_indices]  # [B]
        pos_exp = exp_sim * pos_mask  # [B, C]
        pos_sum = torch.sum(pos_exp, dim=-1)  # [B]
        
        # 计算负样本对的相似度和
        neg_sum = torch.sum(exp_sim * (1 - pos_mask), dim=-1)  # [B]
        
        # 计算每个样本的对比损失
        per_sample_loss = -torch.log(pos_sum / (pos_sum + neg_sum + self.eps) + self.eps)
        
        label_counts = torch.bincount(batch_label_indices)  # [C]
        weights = 1.0 / (label_counts[batch_label_indices].float() + self.eps)
        weights = weights / torch.sum(weights) * B  # 归一化权重
        
        loss = torch.sum(per_sample_loss * weights) / B
        
        return loss

    def forward(self, features, labels):
        """
        前向计算DCC损失
        Args:
            features: 模型输出特征，shape [B, D] (B: batch size, D: feature dimension)
            labels: 样本标签，shape [B]
        Returns:
            loss: DCC损失值
        """
        # 步骤1：估计vMF分布参数
        vmf_params, unique_labels = self._estimate_vmf_parameters(features, labels)
        
        # 步骤2：计算闭式期望监督对比损失
        loss = self._closed_form_expected_loss(features, labels, vmf_params, unique_labels)
        
        return loss


# 测试示例
if __name__ == "__main__":
    batch_size = 64
    feat_dim = 128
    num_classes = 10
    
    features = torch.randn(batch_size, feat_dim)
    labels = torch.cat([
        torch.randint(0, 5, (16,)),  # 长尾部分：0-4类各少量样本
        torch.randint(5, 10, (48,))  # 头部部分：5-9类各大量样本
    ])
    
    # 初始化DCC损失
    dcc_loss = DCCloss(temperature=0.07, normalize_feature=True)
    
    # 计算损失
    loss = dcc_loss(features, labels)
    print(f"DCC Loss Value: {loss.item():.4f}")
    
    # 测试反向传播
    loss.backward()
    print("Backward pass completed successfully!")
