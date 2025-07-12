import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from sklearn.svm import SVC


# ==================模块定义====================================
def precompute_freqs_rotary(dim: int, end: int, theta: float = 10000.0):
    """
    预计算旋转角度 freqs，用于后续生成旋转矩阵。
    dim: 嵌入维度的一半
    end: 序列长度
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)  # [seq_len, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x, freqs_cis):
    """
    对输入 x 应用 RoPE 编码
    x: [B, seq_len, D]
    freqs_cis: [seq_len, D/2] 复数形式
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).expand(x.shape[0], *freqs_cis.shape)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(2)
    return x_out.type(x.dtype)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # [B, C, L]
        # Channels：特征通道数，例如经过 Conv1d(1, 32, ...) 后变为 32
        # Length：光谱数据的时间步/采样点数量（例如经过池化后变成不同长度），取决于输入长度和卷积参数
        y = self.avg_pool(x).view(x.size(0), x.size(1))
        y = self.fc(y).view(x.size(0), x.size(1), 1)
        return x * y.expand_as(x)


# 定义模态特征提取的分支
class FTIREncoder(nn.Module):
    def __init__(self, input_dim):
        super(FTIREncoder, self).__init__()
        self.net = nn.Sequential(      # 输入: [B, input_dim]，样本数*特征数
            # 输出: [B, 1, input_dim]，添加通道维度，变为 [Batch, Channel, Length]
            nn.Unflatten(1, (1, -1)),
            # 输出: [B, 32, L1]， L1 = floor((input_dim - 7) / 2 + 1)
            nn.Conv1d(1, 32, 7, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            SEBlock(32),  # 添加 SE 注意力，输出保持: [B, 32, L1]
            nn.MaxPool1d(3, 2),
            nn.Conv1d(32, 64, 5, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten(),
            nn.Linear(64 * 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # RoPE 参数
        self.freqs_cis = None  # 缓存预计算的 freqs_cis

    def forward(self, feat, feat_axis):
        # 特征加权（乘法）：feat [B, 467] * feature_axis [467] -> [B, 467]
        feat = feat * feat_axis.abs()  # 用绝对值确保权重非负
        # 提取基础特征
        feat = feat.unsqueeze(1)  # [B, 1, 467]
        feat = self.net(feat)
        # 构造 freqs_cis（只在第一次运行时计算）
        if self.freqs_cis is None or self.freqs_cis.shape[0] != feat_axis.shape[0]:
            self.freqs_cis = precompute_freqs_rotary(
                feat_axis.shape[0], dim=self.dim)
        # 将 feat_axis 转为位置编码（RoPE）
        feat = feat.unsqueeze(1)
        freqs_cis = self.freqs_cis.to(feat.device)
        feat = apply_rotary_emb(feat, freqs_cis)  # 注入位置信息
        feat = feat.squeeze(1)
        return feat


class MZEncoder(nn.Module):
    def __init__(self, input_dim):
        super(MZEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (1, -1)),
            nn.Conv1d(1, 32, 5, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            SEBlock(32),
            nn.Conv1d(32, 64, 3, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten(),
            nn.Linear(64 * 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.freqs_cis = None

    def forward(self, feat, feat_axis):
        # 特征加权（乘法）：feat [B, 467] * feature_axis [467] -> [B, 467]
        feat = feat * feat_axis.abs()  # 用绝对值确保权重非负
        # 提取基础特征
        feat = feat.unsqueeze(1)  # [B, 1, 467]
        feat = self.net(feat)
        # 构造 freqs_cis（只在第一次运行时计算）
        if self.freqs_cis is None or self.freqs_cis.shape[0] != feat_axis.shape[0]:
            self.freqs_cis = precompute_freqs_rotary(
                feat_axis.shape[0], dim=self.dim)
        # 将 feat_axis 转为位置编码（RoPE）
        feat = feat.unsqueeze(1)
        freqs_cis = self.freqs_cis.to(feat.device)
        feat = apply_rotary_emb(feat, freqs_cis)  # 注入位置信息
        feat = feat.squeeze(1)
        return feat


class SimpleResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return x + self.net(x)


class GatedFusion(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
        self.bias = nn.Parameter(torch.tensor([0.5, 0.5]))  # 初始为平均融合

    def forward(self, ftir_feat, mz_feat):
        combined = torch.cat([ftir_feat, mz_feat], dim=1)
        weights = self.gate(combined) * self.bias  # 加权融合
        weights = weights / weights.sum(dim=1, keepdim=True)  # 归一化
        fused = weights[:, 0].unsqueeze(
            1) * ftir_feat + weights[:, 1].unsqueeze(1) * mz_feat
        return fused.squeeze(1)


class HybridFusion(nn.Module):
    def __init__(self, dim=128, num_heads=4):
        super().__init__()
        # Gate Fusion
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
        self.gate_bias = nn.Parameter(torch.tensor([0.5, 0.5]))
        # Attention Fusion
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ftir_feat, mz_feat):
        # Gate Fusion Part
        combined_gate = torch.cat([ftir_feat, mz_feat], dim=1)
        weights = self.gate(combined_gate) * self.gate_bias
        weights = weights / weights.sum(dim=1, keepdim=True)
        gate_fused = weights[:, 0].unsqueeze(
            1) * ftir_feat + weights[:, 1].unsqueeze(1) * mz_feat
        # Attention Fusion Part
        ftir_seq = ftir_feat.unsqueeze(1)
        mz_seq = mz_feat.unsqueeze(1)
        cross_ftir, _ = self.attn(ftir_seq, mz_seq, mz_seq)
        cross_mz, _ = self.attn(mz_seq, ftir_seq, ftir_seq)
        attn_fused = (cross_ftir + cross_mz).squeeze(1)
        # 最终融合
        final_fused = torch.cat(
            [gate_fused, self.proj(attn_fused)], dim=-1)  # [B, 256]
        # print("Gate weights:", weights.detach().cpu().numpy())
        return final_fused


# ==================多模态模型定义====================================
class MultiModalModel(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim):
        super(MultiModalModel, self).__init__()
        self.ftir_extractor = FTIREncoder(ftir_input_dim)
        self.mz_extractor = MZEncoder(mz_input_dim)
        self.fuser = HybridFusion(dim=128, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SimpleResidualBlock(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, ftir, mz, ftir_axis, mz_axis):
        ftir_feat = self.ftir_extractor(ftir, ftir_axis)
        mz_feat = self.mz_extractor(mz, mz_axis)
        combined = self.fuser(ftir_feat, mz_feat)
        output = self.classifier(combined)  # [B, 2]
        return output


# ==================单模态模型定义====================================
class SingleFTIRModel(nn.Module):
    def __init__(self, input_dim):
        super(SingleFTIRModel, self).__init__()
        self.ftir_extractor = FTIREncoder(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            SimpleResidualBlock(64),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, ftir, ftir_axis):
        ftir_feat = self.ftir_extractor(ftir, ftir_axis)
        output = self.classifier(ftir_feat)
        return output


class SingleMZModel(nn.Module):
    def __init__(self, input_dim):
        super(SingleMZModel, self).__init__()
        self.mz_extractor = MZEncoder(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            SimpleResidualBlock(64),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, mz, mz_axis):
        mz_feat = self.mz_extractor(mz, mz_axis)
        output = self.classifier(mz_feat)
        return output


# ==================其他消融试验模型定义====================================
# 消融试验1：简单拼接融合
class ConcatFusion(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim):
        super(ConcatFusion, self).__init__()
        self.ftir_extractor = FTIREncoder(ftir_input_dim)
        self.mz_extractor = MZEncoder(mz_input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SimpleResidualBlock(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, ftir, mz, ftir_axis, mz_axis):
        ftir_feat = self.ftir_extractor(ftir, ftir_axis)
        mz_feat = self.mz_extractor(mz, mz_axis)
        combined = torch.cat([ftir_feat, mz_feat], dim=-1)  # [B, 256]
        output = self.classifier(combined)  # [B, 2]
        return output


# 消融试验2：仅保留Gate Fusion
class GateOnlyFusion(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim, dim=128):
        super(GateOnlyFusion, self).__init__()
        self.ftir_extractor = FTIREncoder(ftir_input_dim)
        self.mz_extractor = MZEncoder(mz_input_dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
        self.gate_bias = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            SimpleResidualBlock(64),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, ftir, mz, ftir_axis, mz_axis):
        ftir_feat = self.ftir_extractor(ftir, ftir_axis)
        mz_feat = self.mz_extractor(mz, mz_axis)
        combined = torch.cat([ftir_feat, mz_feat], dim=1)
        weights = self.gate(combined) * self.gate_bias
        weights = weights / weights.sum(dim=1, keepdim=True)
        gate_fused = weights[:, 0].unsqueeze(
            1) * ftir_feat + weights[:, 1].unsqueeze(1) * mz_feat  # [B, 128]
        output = self.classifier(gate_fused)  # [B, 2]
        return output


# 消融试验3：只用了MultiheadAttention
class CoAttnOnlyFusion(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim, dim=128, num_heads=4):
        super(CoAttnOnlyFusion, self).__init__()
        self.ftir_extractor = FTIREncoder(ftir_input_dim)
        self.mz_extractor = MZEncoder(mz_input_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            SimpleResidualBlock(64),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, ftir, mz, ftir_axis, mz_axis):
        ftir_feat = self.ftir_extractor(ftir, ftir_axis)
        mz_feat = self.mz_extractor(mz, mz_axis)
        ftir_seq = ftir_feat.unsqueeze(1)
        mz_seq = mz_feat.unsqueeze(1)
        cross_ftir, _ = self.attn(ftir_seq, mz_seq, mz_seq)
        cross_mz, _ = self.attn(mz_seq, ftir_seq, ftir_seq)
        attn_fused = (cross_ftir + cross_mz).squeeze(1)  # [B, 128]
        output = self.classifier(attn_fused)  # [B, 2]
        return output


# 消融试验4：把 Multi-headAttention 改成 Self-Attention
class SelfAttnFusion(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim, dim=128, num_heads=4):
        super(SelfAttnFusion, self).__init__()
        self.ftir_extractor = FTIREncoder(ftir_input_dim)
        self.mz_extractor = MZEncoder(mz_input_dim)
        # Gate Fusion
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
        self.gate_bias = nn.Parameter(torch.tensor([0.5, 0.5]))
        # Attention Fusion
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SimpleResidualBlock(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, ftir, mz, ftir_axis, mz_axis):
        ftir_feat = self.ftir_extractor(ftir, ftir_axis)
        mz_feat = self.mz_extractor(mz, mz_axis)
        # Gate Fusion
        combined_gate = torch.cat([ftir_feat, mz_feat], dim=1)
        weights = self.gate(combined_gate) * self.gate_bias
        weights = weights / weights.sum(dim=1, keepdim=True)
        gate_fused = weights[:, 0].unsqueeze(
            1) * ftir_feat + weights[:, 1].unsqueeze(1) * mz_feat
        # Attention Fusion
        ftir_seq = ftir_feat.unsqueeze(1)
        mz_seq = mz_feat.unsqueeze(1)
        ftir_attn, _ = self.attn(ftir_seq, ftir_seq, ftir_seq)
        mz_attn, _ = self.attn(mz_seq, mz_seq, mz_seq)
        attn_fused = (ftir_attn + mz_attn).squeeze(1)
        # 最终融合
        final_fused = torch.cat(
            [gate_fused, self.proj(attn_fused)], dim=-1)  # [B, 256]
        output = self.classifier(final_fused)  # [B, 2]
        return output


# 消融试验5：只用了MultiheadAttention，并且是Self-Attention
class SelfAttnOnlyFusion(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim, dim=128, num_heads=4):
        super(SelfAttnOnlyFusion, self).__init__()
        self.ftir_extractor = FTIREncoder(ftir_input_dim)
        self.mz_extractor = MZEncoder(mz_input_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            SimpleResidualBlock(64),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, ftir, mz, ftir_axis, mz_axis):
        ftir_feat = self.ftir_extractor(ftir, ftir_axis)
        mz_feat = self.mz_extractor(mz, mz_axis)
        ftir_seq = ftir_feat.unsqueeze(1)
        mz_seq = mz_feat.unsqueeze(1)
        ftir_attn, _ = self.attn(ftir_seq, ftir_seq, ftir_seq)
        mz_attn, _ = self.attn(mz_seq, mz_seq, mz_seq)
        attn_fused = (ftir_attn + mz_attn).squeeze(1)  # [B, 128]
        output = self.classifier(attn_fused)  # [B, 2]
        return output


# 消融试验6：SVM
class SVMClassifier:
    def __init__(self, kernel='rbf'):
        self.clf = SVC(kernel=kernel, probability=True)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
