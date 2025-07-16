import torch
import torch.nn as nn
from sklearn.svm import SVC


# ==================模块定义====================================
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

    def forward(self, feat):  # [B, C, L]
        # Channels：特征通道数，例如经过 Conv1d(1, 32, ...) 后变为 32
        # Length：经过池化后变成不同长度，取决于输入长度和卷积参数
        y_avg = self.avg_pool(feat).view(feat.size(0), feat.size(1))
        w_channel = self.fc(y_avg)
        return feat * w_channel.view(feat.size(0), feat.size(1), 1)


# 定义模态特征提取的分支
class FTIREncoder(nn.Module):
    def __init__(self, axis_dim):
        super(FTIREncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2),  # 输入 [B,1,467] -> [B,32,230]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            SEBlock(32),
            nn.Conv1d(32, 64, 5, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten(),
            nn.Linear(64 * 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, feat, feat_axis):
        feat = feat.unsqueeze(1)    # (32,467) -> (32,1,467)
        feat = self.net(feat)
        return feat


class MZEncoder(nn.Module):
    def __init__(self, axis_dim):
        super(MZEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            SEBlock(32),
            nn.Conv1d(32, 64, 5, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten(),
            nn.Linear(64 * 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, feat, feat_axis):
        feat = feat.unsqueeze(1)    # (32,2838) -> (32,1,2838)
        feat = self.net(feat)
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
        return final_fused


# ==================多模态模型定义====================================
class MultiModalModel(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim):
        super(MultiModalModel, self).__init__()
        self.ftir_extractor = FTIREncoder(ftir_input_dim)
        self.mz_extractor = MZEncoder(mz_input_dim)
        self.fuser = HybridFusion(dim=256, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            SimpleResidualBlock(256),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2),
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
