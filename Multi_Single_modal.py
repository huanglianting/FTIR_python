from scipy.interpolate import interp1d
import torch
import torch.nn as nn
from sklearn.svm import SVC
import numpy as np
import cv2

# ==================模块定义====================================
# 定义模态特征提取的分支


class FTIREncoder(nn.Module):
    def __init__(self, axis_dim):
        super(FTIREncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2),  # 输入 [B,1,467] -> [B,32,230]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(64 * 32, 256),
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
            nn.Conv1d(32, 64, 5, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(64 * 32, 256),
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
            nn.Linear(256, 2),
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
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SimpleResidualBlock(128),
            nn.Linear(128, 2),
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
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SimpleResidualBlock(128),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, mz, mz_axis):
        mz_feat = self.mz_extractor(mz, mz_axis)
        output = self.classifier(mz_feat)
        return output


# ==================其他消融试验模型定义====================================
# 消融试验1：简单拼接融合
class ConcatFusion(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim, dim=256):
        super(ConcatFusion, self).__init__()
        self.ftir_extractor = FTIREncoder(ftir_input_dim)
        self.mz_extractor = MZEncoder(mz_input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            SimpleResidualBlock(dim),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, ftir, mz, ftir_axis, mz_axis):
        ftir_feat = self.ftir_extractor(ftir, ftir_axis)
        mz_feat = self.mz_extractor(mz, mz_axis)
        combined = torch.cat([ftir_feat, mz_feat], dim=-1)  # [B, 512]
        output = self.classifier(combined)  # [B, 2]
        return output


# 消融试验2：仅保留Gate Fusion
class GateOnlyFusion(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim, dim=256):
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
            nn.Linear(dim, dim//2),
            nn.BatchNorm1d(dim//2),
            nn.ReLU(),
            SimpleResidualBlock(dim//2),
            nn.Linear(dim//2, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, ftir, mz, ftir_axis, mz_axis):
        ftir_feat = self.ftir_extractor(ftir, ftir_axis)
        mz_feat = self.mz_extractor(mz, mz_axis)
        combined = torch.cat([ftir_feat, mz_feat], dim=1)
        weights = self.gate(combined) * self.gate_bias
        weights = weights / weights.sum(dim=1, keepdim=True)
        gate_fused = weights[:, 0].unsqueeze(
            1) * ftir_feat + weights[:, 1].unsqueeze(1) * mz_feat  # [B, 256]
        output = self.classifier(gate_fused)  # [B, 2]
        return output


# 消融试验3：只用了MultiheadAttention
class CoAttnOnlyFusion(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim, dim=256, num_heads=4):
        super(CoAttnOnlyFusion, self).__init__()
        self.ftir_extractor = FTIREncoder(ftir_input_dim)
        self.mz_extractor = MZEncoder(mz_input_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.BatchNorm1d(dim//2),
            nn.ReLU(),
            SimpleResidualBlock(dim//2),
            nn.Linear(dim//2, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, ftir, mz, ftir_axis, mz_axis):
        ftir_feat = self.ftir_extractor(ftir, ftir_axis)
        mz_feat = self.mz_extractor(mz, mz_axis)
        ftir_seq = ftir_feat.unsqueeze(1)
        mz_seq = mz_feat.unsqueeze(1)
        cross_ftir, _ = self.attn(ftir_seq, mz_seq, mz_seq)
        cross_mz, _ = self.attn(mz_seq, ftir_seq, ftir_seq)
        attn_fused = (cross_ftir + cross_mz).squeeze(1)  # [B, 256]
        output = self.classifier(attn_fused)  # [B, 2]
        return output


# 消融试验4：把 Multi-headAttention 改成 Self-Attention
class SelfAttnFusion(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim, dim=256, num_heads=4):
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
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            SimpleResidualBlock(dim),
            nn.Linear(dim, 2),
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
            [gate_fused, self.proj(attn_fused)], dim=-1)  # [B, 512]
        output = self.classifier(final_fused)  # [B, 2]
        return output


# 消融试验5：只用了MultiheadAttention，并且是Self-Attention
class SelfAttnOnlyFusion(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim, dim=256, num_heads=4):
        super(SelfAttnOnlyFusion, self).__init__()
        self.ftir_extractor = FTIREncoder(ftir_input_dim)
        self.mz_extractor = MZEncoder(mz_input_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.BatchNorm1d(dim//2),
            nn.ReLU(),
            SimpleResidualBlock(dim//2),
            nn.Linear(dim//2, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, ftir, mz, ftir_axis, mz_axis):
        ftir_feat = self.ftir_extractor(ftir, ftir_axis)
        mz_feat = self.mz_extractor(mz, mz_axis)
        ftir_seq = ftir_feat.unsqueeze(1)
        mz_seq = mz_feat.unsqueeze(1)
        ftir_attn, _ = self.attn(ftir_seq, ftir_seq, ftir_seq)
        mz_attn, _ = self.attn(mz_seq, mz_seq, mz_seq)
        attn_fused = (ftir_attn + mz_attn).squeeze(1)  # [B, 256]
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

    def decision_function(self, X):
        return self.clf.decision_function(X)


# 横向对比模型1:CMACF
class ModalityMLP(nn.Module):
    def __init__(self, input_dim, output_dim=70):  # 输出70维（论文统一模态维度）
        super(ModalityMLP, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(0.1),
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, dim=70):
        super(SingleHeadSelfAttention, self).__init__()
        self.dim = dim
        # 论文Eq.3：Q/K/V线性映射矩阵
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)  # 论文Eq.5：注意力输出映射

    def forward(self, x):
        # 论文Eq.3：Q/K/V映射
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        # 论文Eq.4：单头注意力计算（缩放点积）
        attn_weights = torch.matmul(
            q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        # 仅输出映射后的注意力特征，无其他处理
        return self.W_o(attn_output)


class CrossModalFusion(nn.Module):
    def __init__(self, dim=70, hidden_dim=210):
        super(CrossModalFusion, self).__init__()
        self.dim = dim
        # 论文Table3：
        # 1. Attention块相关层
        self.layer_norm_before_attn = nn.LayerNorm(
            dim)  # Table3：LayerNorm_before（Attention前）
        # Table3：Dropout=0.1（Attention后）
        self.dropout_attn = nn.Dropout(0.1)
        self.layer_norm_after_attn = nn.LayerNorm(
            dim)  # Table3：LayerNorm_after（Attention后）

        # 2. 单头注意力核心模块（仅负责计算，无其他层）
        self.self_attention = SingleHeadSelfAttention(dim=dim)

        # 3. FFN块相关层（Table3后半部分）
        # Table3：LayerNorm_before（FFN前）（表格64为笔误）
        self.layer_norm_before_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.1)
        )
        self.layer_norm_after_ffn = nn.LayerNorm(
            dim)    # Table3：LayerNorm_after（FFN后）

    def forward(self, z1, z2):  # z1=FTIR(70维), z2=代谢组学(70维)
        def single_head_transformer(x):
            # self attention
            x_norm_attn = self.layer_norm_before_attn(x)
            attn_output = self.self_attention(x_norm_attn)
            attn_output = self.dropout_attn(attn_output)
            # Residual link（原始x + 注意力输出）
            x_res_attn = x + attn_output
            x_after_attn = self.layer_norm_after_attn(x_res_attn)
            # LayerNorm_before（FFN前）
            x_norm_ffn = self.layer_norm_before_ffn(x_after_attn)
            # FFN
            ffn_output = self.ffn(x_norm_ffn)
            # Residual link（FFN输入x_norm_ffn + FFN输出）
            x_res_ffn = x_norm_ffn + ffn_output
            x_final = self.layer_norm_after_ffn(x_res_ffn)
            return x_final

        # 步骤1：每个模态经过图6的Transformer块
        a1 = single_head_transformer(z1)
        a2 = single_head_transformer(z2)
        # 步骤2：论文Eq.6：克罗内克积构建双模态特征对（IM特征对）
        cross_feature = torch.kron(a1, a2)  # 输出：70×70=490维（匹配Table3输入）
        return cross_feature


# 看到这
class BimodalMapping(nn.Module):
    """阶段3：双模态特征映射（仅保留论文PrivateLinear，去掉相似性验证）"""

    def __init__(self, cross_dim=490, hidden_dim=20):  # hidden_dim=20（论文Table3）
        super(BimodalMapping, self).__init__()
        # 论文Eq.7：PrivateLinear私有非线性映射（降维+增强互补性）
        self.private_linear = nn.Sequential(
            nn.Linear(cross_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, cross_feature):
        return self.private_linear(cross_feature)  # 输出：20维

# -------------------------- 3. 完整双模态CMACF模型（论文四阶段框架）--------------------------


class BiModalCMACF(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim, num_classes=2):
        super(BiModalCMACF, self).__init__()
        # 阶段1：模态内特征提取（论文3.1节）
        self.ftir_mlp = ModalityMLP(ftir_input_dim, output_dim=70)
        self.mz_mlp = ModalityMLP(mz_input_dim, output_dim=70)

        # 阶段2：跨模态注意力交叉融合（论文3.2节）
        self.cross_fusion = CrossModalFusion(dim=70)

        # 阶段3：双模态特征映射（论文3.3节简化版，无相似性）
        self.bimodal_mapping = BimodalMapping(cross_dim=70*70, hidden_dim=20)

        # 阶段4：序列交互特征级融合+分类（论文3.4节）
        self.bilstm = nn.LSTM(
            input_size=70+70+20,  # 融合维度：FTIR(70)+MZ(70)+映射特征(20)
            hidden_size=70,        # 论文Table3：LSTM隐藏层70维
            num_layers=1,
            bidirectional=True,    # 双向LSTM
            batch_first=True
        )
        self.fc = nn.Linear(70*2, num_classes)  # 双向→140维，二分类输出
        self.softmax = nn.Softmax(dim=1)  # 论文Eq.14：softmax分类

    def forward(self, ftir, mz):
        # 阶段1：模态内特征提取（论文Eq.2）
        z1 = self.ftir_mlp(ftir)  # FTIR→70维
        z2 = self.mz_mlp(mz)      # MZ→70维

        # 阶段2：跨模态交叉融合（Transformer+克罗内克积，论文Eq.6）
        cross_feature = self.cross_fusion(z1, z2)  # 490维

        # 阶段3：双模态特征映射（论文Eq.7）
        mapped_feature = self.bimodal_mapping(cross_feature)  # 20维

        # 阶段4：序列交互融合（论文Eq.10：特征拼接）
        fusion = torch.cat([z1, z2, mapped_feature], dim=-
                           1).unsqueeze(1)  # [batch, 1, 160]
        lstm_output, _ = self.bilstm(fusion)
        # 论文Eq.13：使用全时间步信息（展平所有时间步）
        lstm_output = lstm_output.reshape(lstm_output.shape[0], -1)

        # 分类（论文Eq.14）
        output = self.fc(lstm_output)
        output = self.softmax(output)
        return output
