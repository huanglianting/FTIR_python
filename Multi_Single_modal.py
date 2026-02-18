from scipy.interpolate import interp1d
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler


# ==================模块定义====================================
# 定义模态特征提取的分支
class FTIREncoder(nn.Module):
    def __init__(self, axis_dim):
        super(FTIREncoder, self).__init__()
        # Simplified FTIREncoder: 1 Conv layer + MLP
        # CNN might be better for spectral data than MLP
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Flatten()
        )
        
        # Calculate output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, axis_dim)
            dummy_output = self.features(dummy_input)
            flattened_dim = dummy_output.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, feat, feat_axis):
        feat = feat.unsqueeze(1) # [B, 1, Dim]
        feat = self.features(feat)
        feat = self.classifier(feat)
        return feat


class MZEncoder(nn.Module):
    def __init__(self, axis_dim):
        super(MZEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(axis_dim, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, feat, feat_axis):
        feat = feat.unsqueeze(1) # Add channel dimension for consistency, though it will be flattened
        feat = self.net(feat)
        return feat



class HybridFusion(nn.Module):
    def __init__(self, dim=64, num_heads=2):
        super().__init__()
        # Gate Fusion
        self.projection = nn.Linear(dim * 2, dim) # Add projection layer
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
        self.gate_bias = nn.Parameter(torch.tensor([0.5, 0.5]))
        # Attention Fusion
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=0.2)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ftir_feat, mz_feat):
        # Gate Fusion Part
        combined_gate = torch.cat([ftir_feat, mz_feat], dim=1)
        combined_gate = self.projection(combined_gate) # Apply projection
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
        self.fuser = HybridFusion(dim=64, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
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
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
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
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, mz, mz_axis):
        mz_feat = self.mz_extractor(mz, mz_axis)
        output = self.classifier(mz_feat)
        return output


# ==================其他消融试验模型定义====================================
# 消融试验1：简单拼接融合
class ConcatFusion(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim, dim=64):
        super(ConcatFusion, self).__init__()
        self.ftir_extractor = FTIREncoder(ftir_input_dim)
        self.mz_extractor = MZEncoder(mz_input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, 2)
        )

    def forward(self, ftir, mz, ftir_axis, mz_axis):
        ftir_feat = self.ftir_extractor(ftir, ftir_axis)
        mz_feat = self.mz_extractor(mz, mz_axis)
        combined = torch.cat([ftir_feat, mz_feat], dim=-1)  # [B, 512]
        output = self.classifier(combined)  # [B, 2]
        return output


# 消融试验2：仅保留Gate Fusion
class GateOnlyFusion(nn.Module):
    def __init__(self, ftir_input_dim, mz_input_dim, dim=64):
        super(GateOnlyFusion, self).__init__()
        self.ftir_extractor = FTIREncoder(ftir_input_dim)
        self.mz_extractor = MZEncoder(mz_input_dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2)
        )
        self.gate_bias = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.BatchNorm1d(dim//2),
            nn.ReLU(),
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
    def __init__(self, ftir_input_dim, mz_input_dim, dim=64, num_heads=2):
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
            nn.Linear(dim//2, 2)
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
    def __init__(self, ftir_input_dim, mz_input_dim, dim=64, num_heads=2):
        super(SelfAttnFusion, self).__init__()
        self.ftir_extractor = FTIREncoder(ftir_input_dim)
        self.mz_extractor = MZEncoder(mz_input_dim)
        # Gate Fusion
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2)
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
    def __init__(self, ftir_input_dim, mz_input_dim, dim=64, num_heads=2):
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
            nn.Linear(dim//2, 2)
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


# --------------------------传统机器学习模型--------------------------
# SVM
class SVMClassifier:
    def __init__(self, C=1.0, kernel='rbf', probability=True, random_state=42):
        self.clf = SVC(C=C, kernel=kernel, probability=probability,
                       random_state=random_state)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def decision_function(self, X):
        return self.clf.decision_function(X)


# 逻辑回归
class LogRegClassifier:
    def __init__(self, C=1.0, solver='lbfgs', max_iter=1000, random_state=42):
        self.clf = LogisticRegression(
            C=C, solver=solver, max_iter=max_iter, random_state=random_state)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def decision_function(self, X):
        if hasattr(self.clf, "decision_function"):
            return self.clf.decision_function(X)
        proba = self.clf.predict_proba(X)[:, 1]
        return proba


# 随机森林
class RFClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def decision_function(self, X):
        proba = self.clf.predict_proba(X)[:, 1]
        return proba


# GBDT（梯度提升树）
class GBDTClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, subsample=1.0, max_features=None, random_state=42):
        self.clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state
        )

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def decision_function(self, X):
        if hasattr(self.clf, "decision_function"):
            return self.clf.decision_function(X)
        proba = self.clf.predict_proba(X)[:, 1]
        return proba


# KNN
class KNNClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.clf = None

    def fit(self, X, y):
        # Dynamically adjust n_neighbors if sample size is small
        n_samples = X.shape[0]
        actual_n_neighbors = min(self.n_neighbors, n_samples) if n_samples > 0 else 1
        # Ensure at least 1 neighbor
        actual_n_neighbors = max(1, actual_n_neighbors)
        
        self.clf = KNeighborsClassifier(
            n_neighbors=actual_n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm
        )
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


# Gaussian Naive Bayes
class NBClassifier:
    def __init__(self, var_smoothing=1e-9):
        self.clf = GaussianNB(var_smoothing=var_smoothing)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X) 


# --------------------------横向对比模型1:zhou2024cmacf--------------------------
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
        batch_size = a1.size(0)
        # 使用更稳定的克罗内克积实现
        cross_feature = torch.einsum(
            "bi,bj->bij", a1, a2).view(batch_size, -1)  # [B, 70*70=4900]

        return cross_feature


class BimodalMapping(nn.Module):
    def __init__(self, cross_dim=4900, hidden_dim=20):  # hidden_dim=20（论文Table3）
        super(BimodalMapping, self).__init__()
        # 论文Eq.7：PrivateLinear私有非线性映射（降维+增强互补性）
        self.private_linear = nn.Sequential(
            nn.Linear(cross_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, cross_feature):
        return self.private_linear(cross_feature)  # 输出：20维


class BiModalCMACF(nn.Module):      # 完整双模态CMACF模型
    def __init__(self, ftir_input_dim, mz_input_dim, num_classes=2):
        super(BiModalCMACF, self).__init__()
        # 阶段1：模态内特征提取
        self.ftir_mlp = ModalityMLP(ftir_input_dim, output_dim=70)
        self.mz_mlp = ModalityMLP(mz_input_dim, output_dim=70)
        # 阶段2：跨模态注意力交叉融合
        self.cross_fusion = CrossModalFusion(dim=70)
        # 阶段3：双模态特征映射
        self.bimodal_mapping = BimodalMapping(cross_dim=70*70, hidden_dim=20)
        # 阶段4：序列交互特征级融合+分类
        # 对应论文BatchNorm1D（输入160=70+70+20）
        self.batch_norm = nn.BatchNorm1d(160)
        self.bilstm = nn.LSTM(
            input_size=160,  # 融合维度：FTIR(70)+MZ(70)+映射特征(20)，对应论文270维
            hidden_size=70,    # 论文Table3：LSTM隐藏层70维（完全沿用）
            num_layers=1,
            bidirectional=True,    # 双向LSTM
            batch_first=True
        )
        self.linear1 = nn.Linear(70*2, 70)  # 第一Linear：140（双向）→70（论文中间维度）
        self.linear2 = nn.Linear(70, num_classes)  # 第二Linear：70→2（二分类，论文是70→3）
        self.softmax = nn.Softmax(dim=1)  # 论文Eq.14：softmax分类
        # 关键：添加更好的权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重以提高收敛性"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用 Xavier 初始化
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                # LSTM权重初始化
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        torch.nn.init.constant_(param, 0)

    def forward(self, ftir, mz, ftir_axis=None, mz_axis=None):
        # 阶段1：模态内特征提取（论文Eq.2）
        z1 = self.ftir_mlp(ftir)  # FTIR→70维
        z2 = self.mz_mlp(mz)      # MZ→70维
        # 阶段2：跨模态交叉融合（Transformer+克罗内克积，论文Eq.6）
        cross_feature = self.cross_fusion(z1, z2)  # 490维
        # 阶段3：双模态特征映射（论文Eq.7）
        mapped_feature = self.bimodal_mapping(cross_feature)  # 20维
        # 阶段4：序列交互融合（完全对齐论文Table3流程）
        # 步骤1：特征拼接（论文Eq.10）
        fusion = torch.cat([z1, z2, mapped_feature], dim=-1)  # [batch, 160]
        # 步骤2：BatchNorm1D（论文Table3首层）
        fusion_bn = self.batch_norm(fusion)
        # 步骤3：适配LSTM输入格式 [batch, seq_len, input_size]
        fusion_bn = fusion_bn.unsqueeze(1)  # [batch, 1, 160]
        # 步骤4：LSTM
        lstm_output, _ = self.bilstm(fusion_bn)
        # 步骤5：使用全时间步信息（论文Eq.13，展平）
        lstm_output = lstm_output.reshape(
            lstm_output.shape[0], -1)  # [batch, 140]
        # 步骤6：第一个Linear层（论文Table3）
        linear1_out = self.linear1(lstm_output)
        # 步骤7：第二个Linear层（论文Table3）
        output = self.linear2(linear1_out)
        # 步骤8：Softmax（论文Eq.14）
        output = self.softmax(output)
        return output


# --------------------------横向对比模型2:chen2024diagnosis--------------------------
# 2.1 独立深度自编码器（论文2.3节）：将高维IR/Met映射到统一低维L=70
class IndependentDeepAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=70, dropout=0.25):
        super().__init__()
        # 编码器：输入维度→隐藏层→低维表示L=70（论文3层非线性变换）
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),  # 论文的归一化
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        z = self.encoder(x)  # 低维表示 (batch, L)
        return z


# 2.2 跨模态特定迁移融合模块（论文2.4节）：Specific Network + Common Network + 拼接融合
class CrossModalSpecificTransferFusion(nn.Module):
    def __init__(self, latent_dim=70, spec_dim=35, comm_dim=70):
        super().__init__()
        # 2.2.1 Specific Network：提取单模态特有特征 (L→K=35)
        self.ir_specific = nn.Sequential(
            nn.Linear(latent_dim, spec_dim), nn.ReLU())
        self.met_specific = nn.Sequential(
            nn.Linear(latent_dim, spec_dim), nn.ReLU())
        # 2.2.2 Common Network：提取跨模态共有特征 (2L→N=70)（论文2.4.2节）
        self.common = nn.Sequential(
            nn.Linear(2 * latent_dim, 2 * comm_dim),
            nn.ReLU(),
            nn.Linear(2 * comm_dim, comm_dim),
            nn.ReLU()
        )

    def forward(self, ir_z, met_z):
        # 1. 提取特有特征
        ir_spec = self.ir_specific(ir_z)  # (batch, K)
        met_spec = self.met_specific(met_z)  # (batch, K)
        # 2. 提取共有特征并切分（论文公式9/10）
        comm_input = torch.cat([ir_z, met_z], dim=1)  # (batch, 2L)
        comm_feat = self.common(comm_input)  # (batch, N=L)
        # 线性切分共有特征为IR/Met部分（论文2.4.3节）
        ir_comm, met_comm = torch.chunk(
            comm_feat, 2, dim=1)  # 各(batch, L/2)
        # 3. 迁移特征融合：VI^T = VI^C ⊕ VM^S ; VM^T = VM^C ⊕ VI^S（论文公式9/10）
        ir_transfer = torch.cat([ir_comm, met_spec], dim=1)  # (batch, L)
        met_transfer = torch.cat([met_comm, ir_spec], dim=1)  # (batch, L)
        return ir_transfer, met_transfer


# 2.3 决策层融合模块（论文2.5节）：迁移特征→分类器→结果平均
class DecisionLevelFusion(nn.Module):
    def __init__(self, latent_dim=70, num_classes=2, dropout=0.25):
        super().__init__()

        # 解码器：Dropout + 两层FC(70)+ReLU ；分类器：一层FC+Softmax（公式11）
        def build_modal_branch():
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, num_classes),
                nn.Softmax(dim=1)
            )

        # 红外光谱/代谢组学 独立分支（权重不共享，贴合论文）
        self.ir_branch = build_modal_branch()    # IR的Decoder+Classifier
        self.met_branch = build_modal_branch()   # Met的Decoder+Classifier

        # 关键：初始化权重，使初始预测接近均匀分布
        for layer in self.ir_branch:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0)
        for layer in self.met_branch:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, ir_transfer, met_transfer):
        # 单模态独立推理：Decoder+Classifier → 得到Softmax概率（公式11）
        ir_prob = self.ir_branch(ir_transfer)    # (batch, C)
        met_prob = self.met_branch(met_transfer)  # (batch, C)
        # 决策层融合：算术平均概率（公式12，论文核心要求）
        final_prob = (ir_prob + met_prob) / 2
        return final_prob


# 2.4 CMSTF主模型（整合所有模块，论文核心框架）
class CMSTF(nn.Module):
    def __init__(self, ir_dim, met_dim):
        super().__init__()
        # 1. 独立深度自编码器
        self.ir_ae = IndependentDeepAutoencoder(ir_dim)
        self.met_ae = IndependentDeepAutoencoder(met_dim)
        # 2. 跨模态特定迁移融合
        self.cmstf = CrossModalSpecificTransferFusion()
        # 3. 决策层融合
        self.dlf = DecisionLevelFusion()

    def forward(self, ir_x, met_x, ir_axis=None, met_axis=None):
        # 步骤1：自编码器提取低维表示
        ir_z = self.ir_ae(ir_x)
        met_z = self.met_ae(met_x)
        # 步骤2：跨模态迁移融合
        ir_transfer, met_transfer = self.cmstf(ir_z, met_z)
        # 步骤3：决策层融合得到最终预测
        final_logits = self.dlf(ir_transfer, met_transfer)
        return final_logits


# --------------------------横向对比模型3:leng2023raman--------------------------
# 1. MFCNN 核心模型
class MFCNN(nn.Module):
    def __init__(self, num_classes=2, latent_dim=54, dropout=0.5, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.filters = 64     # 论文固定滤波器数量64
        # 尺度1：1*1卷积 + BN + LeakyReLU + MaxPool1d(2)（First-Conv-1D）
        self.scale1 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.filters,
                      kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm1d(self.filters),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        # 尺度2：3*1→5*1→7*1卷积 + BN + LeakyReLU + MaxPool1d(2)（Second-Conv-1D）
        self.scale2 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.filters,
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(self.filters),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(self.filters, self.filters,
                      kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(self.filters),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(self.filters, self.filters,
                      kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm1d(self.filters),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 2, padding=0)
        )
        # 尺度3：3*1卷积 + BN + LeakyReLU + MaxPool1d(2)（Third-Conv-1D）
        self.scale3 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.filters,
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(self.filters),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 2, padding=0)
        )
        # 尺度4：直接MaxPool1d(2)（提取全局特征，Forth-Conv-1D）
        self.scale4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # 计算拼接后特征维度，适配任意latent_dim
        self.fc_in_dim = (self.filters * 3 +
                          self.in_channels) * (latent_dim // 2)
        # 全连接分类头（论文：Dense2048 + Dropout0.5 + 分类层）
        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_in_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x, axis=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 增加通道维度 (B,1,latent_dim)
        s1 = self.scale1(x)  # (B,64,27)
        s2 = self.scale2(x)  # (B,64,27)
        s3 = self.scale3(x)  # (B,64,27)
        s4 = self.scale4(x)  # (B,1,27)
        fusion_feat = torch.cat([s1, s2, s3, s4], dim=1)  # 通道维度拼接 (B,193,27)
        final_prob = self.fc_head(fusion_feat)
        return final_prob


# 2. CNN-LSTM 核心模型
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=2, raw_fusion_dim=37, lstm_hid=64, dropout=0.2):
        super().__init__()
        self.in_channels = 1  # 光谱为单通道1D序列
        self.lstm_hid = lstm_hid

        # CNN特征提取层（严格匹配论文：16→16→32→64滤波器，核3→5→6→4）
        self.cnn_backbone = nn.Sequential(
            # 第一组：16滤波器+3*1/5*1卷积 + BN + LeakyReLU + MaxPool1d(2)
            nn.Conv1d(self.in_channels, 16, kernel_size=3,
                      stride=1, padding='same'),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 2, padding=0),
            # 第二组：32滤波器+6*1卷积 + BN + LeakyReLU + MaxPool1d(2)
            nn.Conv1d(16, 32, kernel_size=6, stride=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 2, padding=0),
            # 第三组：64滤波器+4*1卷积 + BN + LeakyReLU + MaxPool1d(2)
            nn.Conv1d(32, 64, kernel_size=4, stride=1, padding='same'),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 2, padding=0),
        )

        # LSTM时序特征挖掘
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=lstm_hid, num_layers=1,
            batch_first=True, dropout=dropout, bidirectional=False
        )

        # 计算LSTM后全连接层输入维度，适配任意raw_fusion_dim
        self.lstm_out_dim = (raw_fusion_dim // 8) * lstm_hid
        # 全连接分类头（论文：Dense2048 + Dropout0.2 + 分类层）
        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.lstm_out_dim, 2048),
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重以避免模型偏向某一类"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 使用Xavier初始化卷积层权重
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # BatchNorm层权重初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 使用Xavier初始化线性层权重
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # 对于最后一层分类器，可以稍微初始化偏向中性值
                    if m.out_features == 2:  # 最后一层
                        nn.init.normal_(m.bias, mean=0.0, std=0.01)
                    else:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                # LSTM权重初始化
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)  # 使用正交初始化
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x, axis=None):
        # CNN提取局部光谱特征
        cnn_feat = self.cnn_backbone(x)  # (B,64, raw_fusion_dim//8)
        # 维度转置适配LSTM：(batch, seq_len, feat_dim)
        cnn_feat = cnn_feat.permute(0, 2, 1)
        # LSTM挖掘时序关联特征
        lstm_feat, _ = self.lstm(cnn_feat)  # (B, raw_fusion_dim//8, 64)
        # 全连接分类
        final_prob = self.fc_head(lstm_feat)
        return final_prob


def extract_pls_features(ftir_train, mz_train, y_train, ftir_val, mz_val, y_val,
                         ftir_components=6, mz_components=48):
    # 特征融合: 使用PLS提取特征
    ftir_scaler = MinMaxScaler(feature_range=(0, 1))
    ftir_pls = PLSRegression(n_components=ftir_components, scale=False)
    ftir_train_scaled = ftir_scaler.fit_transform(ftir_train.numpy())
    ftir_train_pls = ftir_pls.fit_transform(ftir_train_scaled, y_train.numpy())

    mz_scaler = MinMaxScaler(feature_range=(0, 1))
    mz_pls = PLSRegression(n_components=mz_components, scale=False)
    mz_train_scaled = mz_scaler.fit_transform(mz_train.numpy())
    mz_train_pls = mz_pls.fit_transform(mz_train_scaled, y_train.numpy())

    # 处理可能的元组返回值
    if isinstance(ftir_train_pls, tuple):
        ftir_train_pls = ftir_train_pls[0]
    if isinstance(mz_train_pls, tuple):
        mz_train_pls = mz_train_pls[0]

    # 验证集也需要转换
    ftir_val_scaled = ftir_scaler.transform(ftir_val.numpy())
    ftir_val_pls = ftir_pls.transform(ftir_val_scaled)
    mz_val_scaled = mz_scaler.transform(mz_val.numpy())
    mz_val_pls = mz_pls.transform(mz_val_scaled)

    if isinstance(ftir_val_pls, tuple):
        ftir_val_pls = ftir_val_pls[0]
    if isinstance(mz_val_pls, tuple):
        mz_val_pls = mz_val_pls[0]

    def ensure_2d(arr):
        if len(arr.shape) == 1:
            return arr.reshape(-1, 1)
        return arr

    # 确保二维数组
    ftir_train_pls = ensure_2d(ftir_train_pls)
    mz_train_pls = ensure_2d(mz_train_pls)
    ftir_val_pls = ensure_2d(ftir_val_pls)
    mz_val_pls = ensure_2d(mz_val_pls)

    # 将NumPy数组转换为PyTorch张量
    ftir_train_pls = torch.tensor(ftir_train_pls, dtype=torch.float32)
    ftir_val_pls = torch.tensor(ftir_val_pls, dtype=torch.float32)
    mz_train_pls = torch.tensor(mz_train_pls, dtype=torch.float32)
    mz_val_pls = torch.tensor(mz_val_pls, dtype=torch.float32)

    # 拼接特征，改用torch.cat，确保返回张量
    train_features = torch.cat([ftir_train_pls, mz_train_pls], dim=1)
    val_features = torch.cat([ftir_val_pls, mz_val_pls], dim=1)

    return train_features, val_features, ftir_scaler, ftir_pls, mz_scaler, mz_pls


def extract_raw_fusion_pls_features(ftir_train, mz_train, y_train, ftir_test, mz_test, y_test, n_components=37):
    # 拼接原始特征
    train_concat = np.hstack([ftir_train.numpy(), mz_train.numpy()])
    test_concat = np.hstack([ftir_test.numpy(), mz_test.numpy()])
    # PLS降维
    scaler = MinMaxScaler(feature_range=(0, 1))
    pls = PLSRegression(n_components=n_components, scale=False)
    # 训练集处理
    train_scaled = scaler.fit_transform(train_concat)
    train_pls = pls.fit_transform(train_scaled, y_train.numpy())
    # 测试集处理
    test_scaled = scaler.transform(test_concat)
    test_pls = pls.transform(test_scaled)
    # 处理可能的元组返回值
    if isinstance(train_pls, tuple):
        train_pls = train_pls[0]
    if isinstance(test_pls, tuple):
        test_pls = test_pls[0]

    # 确保二维数组
    def ensure_2d(arr):
        if len(arr.shape) == 1:
            return arr.reshape(-1, 1)
        return arr
    train_pls = ensure_2d(train_pls)
    test_pls = ensure_2d(test_pls)
    print(f"train_pls shape: {train_pls.shape}")
    print(f"test_pls shape: {test_pls.shape}")

    train_pls = torch.tensor(train_pls, dtype=torch.float32)
    test_pls = torch.tensor(test_pls, dtype=torch.float32)

    return train_pls, test_pls, scaler, pls
