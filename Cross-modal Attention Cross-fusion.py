import torch
import torch.nn as nn
import torch.nn.functional as F


class FullDimensionalFeatureSubnet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FullDimensionalFeatureSubnet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)


class CrossModalAttentionCrossFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossModalAttentionCrossFusion, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x.unsqueeze(0)  # Add sequence dimension
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.squeeze(0)
        return self.layer_norm(x + attn_output)


class CMACF(nn.Module):
    def __init__(self, input_dim_mz, input_dim_ftir, hidden_dim, num_classes):
        super(CMACF, self).__init__()
        self.mz_subnet = FullDimensionalFeatureSubnet(input_dim_mz, hidden_dim)
        self.ftir_subnet = FullDimensionalFeatureSubnet(input_dim_ftir, hidden_dim)
        self.mz_attention = CrossModalAttentionCrossFusion(hidden_dim)
        self.ftir_attention = CrossModalAttentionCrossFusion(hidden_dim)
        self.bilstm = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, mz_data, ftir_data):
        mz_features = self.mz_subnet(mz_data)
        ftir_features = self.ftir_subnet(ftir_data)

        mz_attn = self.mz_attention(mz_features)
        ftir_attn = self.ftir_attention(ftir_features)

        combined_features = torch.cat((mz_attn, ftir_attn), dim=1)
        combined_features = combined_features.unsqueeze(1)  # Add sequence dimension

        lstm_out, _ = self.bilstm(combined_features)
        lstm_out = lstm_out.squeeze(1)

        output = self.fc(lstm_out)
        return F.softmax(output, dim=1)


def train_model(model, X_train_mz, X_train_ftir, y_train, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_mz, X_train_ftir)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def test_model(model, X_test_mz, X_test_ftir, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_mz, X_test_ftir)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Test Accuracy: {accuracy * 100:.2f}%')