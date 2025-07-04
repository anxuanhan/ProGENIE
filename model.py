import torch
import torch.nn as nn
from einops import rearrange

class FeatureHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        tanh_out = torch.tanh(self.linear1(x))
        sigmoid_out = torch.sigmoid(self.linear2(x))
        return torch.cat([tanh_out, sigmoid_out], dim=-1)

class AttentionPooling(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled

class GeneExpressionModelHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=16059, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.pos_emb1D = nn.Parameter(torch.randn(100, input_dim))

        self.feature_heads = nn.ModuleList([
            FeatureHead(input_dim, hidden_dim) for _ in range(num_heads)
        ])
        self.attn_pooling_heads = nn.ModuleList([
            AttentionPooling(input_dim) for _ in range(num_heads)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(input_dim * num_heads, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, output_dim)
        )

    def forward(self, x):
        x = rearrange(x, 'b ... d -> b (...) d') + self.pos_emb1D

        pooled_outputs = []
        for head, attn in zip(self.feature_heads, self.attn_pooling_heads):
            features = head(x)
            pooled = attn(features)
            pooled_outputs.append(pooled)

        concatenated = torch.cat(pooled_outputs, dim=1)
        output = self.mlp(concatenated)
        return output
