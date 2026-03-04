"""
GoldTransformer: 基于 Transformer Encoder 的金价时序预测模型
纯 PyTorch 实现，无额外依赖
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """标准正弦/余弦位置编码"""
    
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # handle odd d_model
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class GoldTransformer(nn.Module):
    """
    Transformer Encoder 时序预测模型
    
    架构:
      Input (seq_len, input_size) 
        → Linear Projection → d_model
        → Positional Encoding
        → N × Transformer Encoder Layer (Multi-Head Attention + FFN)
        → Global Average Pooling
        → MLP Head → 1 (预测下一天收盘价)
    """
    
    def __init__(self, input_size=7, seq_length=60, d_model=64, nhead=4, 
                 num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        
        self.input_size = input_size
        self.seq_length = seq_length
        self.d_model = d_model
        
        # 输入投影: (batch, seq, input_size) → (batch, seq, d_model)
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length + 10)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 预测头
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Xavier 权重初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_size)  -- 例如 (32, 60, 7)
        output: (batch, 1)               -- 预测下一天收盘价
        """
        # 投影到 d_model 维
        x = self.input_projection(x)          # (batch, seq, d_model)
        
        # 加入位置编码
        x = self.pos_encoder(x)               # (batch, seq, d_model)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)       # (batch, seq, d_model)
        
        # 全局平均池化（取所有时间步的均值）
        x = x.mean(dim=1)                     # (batch, d_model)
        
        # 预测头
        out = self.head(x)                    # (batch, 1)
        return out


if __name__ == "__main__":
    # 快速验证模型形状
    model = GoldTransformer(input_size=7, seq_length=60)
    dummy = torch.randn(4, 60, 7)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
