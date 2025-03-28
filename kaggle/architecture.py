import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import altair as alt
import numpy ans np
alt.data_transformers.disable_max_rows()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q: [batch_size, num_heads, seq_len, head_dim]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(attn_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=0.1)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
            
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        batch_size = src.size(0)
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)
        
        src_emb = self.positional_encoding(self.encoder_embedding(src))
        enc_output = src_emb
        
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        tgt_emb = self.positional_encoding(self.decoder_embedding(tgt))
        dec_output = tgt_emb
        
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        output = self.fc_out(dec_output)
        return output
    
def test_transformer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    src_vocab_size = 100
    tgt_vocab_size = 100
    num_layers = 2

    src = torch.randint(0, src_vocab_size, (batch_size, seq_len)).to(device)
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, seq_len)).to(device)
    
    src_mask = torch.ones(batch_size, 1, 1, seq_len).to(device)
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).expand(batch_size, 1, seq_len, seq_len).to(device)

    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)

    print("="*50)
    print("1. Тест Positional Encoding")
    pe = PositionalEncoding(d_model, dropout=0.1).to(device)
    x = torch.randn(1, seq_len, d_model).to(device)
    print(f"До PE: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
    x_pe = pe(x)
    print(f"После PE: mean={x_pe.mean().item():.4f}, std={x_pe.std().item():.4f}")
    print(f"Форма PE: {x_pe.shape} (должна быть [1, {seq_len}, {d_model}])")
    
    print("\n2. Тест Multi-Head Attention")
    mha = MultiHeadAttention(d_model, num_heads).to(device)
    q = k = v = torch.randn(batch_size, seq_len, d_model).to(device)
    attn_output = mha(q, k, v)
    print(f"Форма выхода внимания: {attn_output.shape} (должна быть {q.shape})")
    print(f"Максимальное значение: {attn_output.max().item():.4f}")
    print(f"Минимальное значение: {attn_output.min().item():.4f}")

    print("\n3. Тест Encoder Layer")
    encoder_layer = EncoderLayer(d_model, num_heads).to(device)
    enc_input = torch.randn(batch_size, seq_len, d_model).to(device)
    enc_output = encoder_layer(enc_input)
    print(f"Форма выхода энкодера: {enc_output.shape} (должна быть {enc_input.shape})")
    print(f"Изменение данных: {torch.allclose(enc_input, enc_output, atol=1e-4)} (должно быть False)")

    print("\n4. Тест Decoder Layer")
    decoder_layer = DecoderLayer(d_model, num_heads).to(device)
    dec_input = torch.randn(batch_size, seq_len, d_model).to(device)
    dec_output = decoder_layer(dec_input, enc_output, src_mask, tgt_mask)
    print(f"Форма выхода декодера: {dec_output.shape} (должна быть {dec_input.shape})")
    print(f"Норма выходных данных: {dec_output.norm().item():.4f}")

    print("\n5. Полный тест Transformer")
    print("Входные данные:")
    print(f"src: {src.shape} (max={src.max().item()}, min={src.min().item()})")
    print(f"tgt: {tgt.shape} (max={tgt.max().item()}, min={tgt.min().item()})")
    
    output = transformer(src, tgt, src_mask, tgt_mask)
    print("\nПроверка формы выхода:")
    print(f"Ожидаемая форма: ({batch_size}, {seq_len}, {tgt_vocab_size})")
    print(f"Реальная форма:   {output.shape}")
    
    print("\nПроверка градиентов:")
    dummy_loss = output.sum()
    dummy_loss.backward()
    has_gradients = any(p.grad is not None for p in transformer.parameters())
    print(f"Градиенты вычислены: {has_gradients} (должно быть True)")

    print("\n6. Проверка параметров модели:")
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"Всего параметров: {total_params}")
    print(f"Параметры энкодера: {sum(p.numel() for p in transformer.encoder_embedding.parameters())}")
    print(f"Параметры декодера: {sum(p.numel() for p in transformer.decoder_embedding.parameters())}")

    print("\nТест завершен!")