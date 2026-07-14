"""
Positional Encoding - 从零实现到 PyTorch 版本
============================================
目标：彻底搞懂 Transformer 的位置编码
  1. 为什么需要位置编码
  2. 公式的直觉：不同频率的"时钟指针"
  3. 从零实现（纯 numpy）
  4. PyTorch 实现（可插入 Transformer 模型）
  5. 验证关键性质：唯一性 & 相对位置线性变换
  6. 可视化（ASCII 热力图）
"""

import math
import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: 为什么需要位置编码？
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PART 1: Self-Attention 是「排列无关」的 —— 演示")
print("=" * 65)

# 模拟 3 个 token 的 embedding（d_model=4）
# 句子 A: ["我", "打", "你"]
# 句子 B: ["你", "打", "我"]  （顺序打乱）

def simple_self_attention(X):
    """
    极简化 Self-Attention（Q=K=V=X，不带 weight matrix）
    只是为了演示"排列无关性"
    """
    # attention scores: (seq_len, seq_len)
    scores = X @ X.T / math.sqrt(X.shape[1])
    # softmax
    exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
    return attn @ X  # weighted sum

# 假设三个词的 embedding
wo  = np.array([0.1, 0.8, 0.2, 0.5])  # "我"
da  = np.array([0.9, 0.1, 0.7, 0.3])  # "打"
ni  = np.array([0.3, 0.6, 0.1, 0.9])  # "你"

sentence_A = np.stack([wo, da, ni])  # "我打你"
sentence_B = np.stack([ni, da, wo])  # "你打我"

out_A = simple_self_attention(sentence_A)
out_B = simple_self_attention(sentence_B)

print("\n句子 A「我打你」的 attention 输出：")
print(np.round(out_A, 4))

print("\n句子 B「你打我」的 attention 输出（顺序不同，但各 token 输出完全一样）：")
print(np.round(out_B, 4))

print('\n结论：out_A[0]（"我"的输出）== out_B[2]（"我"的输出）：',
      np.allclose(out_A[0], out_B[2]))
print("→ 没有位置编码时，模型完全感知不到词序！\n")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: 公式拆解 ——「不同频率的时钟指针」
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PART 2: 公式直觉 —— 不同频率的正弦波")
print("=" * 65)

print("""
公式（原论文）：
  PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
  PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

变量含义：
  pos      → token 在序列中的位置（0, 1, 2, ...）
  i        → embedding 维度的「对」索引（0 ~ d_model/2 - 1）
  d_model  → embedding 总维度（如 512）

关键：分母 10000^(2i/d_model) 控制频率
  - i=0  时分母=1,        波极短（每个位置都剧变）← 秒针
  - i=255时分母≈10000,   波极长（几千个位置才变）← 时针
""")

# 可视化：展示不同维度对「0~20号 token」的编码值
d_model = 16
positions = np.arange(20)

print("不同维度（i=0,1,2,3）对位置 0~9 的编码值：\n")
print(f"{'pos':>4}", end="")
for i in range(4):
    print(f"  dim{2*i}(sin)  dim{2*i+1}(cos)", end="")
print()
print("-" * 80)

for pos in range(10):
    print(f"{pos:>4}", end="")
    for i in range(4):
        freq = 1.0 / (10000 ** (2 * i / d_model))
        s = math.sin(pos * freq)
        c = math.cos(pos * freq)
        print(f"  {s:+.4f}      {c:+.4f}", end="")
    print()

print("\n注意：dim0（i=0，高频）的值在每个 pos 变化剧烈；")
print("      dim6（i=3，低频）的值几乎不变，要更长的序列才能观察到变化。\n")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: 从零实现位置编码（纯 numpy，无 PyTorch）
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PART 3: 从零实现（纯 numpy）")
print("=" * 65)

def positional_encoding_numpy(max_len: int, d_model: int) -> np.ndarray:
    """
    从零实现正余弦位置编码

    Args:
        max_len: 支持的最大序列长度
        d_model: embedding 维度（必须为偶数）

    Returns:
        PE 矩阵，shape = (max_len, d_model)
    """
    assert d_model % 2 == 0, "d_model 必须为偶数"

    # 第一步：构造 pos 列向量 shape=(max_len, 1)
    pos = np.arange(max_len).reshape(-1, 1)   # [[0], [1], [2], ...]

    # 第二步：构造频率向量 shape=(d_model//2,)
    # 原始公式的分母：10000^(2i/d_model)
    # 取对数计算更稳定：exp(2i/d_model * ln(10000))
    i = np.arange(d_model // 2)               # [0, 1, 2, ..., d_model/2 - 1]
    div_term = np.exp(i * (-math.log(10000.0) / d_model))

    # 第三步：广播乘法，得到 (max_len, d_model//2) 的角度矩阵
    angles = pos * div_term                   # broadcast: (max_len,1) × (d_model//2,)

    # 第四步：构造输出矩阵，偶数列=sin，奇数列=cos
    PE = np.zeros((max_len, d_model))
    PE[:, 0::2] = np.sin(angles)              # 所有偶数维度
    PE[:, 1::2] = np.cos(angles)              # 所有奇数维度

    return PE

# 演示
PE_np = positional_encoding_numpy(max_len=50, d_model=16)
print(f"\n生成的 PE 矩阵 shape: {PE_np.shape}")
print("\n前 5 个 token 的位置编码（d_model=16）：")
print(np.round(PE_np[:5], 4))

# 验证：每个位置的编码都是唯一的
print("\n验证「每个位置拥有唯一指纹」：")
print(f"  pos=0 的 L2 距离到 pos=1: {np.linalg.norm(PE_np[0] - PE_np[1]):.4f}")
print(f"  pos=0 的 L2 距离到 pos=5: {np.linalg.norm(PE_np[0] - PE_np[5]):.4f}")
print(f"  pos=0 的 L2 距离到 pos=49:{np.linalg.norm(PE_np[0] - PE_np[49]):.4f}")
print("  → 每个位置都不同！\n")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: PyTorch 实现（可接入真实 Transformer）
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PART 4: PyTorch 实现（可插入 Transformer 模型）")
print("=" * 65)

class PositionalEncoding(nn.Module):
    """
    标准正余弦位置编码（来自 "Attention Is All You Need"）

    使用方式：
        pe_layer = PositionalEncoding(d_model=512, max_len=5000, dropout=0.1)
        x = embedding(tokens)       # shape: (batch, seq_len, d_model)
        x = pe_layer(x)             # 注入位置信息后输出同 shape
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # ── 构造 PE 矩阵（一次性计算，永久缓存）──
        # shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # pos: (max_len, 1)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)

        # div_term: (d_model//2,)
        # 等价于 1/10000^(2i/d_model)，用 exp(log) 保证数值稳定
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # 偶数列 → sin，奇数列 → cos
        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model//2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model//2)

        # 增加 batch 维度 → (1, max_len, d_model)，方便与 (B, T, D) 广播
        pe = pe.unsqueeze(0)

        # register_buffer：不是参数（不参与梯度），但随模型保存/加载
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embedding，shape = (batch_size, seq_len, d_model)
        Returns:
            x + PE[:seq_len]，shape 不变
        """
        # self.pe shape: (1, max_len, d_model)
        # x[:, :seq_len] 自动截取到当前序列长度
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── 测试 ──
d_model = 16
batch_size = 2
seq_len = 10

pe_layer = PositionalEncoding(d_model=d_model, max_len=100, dropout=0.0)

# 模拟 token embedding 输出
fake_embedding = torch.randn(batch_size, seq_len, d_model)
output = pe_layer(fake_embedding)

print(f"\nInput  shape: {fake_embedding.shape}  ← (batch, seq_len, d_model)")
print(f"Output shape: {output.shape}          ← 位置编码后，shape 不变")
print(f"\n缓存的 PE 矩阵 shape: {pe_layer.pe.shape}  ← (1, max_len, d_model)")
print(f"\n前 3 个位置的 PE 向量（前 8 维）：")
for pos in range(3):
    vals = pe_layer.pe[0, pos, :8].numpy()
    print(f"  pos={pos}: {np.round(vals, 4)}")

print("\nPE 是否参与梯度：", pe_layer.pe.requires_grad)
print("→ register_buffer 确保它不会被 optimizer 更新，但随 state_dict 保存。\n")


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: 验证最关键的数学性质 —— 相对位置可以线性变换表达
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PART 5: 验证「相对位置的线性变换」性质")
print("=" * 65)

print("""
核心定理：PE(pos+k) = M(k) · PE(pos)
  其中 M(k) 是只依赖偏移量 k，不依赖绝对位置 pos 的变换矩阵。

对每个维度对 (2i, 2i+1)，变换矩阵是 2×2 旋转矩阵：
  M_i(k) = [[cos(k·ω_i),  sin(k·ω_i)],
             [-sin(k·ω_i), cos(k·ω_i)]]
其中 ω_i = 1/10000^(2i/d_model)

这正是后来 RoPE 旋转位置编码的灵感来源！
""")

def verify_linear_transform(d_model=8, pos=7, k=3):
    """验证 PE(pos+k) 确实等于 M(k) @ PE(pos)"""

    PE = positional_encoding_numpy(max_len=100, d_model=d_model)

    pe_pos   = PE[pos]        # PE(pos)
    pe_pos_k = PE[pos + k]    # PE(pos+k)，这是目标

    # 构造 M(k)：每对 (sin, cos) 维度都有一个 2x2 旋转子块
    M = np.zeros((d_model, d_model))
    for i in range(d_model // 2):
        omega_i = 1.0 / (10000 ** (2 * i / d_model))
        angle = k * omega_i
        cos_k = math.cos(angle)
        sin_k = math.sin(angle)
        # 第 2i 行/列
        M[2*i,   2*i]   =  cos_k
        M[2*i,   2*i+1] =  sin_k
        # 第 2i+1 行/列
        M[2*i+1, 2*i]   = -sin_k
        M[2*i+1, 2*i+1] =  cos_k

    pe_predicted = M @ pe_pos  # 用线性变换预测 PE(pos+k)

    error = np.max(np.abs(pe_predicted - pe_pos_k))
    print(f"  pos={pos}, k={k}:")
    print(f"    PE(pos+k)  实际值: {np.round(pe_pos_k[:6], 5)}")
    print(f"    M(k)@PE(pos) 预测: {np.round(pe_predicted[:6], 5)}")
    print(f"    最大误差: {error:.2e}  {'✓ 完全吻合！' if error < 1e-10 else '✗ 不对'}")

print("验证对多个 (pos, k) 组合：")
verify_linear_transform(d_model=8, pos=5,  k=3)
print()
verify_linear_transform(d_model=8, pos=15, k=7)
print()
verify_linear_transform(d_model=8, pos=1,  k=10)

print("""
这意味着什么？
  Transformer 的 QK^T 注意力分数可以通过学习权重矩阵，
  自动捕捉任意偏移 k 下两个 token 的关系 —— 无论它们在句中的绝对位置在哪。
  这就是 Transformer 能泛化到不同长度句子的原因之一。
""")


# ─────────────────────────────────────────────────────────────────────────────
# PART 6: ASCII 热力图可视化
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PART 6: ASCII 热力图（在终端感受频率梯度）")
print("=" * 65)

def ascii_heatmap(matrix, title="", row_label="pos", col_label="dim",
                  width=50, height=20):
    """
    将 2D 矩阵渲染为 ASCII 热力图
    颜色：█ (高) → ▓ ▒ ░ → ' ' (低)
    """
    chars = "█▓▒░ "

    # 降采样到 (height, width)
    rows, cols = matrix.shape
    r_idx = (np.linspace(0, rows-1, height)).astype(int)
    c_idx = (np.linspace(0, cols-1, width)).astype(int)
    sampled = matrix[np.ix_(r_idx, c_idx)]

    # 归一化到 [0, 1]
    mn, mx = sampled.min(), sampled.max()
    norm = (sampled - mn) / (mx - mn + 1e-9)

    print(f"\n{title}")
    print(f"  {col_label} →   (共 {cols} 维，低维=高频，高维=低频)")
    print(f"  " + "─" * (width + 4))
    for i, row_val in enumerate(r_idx):
        bar = "".join(chars[int((1 - norm[i, j]) * (len(chars)-1))]
                      for j in range(width))
        print(f"  {row_label}={row_val:3d} │{bar}│")
    print(f"  " + "─" * (width + 4))
    print(f"  {'←高频（i小）':^{width//2}}{'低频（i大）→':^{width//2}}")

PE_vis = positional_encoding_numpy(max_len=50, d_model=64)
ascii_heatmap(
    PE_vis,
    title="正余弦位置编码热力图 (max_len=50, d_model=64)",
    row_label="pos",
    col_label="dim",
    width=60,
    height=25
)

print("""
热力图解读：
  纵轴：token 位置（0 在上，49 在下）
  横轴：embedding 维度（左边=低维=高频，右边=高维=低频）

  左侧呈现密集的明暗交替 → 高频，每个位置都有剧烈变化（秒针）
  右侧几乎是纯色 → 低频，需要很长的序列才能看出变化（时针）

  每一行（每个 pos）都是独一无二的"条纹指纹"！
""")


# ─────────────────────────────────────────────────────────────────────────────
# PART 7: 完整 Mini-Transformer 演示（把 PE 接入真实模型）
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PART 7: 把 PE 接入一个完整的 Mini-Transformer Encoder")
print("=" * 65)

class MiniTransformerEncoder(nn.Module):
    """
    最小化可运行的 Transformer Encoder
    架构：Embedding → PositionalEncoding → TransformerEncoderLayer × N → Linear
    """

    def __init__(self, vocab_size: int, d_model: int, nhead: int,
                 num_layers: int, max_len: int, num_classes: int):
        super().__init__()

        # 1. Token Embedding：把词 id 映射到向量空间
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding：注入位置信息
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout=0.1)

        # 3. Transformer Encoder（多头注意力 + FFN）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True   # 期望输入 (B, T, D) 而非 (T, B, D)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 分类头
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, token_ids: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            token_ids:    (batch, seq_len) 的整数 tensor
            padding_mask: (batch, seq_len) bool，True=忽略该位置
        Returns:
            logits: (batch, num_classes)
        """
        # Embedding：(B, T) → (B, T, D)
        x = self.embedding(token_ids) * math.sqrt(self.embedding.embedding_dim)
        # √d_model 缩放：防止 embedding 值太小被 PE 的量级淹没

        # 注入位置编码：(B, T, D) → (B, T, D)，shape 不变
        x = self.pos_enc(x)

        # Transformer Encoder
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # 取 [CLS] token（位置 0）的输出做分类
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


# ── 运行一次前向传播验证 ──
torch.manual_seed(42)
model = MiniTransformerEncoder(
    vocab_size=1000,
    d_model=32,
    nhead=4,
    num_layers=2,
    max_len=128,
    num_classes=3
)

# 模拟一个 batch：2 条句子，长度 10
token_ids = torch.randint(0, 1000, (2, 10))
logits = model(token_ids)

print(f"\nMini-Transformer Encoder 前向传播：")
print(f"  输入 token_ids shape: {token_ids.shape}  ← (batch=2, seq_len=10)")
print(f"  输出 logits    shape: {logits.shape}     ← (batch=2, num_classes=3)")
print(f"\n模型参数量：{sum(p.numel() for p in model.parameters()):,}")

print("""
数据流：
  token_ids (2, 10)
       ↓  Embedding × √d_model
  x    (2, 10, 32)   ← 语义向量，但还不知道谁在哪里
       ↓  PositionalEncoding（相加，不改变 shape）
  x    (2, 10, 32)   ← 现在每个 token 都携带了位置指纹
       ↓  TransformerEncoder（2 层）
  x    (2, 10, 32)   ← 每个 token 都「看过了」整个序列
       ↓  取 x[:, 0, :]（CLS token）
  cls  (2, 32)
       ↓  Linear
  logits (2, 3)      ← 分类输出
""")

print("=" * 65)
print("总结")
print("=" * 65)
print("""
┌──────────────────────────────────────────────────────────────┐
│  位置编码的三大关键设计决策                                  │
├──────────────────────────────────────────────────────────────┤
│ 1. 为什么是正余弦？                                          │
│    → 三角函数和差公式保证：PE(pos+k) = M(k) @ PE(pos)       │
│      相对距离 k 可以用与绝对位置无关的线性变换表达          │
│                                                              │
│ 2. 为什么用不同频率（10000^(2i/d_model) 作分母）？          │
│    → 不同维度扮演不同精度的"时钟"                           │
│      高频维度 → 区分相邻位置                                 │
│      低频维度 → 区分长距离位置                               │
│      组合在一起 → 每个位置有唯一"指纹"                      │
│                                                              │
│ 3. 为什么是相加而不是拼接？                                  │
│    → 高维空间中随机向量近似正交，线性层可以解耦             │
│      拼接会导致维度翻倍 → 参数量和计算量平方爆炸            │
└──────────────────────────────────────────────────────────────┘

进阶：RoPE（旋转位置编码）
  → 把相加改成「旋转」，把 M(k) 直接融入 Q/K 矩阵乘法
  → 更好的外推性，LLaMA/GPT-NeoX 等现代大模型都在用
  → 本质：把正余弦 PE 的线性变换性质发挥到极致
""")
