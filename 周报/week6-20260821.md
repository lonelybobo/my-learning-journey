# 周报 - 2026.08.21 - 2026.08.27

## 一、本周核心工作/学习内容

- **理论学习**：
  - 学习 CLIP（Contrastive Language-Image Pretraining）的完整方法解析，从双编码器模型结构、对比学习预训练方式，到相似度矩阵与双向对比损失的数学原理。
  - 结合上周 MLLM 五阶段架构，进一步解构 LLaVA（Large Language and Vision Assistant）——"LLM + CLIP-ViT + MLP 投影层"的多模态大模型，用 PyTorch 打印模型结构逐层分析数据流。

  - 核心内容：
    - **CLIP 模型结构（双塔结构）**：
      - 图像编码器：ResNet / ViT，其中 ViT 将图像切分为固定大小的 Patch（如 16×16）后输入 Transformer；文本编码器：标准 Transformer（类 BERT），用 BPE 作为分词器
      - 对齐空间：通过训练把图像和文本映射到同一个 d 维语义空间，匹配的图文对距离近、不匹配的距离远
      - 投影层（Projection Layer）：在各自编码器之后设置可训练的线性投影层，将特征统一投影到固定维度的共享嵌入空间
    - **CLIP 对比学习预训练**：
      - 训练数据：从互联网收集的 4 亿个（图像, 文本）配对数据
      - 批次处理：N 个图文对得到 N 个图像特征 + N 个文本特征，两两计算余弦相似度得到 N×N 相似度矩阵
      - 正样本为对角线（原本匹配的对），负样本为非对角线（错误匹配的对），通过对称的对比损失同时优化图像和文本编码器
      - 温度系数 τ：可学习的缩放参数（默认 0.07），控制 softmax 的尖锐程度；τ 越小越"自信"、越尖锐，τ 越大越平滑
    - **相似度矩阵与损失函数**：
      - L2 归一化后模长为 1，点积即余弦相似度，计算更快
      - 相似度矩阵 S[i][j] 表示图像 I_i 与文本 T_j 的余弦相似度；行是"图像在提问"，列是"文本在提问"
      - 损失采用双向对比损失（InfoNCE）：I→T 对每一行做 softmax 取对角线负对数，T→I 对每一列做同样操作，总损失取两者平均
    - **LLaVA 模型结构逐层解析**：
      - 视觉塔 `CLIPVisionModel`：ViT-L/14 架构（patch_size=14、embed_dim=1024、24 层），图像切 14×14 patch 线性投影到 1024 维，位置编码 577 = 1 CLS + 24×24 patch，输出 `[batch, 577, 1024]`
      - 多模态投影器 `LlavaMultiModalProjector`：2 层 MLP（1024 → 4096 → 4096，中间 GELU），把视觉特征对齐到语言模型的 4096 维
      - 语言模型 `LlamaModel`：32 层、hidden_size=4096、词表 32064，视觉 token 拼在文本 token 前面一起送入，使用 RoPE 与 SwiGLU
      - 输出层 `lm_head`：Linear(4096, 32064)，将 LLM 输出映射到词表用于生成下一个 token
      - 整体数据流：图像 → CLIPVisionModel → `[577,1024]` → Projector → `[577,4096]` → 与文本嵌入拼接 → LlamaModel → lm_head 预测

- **实践/实验**：
  - 用 PyTorch 打印 LLaVA-1.5-7B 的完整模型结构，逐层核对视觉塔、投影器、语言模型的维度与层数，理解多模态 token 的拼接过程
  - 编写 LLaVA 推理代码：`AutoProcessor` 处理图片与文本（应用 chat_template 但不 tokenize），`LlavaForConditionalGeneration` 生成并解码输出，跑通图文问答流程
  - 编写 CLIP 图文检索代码：加载 `clip-vit-base-patch16`，构建图库编码索引、文本编码、Top-K 检索，在 COCO128 数据集上实现文字检索图片
  - 编写 CLIP 零样本分类代码：构造类别 prompt 一次性编码，图像特征与文本特征算相似度、softmax 后取 Top-K，无需训练即可分类

- **技能/工具**：
  - `transformers`：`CLIPModel`/`CLIPProcessor`（`get_image_features`、`get_text_features`、`logit_scale`）、`LlavaForConditionalGeneration`/`AutoProcessor`（`apply_chat_template`、`batch_decode`）
  - 数学与实现：L2 归一化、N×N 余弦相似度矩阵、softmax 温度缩放、InfoNCE/交叉熵等价代码实现（`F.cross_entropy`）
  - 模型结构分析：`print(model)` 打印完整结构、逐层核对维度理解多模态模型数据流

- **其他**：无

## 二、问题与解决

* 在总结之后是本周学习工作的详细可供复现的笔记，选择学习 CLIP 和 LLaVA 的内容是因为他们对于 MLLMs 十分重要，许多模型结构沿用了 LLaVA 结构，而 CLIP 可以说改变了传统的图文检索任务，并提供了一个很好的编码器制作思路。

## 三、下周计划

- [x] 深入学习 BLIP、BLIP-2 内容，了解它们的核心内容。
- [x] 逐步学习到可以训练一个轻量的多模态模型。

## 四、总结

- 本周在"图文多模态"方向上迈出了关键一步：以 CLIP 为主线，从双塔结构、对比学习预训练，到相似度矩阵、温度缩放、双向对比损失等。
- 通过 LLaVA 的模型结构逐层解析，加深上周 MLLM 五阶段架构的第一、二、三阶段的理解。

---

# Part3-2 CLIP 解析

### 1. 模型结构

CLIP本质上是双编码器“两塔结构（dual-encoder）”：

- 图像编码器：
  - 主流架构是 ResNet / Vision Transformer（ViT），其中 VIT 将图像切分成固定大小的Patch（如16x16像素），并将这些Patch序列输入到Transformer中进行处理。这是 CLIP 最常用、效果较好的方案
  - 输入：图像，原始的像素矩阵
  - 输出：富含语义的图像特征向量
- 文本编码器：
  - 基于标准的 Transformer 架构，其设计类似于 BERT，使用字节对编码 (BPE) 作为分词器
  - 输入：文本序列
  - 输出：语义向量
- 对齐空间：
  - 通过训练，把图像和文本映射到同一个 d 维语义空间
  - 相同语义的图文 pair 在该空间中距离接近，不同语义的距离远
  - 投影层 (Projection Layer) 的作用是在各自编码器之后，设置一个可训练的线性投影层。它将图像和文本编码器输出的特征维度，统一投影到一个固定维度的共享嵌入空间，所以 CLIP 算是涵盖了输入投影层的工作。

### 2. 预训练方式

- CLIP不直接进行图文生成，而是通过对比学习来判断图文是否匹配。
  - **训练数据**: 使用从互联网收集的4亿个(图像, 文本)配对数据。
  - **批次处理**: 对于一个包含N个图文对的批次，模型会生成N个图像特征向量和N个文本特征向量。
  - **计算相似度矩阵**: 计算N个图像向量和N个文本向量两两之间的余弦相似度，得到一个N x N的矩阵。
  - **损失函数**:
    - 正样本: 矩阵对角线上的N对，它们是原本就匹配的图文对，训练目标是最大化它们的相似度。
    - 负样本: 矩阵非对角线上的N² - N对，它们是错误匹配的图文对，训练目标是最小化它们的相似度。
  - 通过这种对称的对比损失（通常为对称交叉熵损失），模型同时优化图像和文本编码器。
  - **温度系数 (Temperature)**: 在计算损失时，会使用一个可学习的温度参数 τ 来缩放相似度分数，控制模型对难易样本的区分度。该参数通常初始化为 0.07。

## CLIP应用：LLaVA（Large Language and Vision Assistant）

LLaVA的核心路线是：LLM（如 LLaMA） + 视觉编码器（CLIP-ViT） + 一个简单线性（MLP）投影层，配合高质量多模态 instruction 数据。

### 1. pytorch查看模型结构

```python
LlavaForConditionalGeneration(
  (model): LlavaModel(
    (vision_tower): CLIPVisionModel(
      (embeddings): CLIPVisionEmbeddings(
        (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
        (position_embedding): Embedding(577, 1024)
      )
      (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True, bias=True)
      (encoder): CLIPEncoder(
        (layers): ModuleList(
          (0-23): 24 x CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True, bias=True)
            (mlp): CLIPMLP(
              (activation_fn): QuickGELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True, bias=True)
          )
        )
      )
      (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True, bias=True)
    )
    (multi_modal_projector): LlavaMultiModalProjector(
      (linear_1): Linear(in_features=1024, out_features=4096, bias=True)
      (act): GELUActivation()
      (linear_2): Linear(in_features=4096, out_features=4096, bias=True)
    )
    (language_model): LlamaModel(
      (embed_tokens): Embedding(32064, 4096)
      (layers): ModuleList(
        (0-31): 32 x LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
          (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        )
      )
      (norm): LlamaRMSNorm((4096,), eps=1e-05)
      (rotary_emb): LlamaRotaryEmbedding()
    )
  )
  (lm_head): Linear(in_features=4096, out_features=32064, bias=False)
)
```

### 2. 视觉编码器（Vision Tower）
```python
(vision_tower): CLIPVisionModel
```
- 使用 **CLIP 的 ViT-L/14** 架构（因为 patch_size=14，embed_dim=1024，24层）。
- **输入**：图像 `[batch, 3, H, W]`（通常 224×224 或 336×336）
- **处理流程**：
  - `patch_embedding`：将图像切成 14×14 的 patch，线性投影到 1024 维。
  - `position_embedding`：位置编码，577 = 1（CLS token）+ 24×24（patch 数，假设输入 336/14=24）。
  - `pre_layrnorm` + 24层 `CLIPEncoderLayer`（自注意力 + FFN）+ `post_layernorm`。
- **输出**：视觉特征 `[batch, 577, 1024]`

### 3. 多模态投影器（Multi-modal Projector）
```python
(multi_modal_projector): LlavaMultiModalProjector
```
- 一个 **2层 MLP**：`1024 → 4096 → 4096`，中间有 GELU 激活。
- **作用**：将视觉特征从 CLIP 的 1024 维**对齐到语言模型的 4096 维**，使视觉 token 能直接输入 LLM。
- **输出**：`[batch, 577, 4096]`

### 4. 语言模型（Language Model）
```python
(language_model): LlamaModel
```
- 使用 **Llama 架构**（32层，hidden_size=4096，词表 32064）。
- **输入**：文本 token（`[batch, seq_len]`）经过 `embed_tokens` 得到 `[batch, seq_len, 4096]`
- **特殊机制**：
  - 视觉 token（来自 projector）会**拼在文本 token 前面**，一起送入 LLM。
  - LLM 的 `LlamaAttention` 使用旋转位置编码（RoPE）。
  - FFN 使用 SwiGLU（`gate_proj` + `up_proj` + `down_proj`）。
- **输出**：最后一个 token 的隐状态 `[batch, seq_len, 4096]`

### 5. 输出层（LM Head）
```python
(lm_head): Linear(4096, 32064, bias=False)
```
- 将 LLM 输出映射到词表大小，用于生成下一个 token。

### 整体数据流（推理时）
```
图像 → CLIPVisionModel → [577, 1024] 
                          ↓
                    Multi-modal Projector → [577, 4096]
                                          ↓
文本 → embed_tokens → [seq_len, 4096] → 拼接 → [577+seq_len, 4096]
                                                ↓
                                          LlamaModel (32层)
                                                ↓
                                          lm_head → 预测下一个 token
```

推理代码，参考 https://hf-mirror.com/llava-hf/llava-1.5-7b-hf：

```python
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# 1. 模型和处理器
model_id = "models/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# 2. 准备图片和文本
image = Image.open("image.png").convert("RGB") 

# prompt = "USER: <image>\n请用中文描述这张图片。\nASSISTANT:"
# inputs = processor(
#     text=prompt,
#     images=image,
#     return_tensors="pt"
# ).to(device)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"}
            {"type": "text", "text": "请用中文描述这张图片。"},
        ],
    },
]

# 应用模板，但不进行 tokenize（先用模板生成文本）
prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False  # 先不 tokenize，得到文本字符串
)

inputs = processor(
    text=prompt,
    images=image,  # 传入图片对象
    return_tensors="pt"
).to(device)

# 3. 推理
generate_ids = model.generate(
    **inputs,
    max_new_tokens=200
)

# 4. 解码输出
output = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(output)
```

# Part4. CLIP 方法解析 + 图文检索
参考 https://github.com/openai/CLIP

CLIP（Contrastive Language-Image Pretraining）使用了 **对比学习（Contrastive Learning）** 的方式来训练图像和文本的匹配关系，其核心思想是让正确的图像-文本对在特征空间中靠近，而错误的对则远离。

## 1. CLIP 方法解析

CLIP 的目标是学习一个 **共享语义空间**，使得匹配的图像和文本更接近，不匹配的更远。

给定一个 batch：

$$
{(I_i, T_i)}_{i=1}^N
$$

双塔结构编码与归一化

- 图像编码：

$$
v_i = f_{\text{image}}(I_i)
$$

- 文本编码：

$$
t_i = f_{\text{text}}(T_i)
$$

- L2 归一化：

$$
v_i \leftarrow \frac{v_i}{|v_i|}, \quad t_i \leftarrow \frac{t_i}{|t_i|}
$$

 归一化后：

$$
v_i^\top t_j = \cos(\theta_{i,j})
$$

L2 归一化后所有向量的模长：

$$
∥x∥=1∥y∥=1
$$

于是求余弦相似度会出现以下结果：

- 原本：

$$dot(x,y)=∥x∥∥y∥cos⁡θ$$

- 归一化后：

$$
dot(x,y)=cos⁡θ
$$

假设一张“狗”的图片，经过图像编码器后，得到了原始特征向量 **v**。

- 第1步：求平方和。把向量里每个数字平方后加起来。比如 [3, 4] 平方和 = 9+16=25。

- 第2步：开根号求模长。√25 = 5。

- 第3步：每个数字都除以模长。[3÷5, 4÷5] = [0.6, 0.8]。

归一化后，[3,4] 和 [0.6,0.8] 方向完全一样（都是朝右上），算相似度时只看“指向哪里”，不管“多长”。两个长度都是1的向量，它们的点积结果就是余弦相似度（值在-1到1之间）。计算机做点积比做余弦公式快得多。

## 2. 相似度矩阵（logits）

### 2.1 构造相似度矩阵

把所有图像特征与所有文本特征**两两做点积**，得到 N×N 矩阵：

$$
S \in \mathbb{R}^{N \times N}, \quad S_{i,j} = v_i^\top t_j
$$

因为特征已经 L2 归一化，所以 `S[i][j]` 其实就是图像 I_i 和文本 T_j 的**余弦相似度**：

$$
S_{i,j} = \cos(\theta_{i,j})
$$

### 2.2 矩阵里每个位置代表什么

以 batch = 4 为例：

```
         T_0     T_1     T_2     T_3    ← 文本作为列
I_0    [ 0.91  | 0.12  | 0.05  | -0.03 ]
I_1    [ 0.08  | 0.85  | 0.11  | 0.02  ]
I_2    [ 0.10  | 0.09  | 0.93  | 0.07  ]
I_3    [ 0.04  | -0.02 | 0.06  | 0.88  ]
↑
图像作为行
```

- **第 i 行**：图像 I_i 与所有文本的相似度（"这张图像什么文本？"）
- **第 j 列**：文本 T_j 与所有图像的相似度（"这段文本描述哪张图？"）
- **对角线** S_{i,i}：匹配对（正样本），期望值最大
- **非对角线** S_{i,j} (i≠j)：不匹配对（负样本），期望值尽量小

```
I_0 和 T_0 是一对 → S[0][0] = 0.91   应该最大
I_0 和 T_2 不匹配 → S[0][2] = 0.05   应该很小
```

训练的最终目标，就是让对角线亮起来、非对角线暗下去。

### 2.3 为什么还要除以 temperature？

除以 τ 之前，相似度范围大约在 [-1, 1]，直接 softmax 出来的概率分布太平坦，模型难以区分谁是谁。所以加入一个可学习的缩放系数：

$$
\text{logits}_{i,j} = \frac{S_{i,j}}{\tau}
$$

τ 的作用：

- **τ 越小** → logits 越大 → softmax 越尖锐 → 对微小差异更敏感（更"自信"）
- **τ 越大** → logits 越小 → softmax 越平滑 → 更保守

```python
import torch
import torch.nn.functional as F

s = torch.tensor([0.9, 0.1, 0.05, 0.02])

for tau in [0.5, 0.07, 0.01]:
    probs = F.softmax(s / tau, dim=0)
    print(f"tau={tau:.2f}: {probs.tolist()}")
```

```
tau=0.50: [0.40, 0.24, 0.21, 0.15]   太平坦，几乎区分不出来
tau=0.07: [0.93, 0.03, 0.02, 0.01]   有区分度，又不过度
tau=0.01: [1.00, 0.00, 0.00, 0.00]   太尖锐，可能过拟合噪声
```

τ = 0.07 是 CLIP 论文的默认值（等价于 `logit_scale ≈ 1/0.07 ≈ 14.29`，代码里用 `model.logit_scale.exp()` 得到，训练时是可学习的）。

### 2.4 为什么叫 logits？

在分类任务里，**logits 就是 softmax 的输入**（未经 softmax 的原始打分）。

- 相似度 S 是原始的相似度分数
- 除以 τ 之后变成 logits，可以直接丢给 `F.cross_entropy`
- 任何一行 logits 经 softmax 后都变成一个概率分布（每行之和为 1）

$$
p_{i,j} = \mathrm{softmax}(\mathrm{logits})_{i,j}
= \frac{\exp(S_{i,j}/\tau)}{\sum_{k=1}^{N} \exp(S_{i,k}/\tau)}
$$

这就为第 3 节的损失函数做好了准备：每行对应对角线位置的"正确类别"。

## 3. 损失函数（InfoNCE / Cross Entropy）

CLIP 使用 **双向对比损失（bidirectional contrastive loss）**：把每个 batch 当成一个 N 分类问题，正样本（对角线）要赢过所有负样本（非对角线）。

### 3.1 Image → Text（I→T）

**视角**：拿着图像 I_i，在 batch 里的 N 个文本中找出它的正确文本 T_i。第 i 个文本是"正确答案"，其余 N-1 个文本都是**负样本**。

**做法**：对相似度矩阵的**每一行**做 softmax，得到"图像 I_i 选到每个文本的概率"，然后取对角线的负对数：

$$
\mathcal{L}_{i \to t} = - \frac{1}{N} \sum_{i=1}^{N}
\log \frac{\exp(S_{i,i}/\tau)}{\sum_{j=1}^{N} \exp(S_{i,j}/\tau)}
$$

**逐步拆解**（用第 2 节的例子，第 0 行 `[0.91, 0.12, 0.05, -0.03]`，τ=0.07）：

1. 把 logits 全部除以 τ：`[13.0, 1.7, 0.7, -0.4]`
2. 做 softmax：得到概率分布，例如 `[0.9997, 0.0002, 0.0001, 0.0000]`
3. 取对角线上正确位置的概率：`p(选到 T_0) = 0.9997`
4. 算负对数：`-log(0.9997) ≈ 0.0003`，越接近 0 说明越匹配
5. 对 N 行都这么做，再取平均

```python
# I→T 等价实现（一次矩阵运算）
loss_i2t = F.cross_entropy(logits, torch.arange(N))   # logits: (N, N)，按行做 softmax
```

含义：

- 固定图像 ( I_i )
- 在所有文本中找 ( T_i )
- 要让"正确文本"的概率比其他 N-1 个文本都高

### 3.2 Text → Image（T→I）

**视角**：拿着文本 T_j，在 batch 里的 N 张图像中找出正确图像 I_j。

**做法**：对相似度矩阵的**每一列**做 softmax（等价于对 logits 转置后的每行做 softmax）：

$$
\mathcal{L}_{t \to i} = - \frac{1}{N} \sum_{j=1}^{N}
\log \frac{\exp(S_{j,j}/\tau)}{\sum_{i=1}^{N} \exp(S_{i,j}/\tau)}
$$

```python
# T→I 等价实现（转置后再按行做 softmax）
loss_t2i = F.cross_entropy(logits.T, torch.arange(N))   # logits.T: (N, N)
```

含义：

- 固定文本 ( T_j )
- 在所有图像中找 ( I_j )

**为什么分母用 S_{i,j} 求和而不是 S_{j,i}？** 因为分母要遍历"第 j 列的所有 i"，即固定文本 T_j，对比它与所有图像的相似度。

### 3.3 总损失

把两个方向取平均：

$$
\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{i \to t} + \mathcal{L}_{t \to i} \right)
$$

为什么两个方向都要算？第 5 节会详细讲：只用一个方向会让多个图像塌缩到同一个文本（embedding collapse），双向约束能强制**一一匹配**。

**一个具体 batch 的完整流程**（N=16, dim=512）：

```python
logits = image_features @ text_features.T / tau   # (16, 16)
labels = torch.arange(16)                          # 对角线上才是匹配对
loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
```

- `F.cross_entropy` 内部自动做 softmax + 负对数似然，无需手动 softmax
- `logits[i]`：第 i 张图对 16 个文本的"打分"
- `labels[i] = i`：告诉模型第 i 张图的正确文本是第 i 个

## 4. 与 Cross Entropy 的等价性

### 4.1 Cross Entropy 在做什么

标准的交叉熵多分类损失：

$$
\mathcal{L}_{CE} = - \frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(z_{i, y_i})}{\sum_{k=1}^{C} \exp(z_{i,k})}
$$

- z 是打分矩阵，(N 个样本 × C 个类别)
- y_i 是第 i 个样本的真实类别

把它和第 3 节的 InfoNCE 对照着看，结构几乎一模一样：

|              | InfoNCE（第 3 节）                 | Cross Entropy              |
| ------------ | ---------------------------------- | -------------------------- |
| 打分矩阵     | 相似度 logits (N, N)               | 分类打分 z (N, C)          |
| 每个"样本"   | 一张图像 / 一段文本                | 一个数据点                 |
| 正样本       | 对角线 S_{i,i}                     | 真实类别那一项 z_{i, y_i}  |
| 负样本       | 同一行 / 列的其余 N-1 项           | 其余 C-1 个类别            |

### 4.2 关键：把标签设成对角线下标

因为 CLIP 的数据是**对齐的**（第 i 张图配第 i 段文本），所以"真实类别"就是它自己的下标：

$$
y_i = i, \qquad \text{logits} = S / \tau
$$

于是两个方向的损失可以直接写成两个 `F.cross_entropy`：

```python
labels = torch.arange(N)
loss_i2t = F.cross_entropy(logits, labels)     # logits: (N, N)，每行是一个 N 分类
loss_t2i = F.cross_entropy(logits.T, labels)   # logits.T: (N, N)，转置后每行也是 N 分类
```

`F.cross_entropy` 内部自动完成 softmax + 负对数似然 + 取平均，所以这一行代码和第 3 节的两个公式完全等价。

### 4.3 本质：InfoNCE ≡ CrossEntropy

$$
\text{InfoNCE} \equiv \text{CrossEntropy}
$$

只是视角不同：

| 视角          | 含义                 | 正样本来源           |
| ------------- | -------------------- | -------------------- |
| Cross Entropy | 多分类问题           | 类别标签 y_i         |
| InfoNCE       | 正样本 vs 多个负样本 | batch 内配对好的样本 |

InfoNCE 是更通用的表述：它不需要离散的类别，只要能构造出"正样本对 + 负样本对"就能用。CLIP 恰好让"类别"等于 batch 下标，所以两者完全等价。

## 5. 为什么需要双向损失？

### 5.1 先看单向 loss 会出什么问题

如果只保留 I→T：

$$
\mathcal{L} = \mathcal{L}_{i \to t}
$$

单向损失只约束"行"，行与行之间、列与列之间都没有约束。假设训练中图像特征开始互相靠近（collapse 的苗头），可能出现这样的相似度矩阵：

```
          T_0    T_1    T_2
I_0   [  0.9  | 0.1  | 0.8 ]   ← I_0 对 T_0 和 T_2 都很接近
I_1   [  0.1  | 0.9  | 0.1 ]
I_2   [  0.1  | 0.1  | 0.6 ]
```

- 从 **I→T** 看：每一行都是"自己那一列"最大（0.9、0.9、0.6 分别是行内最大），单向损失已经很小，"看起来没问题"
- 从 **T→I** 看：T_2 最匹配的是 I_0（0.8 > 0.6）而不是 I_2，配对是错的

结论：**行能分对，不等于列也能分对**。只训练单向损失，这种"看似合理、实则模糊"的状态不会被惩罚，长期下去可能走向：

- 多个图像匹配同一个文本（列冗余）
- embedding 空间塌缩（collapse）：图像 / 文本特征互相挤到同一个区域，语义区分度消失

### 5.2 logits 是方阵，但行和列的语义不同

相似度矩阵：

$$
S \in \mathbb{R}^{N \times N}
$$

- 第 i 行：图像 I_i vs 所有文本（图像在提问）
- 第 j 列：文本 T_j vs 所有图像（文本在提问）

虽然矩阵是方阵，但行和列是**两个不同的问题**，softmax 的分母都不一样：

I→T（行 softmax）：

$$
\sum_{j} \exp(S_{i,j}/\tau)
$$

T→I（列 softmax）：

$$
\sum_{i} \exp(S_{i,j}/\tau)
$$

分母不同 ⇒ 两个 loss 不等价，一个方向的损失低不代表另一个方向也低。

### 5.3 双向 loss 的作用

1. **强制一一匹配**

目标从"每个图像都有文本要"变成"互相对应"：

$$
I_i \leftrightarrow T_i
$$

而不是：

$$
I_1, I_2 \rightarrow T_3
$$

2. **提供双重约束**

- 行约束（image query）：每个图像必须认出自己的文本
- 列约束（text query）：每个文本必须认出自己的图像

两个方向互相牵制，任何一方的"偷懒"（比如图像互相挤在一起）都会被另一方惩罚。

3. **更稳定的梯度**

每个样本被优化两次：

- 一次作为 query（出现在某一行）
- 一次作为 target（出现在某一列）

梯度信息从两个方向回流，训练更稳，batch 也用得更充分。

4. **提升对称性**

学到的是双向的对齐关系：

$$
\text{image} \leftrightarrow \text{text}
$$

既可以用图检索文，也可以用文检索图，而不是只能单向检索。

## 6. 总结

CLIP 的损失函数可以概括为：

- 使用 **cosine similarity + temperature scaling**
- 构造 **N×N 相似度矩阵**
- 使用 **InfoNCE（= CrossEntropy）**
- 采用 **双向对比损失（I→T + T→I）**

核心优势：

1. **负样本免费**：利用 batch 内其他样本作为负样本，无需额外构造，训练高效
2. **双向约束**：行、列互相牵制，防止 embedding 塌缩
3. **统一语义空间**：学到的是可双向检索的跨模态语义空间，天然支持图文互检

从公式到代码的对照：

| 概念     | 公式                            | 代码                                      |
| -------- | ------------------------------- | ----------------------------------------- |
| 相似度   | S_{i,j} = v_i^T t_j             | `image_features @ text_features.T`        |
| 温度缩放 | logits = S / τ                  | `/ self.temperature`                      |
| 标签     | y_i = i                         | `torch.arange(N)`                         |
| 双向损失 | (L_{i→t} + L_{t→i}) / 2         | `(CE(logits) + CE(logits.T)) / 2`         |

接下来用代码了解CLIP的一些简单应用，CLIP完全可以看作一个编码器。

* 简单的图片检索实现（文本检索图片）

```python
import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import time

# 1. 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("../models/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("../models/clip-vit-base-patch16")

model.eval()

# 2. 构建图库，对所有图片进行编码
def build_image_index(image_paths):
    image_features_list = []

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            # 只运行图像编码器部分
            feat = model.get_image_features(**inputs)
            feat = feat.pooler_output

        # 归一化
        feat = feat / feat.norm(dim=-1, keepdim=True)
        image_features_list.append(feat)

    # 拼接得到张量 形状：(N, 512)
    image_features = torch.cat(image_features_list, dim=0)

    return image_features

# 3. 文本编码
def encode_text(query):
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        text_features = text_features.pooler_output
    # 归一化
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features    # 形状：(1, 512)

# 4. 检索 Top-K
def retrieve_topk(query, image_features, image_paths, k=3):
    text_feat = encode_text(query)
    # 相似度计算
    logits = text_feat @ image_features.T   # 形状：(1, N)

    # 温度系数 这在CLIP里是一个可学习的参数
    logit_scale = model.logit_scale.exp()
    logits = logits * logit_scale

    # Top-K
    topk_scores, topk_indices = torch.topk(logits, k=k, dim=1)

    results = []
    for score, idx in zip(topk_scores[0], topk_indices[0]):
        results.append({
            "image_path": image_paths[idx],
            "score": f"{score.item():.4f}"
        })

    return results


if __name__ == "__main__":
    # 图库位置
    image_dir = "../dataset/coco128/images/train2017"
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]

    print("Building image index...")
    image_features = build_image_index(image_paths)
    # torch.save(image_features, "image_index.pt")
    print("Image index built with shape:", image_features.shape)
    # 查询
    query = "a photo of a cat"

    for i in range(3):
        t1 = time.time()
        results = retrieve_topk(query, image_features, image_paths, k=3)
        print(f"Retrieval done in {time.time() - t1:.4f} seconds")

    print("\nQuery:", query)
    print("Top-K results:")
    for r in results:
        print(f"{r['image_path']}  score={r['score']}")
```

* 简单的零样本分类

```python
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# 1. 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("../models/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("../models/clip-vit-base-patch16")

model.eval()

# 2. 构造文本 prompts
def build_prompts(class_names, template="a photo of a {}"):
    return [template.format(name) for name in class_names]


# 3. 编码文本（一次性）
def encode_text(prompts):
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        text_features = text_features.pooler_output

    # 归一化
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features  # 形状：(num_classes, 512)


# 4. 编码图像
def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features.pooler_output

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features  # 形状：(1, 512)

# 零样本分类
def zero_shot_classify(image_path, class_names, topk=3):
    # 构造 prompt
    prompts = build_prompts(class_names)

    # 编码
    text_features = encode_text(prompts)
    image_features = encode_image(image_path)

    # 相似度
    logits = image_features @ text_features.T  # 形状：(1, num_classes)

    # CLIP temperature
    logit_scale = model.logit_scale.exp()
    logits = logits * logit_scale

    # 概率
    probs = logits.softmax(dim=-1)

    # Top-K
    topk_probs, topk_indices = torch.topk(probs, k=topk, dim=-1)

    results = []
    for prob, idx in zip(topk_probs[0], topk_indices[0]):
        results.append({
            "label": class_names[idx],
            "prob": prob.item()
        })

    return results


if __name__ == "__main__":
    image_path = "../dataset/coco128/images/train2017/000000000074.jpg"

    class_names = [
        "cat",
        "dog",
        "car",
        "truck",
        "airplane",
        "person"
    ]

    results = zero_shot_classify(image_path, class_names, topk=3)

    print("Top predictions:")
    for r in results:
        print(f"{r['label']}: {r['prob']:.4f}")
```

代码示例如上，clip做图像检索和零样本分类时，最大的区别是：
- 在做图像检索时是用text和image_features矩阵的转至计算相似度，用文字检索图像。
- 而在做零样本分类时，实际上是在用图片检索文本，计算的是image_feature和texts转至的相似度。
 
传统分类模型使用固定类别的分类头，对图像进行特征提取后，由分类头进行输出，并经过softmax转换成对应类别的概率。与clip的检索方式相比类别固定不变，若要增加类别需重新训练分类头。
