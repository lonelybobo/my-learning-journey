# 周报 - 2026.08.28 - 2026.09.03

## 一、本周核心工作/学习内容

- **理论学习**：
  - 学习 BLIP（Bootstrapping Language-Image Pre-training）的完整方法，理解 MED（Multimodal Mixture of Encoder-Decoder）这一"一模型三任务"复合架构的设计动机；进而学习 BLIP-2 用冻结 Image Encoder + 冻结 LLM + 可训练 Q-Former 的轻量对齐范式。
  - 承接前几周 MLLM 段架构与 LLaVA 结构，动手从零搭建一个最简化的 MiniLLaVA（CLIP-ViT + Projector + Qwen1.5），并完善"数据构建 → 训练循环 → 推理"的完整训练流程，让没有多模态能力的 Qwen1.5 初步"看懂"图片。

  - 核心内容：
    - **BLIP 的 MED 架构**：
      - MED 共用同一套 Transformer 层，通过不同功能层组合完成三类任务：单模态编码器（ITC）、基于图像的文本编码器（ITM）、基于图像的文本解码器（LM）
    - **BLIP-2 的 Q-Former**：
      - 三阶段架构：阶段一冻结 Image Encoder 提图像特征；阶段二用可训练 Q-Former（少量 Learnable Queries + Cross-Attention）压缩并"翻译"图像信息；阶段三冻结 LLM，把 Q-Former 输出作为前缀输入文本生成
    - **MiniLLaVA 三大组件**：
      - **Vision Encoder**（CLIP-ViT-B/16）：取 `last_hidden_state` 并去掉 CLS，只保留 patch 特征 `[B, 196, 768]`（pooler_output 是整图全局特征，不适合图像理解任务）；可冻结参数
      - **Projector**：两层全连接（768 → 2048 → LLM hidden）+ LayerNorm + GELU，把视觉特征映射到文本语义空间与文本 embedding 拼接
      - **LLM Decoder**（Qwen1.5-1.8B）：通过 `get_input_embeddings()` 取文本 embedding，用 `model(inputs_embeds=...)` 接收拼接后的多模态 embedding
      - **labels 与 attention_mask 的区别**：图片部分 labels 置 -100（CrossEntropyLoss ignore，图片 token 不用于预测下一个词），attention_mask 图片部分置 1（图片信息参与 Self-Attention）
    - **MiniLLaVA 训练 pipeline**：
      - dataset 解析 LLaVA-CC3M 的 chat.json（human/gpt 角色、\<image> 占位符），统一用 "User：问题\nAssistant 回答" 的 prompt 模板（训练与推理保持一致）
      - DataCollator 动态 padding，answer 部分保留真实 token id、prompt 与 padding 部分置 -100
      - train loop：`Accelerator` 统一接管多卡分片、混合精度、梯度裁剪、checkpoint 保存；scheduler 总步数按 prepare 后的 dataloader 长度计算
      - checkpoint 只保存 `requires_grad` 或 `lora_` 的参数（冻结的 clip/qwen 基座不存），加载时 `strict=False` 按名称匹配

- **实践/实验**：
  - 编写 BLIP caption 推理代码：`BlipProcessor` + `BlipForConditionalGeneration`，加载 `blip-image-captioning-base` 对 COCO128 图片生成描述，并封装成类便于复用
  - 对比实验：同一张图，BLIP 输出完整句子、BLIP-2 输出简短答案（two cats）——并非 BLIP-2 更弱，而是两者任务不同（BLIP 偏图像描述、BLIP-2 偏 VQA 简洁作答），换用 "Describe the image in detail" 类 prompt 可引导更丰富的输出
  - 编写 BLIP-2 caption 推理代码：`Blip2Processor` + `Blip2ForConditionalGeneration`（`blip2-flan-t5-xl`，fp16），验证 prompt 引导生成
  - 编写 BLIP caption 的 LoRA SFT 实验：冻结 `vision_model`，LoRA 只作用在文本部分 attention 的 query/value（不能设 task_type，多模态输入含 pixel_values，PEFT 会错误构建 inputs_embeds），在 COCO Caption 数据上用 Trainer 训练并保存 LoRA
  - 从零搭建 MiniLLaVA：分别实现 VisionEncoder / Projector / LLMDecoder 三个模块并串成完整前向，跑通 `outputs.logits`、`past_key_values`、`loss` 的形状验证

- **技能/工具**：
  - `transformers`：`BlipProcessor`/`BlipForConditionalGeneration`、`Blip2Processor`/`Blip2ForConditionalGeneration`、`CLIPVisionModel`/`CLIPImageProcessor`、`AutoModelForCausalLM`（`get_input_embeddings`、`inputs_embeds` 前向、`generate`）
  - `peft`：多模态模型上挂 LoRA 的注意事项（冻结视觉塔、target_modules 选 query/value、不设 task_type）
  - `accelerate`：`Accelerator`/`prepare`/`backward`/`clip_grad_norm_`/`gather_for_metrics`/`wait_for_everyone`，多进程安全的 checkpoint 保存
  - 训练细节：labels 的 -100 掩码（prompt/padding/图像部分不计 loss）、动态 padding、YAML 配置化参数管理、`accelerate launch` 启动训练

- **其他**：无

## 二、问题与解决

* 在总结之后是本周学习工作的详细可供复现的笔记

## 三、下周计划

- [ ] 因为竞赛需求，了解 PCB 电路图的基础内容，搜集 AI 在读 PCB 图方面的应用。

## 四、总结

- 本周沿着"图文多模态"主线继续推进：先从 BLIP 的 MED 复合架构理解"一模型三任务"（ITC/ITM/LM）与自举洗数据，再到 BLIP-2 用冻结强模型 + 轻量 Q-Former 的对齐范式，清楚了 MLLM 五阶段架构中输入投影这一环的两种主流实现（LLaVA 的简单 MLP 与 BLIP-2 的 Q-Former）。
- 实践上实现了从"复现别人模型"到"自己搭一个可训练的多模态模型"的跨越：用 CLIP-ViT + Projector + Qwen1.5 从零串起 MiniLLaVA，补齐 dataset/DataCollator/train loop/checkpoint/infer 全流程，并真正在 RTX 3060 上用小样本微调让 Qwen1.5 初步具备看图说话能力，把前几周学的 LoRA、CLIP、LLaVA、五阶段架构、Transformers/Trainer/accelerate 等知识完整串联了起来。
- 印象最深的是 BLIP vs BLIP-2 输出的"反差"实验——同样一张图、回答风格迥异，提醒我评价生成模型时要先弄清任务设定与 prompt 引导；而给多模态模型挂 LoRA、区分 labels 与 attention_mask 这类细节，也让我对"图像 token 如何进入并影响 LLM"有了更本质的理解。下一步打算用完整数据训练 MiniLLaVA 并对照更多视觉语言对齐方案。

---

# Part5. BLIP / BLIP-2

## BLIP

论文参考：[《BLIP: Bootstrapping Language-Image Pre-training for Unifified Vision-Language Understanding and Generation》](https://proceedings.mlr.press/v162/li22n.html)

### MED

全称为Multimodal Mixture of Encoder-Decoder，是一个可以完成三个任务的复合型模型。

MED主要由四部分组成，每个部分相同模型层共用参数，不同功能的模型层组合可以实现不同的功能。 
与CLIP类似，VIT作图像编码，能完成CLIP的检索任务，也能完成生成任务。
图像编码器的输出会流向以下三个不同的结构。

#### 1、ITC-单模态文本编码器

这个结构类似BERT使用的编码器。将文本开头加入一个[CLS]token之后输入编码器进行编码，然后其输出与图片编码器的输出进行image-text contrastive任务，类似CLIP中的图片和文本特征计算相似度。

这个任务本质就是对比学习的做法，ITC loss借鉴了ALBEF中的做法，引入动量编码器。
其实作者准备了两个编码器

- 主编码器（student）正常训练，参与反向传播，参数：θ

- 动量编码器（teacher）不参与反向传播，参数：θ_m，更新方式：

$$
\theta_m \leftarrow m \cdot \theta_m + (1 - m) \cdot \theta
$$
m 通常是 0.995 / 0.999

使用软标签可以让模型学习到更细粒度的语义信息，而不是简单地二值化判断

#### 2、ITM-基于图像的文本编码器

这个编码器与上一个的唯一不同点就在于多加了一个cross attention(CA)操作。将图片编码器的输出embedding作为query，文本编码器中self attention(SA)之后的embedding作为key和value进行CA操作，这样可以学习到图片和文本的多模态表达以用于捕捉更加精细的视觉与语言之间的对应关系。

这个结构对应的则是image-text matching任务。这是一个二分类任务，通过添加一个线性层输出positive或者negative来表示图片和文本是否匹配。输入时，须在开头处添加[Encode]token表示起始

#### 3、LM-基于图像的文本解码器

双向自注意力机制改为了因果自注意力机制，即Transformer中的decoder只与之前出现的token进行attention操作。这个结构是用于基于给定图片生成对应的文本描述。

它对应语言模型Language Modeling任务，损失函数是交叉熵。同样地，输入文本开头需要添加一个[Decode]token表示开头，结尾需要添加一个[EOS]字符。

训练方式和标准的文本生成模型（类似 GPT）是一样的，只不过它是以图像特征 + 文本前缀作为条件来生成文本。label就是文本序列本身（右移后的 token），也就是做teacher forcing的next-token prediction。

#### Captioner 和 Filter 解决数据集噪声问题

作者利用标注精准的数据集训练了一个MED后，利用BLIP的图像描述能力，在网图（标注不太准的数据集）中生成了较为准确的图像描述句子。再利用BLIP的图像文本比对能力做一次筛选。用洗过的数据又训了一次MED。也就是作者提到的boostrapping，自举法，先用准数据训练模型，用该模型洗数据后，再用于模型训练。

#### BLIP caption实验

模型各种方法使用参考https://hf-mirror.com/docs/transformers/main/en/model_doc/blip

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

def generate_caption(pil_image=None,
                     model_path="../models/blip-image-captioning-base",
                     max_length=50, num_beams=5,temperature=1.0):
    # 1.加载模型和处理器
    processor = BlipProcessor.from_pretrained(model_path)
    # ConditionalGeneration图像描述生成模型
    model = BlipForConditionalGeneration.from_pretrained(model_path)

    # 2.加载图片
    if pil_image == None:
        print("No image provided, using default image.")
        img_url = "../dataset/coco128/images/train2017/000000000030.jpg"
        image = Image.open(img_url).convert("RGB")
    else:
        image = pil_image[0].convert("RGB")

    # 3. 预处理输入
    # 将图片转换为 PyTorch tensor，并自动做 resize / normalize 等处理
    # return_tensors="pt" 表示返回 PyTorch 格式
    inputs = processor(images=image, return_tensors="pt")
    # inputs = processor(images=image, text="用中文描述这张图片：", return_tensors="pt")

    # 4. 将模型和数据移动到 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 5. 生成 caption
    out = model.generate(
        **inputs,          # 输入图片特征
        max_length=max_length,     # 生成文本的最大长度
        num_beams=num_beams,       # Beam Search
        temperature=temperature    # 控制随机性（1.0为默认，越小越保守）
    )

    # 6. 解码输出
    caption = processor.decode(out[0], skip_special_tokens=True)
    print("Caption:", caption)
    return caption


# 可以封装成类，便于修改
class BLIPCaptioner:
    def __init__(self, model_path="../models/blip-image-captioning-base"):
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate_caption(self, pil_image, text=None, max_length=50, num_beams=5, temperature=1.0):
        image = pil_image.convert("RGB")
        if text==None:
            inputs = self.processor(images=image, return_tensors="pt")
        else:
            inputs = self.processor(images=image, text=text, return_tensors="pt")
        # print(inputs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature
        )
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption


if __name__ == "__main__":
    # generate_caption()
    captioner = BLIPCaptioner()
    image = Image.open("../dataset/coco128/images/train2017/000000000030.jpg")
    caption = captioner.generate_caption(image, "discribe this figure:")
    print("Caption:", caption)

```

#### BeamSearch 束搜索

可参考[《动手学深度学习》](https://zh-v2.d2l.ai/)的 **9.8** 部分

核心思想：
每一步保留K个最优候选（K = num_beams）
假设：
num_beams = 2

**Step 1**：保留概率最高的2个词:

a (0.6)  
the (0.4)  

**Step 2**：扩展每个候选（计算的是条件概率）：  
模型生成一句话 ( y = (y_1, y_2, ..., y_T) ) 的概率是：
$$
P(y) = P(y_1) \cdot P(y_2 | y_1) \cdot P(y_3 | y_1, y_2) \cdots P(y_T | y_1,...,y_{T-1})
$$
每一步都是**条件概率**
a → a cat (0.3)  
a → a dog (0.2)  
the → the best (0.36)  
the → the man (0.1)  
计算方法，采用log相加的近似法  
$$
\log P(y) = \log P(y_1) + \log P(y_2|y_1) + \cdots
$$
实际排序用的是：
$$
\text{score} = \sum_{t=1}^{T} \log P(y_t | y_{<t})
$$

**Step 3**：从所有候选中选 Top k(例如k=2)：
the best (0.36)
a cat (0.3)
**Step 4**：继续扩展……

**num_beams**  
优点  
比 Greedy 更准确  
能避免明显错误句子  
更稳定，不像采样那么随机

缺点  
计算成本高  
缺乏多样性，容易生成无聊句子  
不一定全局最优，仍是近似搜索，

## BLIP-2

论文参考：[《BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models》](https://proceedings.mlr.press/v202/li23q)

### 架构（三个阶段）

**阶段1：视觉特征提取（冻结的Image Encoder）**

- 输入一张图片，先交给视觉编码器（如ViT）提取出图像特征。
- **注意**：此时输出的是一系列视觉token（比如256个），它们只是“像素级”的原始信息。

**阶段2：视觉-语言特征对齐（可训练的Q-Former）——这是创新的核心**

- Q-Former接收两组输入：
- 来自Image Encoder的图像特征（作为Key和Value）。
  1. 一组可学习的查询向量（Learnable Queries），通常数量很少（比如32个）。
  2. Q-Former通过注意力机制，让这少量查询向量从图像特征中提取最关键的视觉信息，并“翻译”成与文本空间对齐的语义特征。
- 为什么这么做？ 它起到了“信息压缩”和“语义转译”的作用。它将海量的图像特征压缩为少量（如32个）高语义的视觉查询向量，大大降低了后续LLM处理视觉信息的难度。

**阶段3：文本生成（冻结的LLM）**

- Q-Former输出的视觉查询向量，会直接作为前缀（Prefix），输入给一个冻结的大语言模型（比如Flan-T5-XL）。
- LLM基于这些视觉特征和输入的文本提示（Prompt），生成最终的文本描述。
- **关键**：Q-Former在这里相当于一个“视觉到文本的适配器”，把图像信息翻译成LLM能听懂的语言，让LLM发挥其强大的文本生成能力。

### 核心：Q-former

BLIP-2的核心思想是冻结两个强大的预训练模型，只训练一个轻量级的Q-Former作为“桥梁”。

Q-Former 不在每层使用 cross-attention，主要有三个原因：

1. 避免过度依赖视觉信息，保证语义逐层抽象
2. 保留语言模型结构，使输出更适配 LLM
3. 降低计算复杂度，提高训练稳定性

同时，间隔插入 cross-attention 可以形成“信息提取 + 语义融合”的交替过程，从而更高效地完成跨模态对齐。

可以认为是BLIP定义三个模型为一个复合模型太麻烦了，BLIP-2就设置了一个通用的Qformer进行图片和文本对齐的计算，对应不同任务只需要设计不同的掩码，后续复合模型的任务就交给LLM.

Qformer 拆解实验

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 单层 Block（论文结构）
# Query 和 Text 共享 SA
# CA 只作用在 Query 上
class QFormerBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, cross_attn=True):
        super().__init__()

        self.cross_attn_enabled = cross_attn

        # Self-Attn（Query + Text 共享参数）
        # MultiheadAttention输入分别是Q、K、V
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Cross-Attn（只输入 Query）
        # CA要提取图片的重要信息
        if cross_attn:
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
            )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, query_len, image_embeds, attn_mask=None):
        """
        hidden_states: [B, Q+T或Q, D]
        query_len: Q

        Query长度Q，Text长度T
        """

        # 1. Self-Attention
        residual = hidden_states
        attn_out, _ = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=attn_mask
        )
        hidden_states = self.norm1(residual + attn_out)

        # 2. Cross-Attn
        if self.cross_attn_enabled:
            q = hidden_states[:, :query_len, :]
            residual = q

            q_attn, _ = self.cross_attn(q, image_embeds, image_embeds)
            q = self.norm2(residual + q_attn)

            hidden_states = torch.cat([q, hidden_states[:, query_len:, :]], dim=1)

        # 3. FFN
        residual = hidden_states
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.norm3(residual + hidden_states)

        return hidden_states


# 完整版的Q-former
class QFormer(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        hidden_dim=768,
        num_queries=32,
        num_layers=12,
        num_heads=12,
        max_txt_len=64,
        cross_attn_freq=2
    ):
        super().__init__()

        self.num_queries = num_queries

        # ============================
        # Query Tokens（可学习）
        # ============================
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim)
        )

        # ============================
        # Text Embedding（论文用BERT）
        # ============================
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_txt_len, hidden_dim)

        # ============================
        # Transformer Layers
        # ============================
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            use_cross_attn = (i % cross_attn_freq == 0)
            self.layers.append(
                QFormerBlock(hidden_dim, num_heads, use_cross_attn)
            )

        self.norm = nn.LayerNorm(hidden_dim)

    # ======================================
    # 构造 Attention Mask（论文核心）
    # ======================================
    def build_attention_mask(self, query_len, text_len, mode):
        """
        mode:
            "itc" : query <-> query, text <-> text（隔离）
            "itm" : query <-> text（全连接）
            "itg" : text causal（生成）
        """
        total_len = query_len + text_len
        mask = torch.zeros(total_len, total_len)

        if mode == "itc":
            # Query 和 Text 不互相看
            mask[:query_len, query_len:] = float('-inf')
            mask[query_len:, :query_len] = float('-inf')

        elif mode == "itm":
            # 全部互相看
            pass

        elif mode == "itg":
            # 1. Text causal
            causal = torch.triu(
                torch.ones(text_len, text_len) * float('-inf'),
                diagonal=1
            )
            mask[query_len:, query_len:] = causal

            # 2. Text 不能看 Query
            mask[query_len:, :query_len] = float('-inf')

        return mask

    # ======================================
    # Forward
    # ======================================
    def forward(self, image_embeds, input_ids=None, mode="itm"):
        """
        image_embeds: [B, N, D]
        input_ids: [B, T]
        """

        B = image_embeds.size(0)

        # -----------------------------
        # Query
        # -----------------------------
        query_tokens = self.query_tokens.expand(B, -1, -1)
        query_len = query_tokens.size(1)

        # -----------------------------
        # Text
        # -----------------------------
        if input_ids is not None:
            T = input_ids.size(1)

            pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
            text_embeds = self.token_embed(input_ids) + self.pos_embed(pos_ids)

            hidden_states = torch.cat([query_tokens, text_embeds], dim=1)
            attn_mask = self.build_attention_mask(query_len, T, mode).to(input_ids.device)

        else:
            hidden_states = query_tokens
            attn_mask = None
            T = 0

        # -----------------------------
        # Transformer
        # -----------------------------
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                query_len,
                image_embeds,
                attn_mask
            )

        hidden_states = self.norm(hidden_states)

        # -----------------------------
        # 输出拆分
        # -----------------------------
        query_output = hidden_states[:, :query_len, :]

        if input_ids is not None:
            text_output = hidden_states[:, query_len:, :]
            return query_output, text_output

        return query_output


# ======================================
# Dummy Vision Encoder
# ======================================
class DummyVisionEncoder(nn.Module):
    def __init__(self, img_dim=256, hidden_dim=768):
        super().__init__()
        self.linear = nn.Linear(img_dim, hidden_dim)

    def forward(self, x):
        x = self.linear(x)
        return x.unsqueeze(1).repeat(1, 16, 1)


# ======================================
# Demo
# ======================================
def demo():
    B = 2
    img = torch.randn(B, 256)
    text = torch.randint(0, 30522, (B, 10))

    vision = DummyVisionEncoder()
    qformer = QFormer()

    image_embeds = vision(img)

    # ITC
    q, t = qformer(image_embeds, text, mode="itc")
    print("ITC:", q.shape, t.shape)

    # ITM
    q, t = qformer(image_embeds, text, mode="itm")
    print("ITM:", q.shape, t.shape)

    # ITG
    q, t = qformer(image_embeds, text, mode="itg")
    print("ITG:", q.shape, t.shape)


if __name__ == "__main__":
    demo()
```

### BLIP2 caption实验实验代码

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

# 1. 加载模型和预处理器
processor = Blip2Processor.from_pretrained("../models/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "../models/blip2-flan-t5-xl",
    torch_dtype=torch.float16  # 半精度
)

# 2. 加载图片
img_url = "../dataset/coco128/images/train2017/000000000030.jpg"
image = Image.open(img_url).convert("RGB")

# 3. 预处理输入
# BLIP-2 支持 prompt 可以引导生成
# prompt = "Question: What is in the image? Answer:"
prompt = "Question: Describe the image in detail. Answer:"

inputs = processor(images=image, text=prompt, return_tensors="pt")
# inputs = processor(images=image, text="用中文描述这张图片：", return_tensors="pt")

# 4. 将模型和数据移动到 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 5. 生成 caption
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=50,   # 注意这里是 max_new_tokens BLIP-2常用
        num_beams=5,
        temperature=1.0
)

# 6. 解码输出
caption = processor.decode(out[0], skip_special_tokens=True)
print("Caption:", caption)
```

## BLIP和BLIP2对比

对比BLIP和BLIP2的结果令人吃惊  
BLIP的输出：  
two cats sleeping on a couch with a remote control  
BLIP2的输出：  
two cats  
当我用"Question: What is in the image? Answer:"的prompt给BLIP2，让他解释一下图片内容时，本以为会比BLIP2有大模型的加持，回答的会更丰富，但他却给我两个词。。。

当我换一个Prompt后"Question: Describe the image in detail. Answer:"，BLIP2的回答开始丰富了：  
Two cats are sleeping on a couch next to remotes  

对比后的结论：**不是BLIP2比BLIP弱，而是两种模型在做不同任务，所以输出风格完全不同。**  
BLIP的原生功能是**图像描述（Image Captioning）任务**  
专门训练成：看图，并生成**完整、自然、尽量详细的句子**，所以BLIP会**尽可能丰富地讲故事**  
BLIP2本质是在做 **视觉问答（VQA, Visual Question Answering）任务**  
先理解问题，再给出**最直接、最简洁的答案**  
对于第一个问题：What is in the image?，标准答案就是two cats  
所以BLIP2 本质是问答驱动的  
而BLIP：是端到端训练的caption模型，更偏生成描述  
BLIP更像是在看**图写话**，BLIP2更像是在**答题**

## sft_BLIP

```python
import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer, default_data_collator
from peft import LoraConfig, get_peft_model 

# ======================
# 1. 加载模型
# ======================
model_name = "models/blip-image-captioning-base"

processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# ======================
# 2. 冻结视觉编码器
# ======================
for param in model.vision_model.parameters():
    param.requires_grad = False
    
# debug: 打印模型结构，确认视觉编码器部分被冻结，查看文本部分名字用于挂载 LoRA
for name, module in model.named_modules():
    print(name)
    
print("Vision encoder frozen ✅")

# ======================
# 3. 配置 LoRA（只作用在文本部分）
# ======================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "query", "value",  # attention层
    ],
    
    # 这里不能设置task_type，因为 虽然 BLIP 的文本部分是 encoder-decoder 结构，但是他是多模态模型，输入有pixcel_val 和 input_ids,
    # 不能直接当作语言模型训练
    # task_type后peft会自动在forward时构建一个inputs_embeds,和BlipForConditionalGeneration的输入不一致了。
    lora_dropout=0.05,
    bias="none",
    # task_type="SEQ_2_SEQ_LM" 
    
)


model = get_peft_model(model, lora_config)

# debug: 打印可训练参数，确认 LoRA 已正确挂载在文本部分的 attention 层
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
        
model.print_trainable_parameters()

# ======================
# 4. 构造数据集
# ======================

class CocoCaptionDataset(Dataset):
    def __init__(self, annotation_file, image_dir, processor):
        """
        annotation_file: captions_val2017.json
        image_dir: val2017/
        """
        with open(annotation_file, 'r') as f:
            coco = json.load(f)

        self.image_dir = image_dir
        self.processor = processor

        # 构建 image_id -> file_name
        self.id2file = {img["id"]: img["file_name"] for img in coco["images"]}

        # 展平 annotations（每条 caption 作为一个样本）
        self.samples = []
        for ann in coco["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]
            file_name = self.id2file[image_id]

            self.samples.append((file_name, caption))

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, caption = self.samples[idx]

        img_path = os.path.join(self.image_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # 将padding token的 label 设置为 -100，避免计算 loss 时对 padding 部分进行梯度更新
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        
        # debug: 打印输入的 shapes，确认视觉特征和文本输入正确
        # for k,v in inputs.items():
        #     print(k,inputs[k].shape)
        
        return inputs
        

dataset = CocoCaptionDataset("dataset/COCOCaption/annotations/captions_val2017.json", "dataset/COCOCaption/val2017", processor)

# dataloader = DataLoader(dataset, batch_size=2, shuffle=True) 在trainer中会自动处理 dataloader，无需手动创建

# ======================
# 5. 训练
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5) trainer会自动创建优化器，无需手动创建

model.train() 

# 训练参数配置
training_args = TrainingArguments(
    output_dir="week05_blip_blip2_caption/sft_blip_outputs",
    num_train_epochs=3,
    per_device_train_batch_size=10,
    logging_steps=10,
    save_steps=500,
    learning_rate=5e-5,

    fp16=True,  # 或 bf16=True（如果支持）
    gradient_accumulation_steps=1,
)

# Trainer 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.tokenizer, 
    data_collator=default_data_collator,
)

# 开始训练
trainer.train()

# ======================
# 6. 保存 LoRA 权重
# ======================
model.save_pretrained("week05_blip_blip2_caption/sft_blip_lora")
```

# Part6-1 搭建一个minillava模型

## Vision Encoder

vision encoder采用前面学习的clip模型，选用clip-vit-base-patch16做为minillava的图像特征提取。  

在搭建时，有两点需要注意：  

1. 取clip中vision encoder的last_hidden_state输出，并且去掉cls的整体patch特征输出，只保留图像patch的特征输出，outputs中还有一个pooler_output是将cls token的输出进行layernorm之后的结果，代表整张图的全局特征，不适合用在理解图像，反而patch中的特征更符合llm理解。  

2. 给模型增加freeze的功能，确保训练时可以通过配置文件冻结或解冻视觉编码器的模型参数。  

代码如下：

参考 https://hf-mirror.com/docs/transformers/main/en/model_doc/clip

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image

# 视觉编码器
class VisionEncoder(nn.Module):
    def __init__(self, model_path, freeze=True, device="cuda"):
        super().__init__()
        self.vision_model = CLIPVisionModel.from_pretrained(model_path).to(device)
        self.processor = CLIPImageProcessor.from_pretrained(model_path)
        self.device = device
        self.freeze = freeze

        # 冻结vision encoder参数，只训练projector和llm decoder的参数
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)

        if self.freeze:
            with torch.no_grad():
                outputs = self.vision_model(pixel_values=pixel_values)
        else:
            outputs = self.vision_model(pixel_values=pixel_values)

        # 输入形状：(B, 197, 768) 其中197 = 16*16+1（CLS）
        # 输出取clip中vision encoder的last_hidden_state，去掉cls的整体patch特征输出，只保留图像patch的特征输出
        last_hidden_state = outputs.last_hidden_state
        # outputs中还有一个pooler_output是将cls token的输出进行layernorm之后的结果，代表整张图的全局特征，不适合用在图像理解任务
        # 去掉CLS后输出形状：(B, 196, 768)
        patch_features = last_hidden_state[:, 1:, :]
        return patch_features

if __name__ == "__main__":
    vision_encoder = VisionEncoder(model_path="../models/clip-vit-base-patch16", device="cuda")
    images_file = "../dataset/coco128/images/train2017/000000000009.jpg"
    image = Image.open(images_file).convert("RGB")
    features = vision_encoder([image])
    print(features.shape)   # torch.Size([1, 196, 768])

```

### projector

* projector一般为两层全连接层，并增加LayerNorm把不同模态的特征标准化到一个可对齐、可训练、稳定的空间里。  
  * 视觉特征通过projector将特征映射到能和文本embedding拼接的相同维度大小。


```python
class Projector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.init_weights()

    # 权重初始化
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x
```

## llm decoder

* llm decoder中，选用Qwen1.5-1.8B模型。

**这里需要注意**：  

1. 类中要有get_input_embeddings函数来返回输入token的embedding信息。  
因为我们需要将图像信息和文本信息拼接到一起输入到llm中，简单的tokenid和图像信息无法拼接到一起。
所以可以调用模型的  

```python
self.model(inputs_embeds=inputs_embeds,attention_mask=attention_mask,labels=labels)  
```

- 直接将图像和文本输入的embedding拼接好后传给llm模型。  
在拼接时会用到get_input_embeddings来返回文本的embedding信息，用于和图像embedding拼接。  

2. 构建labels时，应将原始label的长度扩展到image token和text token长度的总和，并将image token处的labels设置为-100。文本部分用实际 token id 。  
因果语言模型的训练目标是预测文本部分的下一个token，所以图片部分的 labels 设置为 -100，表示这些位置的损失将被忽略。这样模型在训练时只会关注文本部分的预测，而不会受到图片部分的影响。  
-100的来源：在 HuggingFace / PyTorch 训练中： CrossEntropyLoss(ignore_index=-100)。  
即使vision encoder的参数更新，图片的label依然是-100，因为图片patch的token id本身就是没有意义的，不应该对模型预测文本的能力产生影响。  

3. 构建attention_mask时，直接构建一个长度为image token和text token总和的全1向量即可。代表图片和文本的信息都是有效输入。  
这里的attention_mask区别于label中的使用-100的强制忽略，attention_mask决定llm在做self attention时是否使用该位置的tensor，所以图片和文本信息的mask全为1，代表都要做self attention。  
而label中的-100是在计算交叉熵时，强制忽略对应tensor对loss的影响，最终目的是让图像的token不用来预测下一个token是什么。

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# LLM部分
class LLMDecoder(nn.Module):
    def __init__(self, model_path, freeze=False, device="cuda"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.freeze = freeze
        self.device = device

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self, inputs_embeds, attention_mask, labels=None):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

if __name__ == "__main__":
    llm_decoder = LLMDecoder(model_path="../models/Qwen1.5-1.8B")
    # 模拟图片特征经过project的输出，维度与llm对齐
    projected_input = torch.randn(1, 196, llm_decoder.model.config.hidden_size).to(llm_decoder.device)

    # 输入文本进行tokenizer，获取input_ids，并通过get_input_embeddings获取文本的embedding
    tokenizer = llm_decoder.tokenizer
    text = "What is in the image?"
    text_inputs = tokenizer(text, return_tensors="pt").to(llm_decoder.device)
    text_input_ids = text_inputs.input_ids
    text_attention_mask = text_inputs.attention_mask
    text_embeddings = llm_decoder.get_input_embeddings()(text_input_ids)
    print(text_embeddings.shape)  # 形状： (1, seq_len, model.config.hidden_size)

    # 直接在特征维度上对图片和文本的embedding进行拼接，并将mask延长，适应新的输入长度。
    # mask中图片部分也全是1，表示图片和文本都是有效输入。
    combined_inputs = torch.cat([projected_input, text_embeddings], dim=1)
    combined_attention_mask = torch.cat([torch.ones(projected_input.size(0), projected_input.size(1)).to(llm_decoder.device), text_attention_mask], dim=1)
    print(combined_inputs.shape)  # 形状：(1, 196 + seq_len, model.config.hidden_size)
    print(combined_attention_mask.shape)  # 形状： (1, 196 + seq_len)
```

## minillava

* 将以上三个部分串起来可实现一个最简化版本的minillava模型，使用yaml对象存放一些变量。 

```python
import torch
from vision_encoder import VisionEncoder, Projector
from llm_decoder import LLMDecoder
import yaml

class MiniLlavaModel(torch.nn.Module):
    def __init__(self, config_path):
        super(MiniLlavaModel, self).__init__()
        self.config = self.load_config(config_path)
        self.vision_encoder = VisionEncoder(model_path=self.config['MINILLAVA']['VISION_ENCODER']['MODEL_PATH'], freeze=self.config['MINILLAVA']['VISION_ENCODER']['FREEZE'], device=self.config['DEVICE'])
        self.language_decoder = LLMDecoder(model_path=self.config['MINILLAVA']['LLM_DECODER']['MODEL_PATH'], device=self.config['DEVICE'])
        self.projector = Projector(input_dim=self.config['MINILLAVA']['PROJECTOR']['INPUT_DIM'], hidden_dim=self.config['MINILLAVA']['PROJECTOR']['HIDDEN_DIM'], output_dim=self.language_decoder.model.config.hidden_size).to(self.config['DEVICE'])
        self.device = self.config['DEVICE'].lower()

    def load_config(self,config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _build_multimodal_inputs(self, images, input_ids, attention_mask):
        """构造 LLM 可直接接收的多模态 embedding 和 attention mask。"""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # 1. 图片 -> CLIP patch 特征 -> projector -> LLM hidden_size。
        image_features = self.vision_encoder(images)
        projected_image_features = self.projector(image_features)

        # 2. input_ids -> 文本 embedding。这里复用 LLM 自己的词向量表，保证空间一致。
        text_embeddings = self.language_decoder.get_input_embeddings()(input_ids)

        # 3. 图片 patch 没有 padding，attention mask 全部置 1。
        image_attention_mask = torch.ones(
            projected_image_features.size(0),
            projected_image_features.size(1),
            dtype=attention_mask.dtype,
            device=self.device
        )
        combined_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        # 4. 序列维度拼接：图片 token 放在文本 token 前面，作为视觉上下文。
        combined_inputs = torch.cat([projected_image_features, text_embeddings], dim=1)
        return combined_inputs, combined_attention_mask, projected_image_features.size(1)

    def forward(self, images, texts):
        # 图片特征
        image_features = self.vision_encoder(images)
        projected_image_features = self.projector(image_features)
        
        # 文本特征
        text_inputs = self.language_decoder.tokenizer(texts, return_tensors="pt").to(self.device)
        text_embeddings = self.language_decoder.get_input_embeddings()(text_inputs.input_ids)
        
        # attention mask
        image_attention_mask = torch.ones(projected_image_features.size(0),projected_image_features.size(1)).to(self.device)
        text_attention_mask = text_inputs.attention_mask
        combined_attention_mask = torch.cat([image_attention_mask, text_attention_mask], dim=1)
        
        # 拼接多模态embeding
        combined_inputs = torch.cat([projected_image_features, text_embeddings], dim=1)
        image_labels = torch.full((projected_image_features.size(0), projected_image_features.size(1)), -100, dtype=torch.long).to(self.device)
        combined_labels = torch.cat([image_labels, text_inputs.input_ids], dim=1)
        
        outputs = self.language_decoder(inputs_embeds=combined_inputs, attention_mask=combined_attention_mask, labels=combined_labels)
        
        return outputs

    @torch.no_grad()
    def generate(self, images, prompts, max_new_tokens=128, **generation_kwargs):
        """多模态生成接口，用于训练后的简单验证。"""
        self.eval()
        tokenized = self.language_decoder.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["DATA"]["PREPROCESS"]["MAX_TEXT_LENGTH"]
        )
        inputs_embeds, attention_mask, _ = self._build_multimodal_inputs(
            images=images,
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask
        )
        output_ids = self.language_decoder.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.language_decoder.tokenizer.pad_token_id,
            eos_token_id=self.language_decoder.tokenizer.eos_token_id,
            **generation_kwargs
        )
        return self.language_decoder.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
if __name__ == "__main__":
    model = MiniLlavaModel(config_path="config.yaml")
    from PIL import Image
    image = Image.open("../dataset/coco128/images/train2017/000000000009.jpg").convert("RGB")
    dummy_images = [image,image,image]  # 模拟一个批次三张图片
    dummy_texts = ["What is in the image?"] * 3  # 模拟一批次的三个prompt
    outputs = model(dummy_images, dummy_texts)
    # print(outputs)
    print(outputs.logits.shape) # logits.shape： (batch_size, seq_len, vocab_size) 其中vocab_size是词表大小
    # torch.Size([3, 202, 151936])
    print(outputs.past_key_values[0][0].shape) # past_key_values 是 Transformer attention 的 KV 缓存（Key-Value cache）
    # past_key_values = [
    #     (k1, v1),   # layer 0
    #     (k2, v2),   # layer 1
    #     ...
    #     ]
    # 每个 k/v shape：
    # (batch, num_heads, seq_len, head_dim)
    # torch.Size([3, 16, 202, 128])
    
    print(outputs.loss)
    # tensor(9.4443, device='cuda:0', grad_fn=<NllLossBackward0>)
    
```

config.yaml：

```yaml
# MiniLLaVA 配置文件
# 用于参数化配置模型的所有参数

# 基础配置
DEVICE: "cuda"  # 运行设备: "cuda" 或 "cpu"

# 模型配置
MINILLAVA:
  # 视觉编码器配置
  VISION_ENCODER:
    MODEL_PATH: "../models/clip-vit-base-patch16"  # CLIP ViT 模型路径
    FREEZE: true  # 是否冻结视觉编码器参数
    OUTPUT_DIM: 768  # 视觉特征维度 (CLIP ViT base patch16)

  # 语言解码器配置
  LLM_DECODER:
    MODEL_PATH: "../models/Qwen1.5-1.8B"  # LLM 模型路径
    FREEZE: false  # 是否冻结LLM参数 (通常训练时解冻)
    MAX_NEW_TOKENS: 512  # 生成时最大新token数
    TEMPERATURE: 0.7  # 生成温度
    DO_SAMPLE: true  # 是否采样生成

  # 投影层配置 (将视觉特征映射到语言空间)
  PROJECTOR:
    INPUT_DIM: 768  # 输入维度 (视觉特征)
    HIDDEN_DIM: 2048  # 隐藏层维度

# 数据配置
DATA:
  # 训练数据
  TRAIN_DATASET:
    PATH: "../dataset/LLaVA-CC3M-Pretrain-595K"  # 数据集路径
    IMAGE_DIR: "images"  # 图片目录
    ANNOTATION_FILE: "chat.json"  # 标注文件
    MAX_LENGTH: 512  # 最大序列长度
    BATCH_SIZE: 4  # 批大小
    NUM_WORKERS: 4  # 数据加载器工作进程数

  # 验证数据 (可选)
  VAL_DATASET:
    PATH: "../dataset/coco128"  # 验证数据集路径
    BATCH_SIZE: 4

  # 数据预处理
  PREPROCESS:
    IMAGE_SIZE: 224  # 图片resize大小
    MAX_TEXT_LENGTH: 512  # 最大文本长度

# 训练配置
TRAINING:
  # 优化器配置
  OPTIMIZER:
    TYPE: "adamw"  # 优化器类型: adamw, adam, sgd
    LR: 1e-4  # 学习率
    WEIGHT_DECAY: 0.01  # 权重衰减
    BETAS: [0.9, 0.999]  # Adam betas

  # 学习率调度器
  SCHEDULER:
    TYPE: "cosine"  # 调度器类型: cosine, linear, constant
    WARMUP_STEPS: 100  # 预热步数
    NUM_EPOCHS: 3  # 训练轮数

  # 梯度裁剪
  GRAD_CLIP:
    MAX_NORM: 1.0  # 最大梯度范数

  # 检查点保存
  CHECKPOINT:
    SAVE_DIR: "../outputs/checkpoints"  # 保存目录
    SAVE_STEPS: 500  # 每多少步保存一次
    SAVE_TOTAL_LIMIT: 3  # 最多保存多少个检查点

  # 日志配置
  LOGGING:
    LOG_DIR: "../outputs/logs"  # 日志目录
    LOG_STEPS: 10  # 每多少步记录一次

# 推理配置
INFERENCE:
  # 生成参数
  GENERATION:
    MAX_NEW_TOKENS: 512
    TEMPERATURE: 0.7
    DO_SAMPLE: true
    TOP_P: 0.9
    TOP_K: 50
    REPETITION_PENALTY: 1.1

  # 批处理
  BATCH_SIZE: 1  # 推理批大小

# 其他配置
MISC:
  SEED: 42  # 随机种子
  DEBUG: false  # 调试模式
  RESUME_FROM: null  # 从检查点恢复训练 (路径或null)
```

# part6-2 minillava训练流程
搭建了minillava中的三大组件：vision encoder，projector，llmdecoder之后，持续完善训练pipline。  
这部分包括dataset、trainloop、infer三个部分。最终实现完整的minillava的训练和推理。  

实验中我冻结了clip的模型参数，训练projector和lora的qwen1.5。  
在本没有多模态功能的qwen1.5上，经过3个epoch的微调，每个epoch只取LLaVA-CC3M的一千个样本，用时大概半小时。  
从实验结果看，模型能初步看懂图片和文字描述，开始具备多模态能力。  
但仍然比较垃圾😈。。

**user**: "What is in the picture?"  
**minillava**: "pacific bluebird at a nest in the tree ."  
我问他图片里有什么？他回答说太平洋鸟在树旁边...嗯 至少树看懂了。  
相信如果在LLaVA-CC3M上完整训练3个epoch后模型的理解能力会更强。  

下载数据集：

```bash
hf download liuhaotian/LLaVA-CC3M-Pretrain-595K --repo-type datase --local-dir ./LLaVA-CC3M-Pretrain-595K
```

## dataset

数据集格式以LLaVA-CC3M的chat.json标注文件为例：

```json
[
  {
    "id": "GCC_train_002582585",
    "image": "GCC_train_002582585.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "Provide a brief description of the given image.\n<image>"
      },
      {
        "from": "gpt",
        "value": "olive oil is a healthy ingredient used liberally ."
      }
    ]
  },
]
```

标注中包括每个图片名称和对话细节，其中对话角色分为human和gpt，防止角色错乱。  
value是prompt和llm要预测的句子。  
\<image>是一个特殊字符，用于占位图片特征。  
在dataset中，我们需要解析出图片路径，以及对应的prompt和label。  
同时将图像特征和文字prompt结合起来组成多模态输入。  
组合的方式有很多种，可以直接将特殊字符处的text embedding替换为image embedding。  
也可以直接将特殊字符去掉，直接将图片embedding拼接到text embeding前面。  
这里我们采用第二种较为简单的方式，只要保证训练和推理时构建多模态prompt的方式一致即可。  

```python
import json
import os
from dataclasses import dataclass

from PIL import Image
from torch.utils.data import Dataset

def _read_json(path):
    """读取 chat.json 标注文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _clean_text(text):
    """去掉数据中用于占位图片的 <image> 标记。"""
    return text.replace("<image>", "").strip()

def build_prompt(question):
    """
    构造训练和推理保持一致的文本模板。
    这里用简单清晰的 Q/A 模板，便于理解。真正大规模训练时也可以换成Qwen chat template，但要保证训练和推理使用同一套格式。
    """
    question = _clean_text(question)
    return f"User：{question}\nAssistant"

def extract_qa(conversations):
    # conversations 中提取第一轮 human/gpt 问答。
    question = None
    answer = None
    for message in conversations:
        role = message.get("from")
        value = message.get("value","")
        if role == "human" and question is None:
            question = value
        elif role == "gpt" and answer is None:
            answer = value
        # 已经从对话第一轮问答得到了问答数据
        if question is not None and answer is not None:
            break
    if question is None or answer is None:
        raise ValueError("样本缺少 human/gpt 对话轮次，无法构造监督数据。")
```

* 上面的代码是，从json文件中解析出一张图片的标注（根据图片的对话、提问等）后，通过_clean_text直接去掉prompt中的特殊字符。然后拼接处出我们自己的minillava多模态prompt。  

**包装成dataset类**

```python
class LlavaPretrainDataset(Dataset):
    """
    读取 datasets/LLaVA-CC3M-Pretrain-595K/chat.json 的 Dataset。
    每条样本返回 PIL 图片、prompt 和 answer。
    """

    def __init__(self, dataset_path, image_dir, annotation_file, max_samples=None):
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, image_dir)
        annotation_path = os.path.join(dataset_path, annotation_file)
        self.samples = _read_json(annotation_path)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # for循环，如果index处图片无法打开则使用下一张
        for offset in range(len(self.samples)):
            sample_index = (index + offset) % len(self.samples)
            item = self.samples[sample_index]
            image_path = os.path.join(self.image_dir, item["image"])
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                # print(f"无法打开图片文件 {image_path}，将换一张图片。异常信息：{e}")
                continue

            question, answer = extract_qa(item["conversations"])
            return {
                "image": image,
                "prompt": build_prompt(question),
                "answer": answer,
                "image_path": image_path
            }

        raise RuntimeError("所有样本图片都无法打开，请检查图片目录和标注文件。")
```

到这里还没有结束，为了训练时方便，还需要一个函数将text prompt通过tokenizer处理成token id的形式  

**labels 的关键规则**：

- prompt 部分是用户问题和“助手：”前缀，只作为条件输入，不计算 loss。
- answer 部分是模型需要学习生成的目标，保留真实 token id。
- padding 部分也置为 -100，避免 padding token 参与 loss。

**DataCollator**接收一个样本列表（来自 train_dataset 或 eval_dataset），将其整理成模型可直接输入的批次，并在必要时执行动态填充、标签对齐或数据增强等操作。这里使用了dataclass装饰器省去init等

```python
@dataclass
class LlavaCollator:
    """把原始样本拼成可训练 batch。"""
    tokenizer: object
    max_length: int = 512

    def __post_init__(self):
        if self.tokenizer is None:
            raise ValueError("LlavaCollator 需要传入 tokenizer，不能为 None。")

    def __call__(self, features):
        images = [x["image"] for x in features]
        prompts = [x["prompt"] for x in features]
        answers = [x["answer"] for x in features]

        # eos 可以明确告诉模型回答结束；如果 tokenizer 没有 eos，就退化为空字符串。
        eos = self.tokenizer.eos_token or ""
        full_texts = [prompt + answer + eos for prompt, answer in zip(prompts, answers)]

        tokenized = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = tokenized.input_ids.clone()

        # 逐条计算 prompt token 长度，并把 prompt 位置 label 屏蔽为 -100。
        for row, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True
            ).input_ids
            prompt_len = min(len(prompt_ids), labels.size(1))
            labels[row, :prompt_len] = -100

        # padding 不参与训练损失。
        labels[tokenized.attention_mask == 0] = -100

        return {
            "images": images,
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "labels": labels
        }
```

测试:

```python
from transformers import AutoTokenizer

if __name__ == "__main__":
    dataset = LlavaPretrainDataset(
        dataset_path="../datasets/LLaVA-CC3M-Pretrain-595K",
        image_dir="images",
        annotation_file="chat.json",
        max_samples=1
    )
    sample = dataset[0]
    print(sample.keys())
    # dict_keys(['image', 'prompt', 'answer', 'image_path'])
    print(sample["image"], sample["prompt"], sample["answer"])
    # <PIL.Image.Image image mode=RGB size=224x224 at 0x74FB56FF9F90> 
    # User：Provide a brief description of the given image.
    # Assistant olive oil is a healthy ingredient used liberally .
    tokenizer = AutoTokenizer.from_pretrained("../models/Qwen1.5-1.8B")
    collator = LlavaCollator(tokenizer)
    print(collator([sample]).keys())
    # dict_keys(['images', 'input_ids', 'attention_mask', 'labels'])
```

## train loop

在train loop中，需要准备优化器、学习率调度器、模型保存模块，大多数操作其实是参数的包装：

声明头文件和工具函数：

```python
import argparse
import os
import random

import torch
import yaml
# Accelerator 负责把普通 PyTorch 训练脚本扩展到单卡、多卡、混合精度等场景。
from accelerate import Accelerator
# accelerate_set_seed 会额外处理分布式训练中的随机种子同步。
from accelerate.utils import set_seed as accelerate_set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
# 复用 transformers 提供的常见学习率调度器，避免手写 warmup/cosine 逻辑。
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

# 数据集负责读取图片和问答文本，Collator 负责在 batch 阶段 tokenize 并构造 labels。
from dataset import LlavaCollator, LlavaPretrainDataset
# MiniLlavaModel 封装视觉编码器、投影层和语言模型解码器。
from mini_llava import MiniLlavaModel

def set_seed(seed):
    """固定随机种子，方便复现结果。"""
    # Python 自带 random 模块的随机性。
    random.seed(seed)
    # CPU 上的 PyTorch 随机性。
    torch.manual_seed(seed)
    # 所有 CUDA 设备上的 PyTorch 随机性；没有 GPU 时调用也不会影响训练。
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    """读取训练配置。"""
    # 配置文件使用 YAML，训练参数、数据路径、模型路径都从这里读取。
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def should_save_param(name, param):
    """checkpoint 中保留可训练参数和 LoRA adapter 参数。"""
    name = name.lower()
    return param.requires_grad or "lora_" in name or ".lora_" in name
```

**优化器部分**

- 在 `w = w - lr * g` 中，lr是学习率，g是梯度，优化器的作用就是修改梯度g：

```python
def build_optimizer(model, config):
    """根据配置创建优化器，只更新 requires_grad=True 的参数。"""
    # 训练配置中 OPTIMIZER 字段决定优化器类型、学习率和权重衰减等参数。
    optim_config = config["TRAINING"]["OPTIMIZER"]
    # 过滤掉被冻结的视觉编码器或语言模型参数，只训练当前允许更新的部分。
    params = [p for p in model.parameters() if p.requires_grad]
    optim_type = optim_config["TYPE"].lower()
    # AdamW 是大语言模型微调中最常用的优化器，带 decoupled weight decay。
    if optim_type == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(optim_config["LR"]),
            weight_decay=float(optim_config["WEIGHT_DECAY"]),
            betas=tuple(optim_config["BETAS"])
        )
    # Adam 不使用 AdamW 的解耦权重衰减，适合简单调试或对比实验。
    if optim_type == "adam":
        return torch.optim.Adam(params, lr=float(optim_config["LR"]))
    # SGD 一般不用于 LLM 微调，但保留入口便于实验。
    if optim_type == "sgd":
        return torch.optim.SGD(params, lr=float(optim_config["LR"]), momentum=0.9)
    raise ValueError(f"不支持的优化器类型: {optim_type}")
```

**学习率调度**  

- 学习率调度器的作用就是修改学习率lr：

```python
def build_scheduler(optimizer, config, total_steps):
    """创建学习率调度器。"""
    # 调度器配置决定 warmup 步数、总训练步数下学习率如何变化。
    sched_config = config["TRAINING"]["SCHEDULER"]
    warmup_steps = int(sched_config["WARMUP_STEPS"])
    sched_type = sched_config["TYPE"].lower()
    # cosine：warmup 后按余弦曲线逐渐衰减，常用于 Transformer 训练。
    if sched_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # linear：warmup 后线性衰减到 0，行为更直观。
    if sched_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # constant：不使用 scheduler，训练过程中学习率保持 optimizer 初始值。
    if sched_type == "constant":
        return None
    raise ValueError(f"不支持的调度器类型: {sched_type}")
```

**模型保存**  

- 保存模型时可以选择只保存我们开放训练的参数，来提高保存速度，减少硬盘消耗。  
- accelerator prepare后的model可能被DDP/FSDP等包装，保存前要取回原始模型对象。  

```python
def save_checkpoint(accelerator, model, optimizer, scheduler, step, save_dir):
    """保存训练检查点，包含 projector、可训练参数和优化器状态。"""
    # 多卡训练时每个进程都会执行代码；只让主进程写文件，避免多个进程同时覆盖同一路径。
    if not accelerator.is_main_process:
        return

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"step_{step}.pt")
    # prepare 后的 model 可能被 DDP/FSDP 等包装；保存前要取回原始模型对象。
    unwrapped_model = accelerator.unwrap_model(model)
    # 只保存开启梯度的参数和 LoRA adapter 参数，冻结的基础模型权重由初始化模型提供。
    saved_param_names = [
        name
        for name, param in unwrapped_model.named_parameters()
        if should_save_param(name, param)
    ]
    saved_state_dict = {
        name: param.detach().cpu()
        for name, param in unwrapped_model.named_parameters()
        if should_save_param(name, param)
    }
    # accelerator.save 会在分布式环境中安全保存对象，语义类似 torch.save。
    accelerator.save(
        {
            # 记录当前全局步数，方便后续恢复或排查 checkpoint 来源。
            "step": step,
            # 只保存部分参数；加载时用参数名称匹配，并允许未保存的冻结参数缺失。
            "model": saved_state_dict,
            # 显式记录保存了哪些参数，便于检查 checkpoint 内容。
            "saved_param_names": saved_param_names,
            # 保存优化器状态，恢复训练时可以延续动量等内部统计量。
            "optimizer": optimizer.state_dict(),
            # constant scheduler 为 None，其它 scheduler 保存状态用于恢复学习率进度。
            "scheduler": scheduler.state_dict() if scheduler is not None else None
        },
        ckpt_path
    )
    print(f"已保存检查点: {ckpt_path}")
```

之后加载dataset，包装成dataloder之后就可以循环取数据开始训练了。 

训练时，加载了hf的accelerator库，accelerator是一个很便捷的分布式训练管理包  
accelerator可以接管混合精度训练、梯度裁剪、反向传播、梯度累计等常用功能  
通过accelerator可以轻松的实现大餐数量模型在多卡上的分片训练  

```python
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

使用时只需要用accelerator包装好模型、优化器和数据迭代器即可  

另外scheduler的总步数应该按accelerate分片后的dataloader长度来算，否则多卡时学习率会按单卡步数走得偏慢。  
把模型、优化器、dataloader、scheduler 交给 Accelerator.prepare()，反向传播、梯度裁剪、日志和保存都走accelerate的多进程安全接口。  
每个进程拿自己的dataloader shard，反向传播和梯度裁剪用accelerate接管，checkpoint只由主进程写。

**完整train loop**  

```python
def main():
    # 命令行参数只保留配置路径和调试样本数，其他训练参数统一放在 YAML 中管理。
    parser = argparse.ArgumentParser(description="MiniLLaVA 微调脚本")
    parser.add_argument("--config", default="config.yaml", help="训练配置文件路径")
    parser.add_argument("--max-samples", type=int, default=1000, help="调试时只取前 N 条数据")
    args = parser.parse_args()

    # Accelerator 会根据 accelerate launch 的启动方式自动识别进程数、设备和分布式后端。
    accelerator = Accelerator()
    config = load_config(args.config)
    # 保留原本的 PyTorch 随机种子设置。
    set_seed(int(config["MISC"]["SEED"]))
    # 再使用 Accelerate 的种子工具，保证多进程场景下每个进程的随机状态可控。
    accelerate_set_seed(int(config["MISC"]["SEED"]))

    # 初始化 MiniLLaVA，并切换到训练模式，启用 dropout 等训练期行为。
    model = MiniLlavaModel(args.config)
    model.train()


    # 从配置中读取训练数据路径、图片目录、标注文件、batch size 等数据相关参数。
    train_config = config["DATA"]["TRAIN_DATASET"]
    dataset = LlavaPretrainDataset(
        dataset_path=train_config["PATH"],
        image_dir=train_config["IMAGE_DIR"],
        annotation_file=train_config["ANNOTATION_FILE"],
        max_samples=args.max_samples
    )
    # Collator 在 DataLoader 拼 batch 时进行 tokenizer、padding，并构造只监督 answer 的 labels。
    collator = LlavaCollator(
        tokenizer=model.language_decoder.tokenizer,
        max_length=int(train_config["MAX_LENGTH"])
    )
    # DataLoader 仍按普通 PyTorch 写法创建；后面 accelerator.prepare 会自动处理多卡分片。
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_config["BATCH_SIZE"]),
        shuffle=True,
        num_workers=int(train_config["NUM_WORKERS"]),
        collate_fn=collator,
        pin_memory=torch.cuda.is_available()
    )

    # 先在原始 model 上创建优化器，这样 optimizer 能拿到正确的可训练参数列表。
    optimizer = build_optimizer(model, config)
    num_epochs = int(config["TRAINING"]["SCHEDULER"]["NUM_EPOCHS"])
    # prepare 会把 model 放到正确设备，并在多卡时包装为分布式模型；
    # dataloader 也会被切成每个进程各自负责的一份数据。
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    # prepare 之后的 dataloader 长度是当前进程实际迭代步数，用它计算 scheduler 总步数更适合多卡。
    total_steps = len(dataloader) * num_epochs
    scheduler = build_scheduler(optimizer, config, total_steps)
    # scheduler 依赖已经 prepare 过的 optimizer，因此在 optimizer prepare 之后创建并交给 accelerator。
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    # 日志、保存、梯度裁剪等训练控制参数。
    log_steps = int(config["TRAINING"]["LOGGING"]["LOG_STEPS"])
    save_steps = int(config["TRAINING"]["CHECKPOINT"]["SAVE_STEPS"])
    save_dir = config["TRAINING"]["CHECKPOINT"]["SAVE_DIR"]
    max_norm = float(config["TRAINING"]["GRAD_CLIP"]["MAX_NORM"])

    # global_step 记录当前进程执行的优化步数；多卡下各进程同步前进。
    global_step = 0
    for epoch in range(num_epochs):
        for batch in dataloader:
            # batch 来自 LlavaCollator：
            # images 是 PIL 图片列表，input_ids/attention_mask/labels 是已 padding 的张量。
            outputs = model(
                images=batch["images"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            # MiniLlavaModel 最终调用语言模型，labels 存在时 transformers 输出中会包含 loss。
            loss = outputs.loss
            # 使用 Accelerator 进行反向传播，兼容多卡、混合精度和梯度累积等能力。
            accelerator.backward(loss)

            # 梯度裁剪可以缓解训练初期或小 batch 时的梯度爆炸。
            accelerator.clip_grad_norm_(model.parameters(), max_norm)
            # 参数更新。
            optimizer.step()
            # 如果启用了 scheduler，每个优化步后推进一次学习率。
            if scheduler is not None:
                scheduler.step()
            # set_to_none=True 可以减少显存写入，下一次 backward 时再重新分配梯度。
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            # 聚合所有进程上的 loss，日志展示的是多卡平均 loss，而不是单个进程的局部 loss。
            loss_value = accelerator.gather_for_metrics(loss.detach()).mean().item()
            
            # 只让全局主进程打印日志，避免多进程重复输出相同 step。
            if global_step % log_steps == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                print(f"epoch {epoch + 1}/{num_epochs} step={global_step} loss={loss_value:.4f} lr={lr:.8f}")
                
            # 到达保存间隔时，先等待所有进程到同一步，再由主进程写 checkpoint。
            if global_step % save_steps == 0:
                accelerator.wait_for_everyone()
                save_checkpoint(accelerator, model, optimizer, scheduler, global_step, save_dir)

    # 训练结束后再同步一次，确保所有进程都完成最后一个 epoch。
    accelerator.wait_for_everyone()
    # 保存最终 checkpoint；函数内部会判断是否为主进程。
    save_checkpoint(accelerator, model, optimizer, scheduler, global_step, save_dir)
```

模型训练运行方式示例：
```shell
# 直接运行
accelerate launch --num_processes 2 train.py
# 或者先读取配置
accelerate config
accelerate train.py
```

训练过程如下，可以通过lr看warmup的过程和学习率调度的过程。

```shell
epoch 1/3 step=10 loss=10.4333 lr=0.00001000
epoch 1/3 step=20 loss=7.6921 lr=0.00002000
epoch 1/3 step=30 loss=6.4302 lr=0.00003000
epoch 1/3 step=40 loss=5.7462 lr=0.00004000
epoch 1/3 step=50 loss=4.1945 lr=0.00005000
epoch 1/3 step=60 loss=5.1084 lr=0.00006000
epoch 1/3 step=70 loss=4.2338 lr=0.00007000
epoch 1/3 step=80 loss=4.4134 lr=0.00008000
epoch 1/3 step=90 loss=4.5375 lr=0.00009000
epoch 1/3 step=100 loss=4.1681 lr=0.00010000
epoch 1/3 step=110 loss=3.5306 lr=0.00009994
...
epoch 1/3 step=250 loss=3.1899 lr=0.00008743
epoch 2/3 step=260 loss=4.4558 lr=0.00008578
epoch 2/3 step=270 loss=3.3304 lr=0.00008405
...
epoch 3/3 step=730 loss=3.4098 lr=0.00000023
epoch 3/3 step=740 loss=3.4100 lr=0.00000006
epoch 3/3 step=750 loss=3.7553 lr=0.00000000
已保存检查点: ../outputs/checkpoints/step_750.pt
```

## infer

- 推理部分相对简单，加载训练时保存的模型，加载图片，构建prompt后调用generate进行推理即可。

```python
def main():
    parser = argparse.ArgumentParser(description="MiniLLaVA推理脚本")
    parser.add_argument("--config", default="week08_minillava_training_v1/code/config.yaml")
    parser.add_argument("--checkpoint", default="week08_minillava_training_v1/outputs/checkpoints/step_3000.pt", help="训练得到的 .pt 检查点路径")
    parser.add_argument("--image", default="dataset/coco128/images/train2017/000000000009.jpg", help="输入图片路径")
    parser.add_argument("--question", default="description this picture.", help="关于图片的问题")
    args = parser.parse_args()

    model = MiniLlavaModel(args.config)
    if args.checkpoint is not None:
        # 检查点只保存可训练参数和 LoRA adapter；需用同样 LoRA 配置初始化模型后按名称加载。
        state = torch.load(args.checkpoint, map_location=model.device)
        model.load_state_dict(state["model"], strict=False)

    image = Image.open(args.image).convert("RGB")
    prompt = build_prompt(args.question)
    gen_config = model.config["INFERENCE"]["GENERATION"]
    outputs = model.generate(
        images=[image],
        prompts=[prompt],
        max_new_tokens=int(gen_config["MAX_NEW_TOKENS"]),
        temperature=float(gen_config["TEMPERATURE"]),
        do_sample=bool(gen_config["DO_SAMPLE"]),
        top_p=float(gen_config["TOP_P"]),
        top_k=int(gen_config["TOP_K"]),
        repetition_penalty=float(gen_config["REPETITION_PENALTY"])
    )
    print(outputs[0])
```
