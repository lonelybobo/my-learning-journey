## 论文精读：*Image Watermarks are Removable Using Controllable Regeneration from Clean Noise*

> **会议**：ICLR 2025（Published as a conference paper）
> **作者**：Yepeng Liu、Yuheng Bu（University of Florida）、Yiren Song、Hai Ci、Mike Zheng Shou（Show Lab, NUS）、Yu Zhang（同济大学）、Haofan Wang（InstantX Team）
> **领域**：图像水印攻防 / 可控生成 ｜ **关键词**：水印移除、可控再生（controllable regeneration）、干净高斯噪声、ControlNet、IP-Adapter、高扰动水印、Tree-Ring
> **链接**：代码开源 https://github.com/yepengliu/CtrlRegen

### 一句话总结

本文是 paper1（Zhao et al. 2023a/NeurIPS 2024，文中称 **Regen**）再生攻击的直系续作：针对后者对高扰动水印（StegaStamp/TreeRing）失效的痛点，诊断出根源在于**再生起点是"水印潜变量加少量噪声"——水印信息残留在起点里、被反向过程带回重建图**。CtrlRegen 反其道行之：直接从**干净高斯噪声** $z_T\sim\mathcal{N}(0,I)$ 开始去噪，从源头切断潜空间水印残留；一致性则交给从水印图提取的两路"无水印条件"——IP-Adapter 式**语义控制适配器**（冻结 DINOv2 编码 + 可训练投影 + U-Net 内解耦交叉注意力，文本 prompt 置空）管内容，ControlNet 式**空间控制网络**（Canny 边缘图条件）管布局。另提出 CtrlRegen+：对潜变量加 $t^*$ 步噪声再受控去噪，以噪声步数为旋钮在移除强度与一致性间平滑权衡。7 种水印评测：把 StegaStamp 的 TPR@1%FPR 从 1.00 打到 **0.01**、TreeRing 从 0.99 到 **0.12**（Regen 为 0.99/0.87），同时 CLIP-FID/Q-Align/LIQE 全面优于 Regen——连 paper1 建议的"语义水印"防线也被严重削弱，攻防判据被推到"能否在从纯噪声再合成时幸存"。

### 研究背景与动机

- **水印谱系（按嵌入阶段 × 扰动大小）**：post-hoc（对成品图嵌入：HiDDeN/StegaStamp 的 encoder-decoder、SSL 的优化式、DwtDctSvd 的小波变换）vs in-generation（嵌入生成过程：Tree-Ring/RingID 改初始噪声、StableSignature/WMAdapter 微调 VAE 解码器）；低扰动水印（ℓ2 距离小）易被再生攻击，高扰动水印（图 1：StegaStamp/TreeRing 在像素与潜空间 ℓ2 距离都大一个量级）抗操纵强。
- **Regen 的高扰动盲区**：起点噪声 = 水印潜变量 + 少量噪声 ⇒ 水印结构残留在起点中、去噪时"扩散"回图像；增大加噪步数可逼近纯噪声但画质崩坏（可见失真乃至语义丢失）；Rinse（An et al. 2024，把 Regen 迭代两遍）也不够——StegaStamp 仍有 0.94 TPR@1%FPR。**两难：步数少除不掉，步数多图像毁。**
- **对抗攻击的短板作对照**：PGD 类攻击需要知道水印方案/白盒访问生成器/多次查询检测器 API，且逐图逐水印定制、耗时贵——Regen 系的 no-box 姿态（单张水印图、不碰检测器）才是本文延续的路线。
- **解法的一句话**：把"清除"与"重建"解耦——**起点决定清得多干净**（纯噪声 = 零水印残留），**条件决定重建得多像**（语义 + 空间控制引导去噪），两个自由度从 paper1 的一个 $t^*$ 旋钮里解放出来、独立可调。

### 方法：CtrlRegen——从干净噪声出发的可控重建

统一形式（Eq.1）：$\tilde{x} = \mathcal{G}(z,\,x_w,\,\hat{x}_w)$，其中 $\hat{x}_w = f(x_w)$ 为 Canny 边缘检测图。**CtrlRegen** 取 $z=\epsilon\sim\mathcal{N}(0,I_n)$；**CtrlRegen+** 取 $z_{t^*}=\sqrt{\alpha_{t^*}}z_0+\sqrt{1-\alpha_{t^*}}\epsilon$（$z_0=\mathcal{E}(x_w)$，$\alpha$ 随 $t^*$ 从 1 衰减到 0，$t^*$ 按水印鲁棒性选取）。

**① 语义控制适配器（Sec 3.2.1，IP-Adapter 式）**：

- 思路：借鉴 SD 用文本嵌入 + 交叉注意力保证图文对齐——把水印图压成**只含语义**的图像嵌入，经交叉注意力控制生成；嵌入由鲁棒的 DINOv2 提取，水印的高频扰动被压缩掉（正是 paper1 Remark C.7"神经 embedding 衰减小扰动"的同一原理，这里被攻击者拿来当信息瓶颈用）。
- 结构：冻结 DINOv2-giant 图像编码器 → 可训练投影网络 → U-Net 内新增**解耦图像交叉注意力层**：$\mathrm{Attention}(Q,K',V')=\mathrm{softmax}\big(\tfrac{QK'^T}{\sqrt d}\big)V'$，其中 $Q=W_Q\phi(z^t)$ 与文本注意力共享投影、$K'=W'_K\phi(x)$、$V'=W'_V\phi(x)$；总输出 $Z=\mathrm{Attention}(Q,K,V)+\mathrm{Attention}(Q,K',V')$。训练与推理时**文本 prompt 置空**——只有图像嵌入起作用。
- 训练（Eq.3）：标准噪声预测 MSE $\mathcal{L}_{\theta_p,\theta_a}=\mathbb{E}\|\epsilon-\epsilon_\theta^S(z_t,\phi_{\theta_e,\theta_p}(x),t)\|^2$；冻结 $\theta_e$（DINOv2）与 $\theta_d$（SD），只训投影 $\theta_p$ 与交叉注意力 $\theta_a$。

**② 空间控制网络（Sec 3.2.2，ControlNet 式）**：

- 思路：语义控制只对齐内容（粗粒度），管不住细节与布局；**边缘图提供关键空间信息且天然不含水印**（二值轮廓丢弃低幅值扰动）。
- 结构：ControlNet；U-Net 各块输出 $\zeta(z_t^i,\cdot)=\mathcal{F}_{\theta_u}(z_t^i,\phi,t)+\mathcal{H}_{\theta_C}(\hat{x}_t^i,t)$，Canny 图经编码器 + 卷积注入。
- 训练（Eq.5）：固定语义适配器与 SD，只训 $\theta_C$；数据为 MSCOCO 118k 图像-Canny 对。

**③ 推理（Algorithm 1/2）**：单张水印图、一次前向：采初始噪声（CtrlRegen）或对潜变量加 $t^*$ 步噪声（CtrlRegen+）→ 图像嵌入与边缘图作条件 → 冻结骨干 SD 去噪 $T$ 步 → 解码。**no-box**：不需要水印方案知识、检测规则或检测器访问。

### 实验设计与关键数据

- **数据**：post-hoc 水印用 MIRFLICKR 1000 张真实图；in-generation 用 DiffusionDB 1000 条 prompt 生成。训练：语义适配器用 LAION-2B + COYO-700M 抽样 1000 万图；空间网络用 MSCOCO 118k 对。
- **水印（7 种）**：TreeRing/StableSignature/StegaStamp/HiDDeN/SSL/RivaGAN/DwtDctSvd，覆盖 post-hoc + in-generation、低 + 高扰动；比特数（表 4）DwtDctSvd/RivaGAN/SSL = 32、StableSignature = 48、StegaStamp = 96。
- **基线**：Regen（= paper1 的扩散实例，加噪/去噪各 70 步）与 Rinse（Regen 跑两遍）。
- **指标**：TPR@1%FPR（攻前 B/攻后 A）、BitAcc（多比特水印的正确位率）、CLIP-FID、PSNR（全参考）+ Q-Align/LIQE（无参考质量）。
- **实现**：SD-v1.5 冻结骨干；语义适配器 8×A100（batch 8/GPU）、空间网络 8×A100（batch 4/GPU）；推理单卡 RTX 4090。

**TPR@1%FPR 攻后对比（Table 1 核心列，越低越好）**：

| 水印 | Regen | Rinse | **CtrlRegen** |
|---|---|---|---|
| DwtDctSvd | 0.39 | 0.11 | **0.00** |
| RivaGAN | 0.07 | 0.02 | **0.00** |
| SSL | 0.39 | 0.10 | 0.06 |
| StableSignature | 0.02 | 0.10 | **0.02** |
| StegaStamp | 0.99 | 0.94 | **0.01** |
| TreeRing | 0.87 | 0.61 | **0.12** |

关键发现：

1. **高扰动水印的决定性突破**：StegaStamp/TreeRing 上 Regen/Rinse 基本无效（0.99/0.94、0.87/0.61），CtrlRegen 打到 0.01/0.12——干净起点消除了潜空间水印残留，是唯一变量。
2. **低扰动水印同样最优或持平**：DwtDctSvd/RivaGAN 归零；SSL 残余 0.06（相对鲁棒，需更多步）；StableSignature 本就最脆弱（Regen 已 0.02，附录称 20 步加噪即可移除）。
3. **质量指标全面占优、PSNR 反而更低**：CLIP-FID/Q-Align/LIQE 三项 CtrlRegen 全面最好（如 TreeRing CLIP-FID 1.63 vs Regen 2.84、Q-Align 4.17 vs 4.03）；PSNR 却 ~19 vs Regen ~22–26——作者论证像素距离在生成式管线下失义（详见批注③与补充 3）。
4. **CtrlRegen+ 的权衡曲线支配 Regen**：高扰动水印步数扫 {100,…,1000}、低扰动 {20,…,140}（SSL 到 340）——同等移除率下画质/相似度全面更好；且 Regen 超过 500 步后潜变量已近纯噪声、会生成完全无关图像，受控版本到 1000 步仍保一致。
5. **消融（Table 2）**：语义+空间 vs 仅语义全面大幅提升（DwtDctSvd：CLIP-FID 13.06→8.68、PSNR 15.41→19.13、LIQE 2.55→3.76）——空间控制同时提一致性与质量。
6. **开销（Table 3）**：CtrlRegen 2.56s vs 骨干 SD 1.74s（仅 +0.8s）；CtrlRegen+ 步数 <600 时比骨干还快（200 步仅 0.69s，去噪步数少了）。

> ⚠️ **精读批注**：① 4.1 节声称评测"seven diverse watermarking methods"并列出 HiDDeN，但 Table 1 只有 6 种，HiDDeN 结果全文未出现——口径不一致，引用时勿转述"7 种全测"；② TPR 崩塌的机制要看 BitAcc：攻击后 BitAcc 降到 ~0.46–0.56（≈随机猜 0.5），因检出阈值需 23/32（或 59/96）位才让 TPR 归零；对 0-bit 检测的 TreeRing 无 BitAcc 列（表中"–"）；③ PSNR 论证有道理但要警惕自利偏向：同 PSNR 对比（图 8）只是定性证据，无用户研究；"感知质量高"与"内容忠实"是两个轴——从纯噪声再生等于**重新画了一张同类图**，文字、人脸身份等细节的忠实度未被单独评测；④ 与 paper1 的 Regen 实现不同（SD-v2.1+PNDM 步数扫描 vs 本文 SD-v1.5+70 步），两文 Regen 数字不可直接互换——paper1 里大噪声扩散也能攻破 StegaStamp，只是画质崩；⑤ "no-box" 指对水印方案零知识，**不等于零成本**：攻击器与骨干绑定（SD-v1.5 专用），需 1000 万图 + 8×A100 训练——从 paper1 的"免训练、现成模型即用"退化为"为每代骨干预训练攻击器"，作者也承认一致性受骨干能力约束。

### 与 paper1（Zhao et al., NeurIPS 2024）的对照

Zhao et al. 2023a 即 paper1 的 arXiv 版（arXiv:2306.01953，2023.6 首发，NeurIPS 2024 正式发表）；本文直接将其扩散实例命名为 Regen 作基线。两文是同一问题（再生攻击）的两次迭代，路线取向恰好相反：

| | paper1（可证移除） | paper2（CtrlRegen） |
|---|---|---|
| 再生起点 | 水印潜变量 + $t^*$ 噪声 | 干净高斯噪声（CtrlRegen+ 亦可 + $t^*$） |
| 一致性来源 | 去噪器自身（$t^*$ 小则保真） | 语义嵌入 + Canny 边缘两路条件控制 |
| 理论保证 | f-CWF 认证（ℓ2-∆-invisible 水印） | 无，纯经验 |
| 移除范围 | 低扰动水印（高扰动需大噪声/失败） | 低 + 高扰动通吃 |
| 对 Tree-Ring | 无效（检测率 ~1.0，列为"潜在防御"） | 有效削弱（0.99→0.12） |
| 攻击成本 | 免训练，现成扩散模型 | 每骨干需训练适配器（10M 图、8×A100） |
| 威胁模型 | 黑盒（不可查询检测器） | no-box（单张水印图即可） |

叙事线：paper1 证明了"ℓ2 不可见 ⇒ 必可移除"，并把 Tree-Ring 式语义水印当作出路；paper2 用受控再生把这条出路也打掉一半——两文合并的结论是：**像素级与生成期低层水印都不是安全终点，水印必须活过"语义保持的再合成"**。paper1 淹死水印（噪声淹没 ℓ2 预算内的扰动），paper2 干脆不让水印上船（起点里就没有水印信息）；前者给认证、后者给覆盖。

### 研究缺口与未来方向

论文自己给的：换更强骨干（SD-v3）提升一致性上限、改进控制技术；ethics 声明把方法定位为"未来水印鲁棒性的评测基准"。结合全文，实际打开的缺口：

1. **攻击器与骨干的版本滞后**：每代生成模型都要重训适配器（10M 图 + 8×A100）——这个滞后窗是防御者唯一的时间杠杆；能否做骨干无关的 zero-shot 攻击值得研究；
2. **"幸存于再生"的水印设计**：攻击的条件通道是 DINOv2 嵌入与 Canny 边缘——水印若嵌入这两者携带的表示（如把消息编进语义嵌入本身），再生时会随条件一起被保留；这是 Tree-Ring 之后"语义水印 2.0"的具体化方向，也接续 paper1 附录 A 的对抗对齐思路；
3. **TreeRing 残余 0.12 的机理**：语义条件引导模型生成同类内容，重建图反演出的初始噪声可能与原指纹部分相关——机理解析与彻底归零是 open problem；
4. **内容忠实度的独立评测**：文字/人脸身份/商标等细节在"再想象"下的保真度需要专门基准（用户研究），否则"视觉质量高"可能掩盖内容篡改；
5. **模态与架构外推**：全部基于 SD-v1.5 U-Net/512 系；DiT 架构（SD3/FLUX）、高分辨率、视频水印的适用性未测。

### 亮点与洞察

- **架构级洞察：清除与重建解耦**。paper1 的一体化再生里，起点既载内容又载水印，移除强度与保真焊死在 $t^*$ 上；本文把两者拆到两个通道——起点纯噪声负责"零残留"，条件负责"高一致"，于是 $t^*$ 旋钮可以拧到 Regen 的可行域之外（1000 步）而不毁图。这是对 paper1 框架的结构性改进，而非参数调优。
- **"无水印条件"的选材讲究**：DINOv2（自监督、扰动鲁棒）+ Canny 二值边缘（天然滤掉低幅值扰动）——两个条件通道都经过"信息瓶颈"过滤，语义和布局通过、水印不通过。反向读就是防御说明书：**水印必须藏进攻击条件所携带的表示里才能幸存**。
- **评测思路升级**：把无参考质量指标（Q-Align/LIQE）与 CLIP-FID 引入攻击评测、并系统质疑 PSNR 在生成式管线下的意义——比 paper1 只报 PSNR/SSIM 更贴近"人眼感知"；同时 CLIP-FID 更低 + PSNR 更低的数据组合本身就是"像素距离失义"的好教材。
- **攻击的工程化定型**：no-box、单图单次前向、watermark-agnostic（一套攻击器通吃给定骨干的 7 种水印）——从研究性 PoC 走向可复用的鲁棒性基准工具，ethics 声明的自我定位（benchmark）是清醒的。
- **对周报主线的延续**：week10 的 NS-DSer 把检测搬进噪声空间、paper1 用噪声毁水印，本文更进一步——把整张图换成从噪声重新采样的受控样本：噪声空间从"战场"变成了"重铸炉"。信息隐藏的攻防判据正在从"扰动是否可检测/可抹除"迁移到"载荷能否在再合成中幸存"。

### 局限与批判性思考

- **训练成本与骨干绑定**：语义适配器 1000 万图、8×A100，空间网络 118k 对——攻击基础设施贵且随骨干换代过时；SD-v1.5 已偏老，对最新模型的效果存疑（作者自己把 SD-v3 列为 future work）。
- **一致性上限 = 生成模型能力**：分布外内容（罕见主体、特定人脸、文字排版、精确构图）在从纯噪声再生时会漂移——对这类内容"移除水印"与"保留内容"可能不可兼得；论文未给出失败案例分析。
- **无理论保证**：与 paper1 相反方向的取舍（放弃认证换覆盖）；对需要可证声明的场景（合规、司法），本文只提供经验证据，且 TPR 残余（SSL 0.06、TreeRing 0.12）表明并非彻底归零。
- **质量论证的定性残余**：PSNR 更低被解释为"改动与内容和谐"（图 8），但缺用户研究与内容忠实度指标；Q-Align/LIQE 这类 CLIP/LMM 系指标可能系统性偏好生成模型自己重绘的风格——自评自夸的风险未被讨论。
- **报告完整性小疵**：HiDDeN 声称评测却无结果；StableSignature 行 Rinse（0.10）劣于 Regen（0.02）反常未解释；TreeRing 残余机理未析。
- **双刃剑老问题**：与 paper1 相同——揭示漏洞推动防御，但工具开源 + 攻击器可复训，合规水印体系的现实风险敞口进一步扩大；作者的责任声明（responsible use）仍是姿态多于机制。

### 评分

- **新颖性**：4/5。核心思想（干净起点 + 可控条件再生）把成熟组件（IP-Adapter/ControlNet）组合进攻击场景，单点技术新意有限；但"高扰动水印为何幸存、如何断其生路"的问题定义与清除/重建解耦的架构是真洞察。
- **技术深度**：4/5。两阶段训练管线完整、解耦交叉注意力与 ControlNet 的集成干净、消融（语义/空间分工）到位；理论分析缺失，TreeRing 残余未解释。
- **实验完备度**：4.5/5。7 水印 × 3 攻击 × 2 数据集 + 步数扫描 + 消融 + 推理时间 + 无参考质量指标；扣分：HiDDeN 缺席、单一骨干/分辨率、内容忠实度未独立评测。
- **参考价值**：4.5/5。与 paper1 对读是水印攻防两条路线（认证 vs 覆盖）的最佳教材；作为鲁棒性基准攻击可复用；给水印设计者的启示具体（藏进条件表示/语义嵌入）。

---

## 补充

### 1. 为什么"从干净噪声出发"是决定性的：起点的信息论

Regen 的起点 $z_{t^*}=\sqrt{\alpha}\,z_0+\sqrt{1-\alpha}\,\epsilon$ 中，水印信息的存活比例由 $\sqrt{\alpha}$ 决定——paper1 的 $\mu=L\Delta/\sigma$ 说的正是同一件事：$t^*$ 小则水印活着，$t^*$ 大则内容也死了。病根在于 **起点与内容是同一个向量**，两者此消彼长。CtrlRegen 把两者拆到两个通道：起点 $z_T$ 纯噪声（零水印信息，残留问题的定义性消解），内容走条件通道（DINOv2 嵌入是语义的低维压缩、Canny 边缘只保留强度剧变的轮廓——两者都是"水印进不去"的瓶颈）。水印要幸存，必须在两个通道都有载体。TreeRing 残余 0.12 的合理解释也在这里：它的指纹会轻微改变生成内容的分布，而语义条件恰好引导模型生成"同类内容"，重建图反演出的初始噪声与原指纹部分相关——条件通道无意中给语义级水印留了窄门。

### 2. 两级控制的分工：语义管内容、边缘管布局

语义控制是 IP-Adapter 的解耦交叉注意力：$Q$ 与文本注意力共享（生成过程不变），$K',V'$ 独立学图像嵌入——空 prompt 下图像嵌入独挑大梁，SD 原有能力零损伤。选 DINOv2 而非 CLIP 是因为其自监督特征对小扰动更鲁棒：嵌入通道稳定，重建的语义就稳定。空间控制是 ControlNet 结构：边缘特征经零卷积逐块注入 U-Net 解码器；Canny 二值图天然丢弃低幅值水印扰动，只留轮廓。消融显示两者叠加不仅提一致性还提质量（Table 2）——语义条件给"画什么"，边缘条件给"画在哪"，SD 只需负责"画得自然"，各司其职。

### 3. PSNR 之辩：像素距离在生成式管线下的失义

PSNR 衡量逐像素保真，前提是"两张图是同一图像的两个处理版本"；而从纯噪声再生等于"按语义重新画一张"，像素级对齐本来就不再成立——如同同一场景的两张不同快照。因此本文 CLIP-FID（分布相似）、Q-Align/LIQE（单图感知质量，无需参考图）更低/更高而 PSNR 更低的组合是自洽的。但要分清两个轴：**感知质量**（看起来自然）≠ **内容忠实**（细节没变）——文字标牌、人脸身份、指纹纹理在再生中的改动，PSNR 会扣分、感知指标却毫不在意。图 8 的同 PSNR 对比只证明了"改动更和谐"，没有证明"内容更忠实"。这一点对 paper1 同样适用（其扩散重建同样改细节），两文的 utility 评测其实都停在"看起来像"，"是不是同一作品"没有回答——对版权场景这是根本缺口。

### 4. CtrlRegen+ 与 paper1 Algorithm 1：同一旋钮的两种拧法

CtrlRegen+ 的管线与 paper1 的扩散实例几乎逐行对应（encode → 加 $t^*$ 步噪声 → 去噪 → decode），差别只在两处：去噪时挂了语义+空间两个控制分支，prompt 置空。于是同一个 $t^*$ 旋钮的意义变了：paper1 里拧大 $t^*$ 是"移除换保真"的零和交易；CtrlRegen+ 里保真端有控制器兜底，$t^*$ 可以拧进 Regen 的禁区（>500 步、起点已近纯噪声、Regen 会生成无关图，受控版本到 1000 步仍与原图一致）。一个可爱的副产品（Table 3）：$t^*<600$ 时 CtrlRegen+ 比骨干 SD 的完整采样还快（200 步 0.69s vs 1.74s）——因为去噪步数本来就少了。

### 5. 复现要点

- **代码**：github.com/yepengliu/CtrlRegen；
- **骨干与环境**：SD-v1.5（冻结）+ DINOv2-giant（冻结）；训练 8×A100（语义 batch 8/GPU、空间 batch 4/GPU），推理单卡 RTX 4090（CtrlRegen 2.56s/图）；
- **训练数据**：语义适配器 = LAION-2B + COYO-700M 抽样 1000 万图；空间网络 = MSCOCO 118k 图像-Canny 对；两阶段：先训语义适配器（Eq.3，只训投影+交叉注意力），冻结后再训空间网络（Eq.5，只训 $\theta_C$）；
- **推理配置**：CtrlRegen 从 $\mathcal{N}(0,I)$ 采样；CtrlRegen+ 高扰动水印步数 {100,200,300,400,500,1000}、低扰动 {20,…,140}（SSL 到 340）；文本 prompt 置空；条件 = DINOv2 嵌入 + Canny 边缘图；
- **水印端**：DwtDctSvd/RivaGAN/SSL 32 bit、StableSignature 48 bit、StegaStamp 96 bit（Table 4）；post-hoc 用 MIRFLICKR 1000 张、in-generation 用 DiffusionDB 1000 条 prompt；
- **评测口径**：TPR@1%FPR 与 BitAcc 均报攻前（B）/攻后（A）；质量 = CLIP-FID + PSNR + Q-Align + LIQE；Regen 基线为 SD-v1.5、70 步，Rinse 为 Regen×2。
