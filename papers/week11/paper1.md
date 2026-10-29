## 论文精读：*Invisible Image Watermarks Are Provably Removable Using Generative AI*

> **会议**：NeurIPS 2024（38th Conference on Neural Information Processing Systems）
> **作者**：Xuandong Zhao\*、Kexun Zhang\*（\*共同一作）、Zihao Su、Saastha Vasan、Ilya Grishchenko、Christopher Kruegel、Giovanni Vigna、Yu-Xiang Wang、Lei Li（UC Berkeley / CMU / UC Santa Barbara / UC San Diego）
> **领域**：图像水印安全 / AIGC 内容溯源 ｜ **关键词**：隐形水印、再生攻击（regeneration attack）、扩散模型、认证移除（certified removal）、高斯差分隐私、语义水印
> **链接**：代码开源 https://github.com/XuandongZhao/WatermarkAttacker

### 一句话总结

该文把水印攻防的"猫鼠游戏"升级为根本性问题——**隐形水印是否必然不鲁棒？**——并对像素级水印给出否定答案：作者提出一族**再生攻击**，先把水印图映射到某个表示 $\phi(x_w)$（恒等 / VAE 潜空间 / 扩散潜空间），加高斯噪声破坏水印（destructive），再用去噪器或生成模型重建图像（constructive）：$\hat{x} = \mathcal{A}(\phi(x_w)+\mathcal{N}(0,\sigma^2 I))$。借助高斯差分隐私（f-DP）的 tradeoff function 语言证明：对任意 $\ell_2$-∆-invisible 水印与**任意**检测算法，攻击后必有 $\epsilon_2 \ge \Phi(\Phi^{-1}(1-\epsilon_1) - L_{x,w}\Delta/\sigma)$——只要 $\sigma$ 相对水印强度足够大，任何检测器在统计上失效；同时重建图与原图的距离不超出去噪器自身误差（质量可保）。在 4 种主流像素级水印（DwtDctSvd/RivaGAN/SSL/StegaStamp）× 8 种攻击 × 2 数据集上，VAE/扩散攻击对前三者移除率 >99%（RivaGAN 98% 移除且 PSNR>30）；StegaStamp 因扰动大需更强噪声；而扰动大、藏进生成过程的语义水印 Tree-Ring 在全部攻击下检测率保持 ≥0.994。结论：不可见性与鲁棒性存在根本权衡，研究与产业应转向语义水印。

### 研究背景与动机

- **监管与产业需求**：DALL-E/Imagen/Stable Diffusion 生成图与真人作品难辨，各国推动 AIGC 标识；Google/Microsoft/OpenAI 承诺为 AI 生成图加水印。隐形水印保画质、非专业人士难去除，成为默认方案——**Stable Diffusion 默认就嵌 DwtDctSvd**。但滥用者同样知晓水印的存在并会刻意移除，鲁棒性是隐形水印的生命线。
- **已有攻击两难**：destructive（亮度/JPEG/高斯噪声——能除水印但画质损失大）vs constructive（高斯模糊/BM3D/学习式去噪——画质好但除不掉鲁棒水印）；学习式水印（RivaGAN/SSL 等）针对已知攻击对抗训练之后，**没见过的攻击怎么办**？本文问的是更根本的：隐形性与"对一切保画质攻击的鲁棒性"之间是否存在根本权衡。
- **讽刺性观察**：生成模型的进步制造了对水印的需求，而生成模型赖以工作的**去噪能力**恰好是拆除水印的最佳工具——与对抗净化（DiffPure）同一机制，换一个立场就是攻击。

### 问题形式化：∆-隐形水印与黑盒威胁模型

- **Definition 2.1（隐形水印）**：$\mathrm{dist}(x, x_w)\le\Delta$，即加嵌扰动幅度有界（本文取 $\ell_2$ 距离）。"隐形"被数学化为**扰动预算 ∆**，这是全文定理的适用边界。
- **Definition 2.2（检测）**：$\mathrm{Detect}:\mathcal{X}\times \mathrm{aux}\to\{0,1\}$；Type I 错误 $\epsilon_1$（假阳率，干净图分布 $P_0$）、Type II 错误 $\epsilon_2$（漏检率，水印图分布 $P_1$）。水印方案的设计目标是 $P_1$ 与 $P_0$ 几乎完美可分（两类错误 ≈0）；**攻击的目标是后处理让两类错误同时升高**。
- **威胁模型（最弱假设的黑盒）**：攻击者只有水印图本身；Watermark/aux/Detect 全部未知、不可查询 Detect，只能利用任意侧信息与算力做后处理——比 DiffWA/自适应攻击等需要模型知识或多次查询的方案假设更弱。
- **k-bit 水印的统计模型**：提取比特与密钥的匹配数在 $H_0$ 下服从 $\mathcal{B}(k,0.5)$，超过阈值 τ 即判水印存在；$p<0.01$ 对应 32-bit 需 ≥23 位正确、96-bit 需 ≥59 位。假阳率有闭式 $\epsilon_1(\tau)=\tfrac{1}{2^k}\sum_{i=\tau+1}^{k}\binom{k}{i}=I_{1/2}(\tau+1,\,k-\tau)$。

### 攻击方法：再生攻击（Regeneration Attack）

统一形式（Eq.1）：$\hat{x} = \underbrace{\mathcal{A}\big(\underbrace{\phi(x_w) + \mathcal{N}(0,\sigma^2 I_d)}_{\text{destructive：加噪毁水印}}\big)}_{\text{constructive：重建}}$——理论保证与 φ/A 的具体选择**解耦**：

1. **实例 1：恒等嵌入 + 去噪重建**：φ=identity，A 取 BM3D/TV 去噪/双边滤波/DnCNN 等任意图像去噪算法；特例 denoising autoencoder 自带加噪，无需额外噪声。
2. **实例 2：VAE 嵌入与重建**：VAE 编码器输出的是高斯分布的 $(\mu(x),\sigma(x))$ 并从中采样——编码阶段自带噪声，直接编码-解码即完成一次再生攻击（denoising autoencoder 是 $\mu(x)$ 为恒等的平凡特例）。
3. **实例 3：扩散嵌入与重建（Algorithm 1）**：$z_0=\phi(x_w)$ → 按 VP-SDE 加噪到时刻 $t^*$（$z_{t^*}=\sqrt{\alpha(t^*)}z_0+\sqrt{1-\alpha(t^*)}\epsilon$）→ 网络 $s(z,t)$ 参数化 score、数值求解器（PNDM）反解 $\hat{z}_0$ → 解码回图像。扩散模型生来就是学去噪的，天然充当 $\mathcal{A}$。

与对抗净化的差别：机制同源（都靠扩散去噪），但净化是防御（去对抗扰动）、此处是攻击（去水印），且本文给出更强的移除保证。

### 理论分析：认证移除（f-CWF）与质量保证

- **Definition 4.1（f-Certified-Watermark-Free）**：若对**任意**检测算法都有 $\epsilon_2 \ge f(\epsilon_1)$（$f$ 非增），则水印被认证移除——攻击后没有算法能区分结果"来自水印图"还是"来自原图"。
- **Definition 4.2（局部水印特定 Lipschitz）**：$\|\phi(x_w)-\phi(x)\| \le L_{x,w}\|x_w - x\|$，衡量 embedding 对该图上水印扰动的压缩比：identity → $L\equiv1$；线性投影 → $0\le L\le1$；神经 embedding 未知但逐对可高效计算。
- **Theorem 4.3（认证移除）**：对 $\ell_2$-∆-invisible 水印，$\mathcal{A}(\phi(\cdot)+\mathcal{N}(0,\sigma^2 I))$ 的输出满足 $f(\epsilon_1)=\Phi\big(\Phi^{-1}(1-\epsilon_1)-\tfrac{L_{x,w}\Delta}{\sigma}\big)$。检测难度等价于**单样本区分 $\mathcal{N}(0,1)$ 与 $\mathcal{N}(L\Delta/\sigma,1)$**；$\sigma \gg L\Delta$ 时 $f\to$ 常数 1（完全不可检）。
- **证明骨架（附录 D）**：$\|\phi(x_0)-\phi(x_w)\|\le L\Delta$ 即 DP 语言中的 ℓ2 局部敏感度 → 平移缩放不变 → 多维各向同性高斯折叠到一维 → 后处理不等式（hockey-stick divergence）→ 再生过程本身也是后处理、不削弱保证。工具箱整体取自 Gaussian DP（Dong-Roth-Su 2022，上承 Wasserman-Zhou 2010 的 tradeoff-function 视角）。
- **Theorem 4.4（质量保证）**：若 $\mathcal{A}$ 能以 $1-\delta$ 概率把干净图的加噪版去噪到 $\xi_{x_0,\sigma,\delta}$ 内，则对水印图同样成立、失败概率增至 $\tilde\delta$；**Corollary 4.5**：$\tilde\delta \le e^{L^2\Delta^2/\sigma^2}\cdot\delta^{1/2}$——$\sigma\asymp L\Delta$ 时指数尾保持。对有理论保证的去噪器（如 TV）直接平移出"MSE 随像素数消失"的保证；对深度去噪器逐图适用：对干净图去噪好 ⇒ 对其水印版也好。
- **两个平凡极端与 t\* 旋钮（Remark C.4/C.5）**：$\alpha(t^*)=0$ 时输出与输入无关（完美无水印但图像信息全失）；$\alpha(t^*)=1$ 时攻击退化为恒等（水印原样保留）。实践取 $L\Delta\sqrt{\alpha(t^*)/(1-\alpha(t^*))}$（即 $L\Delta/\sigma$）为小常数——在彻底移除与保真之间连续调节。
- **φ 的作用（Remark C.7，三例）**：① 像素量化直接抹掉 LSB 水印（$L=0$）；② 低通滤波衰减频域水印（$L\ll1$）；③ 神经 embedding 编码"自然图像流形"，把水印能量从 $n\sigma_w^2$ 压到 $d\sigma_w^2$（$d\ll n$）——信号通过、水印衰减，随后所加噪声即可淹没残余。

### 实验设计与关键数据

- **数据与算力**：MS-COCO 随机 500 张真实图 + SDP 数据集 prompt 经 SD-v2.1 生成的 500 张图；Nvidia A6000。
- **水印设置**：DwtDctSvd（SD 默认）/RivaGAN/SSL/StegaStamp 四种公开像素级水印；检测阈值 $p<0.01$，即 23/32、59/96 位正确。
- **攻击配置**：基线 = 亮度 [2,4,6,8,12]、对比度 [2–7]、JPEG 质量 [10–60]、高斯噪声 σ [5–30]、高斯模糊半径 [2–12]；再生攻击 = CompressAI 压缩模型 Bmshj2018/Cheng2020（compression factor 1–6）+ SD-v2.1 扩散（PNDM，噪声步 [10,30,50,100,150,200] ↔ σ [0.10,0.17,0.23,0.34,0.46,0.57]）。
- **指标**：PSNR/SSIM（画质）、FID（分布保真）、TPR@FPR=0.01（检测鲁棒性，对攻击者越低越好）。

**无攻击时水印性能（Table 1，四种方法 TPR 均为 1.000）**：

| 水印 | MS-COCO PSNR↑ | SSIM↑ | FID↓ | SDP PSNR↑ | SSIM↑ | FID↓ |
|---|---|---|---|---|---|---|
| DwtDctSvd | 39.38 | 0.983 | **5.28** | 37.73 | 0.972 | **9.62** |
| RivaGAN | 40.55 | 0.978 | 10.83 | 40.64 | 0.979 | 13.56 |
| SSL | **41.79** | **0.984** | 18.86 | **41.88** | **0.983** | 23.87 |
| StegaStamp | 28.50 | 0.911 | 35.91 | 28.28 | 0.900 | 41.63 |

关键发现：

1. **移除效果**：VAE/扩散攻击对 DwtDctSvd/RivaGAN/SSL 移除率一致 >99%（质量-可检测性权衡曲线位于右下角：检出率最低 + 画质最高）；RivaGAN 这类抗传统攻击的鲁棒水印被移除 98% 且 PSNR>30。基线要么除不掉（模糊/对比度）要么毁画质（亮度/JPEG）。
2. **StegaStamp 最抗打，且与理论自洽**：其水印图本身 PSNR 仅 28.5、像素/潜空间 ℓ2 距离都是四者最大（图 5）——∆ 大 ⇒ 需要更大 σ ⇒ 只有高噪声扩散有效；逐步加大噪声即可攻破，正是 Remark C.5 的定量直觉。
3. **VAE vs 扩散**：VAE 的 PSNR/SSIM 更高但输出有模糊伪影（PSNR/SSIM 对模糊不敏感），扩散保细节更好——移除率相当，按应用需求选择。
4. **理论-实验互证（图 2/12）**：DwtDctSvd 取经验上界 L=1、σ=1.16∆ 后，经验 tradeoff 曲线劣于理论曲线——界 valid 且保守；"不可能区域"内不存在假阳、漏检双低的检测器。
5. **防御实验（附录 E）**：Tree-Ring 语义水印（水印藏在生成时的初始噪声里）在全部 8 种攻击下检测率保持 0.994–1.000——水印幸存；代价是其 ℓ2 距离远大于像素级水印（图 14），成为"可见"水印。

| 攻击（对 Tree-Ring） | MS-COCO TPR@FPR=0.01 | SDP TPR@FPR=0.01 |
|---|---|---|
| Brightness-2 / Contrast-2 | 1.000 / 1.000 | 1.000 / 1.000 |
| JPEG-50 | 1.000 | 0.994 |
| Gaussian noise-5 | 1.000 | 0.996 |
| Gaussian blur-6 | 1.000 | 1.000 |
| VAE-Bmshj2018-3 / VAE-Cheng2020-3 | 0.998 / 1.000 | 0.994 / 0.994 |
| Diffusion model-60 | 1.000 | 0.998 |

> ⚠️ **精读批注**：① 上表 ↓ 沿用攻击者视角（低 = 水印被除 = 攻击成功），数值 ~1.0 表示 Tree-Ring 幸存，是防御成功而非排版错误；② 附录 C.4 中"$f(\epsilon_1)=1-\epsilon_2$"应为 $f(\epsilon_1)=1-\epsilon_1$ 的笔误（两分布相同时最优检测落在 $\epsilon_1+\epsilon_2=1$ 对角线上）；③ "移除率 98%/99%" 的口径是攻击后 TPR@FPR=0.01 骤降至接近 0，且摘要说 RivaGAN 98%、5.1 节说三种水印均 >99%，应为不同 strength 下的口径，引用时勿当全范围保证；④ Sec.5 说 SSL 用 $k=32$，指标段又写 "23/32, 59/96, and 32/48 bits"——SSL 官方实现是 48-bit 消息、阈值 32，正文内部不一致，复现以 48-bit 为准；⑤ Theorem 4.3 的 $L_{x,w}$ 是逐对（图、水印）定义的，攻击者手里没有干净原图、严格说算不出 $L$，实验用 identity embedding 的上界 $L=1$——"certified"在实操层面依赖这个上界成立，而对神经 embedding（VAE/扩散潜空间）$L$ 甚至可能 ≫1（水印对齐对抗方向时），此时定理不再提供有效保证。

### 研究缺口与未来方向

论文自己给的：研究/产业重心从隐形水印转向语义水印；研究加噪机制的选择（Laplace 噪声、随机量化）；用经典移除攻击构造更低 $L$ 的 embedding φ。结合全文，实际打开的缺口：

1. **语义水印缺理论**：Tree-Ring 只有实验鲁棒性，既没有"语义约束下不可移除"的证明，也没有被证明安全——下一步要么形式化要么攻破它；
2. **对抗对齐水印**：附录 A 指出与对抗扰动对齐的水印可让神经 embedding 的 $L$ 远大于 1、逃出定理适用域（但对不用神经 embedding 的攻击不设防）——这是"在 ∆-invisible 预算内做防御"的少数活路，值得系统研究；
3. **对偶的认证防御**：本文给的是"认证移除"，反问题是能否造出 randomized-smoothing 式"认证鲁棒"的水印检测器（扰动 ≤ r 时检出率有下界）；
4. **超越 ℓ2**：invisible 的定义换到感知距离（LPIPS 等）后，定理与攻击机制如何变化（作者只说思路可推广，未展开）；
5. **实用性代价**：扩散重建是"再想象"，人脸/文字等细节会变——对版权溯源场景"重建图还算不算同一作品"存在法律与评测盲区；高分辨率批量图的算力成本亦未讨论。

### 亮点与洞察

- **问题层级高于方法层级**："隐形水印是否必然不鲁棒"是一个 impossibility 风味的根本问题，答案肯定且带构造性攻击——这类"终结性结论"把猫鼠游戏从逐个攻破升级为划清可能性的边界，是攻防研究的理想范式。
- **理论移植干净漂亮**：把 Gaussian DP 的 tradeoff-function 语言整体搬进水印场景（水印扰动 = 敏感度、加噪+重建 = 高斯机制 + 后处理、f-CWF = f-DP），一个领域的成熟理论直接产出另一个领域的根本性结论，方法论上是"寻找同构问题"的范例。
- **框架的插件化与前瞻性**：φ/A 任意组合、保证与实例解耦——生成模型越强攻击越强。2024 年的结论在 SD3/FLUX 时代只会更成立；生成模型每前进一步，都是同时给攻守双方发武器，而本文证明武器主要在攻方手里。
- **实验与理论闭环**：StegaStamp 的"∆ 大 ⇒ 需 σ 大"与 Remark C.5 定量吻合；理论 tradeoff 曲线与经验曲线同图对照（图 2/12），界的 validity 可视化——经验劣于理论说明保证成立且保守。
- **对隐写主线的镜像价值**：week10 的 NS-DSer 把检测搬进扩散噪声空间，本文把攻击也搬进噪声空间——扩散模型的噪声表示正成为信息隐藏攻防的中枢战场。且 Theorem 4.3 对扰动语义不敏感：它摧毁一切 ℓ2 有界的小扰动，因此同样威胁"微扰式"图像隐写的载荷存活（主动看守者一次再生即可让接收方解不出消息）——只有把信息藏进生成过程本身（噪声空间/语义层）才可能幸免，与 Tree-Ring 的防御思路殊途同归。

### 局限与批判性思考

- **保证范围被 ∆-invisible 严格限定**：结论准确说是"ℓ2 意义下不可见的水印必可移除"，不是"一切水印皆可移除"。防御者把扰动做大或藏进语义即逃出定理（StegaStamp 的部分抗性、Tree-Ring 的完全幸存都源于此）——标题的普适口吻与正文精确声明之间有营销落差。
- **σ 的校准在实操中是盲的**：可证移除需 $\sigma \gtrsim L\Delta$，但攻击者既不知道 ∆ 也算不出 $L$，实验里实际是对噪声步做网格搜索找权衡点；"provably removable"的存在性陈述与"如何选 σ"的算法性陈述之间有缝。
- **实证范围是 post-hoc 像素级水印**：四种被测水印全部是事后贴到图上的方案；生成期水印（Stable Signature 微调解码器、Tree-Ring）只作防御讨论/附录实验，未进主实验——"pixel-level invisible watermarks are vulnerable"的实证支撑集中在 ∆ 最小的那一类。
- **无误差棒/显著性检验**（checklist 自答 No），每配置 500 张图规模有限；VAE 攻击的理论缺口（噪声依赖输入、Theorem 4.3 不直接适用）只给了 smooth sensitivity/PTR 的修补方向而没有落实。
- **攻击工具开源的双刃剑**：作者立场是"揭示漏洞、推动转向语义水印"，但对合规平台水印被批量失效的现实后果与缓解（如水印方案本身的对抗加固）着墨很少；broader impacts 的讨论止于 checklist。

### 评分

- **新颖性**：4.5/5。首个带可证保证的水印移除攻击族，"再生攻击"统一了 destructive/constructive 两类已有攻击；机制与 DiffPure 同源，但立场、目标与保证均新。
- **理论深度**：4.5/5。f-DP 移植严谨、假设诚实（∆-invisible、局部 Lipschitz 均明说）；移除与质量两条定理加两个平凡极端的刻画完整；扣分在 VAE 情形未闭合、$L$ 的可观测性问题被绕开。
- **实验完备度**：4/5。4 水印 × 8 攻击 × 2 数据集 + 强度扫描 + 防御实验 + 理论曲线对照；缺误差棒、缺生成期水印主实验、缺运行时间对比。
- **参考价值**：4.5/5。水印与 AIGC 溯源方向必读：设计者据此自查"∆ 多大、$L$ 多大、σ 多大时会被证伪"，防御端指向语义水印/对抗对齐两条路；代码开源可直接复用。

---

## 补充

### 1. 从差分隐私到认证移除：一张术语对照表

f-DP（Dong-Roth-Su 2022）把"机制在相邻数据集上的隐私"定义为区分两个输出分布 $P,Q$ 的假设检验难度：tradeoff function $T(P,Q)(\alpha)$ = Type I 错误控制在 α 内时**最优**检验（Neyman-Pearson 似然比检验）的 Type II 错误，它是任何检验的 $(\epsilon_1,\epsilon_2)$ 轨迹的下包络。两个性质撑起整个证明：**后处理不等式** $T(h(P),h(Q)) \ge T(P,Q)$（对输出做任何变换只会更难区分）与**高斯机制的闭式解** $\Phi(\Phi^{-1}(1-\alpha)-\mu)$（µ 为噪声信号比）。对照本文：

| 差分隐私 | 本文水印场景 |
|---|---|
| 相邻数据集 $x \sim x'$ | 干净图 $x_0$ vs 水印图 $x_w$ |
| 机制的 ℓ2 敏感度 | 水印扰动经 embedding 后的范数 $\le L_{x,w}\Delta$ |
| 高斯机制 $x + \mathcal{N}(0,\sigma^2 I)$ | 加噪表示 $\phi(x_w)+\mathcal{N}(0,\sigma^2 I)$ |
| 后处理不变性 | 重建 $\mathcal{A}(\cdot)$ 不削弱保证 |
| 高斯机制 tradeoff：$\Phi(\Phi^{-1}(1-\alpha)-\mu)$ | Theorem 4.3：$\mu = L_{x,w}\Delta/\sigma$ |

所以"任何检测算法都失效"不是经验观察而是定理：检测器就是一个假设检验，而它整条 ROC 的下界已被最优检验锁定。直觉上，攻击把"水印图 vs 原图"的差异稀释进一个 σ 尺度的高斯噪声球，检测退化为从单个样本判断它来自 $\mathcal{N}(0,1)$ 还是 $\mathcal{N}(L\Delta/\sigma,1)$——µ 小则物理上不可分。

### 2. 为什么重建能"抹水印、留内容"

水印扰动 $\epsilon = x_w - x_0$ 与图像内容有不同的统计身份：内容在自然图像流形上，ε 是流形外的小扰动。Remark C.7 的三例把 φ 的作用拆开：**像素量化**直接抹掉 LSB 水印（$L=0$）；**低通滤波**衰减频域水印（$L\ll1$）；**神经 embedding** 把信号投影到学到的 $d$ 维自然子空间——若 $\epsilon\sim\mathcal{N}(0,\sigma_w^2 I_n)$，则 $\mathbb{E}\|\phi(\epsilon)\|^2 = d\sigma_w^2 \ll n\sigma_w^2 = \mathbb{E}\|\epsilon\|^2$。VAE/扩散编码器在压缩表示的同时压缩水印能量，随后的高斯噪声只需淹没残余；去噪器重建的是流形上邻近的自然图像——内容语义保留（来自 $x_w$），水印作为"非自然成分"不被重建。这也解释了攻击强度与生成模型质量的正反馈：流形建模越准，信号-水印分离越干净。

### 3. t\* 是"移除彻底性 ↔ 保真"的连续旋钮

Algorithm 1 中 $z_{t^*}=\sqrt{\alpha}z_0+\sqrt{1-\alpha}\,\epsilon$，等效噪声信号比 $\sigma=\sqrt{(1-\alpha)/\alpha}$：α→0 时输出与输入独立（$f(\epsilon_1)=1-\epsilon_1$，完美无水印但图像信息全失）；α→1 时攻击退化为恒等（水印原样保留，好检测器仍可 $\epsilon_1=\epsilon_2=0$）。Remark C.5 的实践准则：取 $t^*$ 使 $L\Delta/\sigma = L\Delta\sqrt{\alpha/(1-\alpha)}$ 为小常数——这个常数正是 Theorem 4.3 的 µ，即**幸存水印的信噪比**。攻击者的全部自由度就是把它压小，而防御者的反制（加大 ∆、抬高 $L$）与攻击者的画质预算之间的拉锯，构成整个攻防的一维刻画。

### 4. Tree-Ring 为什么幸存：水印藏进了生成过程本身

Tree-Ring 不在成品图上"贴"扰动，而是把傅里叶空间的圆环模式刻进采样用的**初始噪声** $z_T$，整张图由它生成——不存在"干净原图 vs 水印图"这种 ∆-invisible 配对，扰动在语义尺度上（图 14 的 ℓ2 距离比像素级水印大一个数量级），直接落在定理适用域之外。检测时用 DDIM 反演回初始噪声查模式，傅里叶结构对卷积/裁剪/翻转/旋转不变，因此再生攻击后模式仍可恢复——实验里 8 种攻击全部 ≥0.994。代价有三：图像与无水印生成结果"可见地"不同；只保护**这个模型**的输出（无法给任意已有图加水印）；鲁棒性暂无理论保证。它印证了本文核心论断：把不可见性预算从 ℓ2 挪到"语义相似"这个宽松得多的约束上，是隐形水印之外的活路——但这条路的安全性目前只有实验背书。

### 5. 复现要点

- **代码**：github.com/XuandongZhao/WatermarkAttacker；
- **攻击端**：SD-v2.1-base（HuggingFace `stabilityai/stable-diffusion-2-1-base`）+ PNDM，噪声步 [10,30,50,100,150,200]；VAE 攻击用 CompressAI 的 Bmshj2018/Cheng2020（compression factor 1–6，值越小退化越重）；
- **水印端**：DwtDctSvd 用 invisible-watermark（SD 默认水印库）、RivaGAN/SSL/StegaStamp 用各自官方实现；比特数 32/32/48/96（SSL 以 48-bit 为准，见批注④）；检出阈值 23/32、59/96 位（p<0.01）；
- **数据**：MS-COCO 随机 500 张；SDP 数据集 prompt + SD-v2.1 生成 500 张；
- **评测**：PSNR/SSIM/FID + TPR@FPR=0.01；画理论 tradeoff 曲线需逐图估计 $L_{x,w}$ 或用上界（identity 时 $L=1$）；
- **算力**：单卡 A6000 可跑。
