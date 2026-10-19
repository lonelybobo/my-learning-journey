# 周报 - 2026.09.18 - 2026.09.24
---
# 一、本周核心工作/学习内容

## 文献阅读

- 【综述】全品类图像隐写，传统 + 深度学习隐写 —— [*Image Steganography Based on Machine Learning Algorithms: A Survey*](./docs/paper1.md)（Journal of Science and Engineering  Applications 2026.4）
- 【综述】生成式隐写，扩散模型为主，含隐写检测 —— [*A review of generative coverless image steganography based on diffusion models*](./docs/paper2.md)（Neurocomputing 2026.5）

## 扩散模型基础内容学习

### DDPM 先导

在阅读 **DDPM（Denoising Diffusion Probabilistic Models）** 论文时，最先看见了 **latent variable models（潜变量模型）**，这里搜集了相关资料。

#### 潜变量模型与边际似然

##### 一、潜变量模型（Latent Variable Models）

**核心思想：观测数据 $x$ 是由某些不可直接观测的隐藏变量 $z$ 生成的。** 我们只能看到最终结果 $x$，看不到背后的"隐藏因素" $z$；通过建模 $z$ 和 $x$ 之间的关系，模型可以学会生成新的数据。

一个典型的潜变量模型包含两部分：

1. **先验分布 $p(z)$**：潜变量 $z$ 本身的分布，通常设为简单的分布（如标准正态分布）。
2. **条件分布 $p(x|z)$**：给定潜变量 $z$ 后，观测数据 $x$ 的生成分布。

模型的目标是最大化观测数据的边际似然：

$$
p(x) = \int p(x|z) p(z) \, dz
$$

常见的潜变量模型例子：

- **变分自编码器（VAE）**：用神经网络参数化 $p(x|z)$ 和近似后验 $q(z|x)$，通过变分下界（ELBO）训练。
- **生成对抗网络（GAN）**：虽然通常不直接称为潜变量模型，但它也从随机噪声 $z$ 生成 $x$，可以看作一种隐式潜变量模型。
- **标准化流（Normalizing Flows）**：通过可逆变换将简单分布映射到复杂数据分布。
- **扩散模型（DDPM）**：也是潜变量模型的一种，只不过它的潜变量是**一系列逐步加噪的中间状态** $x_1, x_2, \dots, x_T$——不是像 VAE 那样只有一个 $z$。

##### 二、边际似然（Marginal Likelihood）

**似然（likelihood）** $p_\theta(x)$：在当前参数 $\theta$ 下，模型"认为"这批数据出现的概率。模型认为真实数据出现概率高 → 参数合理；概率低 → 参数不好。**最大化似然**就是让模型尽可能"解释"观测数据（统计上等价于最大似然估计 MLE）。

**为什么叫"边际"**：数据 $x$ 与参数 $\theta$ 之间隔着潜变量 $z$，完整的联合分布是

$$
p_\theta(x, z) = p_\theta(x|z)\, p_\theta(z)
$$

但我们**只能观测到 $x$，看不到 $z$**（所以也不能直接最大化联合似然），必须把 $z$ 的所有可能性都考虑进去，对 $z$ 积分（或求和）消掉：

$$
p_\theta(x) = \int p_\theta(x, z)\, dz = \int p_\theta(x|z)\, p_\theta(z)\, dz
$$

对 $z$ 积分后剩下的 $p_\theta(x)$，就叫**边际似然（marginal likelihood）**——"边际"来自概率论：对联合分布中的某些变量积分/求和后，剩下的分布就叫边际分布。

##### 三、边际似然不可解怎么办

训练目标是

$$
\max_\theta \; p_\theta(x) = \int p_\theta(x|z)\, p_\theta(z)\, dz
$$

但对神经网络参数化的复杂模型（VAE、扩散模型），这个积分通常**没有解析解**，实际做法：

- **变分推断**：引入近似后验 $q(z|x)$，推导出 ELBO（证据下界），最大化 ELBO 等价于间接最大化边际似然的下界。VAE 就是这样训练的。
- **采样/蒙特卡洛**：用采样估计积分。
- **DDPM 的做法**：把边际似然写成关于整条加噪链的积分，推导出一个简化的去噪损失——它本质是边际似然的变分下界。这条线正是下面「DDPM 理解」部分的解读主线。

##### 四、和 VAE 的关键区别

| 方面 | VAE | DDPM |
|------|-----|------|
| 潜变量维度 | 通常低维 | 与数据同维，且为整条序列 |
| 编码器 | 可学习 | 固定为加噪过程 |
| 解码器 | 可学习 | 可学习（去噪过程） |
| 训练目标 | ELBO | 简化后的去噪得分匹配损失 |
| 生成方式 | 一次解码 | 迭代去噪 |

> **一句话总结**：边际似然就是"把所有可能的潜变量都考虑进去后，观测数据出现的概率"——我们真正想建模的是观测数据 $x$ 的分布，$z$ 只是增强表达能力的工具，必须被积分掉。DDPM 属于潜变量模型这一大类，但潜变量结构不同：它用固定的前向加噪过程定义潜变量（整条马尔可夫链 $x_{1:T}$），用可学习的反向去噪过程生成数据。

##### 五、KL 散度（Kullback-Leibler Divergence）

KL 散度是信息论和统计学里用来衡量**两个概率分布有多不一样**的一个量。在扩散模型里，它几乎是整个损失函数推导的核心工具。

###### 1. 直观理解

假设你有两个分布：

- $p(x)$：真实的分布（比如真实数据的分布）
- $q(x)$：你用来近似它的分布（比如模型学出来的分布）

KL 散度 $D_{\mathrm{KL}}(p \parallel q)$ 衡量的是：**如果用 $q$ 来代替 $p$，你会损失多少信息**。

它不是距离，因为不对称：

$$
D_{\mathrm{KL}}(p \parallel q) \neq D_{\mathrm{KL}}(q \parallel p)
$$

所以它叫"散度"而不是"距离"。

###### 2. 数学定义

**离散分布：**

$$
D_{\mathrm{KL}}(p \parallel q) = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

**连续分布：**

$$
D_{\mathrm{KL}}(p \parallel q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx
$$

也可以写成期望形式：

$$
D_{\mathrm{KL}}(p \parallel q) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]
$$

注意：期望是**对 $p$ 取的**，不是对 $q$。这是理解 KL 散度方向性的关键。

###### 3. 关键性质

1. **非负性**：$D_{\mathrm{KL}}(p \parallel q) \geq 0$，当且仅当 $p = q$ 时取 0。
2. **不对称**：$D_{\mathrm{KL}}(p \parallel q) \neq D_{\mathrm{KL}}(q \parallel p)$。
3. **不满足三角不等式**：所以它不是度量。
4. **对 $q$ 有定义要求**：如果 $q(x)=0$ 但 $p(x)>0$，KL 散度会趋于无穷大。这意味着 $q$ 必须覆盖 $p$ 的支撑集。

###### 4. 两个方向的区别（很重要）

| 方向 | 名称 | 行为 |
|---|---|---|
| $D_{\mathrm{KL}}(p \parallel q)$ | 前向 KL | 倾向于让 $q$ 覆盖 $p$ 的所有模式（mass-covering） |
| $D_{\mathrm{KL}}(q \parallel p)$ | 反向 KL | 倾向于让 $q$ 集中在 $p$ 的某个模式上（mode-seeking） |

在变分推断里，我们通常最小化 $D_{\mathrm{KL}}(q \parallel p)$，其中 $q$ 是变分分布，$p$ 是真实后验。这个方向会导致 $q$ 偏向于拟合 $p$ 的某一个峰。

###### 5. 在扩散模型里的角色

回到 DDPM 第二章，KL 散度出现在公式 (5) 的损失分解里：

$$
L = \mathbb{E}_q\left[D_{\mathrm{KL}}(q(x_T|x_0) \parallel p(x_T)) + \sum_{t>1} D_{\mathrm{KL}}(q(x_{t-1}|x_t,x_0) \parallel p_\theta(x_{t-1}|x_t)) - \log p_\theta(x_0|x_1)\right]
$$

这里每个 $D_{\mathrm{KL}}$ 都在做同一件事：

- **左边的 $q$**：真实的后验分布，是已知的、可计算的高斯分布。
- **右边的 $p_\theta$**：模型学出来的分布，也是高斯分布。

因为两个都是高斯分布，它们的 KL 散度有**闭式解**。这正是 DDPM 能把损失函数写成简单形式的原因。

具体来说，对于两个高斯分布（一维形式；多维时各独立维度逐项求和）：

$$
D_{\mathrm{KL}}(\mathcal{N}(\mu_1,\sigma_1^2) \parallel \mathcal{N}(\mu_2,\sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

在 DDPM 中，方差通常是固定的，所以这个 KL 散度最终简化为**均值的平方误差**，也就是论文里最终用的那个简单损失：

$$
L_{\text{simple}} = \mathbb{E}_{t,\,x_0,\,\epsilon}\left[\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\big)\big\|^2\right]
$$

###### 一句话总结

> **KL 散度衡量两个分布的差异，非负、不对称。在扩散模型里，它把"让模型分布逼近真实后验"这件事，变成了一个可以逐项计算的高斯 KL 散度，最终化简成了预测噪声的均方误差。**

### DDPM 理解

[【较真系列】讲人话-Diffusion Model全解(原理+代码+公式)】](https://www.bilibili.com/video/BV19H4y1G73r/?share_source=copy_web&vd_source=dcedb729c1920a524ddace9deca746fd)

对论文 **第 2 章 Background** 与 **第 3 章 Diffusion models and denoising autoencoders** 的逐公式精读。第 2 章公式 (1)–(7) 的分工：**(1)(2) 定义模型 → (3) 定义训练目标 → (4)–(7) 让目标可计算**；第 3 章把变分下界一步步化简成"预测噪声"的 MSE，并给出完整的训练/采样算法（Algorithm 1、2）。

#### 第 2 章解读：模型如何定义，目标如何可算

##### 全局图景：两条随机过程

- **前向过程（扩散过程）$q$**：把真实图像 $x_0$ 一步步加噪，变成纯噪声 $x_T$。**固定，没有任何可学习参数。**
- **反向过程（生成过程）$p_\theta$**：从纯噪声出发一步步去噪还原图像。**每一步由神经网络预测，是唯一要学的部分。**

扩散模型的来历：最早由 Sohl-Dickstein 等人（2015）提出。DDPM 的关键观察是——**只要每步加的噪声足够小，反向采样链的每步转移也可以设成条件高斯**，于是整个反向过程可以用一个简单的神经网络参数化；它还与 energy based model 的"去噪得分匹配训练 + 退火 Langevin 动力学采样"建立起等价联系（训练 ⇔ 多尺度去噪得分匹配，采样 ⇔ 退火 Langevin 动力学，分别见第 3 章公式 12 与公式 11）。

记号约定：

- $x_0$ 真实图像；$x_1,\dots,x_T$ 与数据**同维度**的潜变量（加噪序列）；$T$ 总步数（论文取 $T=1000$）。
- $\mathcal{N}(x;\mu,\Sigma)$：均值 $\mu$、协方差 $\Sigma$ 的高斯分布；$I$：单位矩阵（每个像素维度加同样方差的噪声），分号前是分布作用的变量，分号后是分布的两个参数。
- `:=` 读作"定义为"：左边是新引入的记号，右边是它的具体内容。

##### 公式 (1)：反向过程（生成模型本体）

$$
p_\theta(x_{0:T}) := p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t), \qquad p_\theta(x_{t-1}|x_t) := \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t))
$$

- 前面的 $p_\theta(x_0) := \int p_\theta(x_{0:T})\, dx_{1:T}$ 就是**边际似然**形式：关心数据 $x_0$ 的概率，潜变量 $x_{1:T}$ 被积分掉。
- 生成 = 从先验 $x_T \sim \mathcal{N}(0, I)$（纯高斯噪声）出发，沿马尔可夫链逐步去噪回到 $x_0$；每步转移是高斯，均值 $\mu_\theta$ 与协方差 $\Sigma_\theta$ 由网络根据当前噪声图 $x_t$ 和时间步 $t$ 预测。
- 马尔可夫性质：$x_{t-1}$ 只依赖 $x_t$，不依赖更早的状态。
- **为什么反向过程需要学习**：真正的反向转移 $q(x_{t-1}|x_t)$ 要对整个数据分布求平均才能得到（不可知），所以只能用带参数的网络 $p_\theta(x_{t-1}|x_t)$ 去近似；而条件在 $x_0$ 之后的 $q(x_{t-1}|x_t, x_0)$ 才有解析形式（公式 6–7）。

##### 公式 (2)：前向过程（加噪过程，固定）

$$
q(x_{1:T}|x_0) := \prod_{t=1}^{T} q(x_t|x_{t-1}), \qquad q(x_t|x_{t-1}) := \mathcal{N}(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I)
$$

- 每步把上一状态缩小 $\sqrt{1-\beta_t}$ 倍，再加方差 $\beta_t$ 的高斯噪声。$\beta_1,\dots,\beta_T$（variance schedule，方差表）是**固定超参数**（论文：从 $10^{-4}$ 线性增到 $0.02$）。
- **重参数化形式**：$q(x_t|x_{t-1})$ 等价于 $x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,z$，$z \sim \mathcal{N}(0,I)$——"加噪"就是缩放旧状态、注入一份可控的噪声，这也是后面逐步展开推导的出发点。
- **$\sqrt{1-\beta_t}$ 缩放因子的作用**：若直接 $x_t = x_{t-1} + \epsilon$，方差会越加越大；缩放后 $\mathrm{Var}(x_t) = (1-\beta_t)\cdot 1 + \beta_t = 1$，方差全程不变——信号按 $\sqrt{1-\beta_t}$ 衰减，噪声按 $\beta_t$ 注入，信噪比平滑下降。
- **与 VAE 的关键区别**：$q$ 就是"编码器"，但完全固定、不可学习。
- 隐含合理性：每步 $\beta_t$ 很小时，加噪一步的"逆操作"也近似是高斯分布，所以反向过程敢用同样形式（高斯）的转移去建模。

##### 公式 (3)：训练目标 = 负的变分下界

$$
\mathbb{E}[-\log p_\theta(x_0)] \;\le\; \mathbb{E}_q\Big[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\Big] = \mathbb{E}_q\Big[-\log p(x_T) - \sum_{t\ge1} \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\Big] =: L
$$

- 左边是真正想最小化的**负对数似然**，但 $p_\theta(x_0)$ 含对 $x_{1:T}$ 的积分——不可解（"边际似然不可解 → 变分推断近似"在 DDPM 里的实例）。
- $\le$ 来自 **Jensen 不等式**（$\log$ 是凹函数）：$\log p_\theta(x_0) = \log\,\mathbb{E}_q\Big[\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\Big] \ge \mathbb{E}_q\Big[\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\Big]$，不等式右边就是 **ELBO**。最小化 $L$（负 ELBO）等价于间接最大化似然。
- 从 ELBO 到 KL 分解（公式 5）的完整推导较长（附录 A），套路是三步：逐项分解乘积 → 贝叶斯重写 → 望远镜求和相消。
- 右边的等式只是展开：分子分母各自取对数（乘积变求和），再逐项相除合并。
- 遗留问题：此时的 $L$ 是"期望套对数比值"，直接用蒙特卡洛估计方差很大 → 需要下面两个关键性质。

##### 公式 (4)：任意时刻的闭式采样（关键性质一）

先定义记号：$\alpha_t := 1-\beta_t$（单步保留的信号比例），$\bar\alpha_t := \prod_{s=1}^{t}\alpha_s$（前 $t$ 步连乘，从 1 单调降到 ≈0）。

$$
q(x_t|x_0) = \mathcal{N}(x_t;\, \sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t) I)
$$

- **意义：不需要迭代 $T$ 次，一步就能从 $x_0$ 直接算出任意时刻 $t$ 的加噪版本 $x_t$。**
- 直观读法：均值 $\sqrt{\bar\alpha_t}\,x_0$ 是"原图残留的信号"，方差 $1-\bar\alpha_t$ 是"累积的噪声"；$t=T$ 时 $x_T \approx \mathcal{N}(0,I)$，与先验吻合。
- 推导思路（先展开两步）：$x_1 = \sqrt{\alpha_1}\,x_0 + \sqrt{1-\alpha_1}\,\epsilon_1$；$x_2 = \sqrt{\alpha_2}\,x_1 + \sqrt{1-\alpha_2}\,\epsilon_2 = \sqrt{\alpha_1\alpha_2}\,x_0 + \sqrt{\alpha_2(1-\alpha_1)}\,\epsilon_1 + \sqrt{1-\alpha_2}\,\epsilon_2$。后两项是两个独立高斯之和，仍是高斯、方差相加：$\alpha_2(1-\alpha_1) + (1-\alpha_2) = 1-\alpha_1\alpha_2$，所以可以合并成 $\sqrt{1-\alpha_1\alpha_2}\,\epsilon$。归纳到第 $t$ 步得 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$——这正是 Algorithm 1 训练步的出处。
- **$\beta_t$ 的直觉作用**：$\sqrt{\bar\alpha_t}$ 与 $\sqrt{1-\bar\alpha_t}$ 分别是"数据"与"噪声"的权重。$\beta_t$ 线性递增时，数据权重单调递减、噪声权重单调递增——$\beta_t$ 本质上是在给数据和噪声在不同时间步分配权重。

##### 公式 (5)：把 $L$ 拆成 KL 散度之和（关键性质二）

$$
L = \underbrace{D_{\mathrm{KL}}(q(x_T|x_0) \,\|\, p(x_T))}_{L_T} + \sum_{t=2}^{T} \underbrace{D_{\mathrm{KL}}(q(x_{t-1}|x_t,x_0) \,\|\, p_\theta(x_{t-1}|x_t))}_{L_{t-1}} \underbrace{- \log p_\theta(x_0|x_1)}_{L_0}
$$

- 拆分技巧（附录 A）：在各求和项的分子分母上同时乘除 $q(x_{t-1}|x_0)/q(x_t|x_0)$，让相邻项 telescoping（连乘相消），最后每项都是两个分布之比——正好认出是 **KL 散度**。
- 三项含义：
  - $L_T$：加噪终点 $x_T$ 偏离先验 $\mathcal{N}(0,I)$ 多少。$\beta_t$ 固定时是常数，训练中忽略。
  - $L_{t-1}$：**真正的训练信号**——让网络学的一步去噪 $p_\theta(x_{t-1}|x_t)$ 去匹配已知"标准答案" $q(x_{t-1}|x_t,x_0)$（公式 6–7）。
  - $L_0$：最后一步 $x_1 \to x_0$ 的重建项（公式 13 的离散解码器）。
- **好处**：高斯之间的 KL 有解析公式，$L$ 可以逐项解析计算（Rao-Blackwellized，方差缩减），不需要高方差的蒙特卡洛估计。

##### 公式 (6)(7)：前向过程后验 = 训练的"标准答案"

$$
q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1};\, \tilde\mu_t(x_t,x_0),\, \tilde\beta_t I)
$$

$$
\tilde\mu_t(x_t,x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\, x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, x_t, \qquad \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t
$$

- **来源**：贝叶斯公式 $q(x_{t-1}|x_t,x_0) \propto q(x_t|x_{t-1})\, q(x_{t-1}|x_0)$——右边两个分布（单步加噪、闭式边际）都是已知高斯，两个密度相乘配方，结果仍是高斯。条件在 $x_0$ 上之后，"往回走一步"从不可算变成**完全可解析**。
- **读 $\tilde\mu_t$**：$x_0$ 与 $x_t$ 的加权组合。$t$ 小（刚开始加噪）时权重几乎全落在 $x_0$ 上（$t=1$ 时均值恰为 $x_0$ 本身）；$t$ 大（接近纯噪声）时权重几乎全落在 $x_t$ 上。直觉：噪声少时图像信息可信，以原图为参照；噪声多时当前状态更可信。
- $\tilde\beta_t$ 是该后验的方差，同样只由 $\beta_t$ 和 $\bar\alpha_t$ 算出，无需学习。

##### 本章逻辑链

- (1)(2) 定义两条马尔可夫链（固定加噪 $q$ + 可学习去噪 $p_\theta$）；
- (3) 用变分下界（负 ELBO）绕开不可解的边际似然积分；
- (4) 闭式采样 → 训练时随机抽 $t$ 一步造出对应噪声程度的样本；
- (5)(6)(7) 把 $L$ 拆成高斯间的 KL；唯一含参数项的标准答案是前向过程后验；
- 通往第 3 章：继续化简，最终得到"预测 $\epsilon$"的简洁 MSE 目标（公式 14）。

#### 第 3 章解读：从变分下界到一行 MSE

本章回答一个实现问题：**变分下界 $L$ 里还留有大量设计自由度（$\beta_t$ 怎么取、反向过程的高斯怎么参数化），该怎么选？** 作者的思路是建立扩散模型与**去噪得分匹配（denoising score matching）**的显式联系，最终把目标化简成极其简单的加权形式。讨论按公式 (5) 的三项展开。

##### 第一步：$\beta_t$ 固定为常数，$L_T$ 直接忽略

$\beta_t$ 理论上可以通过重参数化学习，但直接固定为超参数。于是 $q$ 没有任何可学习参数 → $L_T = D_{\mathrm{KL}}(q(x_T|x_0)\,\|\,p(x_T))$ 对 $\theta$ 是常数，**训练时直接忽略**。

##### 公式 (8)：$L_{t-1}$ 化成"均值匹配"

反向过程第一步设计：方差**不学习**，固定为 $\Sigma_\theta(x_t,t) = \sigma_t^2 I$。两个候选：$\sigma_t^2 = \beta_t$（对 $x_0 \sim \mathcal{N}(0,I)$ 最优）与 $\sigma_t^2 = \tilde\beta_t$（对 $x_0$ 为确定值最优）——分别对应反向过程熵的上、下界，实验中效果相近。

于是 $p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t,t),\, \sigma_t^2 I)$，代入 $L_{t-1}$ 的 KL：

$$
L_{t-1} = \mathbb{E}_q\Big[\frac{1}{2\sigma_t^2}\big\|\tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\big\|^2\Big] + C
$$

- **原因**：两个协方差相同（都是 $\sigma_t^2 I$）的高斯，其 KL 只剩均值之差的二次型，方差相关项全部并入与 $\theta$ 无关的常数 $C$。
- **读法**：最小化 $L_{t-1}$ ⇔ 让网络预测的均值 $\mu_\theta$ 逼近真值 $\tilde\mu_t$（公式 7）。最直接的参数化是让网络直接输出 $\tilde\mu_t$——但下一步会给出更好的选择。

##### 公式 (9)(10)：重参数化，把 $x_0$ 换成 $\epsilon$

利用公式 (4) 的闭式形式 $x_t(x_0,\epsilon) = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$，反解出 $x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\big(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon\big)$，代入公式 (8)：

$$
L_{t-1} - C = \mathbb{E}_{x_0,\epsilon}\Big[\frac{1}{2\sigma_t^2}\Big\|\tilde\mu_t\Big(x_t(x_0,\epsilon),\, \frac{1}{\sqrt{\bar\alpha_t}}\big(x_t(x_0,\epsilon) - \sqrt{1-\bar\alpha_t}\,\epsilon\big)\Big) - \mu_\theta(x_t(x_0,\epsilon), t)\Big\|^2\Big]
$$

再把 $\tilde\mu_t$ 的表达式（公式 7）代入并化简：

$$
= \mathbb{E}_{x_0,\epsilon}\Big[\frac{1}{2\sigma_t^2}\Big\|\frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\Big) - \mu_\theta(x_t, t)\Big\|^2\Big]
$$

- **关键读法**：给定输入 $x_t$，网络需要预测的目标是 $\frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\big)$——一个"$x_t$ 的固定仿射变换**减去**一份缩放后的真噪声 $\epsilon$"。既然 $x_t$ 已经是网络输入，唯一的未知量就是**当初加进去的那个噪声 $\epsilon$** → 干脆让网络直接预测它。

##### 公式 (11)：让网络直接预测噪声 $\epsilon_\theta$

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\Big)
$$

其中 $\epsilon_\theta$ 是从 $x_t$ 预测 $\epsilon$ 的函数逼近器。网络实现（附录 B）：

- **U-Net 主干**：类似 unmasked PixelCNN++（基于 Wide ResNet），全程使用 group normalization；
- **多分辨率特征**：32×32 模型用 4 个特征分辨率（32×32 一直到 4×4），256×256 模型用 6 个；每个分辨率 2 个卷积残差块；
- **self-attention**：加在 16×16 特征图分辨率处；
- **时间步注入**：$t$ 经 Transformer 正弦位置编码进入每个残差块，参数跨时间步共享。

- **采样**：从 $p_\theta(x_{t-1}|x_t)$ 采样就是

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\Big) + \sigma_t z, \qquad z \sim \mathcal{N}(0, I)
$$

这正是 **Algorithm 2** 的核心一行；整个采样过程形似 **Langevin 动力学**，$\epsilon_\theta$ 相当于学到的"数据密度梯度"（得分函数）。

##### 公式 (12)：化简成多尺度去噪得分匹配

把 (11) 代回 (10)，$x_t$ 项相消：

$$
L_{t-1} - C = \mathbb{E}_{x_0,\epsilon}\Big[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\big)\big\|^2\Big]
$$

- **含义**：损失变成"**预测当初加进去的噪声**"的加权 MSE——输入是加噪图像 $\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$，标签就是已知的 $\epsilon$。
- 前面的权重 $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$ 随 $t$ 变化 → 形式上正是**多个噪声尺度上的去噪得分匹配**（NCSN 的做法）。论文指出：优化这种目标 = 用变分推断拟合一条 Langevin 型采样链的有限时间边际。
- **权重的两种具体取值**（对应两种 $\sigma_t$ 候选）：$\sigma_t^2 = \beta_t$ 时权重为 $\frac{\beta_t}{2\alpha_t(1-\bar\alpha_t)}$；$\sigma_t^2 = \tilde\beta_t$ 时权重为 $\frac{\beta_t}{2\alpha_t(1-\bar\alpha_{t-1})}$。两种取法只改变不同 $t$ 项的相对权重，且最终都会在 $L_{\text{simple}}$ 中被去掉（公式 14）。

##### 公式 (13)：数据缩放与离散解码器 $L_0$

- 数据：$\{0,1,\dots,255\}$ 的整数像素线性缩放到 $[-1,1]$，保证网络输入尺度与先验 $\mathcal{N}(0,I)$ 一致。
- 最后一步 $p_\theta(x_0|x_1)$ 用**离散化的高斯解码器**：

$$
p_\theta(x_0|x_1) = \prod_{i=1}^{D} \int_{\delta_-(x_0^i)}^{\delta_+(x_0^i)} \mathcal{N}(x;\, \mu_\theta^i(x_1,1),\, \sigma_1^2)\, dx
$$

其中每个像素维度上，积分区间是以该整数值为中心、宽 $1/255$ 的小格（$\delta_\pm$ 在边界 $\pm 1$ 处延伸到 $\pm\infty$）。

- **含义**：$L_0 = -\log p_\theta(x_0|x_1)$ 度量的是"高斯密度落在该像素取值小格内的概率"，保证变分下界是**离散数据的无损码长**——不用给数据额外加噪，也不用把缩放操作的雅可比计入似然。生成结束时直接输出 $\mu_\theta(x_1,1)$（无噪显示）。

##### 公式 (14)：简化训练目标 $L_{\text{simple}}$

$$
L_{\text{simple}}(\theta) := \mathbb{E}_{t,\,x_0,\,\epsilon}\Big[\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\big)\big\|^2\Big]
$$

- 对公式 (12) **去掉权重** $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$，$t$ 在 $\{1,\dots,T\}$ 上均匀采样：
  - $t=1$ 项对应 $L_0$：把 (13) 的积分近似为"概率密度 × 格宽"（忽略 $\sigma_1^2$ 与边界效应）后，$L_0 = \frac{d}{2}\log 2\pi\beta_1 + \frac{1}{2\alpha_1}\big\|\epsilon - \epsilon_\theta(x_1, 1)\big\|^2$（$d$ 为数据维度）——$L_0$ 同样归结为时间步 1 的噪声预测；
  - $t>1$ 项是 (12) 的无加权版本；
  - $L_T$ 不出现（$\beta_t$ 固定）。
- **为什么敢扔掉权重**：这仍是一个（重新加权的）变分下界，只是改变了不同 $t$ 项的相对重要性。小 $t$ 项训练"去除极小的噪声"——任务简单，且对应人眼不可感知的细节；降权它们，让网络聚焦更大 $t$ 的难任务 → **样本质量更好**。
- **Table 2 消融的三个结论**：① 预测 $\tilde\mu$ 只有用真实变分下界训练时才有效，换成无加权 MSE 会变差；② 学习可变的 $\Sigma_\theta$ 会让训练不稳定、样本质量更差（不如固定方差）；③ 预测 $\epsilon$ 与"变分下界 + 学 $\tilde\mu$"效果相当，而且能用 $L_{\text{simple}}$ 训练 → 最终方案 $\epsilon$ 预测 + $L_{\text{simple}}$ 取得 IS 9.46 / FID 3.17，全面最优。

##### Algorithm 1：训练流程

```text
repeat
  1. x0 ~ q(x0)                     # 抽一张真实图像
  2. t  ~ Uniform({1,...,T})        # 随机抽一个时间步
  3. ε  ~ N(0, I)                   # 抽一份噪声
  4. 对 ‖ε − εθ(√ᾱt·x0 + √(1−ᾱt)·ε, t)‖² 做一步梯度下降
until converged
```

- **重点**：借助公式 (4)，每一步训练**只需要一张图 + 一次网络前向 + 一次 MSE**，完全不需要真的跑 $T$ 步加噪链。
- 训练目标极其简单：给网络看"加了已知噪声的图"，让它猜噪声是多少，猜错就罚 MSE。

##### Algorithm 2：采样流程

```text
1. xT ~ N(0, I)                            # 从纯噪声开始
2. for t = T, ..., 1:
3.     z ~ N(0, I) if t > 1, else z = 0    # 最后一步不再加随机噪声
4.     xt−1 = (1/√αt)·( xt − (βt/√(1−ᾱt))·εθ(xt, t) ) + σt·z
5. return x0
```

- 每次迭代 = **一次网络前向 + 一步高斯采样**；$T=1000$ 次迭代就是 1000 次前向。
- 与训练相反：**训练从不跑链，采样才跑链**。

##### 完整工作流程总结

```text
【训练】（不跑链）
真实图像 x0 ──随机抽 t、抽 ε──(公式4 闭式加噪)──> xt = √ᾱt·x0 + √(1−ᾱt)·ε
xt ──> εθ(xt, t) 预测噪声 ε̂ ──> 损失 = ‖ε − ε̂‖² ──> 反向传播更新 θ

【采样】（跑完整条链）
xT ~ N(0,I) ──> [εθ 预测噪声 → 按公式(11)去掉一部分噪声 + 少量随机 z] × T 次 ──> x0
```

- 两个过程的联系：$\epsilon_\theta$ 在训练中学会"看出图里混了多少噪声"，在采样中就用这个能力一步步把噪声剥掉。
- 整条推导链：公式 (8) 均值匹配 → (9)(10) 重参数化暴露出 $\epsilon$ → (11) 网络改为预测 $\epsilon$ → (12) 加权去噪 MSE → (14) 去掉权重的 $L_{\text{simple}}$。

##### 补充：从实验看模型的直观行为

- **生成过程可视化**（论文 Fig 6/14）：反向过程先出现大尺度结构（轮廓、配色），细节最后才浮现——"先画轮廓，再补细节"。
- **共享中间隐变量**（论文 Fig 7 与附录 D）：从同一个先验样本出发、在不同时刻 $t$ 分叉采样：$t=1000$（先验后立刻分叉）生成的图像完全不同；在更晚的时刻（如 $x_{750}$）分叉，图像虽不同却共享高层属性（论文例子：普遍都戴太阳镜，发色、姿势相似）——说明中间潜变量编码了这些高层语义，尽管它们在人眼看来只是噪声。
- **图像插值**（论文 Fig 8/9）：对两张原图分别加噪得到 $x_t, x_t'$，在潜空间线性插值 $\bar x_t = (1-\lambda)\,x_t + \lambda\,x_t'$ 后解码：$\lambda$ 越大越像右侧原图；插值前加噪的步数 $t$ 越大，信息丢失越多、结果越粗糙越多样，$t=1000$ 时插值等价于生成全新样本。

# 二、问题与解决

- 先学习扩散模型的基本知识。从代码层面了解扩散模型，从训练到推理的全流程，参考 https://github.com/LambdaLabsML/diffusion-from-scratch

# 三、下周计划

- [x] 学习 DDIM 模型，复现 CRoSS 论文的 workflow。

# 四、总结

- 概括 DDPM 的模型范式：
  1. 使用固定的 beta 链和 alpha_bar 链实现 $x_0$ 到任意 $x_t$ 的前向过程。
  2. 训练一个模型，输入  $x_t$ 和 $t$，得到预测噪声 $ε_θ$，这里 $ε_θ$ 与前向过程中使用的标准噪声 $ε$ 计算 loss，输入的 $x_t$ 是利用 1 中的步骤随机采样出来的。
  3. 推理利用模型输出的 $ε_θ$ 计算 $p_\theta(x_{t-1}|x_t) := \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t))$ 中的 $\mu_\theta(x_t, t)$，通过重参数计算得到 $x_{t-1}$。
  4. 重复操作就可以从完全随机的高斯噪声 $x_t$ 逐步去噪 得到 $x_0$。

- 论文的一条重要链路：

```
前向定义为高斯
   ↓
q(x_{t-1}|x_t,x_0) 严格高斯（可精确算）
   ↓
论证 q(x_{t-1}|x_t) 近似高斯
   ↓
据此把 p_θ(x_{t-1}|x_t) 设为高斯
   ↓
目标：算出 p_θ 的均值/方差 → 采样 x_{t-1}
   ↓
均值可用噪声表示 → 只需预测 ε
   ↓
用 ELBO 训练：KL(q(x_{t-1}|x_t,x_0) ‖ p_θ(x_{t-1}|x_t))
   ↓
两个高斯 → KL 闭式解 → 化简为 ‖ε − ε_θ‖²
```