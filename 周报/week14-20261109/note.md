# 周报 - 2026.11.09 - 2026.11.15
---
# 一、本周核心工作/学习内容

## 基础知识

### 1. 基于SDE的分数生成模型

_参考[基于SDE的分数生成模型原理解析](https://www.bilibili.com/video/BV1ZT421v7bx/?share_source=copy_web&vd_source=dcedb729c1920a524ddace9deca746fd)_

#### 前置知识

**为什么学 SDE 扩散模型要先谈 ODE？** 因为同一个扩散过程存在两种等价描述：既能写成**随机微分方程**——每个样本沿一条随机轨迹演化（对应 DDPM 类随机采样器）；也能写成一条不含噪声的**常微分方程**（概率流 ODE）——每个样本走一条确定轨迹，但任意时刻的边缘分布与 SDE 完全相同（对应 DDIM 类确定性采样器）。理解 ODE 不只是作对比，它带来三项 SDE 视角给不了的能力：① 用高阶数值求解器加速采样；② 通过瞬时变量变换精确计算似然；③ 图像 ↔ 隐空间的确定性互转（可插值、可编辑）。此外后续的 flow matching、consistency model 等现代框架都建立在 ODE 视角上，不先掌握 ODE 就无法理解这条演化路线。

##### 1.1 常微分方程（ODE）

ODE 描述一个量随时间的**确定性**演化：状态的变化率由当前状态完全决定，不含随机项。标准形式为

\[
\frac{dx}{dt} = f(x, t), \qquad x(0) = x_0
\]

- $x(t) \in \mathbb{R}^d$：状态（在生成模型里就是图像数据）；
- $f(x, t)$：**向量场**，指出时刻 $t$、位于 $x$ 处的"速度"方向。

给定初值 $x_0$，解是一条确定的轨迹 $x(t)$。以最简单的线性 ODE 为例：

\[
\frac{dx}{dt} = -\theta x \;\Rightarrow\; x(t) = x_0 e^{-\theta t}
\]

所有初值沿各自轨迹被指数拉向原点——这种"全体向一点收缩"的直觉，正是扩散模型概率流 ODE 的原型。

实际中解析解一般不存在，需要数值积分。最简单的**欧拉法**把时间离散成步长 $\Delta t$：

\[
x_{t+\Delta t} = x_t + f(x_t, t)\,\Delta t
\]

扩散模型的采样器（DDIM、概率流 ODE 求解器等）本质上就是在数值积分一条 ODE。

##### 1.2 随机微分方程（SDE）

SDE 在 ODE 的基础上加入随机项，刻画"确定趋势 + 随机扰动"的演化。定义它之前先要定义噪声来源——**维纳过程（布朗运动）** $w(t)$，满足三条性质：

1. $w(0) = 0$；
2. **增量独立**：不相交时间段上的增量互相独立；
3. **增量服从高斯**：$w(t) - w(s) \sim \mathcal{N}(0,\, t-s)$，即扰动大小与时间间隔的平方根 $\sqrt{t-s}$ 成正比。

于是 SDE 的一般形式为

\[
dx = f(x, t)\,dt + g(x, t)\,dw
\]

- $f(x, t)$：**漂移系数**——确定性部分，就是 ODE 里的"速度场"，决定分布中心的移动趋势；
- $g(x, t)$：**扩散系数**——噪声强度，决定分布在演化中被"摊开"的程度；
- $dw = w(t+dt) - w(t) \sim \mathcal{N}(0, dt)$：一小步噪声，方差随时间间隔线性增长，标准差按 $\sqrt{dt}$ 缩放。

把它离散化（**Euler–Maruyama 方法**）最能看清 SDE 到底在做什么：

\[
x_{t+\Delta t} = x_t + f(x_t, t)\,\Delta t + g(x_t, t)\sqrt{\Delta t}\,z, \qquad z \sim \mathcal{N}(0, I)
\]

对比欧拉法可见：SDE 的数值模拟 = 每一步"沿向量场走一小步，再叠加一个方差正比于步长的高斯噪声"。**DDPM 的前向加噪正是这样一个离散 SDE**——每步都是"均值漂移 + 注入高斯噪声"。

一个经典例子是 **Ornstein–Uhlenbeck 过程**（均值回归 + 恒定噪声）：

\[
dx = -\theta x\,dt + \sigma\,dw
\]

无论初值在哪，$x$ 都被指数拉向 0，同时不断被噪声扰动；时间足够长后收敛到与初值无关的平稳分布 $\mathcal{N}\!\left(0, \dfrac{\sigma^2}{2\theta}\right)$。"任意分布 → 固定高斯分布"正是扩散模型前向过程需要的性质，VP-SDE 取的就是 $\theta = \tfrac{1}{2}\beta(t)$、$\sigma = \sqrt{\beta(t)}$ 的形式。

#### 2. 连续化

##### 2.1 DDPM 从离散加噪到连续 SDE（VP SDE）

DDPM 的前向过程为 \(x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon\)。在连续视角下，令时间 \(t \in [0,1]\)，并引入连续函数 \(\beta(t)\) 替代离散序列 \(\beta_t\)。DDPM 的单步更新可以重写为：

\[
x_{t+\Delta t} - x_t = -\frac12 \beta(t)x_t \Delta t + \sqrt{\beta(t)\Delta t}\,\epsilon
\]

当 \(\Delta t \to 0\) 时，上式依分布收敛于如下 VP SDE：

\[
dx = -\frac12 \beta(t)x\,dt + \sqrt{\beta(t)}\,dw
\]

其中 \(w\) 是标准维纳过程。直观上，离散步骤中的衰减项变成了漂移项，而每次加入的独立高斯噪声在连续极限下累积成了布朗运动。

##### 2.2 NCSN 的加噪过程连续化（VE SDE）

NCSN 的离散加噪过程是逐步增加噪声方差。其单步更新可写作 \(x_i = x_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2}\, z_{i-1}\)，其中 \(\sigma_i\) 是递增的噪声标准差序列。

在连续极限下，令 \(\sigma(t)\) 为连续递增函数，上述离散链收敛于如下 SDE：

\[
dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\, dw
\]

这个 SDE 的漂移项 \(f(x,t)=0\)，扩散系数 \(g(t)=\sqrt{d[\sigma^2(t)]/dt}\)。由于没有漂移项来抵消噪声注入，过程的方差会持续增长并趋于无穷，因此得名“方差爆炸”。

##### 2.3 sub-VP SDE
在保持与 VP SDE 相同训练便利性和先验可达性的同时，让前向过程的方差始终被 VP SDE 的方差所上界约束，sub-VP SDE是为生成建模提供了一个更灵活且理论上更优的折中选择。

在 VE SDE（对应 NCSN）和 VP SDE（对应 DDPM）之外，作者引入了 sub-VP SDE。其形式为：

\[
dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)\left(1-e^{-2\int_0^t \beta(s)ds}\right)}\,dw
\]

它与 VP SDE 共享相同的漂移项 \(-\frac12\beta(t)x\) 和噪声调度 \(\beta(t)\)，区别仅在扩散系数的设计。

#### 3. 扰动核训练模型

有了连续化的 SDE 表示后，训练网络的核心思路是：**不去直接拟合未知的分数函数，而是拟合一个已知的、由扰动核给出的条件分数**。扰动核就是前向过程从数据 \(x_0\) 到噪声状态 \(x_t\) 的条件分布 \(p_t(x_t|x_0)\)。三个 SDE 的扰动核都能从各自的线性 SDE 解析求解得到，形式都是高斯分布，因此条件分数可以直接写出，训练就变成了一个可计算的回归问题。

##### 3.1 扰动核是什么

扰动核 \(p_t(x_t|x_0)\) 描述了给定一个干净数据点 \(x_0\) 时，\(t\) 时刻的噪声状态 \(x_t\) 服从什么分布。对于 VE、VP、sub-VP 这三种 SDE，由于漂移项都是线性的（VE 的漂移为 0，VP/sub-VP 为 \(-\frac12\beta(t)x\)），前向 SDE 可以解析求解，得到的扰动核都是高斯分布：

\[
p_t(x_t|x_0) = \mathcal{N}(x_t; \mu_t(x_0), \Sigma_t)
\]

有了这个高斯形式，条件分数 \(\nabla_{x_t}\log p_t(x_t|x_0)\) 就可以直接算出来，它只依赖于 \(x_t\)、\(x_0\) 和已知的系数。

##### 3.2 三个扰动核怎么推出来

推导的共同工具是**线性 SDE 的解析解**。对于一般形式 \(dx = a(t)x\,dt + b(t)dw\)，从 0 积分到 \(t\) 得到：

\[
x_t = \underbrace{e^{\int_0^t a(s)ds}}_{\text{缩放因子}} x_0 + \int_0^t e^{\int_s^t a(u)du} b(s)\,dw_s
\]

后面的积分项是高斯随机变量，均值为 0，方差可以计算。由此得到 \(x_t|x_0\) 的均值和方差。

**VE SDE**：漂移 \(a(t)=0\)，扩散 \(b(t)=\sqrt{d[\sigma^2(t)]/dt}\)。缩放因子为 1，积分方差就是 \(\sigma^2(t)\)，所以扰动核是：

\[
p_t(x_t|x_0) = \mathcal{N}(x_t; x_0, \sigma^2(t)I)
\]

这就是 NCSN 的加噪形式 \(x_t = x_0 + \sigma(t)\epsilon\) 的连续版本。

**VP SDE**：漂移 \(a(t)=-\frac12\beta(t)\)，扩散 \(b(t)=\sqrt{\beta(t)}\)。定义 \(\alpha(t)=e^{-\int_0^t \beta(s)ds}\)，则缩放因子为 \(\sqrt{\alpha(t)}\)。通过积分计算，方差为 \(1-\alpha(t)\)，扰动核是：

\[
p_t(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\alpha(t)}\,x_0, (1-\alpha(t))I)
\]

这正是 DDPM 中 \(x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon\) 的连续形式。

**sub-VP SDE**：漂移与 VP 相同，扩散项被设计为 \(\sqrt{\beta(t)(1-e^{-2\int_0^t\beta(s)ds})}\)。它的扰动核仍然是高斯，但方差形式不同，保证方差不超过 VP 的方差。具体表达式可以从同一套线性 SDE 求解公式中代入对应的 \(a(t),b(t)\) 得到，核心结论是：它同样给出一个可解析计算的高斯扰动核，从而条件分数可算。

##### 3.3 训练网络怎么做

有了扰动核，训练目标就明确了。我们想让网络 \(s_\theta(x_t,t)\) 近似真实分数 \(\nabla_{x_t}\log p_t(x_t)\)，但真实边际分数不知道。**去噪分数匹配**指出，最小化以下目标等价于近似边际分数：

\[
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, x_t}\left[ \lambda(t) \left\| s_\theta(x_t,t) - \nabla_{x_t}\log p_t(x_t|x_0) \right\|^2 \right]
\]

其中条件分数 \(\nabla_{x_t}\log p_t(x_t|x_0)\) 从扰动核直接算出。

对 VP SDE，这个条件分数是 \(-\frac{x_t-\sqrt{\alpha(t)}x_0}{1-\alpha(t)} = -\frac{\epsilon}{\sqrt{1-\alpha(t)}}\)。所以训练网络预测噪声 \(\epsilon\)，与直接做分数匹配是等价的。对 VE SDE，条件分数是 \(-\frac{x_t-x_0}{\sigma^2(t)}\)，网络等价于在预测“加了什么噪声”。

实际训练流程因此非常直接：采样一个数据 \(x_0\)，采样一个时间 \(t\)，根据该 SDE 的扰动核采样 \(x_t|x_0\)（等价于加对应方差的高斯噪声），然后让网络从 \((x_t,t)\) 预测条件分数（或等价的噪声/去噪目标），做回归即可。三个 SDE 的区别只在于扰动核的具体系数不同，训练框架完全一致。

##### 3.4 反向过程

反向过程不再是“一步步加噪的逆操作”，而是一个**由前向 SDE 严格推导出的反向时间随机微分方程**。它同样是一个连续过程，从 \(t=T\) 的纯噪声出发，沿着时间倒流，最终到达 \(t=0\) 的数据分布。这个反向 SDE 的漂移项里包含了前向漂移、扩散系数以及**训练好的分数函数**。

给定前向 SDE \(d\mathbf{x} = \mathbf{f}(\mathbf{x},t)dt + g(t)d\mathbf{w}\)，其对应的反向时间过程满足：

\[
d\mathbf{x} = \left[\mathbf{f}(\mathbf{x},t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right]dt + g(t)d\bar{\mathbf{w}}
\]

其中 \(d\bar{\mathbf{w}}\) 是反向时间（从 \(T\) 到 \(0\)）的布朗运动。论文中更一般的形式（扩散系数为矩阵 \(\mathbf{G}\)）写作：

\[
d\mathbf{x} = \left\{\mathbf{f}(\mathbf{x},t) - \frac{1}{2}\mathbf{G}(t)\mathbf{G}(t)^\mathsf{T}\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\right\}dt + \mathbf{G}(t)d\bar{\mathbf{w}}
\]

这个公式的核心意义是：**只要知道每个时刻的分数函数 \(\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\)，就能构造出一个从噪声生成数据的连续过程**。而分数函数正是前面用扰动核训练网络去近似的东西。


训练好分数网络 \(s_\theta(\mathbf{x},t)\) 后，把它代入反向 SDE 替换未知的真实分数。采样时：

1. 从先验分布采样 \(\mathbf{x}(T) \sim p_T\)（通常是标准高斯）；
2. 用数值 SDE 求解器（如 Euler–Maruyama）从 \(t=T\) 向 \(t=0\) 积分这个反向方程；
3. 最终得到的 \(\mathbf{x}(0)\) 就是生成样本。

论文还提出了两种改进采样方式：**Predictor-Corrector 采样器**交替使用 SDE 求解器和基于分数的 MCMC 校正，以及**概率流 ODE**，后者是一个确定性的常微分方程，采样时没有随机性，还支持精确似然计算。

离散 DDPM 的反向是一串固定的马尔可夫转移 \(p_\theta(\mathbf{x}_{i-1}|\mathbf{x}_i)\)，每一步的更新规则是预设的。连续化后，反向变成了**一个可自由选择数值求解方案的微分方程**——步长、求解器类型、是否加入校正步骤，都成为可设计的超参数，而不是算法结构本身。这就是 SDE 视角在反向过程中带来的实质自由度。

##### 3.5 采样

采样流程就是：**把训练好的分数网络代入反向 SDE，得到一个完全已知的微分方程，再用数值方法把它离散化，逐步积分得到样本。** 具体分三层来看。

**第一层：代入分数，得到可求解的反向 SDE**

训练完成后，网络 \(s_\theta(\mathbf{x},t)\) 近似了真实分数 \(\nabla_\mathbf{x}\log p_t(\mathbf{x})\)。把它代入反向 SDE：

\[
d\mathbf{x} = \left[\mathbf{f}(\mathbf{x},t) - g(t)^2 s_\theta(\mathbf{x},t)\right]dt + g(t)d\bar{\mathbf{w}}
\]

此时方程右边除布朗运动外全部已知，它就是一个**具体的、可以用数值方法求解的 SDE**。这一步是“构造采样器”的起点。

**第二层：离散化——把连续 SDE 变成可执行的迭代**

连续方程不能直接计算，需要选一个数值求解器，把时间轴 \([0,T]\) 切成若干步，得到迭代更新式。最基础的是 **Euler–Maruyama** 离散化。设反向时间步长为 \(\Delta t\)，从 \(t\) 到 \(t-\Delta t\)：

\[
\mathbf{x}_{t-\Delta t} = \mathbf{x}_t - \left[\mathbf{f}(\mathbf{x}_t,t) - g(t)^2 s_\theta(\mathbf{x}_t,t)\right]\Delta t + g(t)\sqrt{\Delta t}\,\mathbf{z}
\]

其中 \(\mathbf{z}\sim\mathcal{N}(0,\mathbf{I})\)。这就是一个可直接执行的采样迭代：每步用网络预测分数，按公式更新状态并注入噪声。

论文还给出了更精细的求解方案，比如用 Heun 方法（二阶）替代 Euler 一阶，或者使用 **Predictor–Corrector**：先用 SDE 求解器走一步（predictor），再用基于分数的 Langevin 校正若干步（corrector），以提高样本质量。

**第三层：实际采样循环**

完整流程是：

1. 从先验采样初始状态 \(\mathbf{x}_T \sim \mathcal{N}(0,\mathbf{I})\)；
2. 设定离散时间网格 \(T=t_N > t_{N-1} > \cdots > t_0=0\)；
3. 对每个相邻时间对，用上述离散更新式从 \(t_i\) 走到 \(t_{i-1}\)，每步调用 \(s_\theta(\mathbf{x}_{t_i},t_i)\)；
4. 最终 \(\mathbf{x}_0\) 即生成样本。

注意：**离散化方式不是唯一的**。同一个反向 SDE，可以用 Euler、Heun、高阶 Runge–Kutta，甚至不同步长调度。选哪种离散化，本身就是设计采样器时的自由度，这也是 SDE 框架相比固定离散链的一个实际优势。另外，论文还从同一前向 SDE 导出了**概率流 ODE**，那是另一种采样器（确定性、无噪声注入），同样通过离散化 ODE 来执行。

_这部分内容想要更深的理解可以参考 [remmsrt/DiffusionSDE](https://github.com/remmsrt/DiffusionSDE.git)_

# 二、问题与解决

# 三、下周计划

- [ ] 探究生成式的图像隐写现有的不足

# 四、总结

