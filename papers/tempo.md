# TEMPO: Makespan-Aware Expert-Parallel Load Balancing Across Memory- and Compute-Bound Regimes

> [arXiv 2608.13057](https://arxiv.org/abs/2608.13057) (v2, 2026-08-14；v1 08-13, cs.DC / cs.AI / cs.CL / cs.GT, 18 页) · **KlingAI Research（快手可灵）** · 8 作者（Jie Li 等，通讯 Chengru Song，`@klingai.com`）
> 代码：⚠ **摘要声称 "Code is available at https://github.com/jeshxxx/TEMPO"，但该仓库 2026-09-07 核实为 404，不存在。无代码可用。**
> ⚠ **硬件被匿名化**：论文明说"GPU 型号、微架构名称、厂商规格在本版本中匿名"。Testbed A = 较老的 8 卡节点，Testbed B = 新一代节点（2×8，CUDA 13.2，RoCE 8×400G/节点）。从 DeepGEMM fp8 / CUDA graph / NCCL / DeepEP 这套工具链可**推断**是 NVIDIA，但论文未确认
> 场景：**MoE 推理服务（decode 为主）**，不是训练。EP 下的**每批次 dispatch 层**，placement 层（EPLB）视为给定
> 对照模型：Qwen3-235B-A22B-FP8、DeepSeek-V3-0324、DSv3-shape 代理、DeepSeek-V2-Lite

---

## TL;DR

**这篇的价值不在 TEMPO 这个调度器，而在它把"grouped GEMM 的每专家耗时"这件事测清楚了，并且给了一个十分钟就能在任意硬件上重新标定的黑盒方法。对我们来说，可迁移的是量纲和方法，不是算法。**

核心测量：DeepGEMM fp8 masked grouped GEMM 的每专家耗时**不是 token 数的线性函数**，而是两段的——

- **低于 n\* ≈ 156–168 token/专家**：耗时**与 token 数无关**，被"把这个专家的权重从 HBM 流进来"支配。代价挂在**激活的副本数**上。
- **高于 n\***：线性，但 grouped GEMM 把每个专家的 token 向上取整到 **128 的 M-tile**，所以把一个专家拆到 k 个副本上会**取整 k 次**——拆分不是重新分配工作，是**凭空制造填充计算**。

一个 max-affine 模型同时刻画两段：`t_g = max(a + b·G_g, c + β·N_g)`，`G_g` 是该卡上激活的 (专家, 副本) 对数，`N_g` 是 token 数。

**最该记住的四条：**

1. **激活地板 `b` 有干净的 roofline 验证，这是全篇最扎实的部分。** 一个专家 FFN 有 `3·d_model·d_ff` 个参数，fp8 下 DSv3 是 44.0 MB、DSv2-Lite 8.7 MB、Qwen3-30B 4.7 MB；除以峰值 HBM 带宽预测地板为 13.1 / 2.6 / 1.4 µs，实测拟合值 14.8 / 3.0 / 1.7 µs，落在预测的 **1.13–1.24 倍**，即**跑在峰值带宽的 80–88%**。`b` 就是权重流式，不是别的。

2. **两个 regime 在同一个 batch 里同时存在，这是全文的枢纽事实。** 真实 decode trace 里 **92–100% 的 batch** 同时有 ≥20% 的专家在平坦段、≥20% 的 token 在线性段。B=512 时，**平坦段专家占激活数的 47%，而线性段专家承载 91% 的 token**。所以"这个部署是访存瓶颈还是计算瓶颈"这个问题本身就问错了。

3. **⚠ 头条数字来自模拟器，不是墙钟。** 相图与规模外推全部是 calibrated simulation（论文 L1 自陈"headline numbers are model-space"，中段迁移误差 2–6 pp），15.5% 的最大增益是 **EP32–64 的外推**。真实墙钟的量级要小一个档：Qwen3-235B 上 GovReport +5.0% 吞吐、p99 TPOT −15.6%，而**饱和 decode 反而比 static 慢 2–4%**。

4. **⚠ 对 DeepSeek-V3 完全无效，而且论文把这个当成卖点。** DSv3（61 层、每卡 32 个专家）落在预测的 win region 之外，所有 workload 都是 −2 到 −3%，与 noop 对照无法区分；SGLang 自带的 dynamic rebalancer 同样一无所获（±1%）。论文的主张明确写成"**a phase diagram, not a universal win**"——**卖的是"部署前能预测会不会有用"，不是"用了就快"**。

**这篇的方法论诚实度是我读过的同类里最高的**，值得单独学（见 §6）：他们撤回了一个没复现的结果、预先声明了窗口剔除规则、报告了模型失效的负对照、用 noop 对照把数据通路与求解器的功劳分开、还做了 like-for-like 移植证明**自己头条增益的大部分来自工程架构而不是自己的想法**。

**一个必须打问号的地方**：`+38–70%` vs LPLB 这个最抢眼的数字，机制是 LPLB 把求解放在图内、每层一个 EP collective、94 层每步重放——**这是在比工程实现，不是在比调度目标**。论文自己在 §5.12 做了 like-for-like 移植并承认"most of the headline gap was architecture"。引用时**只能引 like-for-like 后的那个结论：token-LP 在低/中负载与 TEMPO 打平，TEMPO 的真实优势是在计算瓶颈点的稳定性**（1024 并发下 token-LP 跨窗口在 −11.6% 到 +8.7% 之间摆动，spread 20 pp；TEMPO 每个窗口都在噪声内，spread 4.4 pp）。

---

## 1. 问题与定位

### 1.1 两层结构：placement 与 dispatch

EP 下每个 MoE 层是 `router → dispatch all-to-all → per-expert grouped GEMM → combine all-to-all`。两个 collective 都同步整个 EP 组，所以**每层的 MoE 时间恰好等于 makespan（最慢那张卡的时间）**，其他卡多空闲都不影响。

生产系统分两层处理：

| 层 | 时间尺度 | 代表 | 做什么 |
|---|---|---|---|
| **Placement** | 分钟级 | DeepSeek EPLB | 用长期平均负载复制热专家、在卡间重排 |
| **Dispatch** | 每 batch | LPLB / UltraEP / METRO / EPLB 默认均分 | 决定每个专家的 token 如何在其副本间切分 |

本文只做 **dispatch 层**。理由是批间路由波动远大于长期平均——**某个 batch 的瓶颈卡routinely 是均值的 1.5–2 倍**，这部分不均衡 placement 够不着。

### 1.2 现有 dispatcher 都在优化代理指标，而两个代理各错一半

| 类别 | 代表 | 隐含假设 |
|---|---|---|
| 平衡 token 数 | LPLB（LP）、UltraEP（精确配额）、EPLB 默认均分 | 时间 ∝ token 数 |
| 平衡激活专家数 | METRO | 时间 ∝ 专家数 |

两个假设都不成立，因为有两个 token 数**表达不出来**的硬件效应：权重流式的地板，和 M-tile 的取整。

---

## 2. 核心测量（本篇最有价值的部分）

### 2.1 两段式 max-affine 模型

`t_g = max(a + b·G_g, c + β·N_g)`

- `G_g`：卡 g 上激活的 (专家, 副本) 对数
- `N_g`：卡 g 上的 token 数
- 拟合平均误差 **4–8%**

**激活地板 `b` 随专家变大而变大，跨形状差 8.5 倍：**

| 专家形状 | fp8 权重大小 | roofline 预测地板 | 实测拟合 `b` | 实测/预测 |
|---|---|---|---|---|
| Qwen3-30B | 4.7 MB | 1.4 µs | **1.7 µs** | 1.21× |
| DSv2-Lite | 8.7 MB | 2.6 µs | **3.0 µs** | 1.15× |
| DSv3 | 44.0 MB | 13.1 µs | **14.8 µs** | 1.13× |

即权重流式跑在峰值 HBM 带宽的 80–88%。bf16 的 loop 实现下拐点移到 `n* ≈ 331`，拟合 45 µs vs roofline 26 µs——**两段形状不变，常数反映实现效率**。

### 2.2 tile 阶梯：128 的 M-tile 才是底层机制

在 Testbed B 上细扫 128 边界，发现耗时其实是**阶梯**而非两段：`[1,128]` 内平坦，129 处跳一级，到 256 再平坦。而且**后面的台阶是次线性的**（每个边界 +11–24%），因为同一个专家的后续 tile 能从 L2 复用权重。

于是**两段模型的"平坦段"其实就是这个阶梯的第一级**，只是第一级的台阶高度被权重加载支配。加一个参数就能刻画后面的台阶：

`t = max(a + b·G + b₂·(T − G), c + β·N)`，其中 `T = Σ_e ⌈n_e/128⌉`，`b₂ ≈ b/3`（Qwen3-235B: b=3.95, b₂=1.23 µs；DSv3: b=8.32, b₂=2.65 µs）

tile 项把多 tile 区的模型误差从约 10%（最差 25%）降到 **2.0%**，且不影响 decode 区（`n_e ≤ 128` 已由 `bG` 定价）。

**⚠ 这个 tile 项的适用范围有明确上界**：在 prefill 尺度（4096–8192 token/卡）它会改变 97–99% 的 batch 的解、把建模 block time 中位数砍掉 4.5–6.0%（p95 9.4%），均分策略会**凭空制造 3–10% 的填充 tile**；但**到 16,384 token/卡就消失了**，因为线性项开始主导。

### 2.3 混合 regime 是常态，不是边角情况

用标定出的 `n*` 当分类器，在四组真实 trace（Qwen3-30B、DSv2-Lite × wikitext、GSM8K）的 128–512 token/卡 decode 下：**92–100% 的 batch** 同时满足"≥20% 激活专家在平坦段"和"≥20% token 在线性段"。合成 Zipf(s=1.2) 路由下是 100%。

> **这条是全文的枢纽。** 如果两个 regime 是可分的两种工作点，运维只要按工作点选策略就行；正因为它们在同一个 batch 里共存，才需要一个能同时给两者定价的目标函数。

### 2.4 代理指标在真实 batch 上错多少

Table 1，固定 batch、固定 placement、只改切分方式，建模 block time 相对 TEMPO 解（中位数 / p95）：

| B | static | uniform | token-LP | METRO | 四者极差 |
|---|---|---|---|---|---|
| 128（地板段） | 1.17/1.27 | **1.51/1.60** | 1.17/1.27 | **1.11/1.18**（最好） | 1.37/1.46 |
| 512（计算段） | **1.47/1.58** | 1.12/1.20 | **1.00**（最优） | 1.05/1.10 | 1.47/1.58 |
| 1024（计算段） | **1.56/1.70** | 1.04/1.10 | **1.00**（最优） | 1.06/1.12 | 1.56/1.70 |

**贵的那个代理会随 regime 翻转**：B=128 时均分比建模最优贵 51%、METRO 最省（11%）；B=512–1024 时 static 贵 47–56%、token-LP 就是最优。没有哪个固定代理在整条 batch 梯度上安全。

误定价还是**不对称的**：METRO 在计算段深处最多贵 **2.8×**，而 token-LP 在访存段只贵 5–17%——论文用这条解释"为什么生产里先注意到 METRO 方向的失败"。

---

## 3. 理论：难点恰好在 regime 交界处

| 结论 | 内容 |
|---|---|
| **Theorem 1** | 判定是否存在 makespan ≤ T 的 dispatch 是 **NP-complete**，**2 张卡、全复制、a=c=0 就已经是**。从 Balanced PARTITION 归约 |
| **Lemma 1**（b→0） | 退化成 `min max_g N_g`，即 token LP（LPLB / UltraEP 的目标），多项式 |
| **Lemma 2**（β→0） | 退化成副本集合上的最优 semi-matching，多项式（增广路） |
| **Theorem 2** | 按 token 降序做整专家 round-robin 的 `A₃` 满足 `M(A₃) ≤ OPT + max(b, β·n_max)`（OPT 允许任意切分）。相对误差 ε 在真实 decode batch 上中位 4.1%、最大 7.7%，到 4× token 规模涨到 31% |

> **"在每个纯 regime 下多项式、只有 regime 交互时 NP 难"——这是系统观察的形式化镜像**：每个代理在自己主场都没问题，需要新算法的是过渡带。这个对应关系是本篇最漂亮的结构。

---

## 4. 算法与工程

### 4.1 tempo_fast 四阶段

| 阶段 | 针对的失效模式 |
|---|---|
| 1. 代价感知贪心播种 | 按 token 降序**整专家**放到边际代价最低的副本上。**整专家放置是关键——切碎冷专家正是 token-LP 的错误** |
| 2. 增广链做激活再平衡 | 平坦段瓶颈是 `max_g G_g`，单步移动会抬高目的地而停滞。用 1 步和 2 步增广链（Lemma 2 那个最优算法的截断版） |
| 3. 瓶颈局部搜索 + 部分迁移 | 线性段把 token 质量搬离瓶颈卡：整专家移动 + 三分搜索的部分切分（固定切分大小时目标是分段线性单峰），≤300 轮 |
| 4. 带 1% 切换容差的 ensemble | 把 token-LP 与 `A₃` 也按同一模型打分，只有好过 1% 才切换。**这使"不会比 token-LP 差超过 1%"按构造成立**；近似平局交给模型之外的因素决定 |

**质量与开销**：对 10 秒 HiGHS MILP，makespan 比值均值 1.005 / p95 1.024 / 最大 1.033；求解 **1.9 ms**（B≤256，纯 Python），约为 MILP 的 1/550，且在关键路径之外。

**消融**（相对完整 tempo_fast 的劣化）：去掉部分迁移最多 **16.3%**（EP32 过渡带）、去掉增广链最多 2.7%（中 B 平坦带）、去掉 ensemble 在 EP8 只 ≤0.8% 但 EP64 深计算段到 4.5%。**四个阶段各自在相图的不同区域生效，互为守卫**——在地板段（B=128）播种本身一无所获（1.000），增广链贡献全部的 −9.8%；到流量段角色反转，增广链反而把播种的 0.883 弄坏到 0.940，靠局部搜索修回来。

### 4.2 黑盒标定：十分钟，这是最可迁移的部分

代价表是**拟合出来的，不是推导出来的**：在部署的 MoE 流水（或其 microbenchmark）上扫 `(G, N)` 网格，记录每卡 `(G, N, t)`，交替赋值拟合两段 max-affine。**每个 (kernel, dtype, 硬件) 组合约十分钟**，一轮迭代即收敛。

**标定是承重的不是仪式**：用 GEMM-only 参数去跑一条有未融合 per-token kernel 的流水，B=128 时让 TEMPO 亏 16%。

### 4.3 SGLang 集成：目标是图内零边际开销

- **图内只有一个融合 kernel**：概率性 dispatch 与计数收集融合，采样 kernel 原子地累加持久化的每专家计数器。相对 static 的图内边际开销是**零个 kernel**；相对 LPLB 去掉了图内内点法求解及其每层 EP collective。
- **求解器是独立进程而非线程**：刷新线程在旁路 stream 上快照计数器，用**专用 gloo group**跨 rank 聚合（共享 group 对调度器自己的 collective 不是线程安全的），交给一个纯 numpy 的 worker 进程求解。**线程内版本因 GIL 争用端到端亏 8–10%**。
- **不做双缓冲的一致性**：空闲专家的行不动、无效副本列在新旧表里都是零、kernel 在使用时按行重归一化，于是读到新旧混合的行仍是合法分布；单行撕裂最坏情况下和为零，kernel 退化成在持有该专家的副本上均匀分布。**撕裂只会让某个 batch 用到陈旧或均匀的切分，永远不会路由到没有该专家的副本、也不会除零。**
- 代价是陈旧性（默认 ≤200 ms）。集成触及 7 个 patch 点。

> **顺手发现了 SGLang 已发布 LPLB 路径的一个 bug**：求解器输入在 padding-masking 之前取，CUDA graph 的 padding 行会**捏造出一个幽灵热专家**。

### 4.4 多节点：拓扑决定成败

平坦的 dispatch 表对所有 rank 一视同仁，但**token 发给哪个副本决定它是否跨节点**。两阶段处理，不动求解器内核：先解出每卡份额 `x_{e,g}`，再对每个专家用**同节点优先的运输规则**把 token 源切分到各副本——这在所有能实现该 `x_{e,g}` 的切分里对跨节点流量是最优的。**每卡负载因而与平坦解完全相同，只有源到副本的配对变了。**

2 节点 EP16 上：平坦表在 all-to-all 瓶颈点比 static **倒退 3.5%**，加上拓扑感知切分翻成 **+4%**（Qwen3-235B 端到端 +7.2%），大 batch 下值约 **11 pp** 吞吐。

---

## 5. 实验证据与其强度

论文自己把证据分成四层，并声明每层能主张什么：**测量代价网格 → 标定模拟（头条）→ 8 卡墙钟 → SGLang 服务**。

### 5.1 墙钟微基准（8 卡 Testbed A，EP8，DSv3 形状，真实 Qwen3 路由）

- B=32（访存瓶颈）：TEMPO 胜 EPLB-even **11–14%**、胜 token-LP 7%；token-LP 离最优差 19%
- B=2048（计算瓶颈）：METRO 离最优差 7–11%；TEMPO 与 token-LP 打平
- 全 B 范围：TEMPO 在每个 B 的最优固定策略的 5%（≈ 运行噪声）以内，而**每个固定策略都有 ≥7% 的失效区**

### 5.2 端到端服务：这里才是真话

| 场景 | 结果 |
|---|---|
| **Qwen3-235B（在 win region 内）** GovReport 长 prefill | 吞吐 **+5.0%** 中位，区间不重叠（521–524 vs 549–554 tok/s） |
| 同上，Poisson 8 req/s | **p99 TPOT −15.6%**（191 vs 226 ms），中位 TTFT −12.5% |
| 同上，ShareGPT | 基本打平（吞吐 −0.2%，p99 −8.9%） |
| 同上，饱和 decode-heavy | **比 static 慢 2–4%**（残余数据通路开销） |
| **DSv3-0324（在 win region 外）** 全部 workload | **−2 到 −3%**，与 noop 对照无法区分 |
| token-LP dispatcher，两个模型所有 workload | **−10% 到 −56%**（图内 collective 随层数复合） |

**noop 对照把功劳拆开了**：GovReport 那 +5.0% 里约**三分之二来自均分切分的数据通路本身、只有三分之一来自求解器**；而尾延迟的胜利主要是求解器的。

### 5.3 陈旧 placement 下（dispatch 最该发光的场景）

真实 Qwen3-235B、人为破坏 placement：static 掉 −7.3% ShareGPT 吞吐、p99 TPOT +40%。TEMPO **只找回三分之一的尾部损失**（p99 −10%、TTFT −5%），**吞吐一点没找回**。

> 论文自己的总结值得抄下来：**"Dispatch repairs the tail; only placement repairs the mean."** 漂移下冗余副本坐在错的（冷）专家上，这是**供给错误，只有 placement 能修**；尾部来自热专家的瞬时共调度，dispatch 还能绕。

### 5.4 §5.14：五种策略内核在同一代价模型下的横评

Table 4，相对 static 的 block makespan（越低越好，R=64）：

| 策略内核 | 0.25×（地板段） | 1×（混合） | 4×（token 段） |
|---|---|---|---|
| uniform (dynamic) | 1.299 | 1.052 | 0.659 |
| token-LP（LPLB / UltraEP） | 1.006 | 0.836 | **0.630** |
| METRO（激活） | 0.950 | 0.815 | 0.672 |
| LLEP-R（最少负载） | 0.984 | 0.835 | 0.662 |
| **TEMPO（时间模型）** | **0.856** | **0.764** | **0.630** |

论文自陈"用 TEMPO 优化的模型去给所有策略打分是自评"，所以另外跑了 METRO 与 LLEP-R 的端到端（3 个 drift-16 窗口各一次），墙钟排序与表一致、被 Amdahl 压缩：512/1024 请求下中位吞吐 vs 同窗 static 分别是 METRO −2.1%/−0.2%、LLEP-R +0.1%/−2.2%、TEMPO −0.3%/+2.0%。

**重要边界**：R=16、K≥16 时**所有内核都塌到 1.000**——漂移后的热专家集合根本没有副本可用时，这个问题属于 placement 层。

---

## 6. 方法论：这篇的诚实度值得单独学

这是我读过的同类工作里自我约束最强的一篇。具体做法，建议我们自己的实验报告照抄：

1. **撤回了一个没复现的结果**：§5.12 原文——"an earlier single-window +5% did not replicate; we retract it"。
2. **预先声明的窗口剔除规则**：一个窗口只有在出现"不参考哪个策略赢就能识别的基础设施故障"（崩溃、节点被回收、物理上不可能的 bench 客户端指标）时才剔除，**从不因结果剔除**；所有尾部指标都报告，**包括对自己不利的**。
3. **报告了模型失效的负对照**：Qwen3 形状**锚不住模拟器**——`b ≈ 1.7 µs` 低于测量噪声、重拟合退化出**负的 b**、相同赋值序列跨运行波动 ±11%（DeepGEMM tuner 的不确定性）。论文把它当 limitation 报出来，并指出它的反面：模型分辨不了的地方，硬件上策略选择也无所谓。
4. **noop 对照分离数据通路与求解器**：完整 TEMPO 数据通路、但求解器从不运行、表冻结在均分。
5. **like-for-like 移植证伪自己的头条数字**：把 token-LP 原样移进自己的 worker，发现"most of the headline gap was architecture"。
6. **主张写成相图而不是加速比**："A phase diagram, not a universal win"，并且用两个模型（一个在区内一个在区外）双向验证了这个边界。
7. **L6 自陈"parity-at-zero-tax rather than victory"**——在权衡生效的地方，诚实的读法是打平而非胜利。

---

## 7. 局限（论文 L1–L6 + 我补充的）

**论文自陈：**

- **L1** 头条数字在模型空间。相图与规模图来自标定模拟器，两端有墙钟锚定但**中段迁移误差 2–6 pp**。
- **L2** 测试台规模有限。服务证据只到 8 卡 Testbed A 与 2×8 Testbed B（EP16）；**完整 DSv3 规模（58 MoE 层、EP32+）仍是外推**——逐层效应是实测的，深度相乘的是推算的。
- **L3** win region 是有条件的（已成图）。小专家 bf16 形状实测无增益；placement 新鲜 + 副本充裕时所有策略打平。
- **L4** 静态标定。kernel / 驱动更新需重跑十分钟标定。
- **L5** `(G,N)` 模型有下限。DeepGEMM masked kernel 在**相同 (G,N) 下、每 slot 负载更均匀时更快**——这是模型表达不了的第三个维度，也是 Qwen3 重拟合出负 `b` 的原因。
- **L6** 部署系统不是理论最优，且是有意的。NP 难 → 有界启发式；异步规划用上一窗口的计数（值 22 pp headroom 里的 5–7 pp，端到端约 2 pp）；整 slot 粒度；继承 EPLB 的 placement 而非协同优化；付随深度放大的 remap 税。

**未述但重要的：**

1. **硬件匿名化让所有绝对量纲不可直接引用。** `n* ≈ 156`、`b = 14.8 µs` 这些数只在他们那台未公开的机器上成立。好在方法是黑盒的，我们重标定即可——但**不能把这些数字当常数抄进我们的文档**。
2. **`+38–70% vs LPLB` 这个最抢眼的数字比的是工程架构。** 论文自己澄清了，但摘要与引言仍然把它放在显眼位置。
3. **全篇是 decode / serving。** §5.14 明说训练侧的均衡器（LLEP / TAOT / UltraEP）**无法在服务栈上端到端复现**，因为它们利用的是梯度步之间的权重迁移窗口，服务没有这个窗口。**所以本文的调度算法不能直接搬到训练。**

---

## 8. 与我们的关系

### 8.1 可以直接迁移的：方法，不是算法

1. **在 MI355X 上重做一遍这个测量。** 十分钟的黑盒标定 + `(G, N)` 网格扫描，对我们的 grouped GEMM（CK / hipBLASLt / FlyDSL / AITER 各一条）拟合两段 max-affine。三个问题值得回答：
   - **我们的 M-tile 是不是 128？** CDNA3/CDNA4 的 MFMA tile 与 CUDA 侧不同，阶梯的步长和台阶高度都要重测。这直接影响 [`notes/MegaMoeFlydsl/`](../notes/MegaMoeFlydsl/2026-09-03_1555_flydsl_vs_triton_op_perf_mi355x.md) 里 grouped GEMM 的分桶与 autotune 策略。
   - **拐点 `n*` 在 MI355X 的 HBM3e 8 TB/s 下落在哪？** 带宽比他们的测试台高，地板 `b` 应该更低、`n*` 更小。
   - **roofline 比值是多少？** 他们拿到 80–88% 的峰值带宽。我们的 grouped GEMM 在权重流式阶段能到多少，是一个独立有价值的数字。
2. **roofline 交叉验证的写法可以照抄**：先从参数量算出理论地板，再看拟合值落在几倍——这比单纯报一个拟合误差有说服力得多。
3. **§6 的方法论七条**，尤其是 noop 对照与 like-for-like 移植，建议进我们自己的 kernel 优化流程。这与 [`.cursor/rules/iteration_rules.mdc`](../Primus-Turbo/.cursor/rules/iteration_rules.mdc) 的单变量迭代要求是一路的。

### 8.2 不要误用的地方

**⚠ 我在 [`knowledge/systems/arxiv-digest-2026-09.md`](../knowledge/systems/arxiv-digest-2026-09.md) 里最初写"128 M-tile 填充效应对训练侧 grouped GEMM 分桶同样成立"，读完全文后这个说法需要收窄**：tile 项的价值在 4096–8192 token/卡时最大，**到 16,384 token/卡就消失了**（线性项主导）。典型训练配置的每专家 token 数远高于此，所以：

- **激活地板 `b` 在训练里基本不相关**（训练不会让专家停在 `n < 128`）。
- **tile 填充在训练里只在中等规模下有意义**——比如小 micro-batch、大 EP、或者专家很多导致每专家 token 数被摊薄的配置。
- **真正跨场景成立的是"per-expert 耗时不是 token 数的线性函数"这个事实本身**，以及由此得到的"任何按 token 数做的静态分桶都会在某个 regime 出错"。

### 8.3 与已有笔记的交叉

| 笔记 | 关系 |
|---|---|
| DA-MoE（`2607.23099`，见 [`knowledge/systems/arxiv-digest-2026-08.md`](../knowledge/systems/arxiv-digest-2026-08.md)） | 「静态 token-count 分桶选核是错的」——与本文「没有 regime-free 的固定代理」是同一结论的推理侧与调度侧两个版本，互相印证 |
| [`ultraep/`](./ultraep/README.md) | UltraEP 的精确配额在本文里被归类为 token 平衡代理，在地板段被证明系统性错误（Table 4，0.25× 列 1.006 vs TEMPO 0.856） |
| [`tessera.md`](./tessera.md) | Tessera 记的「代价模型尾部误差 15% 会翻转 MILP 排序」与本文「标定不准让 TEMPO 亏 16%」是同一教训的两次独立确认 |
| [`instella-moe.md`](./instella-moe.md) | Instella-MoE 的 FarSkip 是**架构侧**减通信，TEMPO 是**调度侧**均衡计算；两者正交，但都建立在"EP 的同步点是 makespan"这个前提上 |
| [`knowledge/systems/mi455x-parallelism-strategy-reference.md`](../knowledge/systems/mi455x-parallelism-strategy-reference.md) | 本文 DSv3 的负结果（每卡 32 个专家时统计平均已经抹平不均衡）给了一个反向判据：**每卡专家数够多时，dispatch 层没有可回收的空间**。这对 MI455X 上选 EP 度数是直接输入 |

---

## 9. 术语

- **Makespan** → 一组并行任务里最慢那个的完成时间。EP MoE 层的耗时恰好是它。
- **Dispatch 层 / Placement 层** → 前者每 batch 决定 token 在既有副本间怎么切；后者分钟级决定哪些专家在哪张卡上有几份副本。
- **激活地板 `b`** → 在一张卡上多碰一个专家的权重所需的时间，本质是 HBM 权重流式，与 token 数无关。
- **M-tile 取整** → grouped GEMM 把每个专家的 token 数向上取整到 128 的倍数；拆分专家会让取整发生多次。
- **Fixed-charge makespan** → 每使用一个"资源"要付固定开销、再加与用量成正比的可变开销的最小化最大值问题。
- **Semi-matching** → 二分图上每个左点恰好匹配一个右点、最小化右点最大负载的问题；本文 β→0 极限。
- **METRO / LPLB / UltraEP / EPLB / LLEP / TAOT** → 现有的 EP 负载均衡方案，前四者是 dispatch 层代理，后两者是 placement 层。

---

## 10. 复现清单

| 项 | 状态 |
|---|---|
| 代码 | ❌ **摘要给出的 `github.com/jeshxxx/TEMPO` 2026-09-07 核实为 404**。论文描述了模拟器与 7 个 SGLang patch 点，但都无从取得——**所有"可复现"的说法只能按论文文字采信** |
| 代价模型参数 | ✅ Table 7（Appendix B）列出本文用到的每一组，并解释 GEMM-only 与 full-pipeline 拟合为何合理地不同 |
| 标定流程 | ✅ 黑盒，约 10 分钟每 (kernel, dtype, 硬件) |
| 路由 trace | ⚠ 用了"recorded proxy routing"与真实 trace，未说明是否释出 |
| **硬件** | ❌ **型号、微架构、厂商规格全部匿名化** |
| 墙钟 vs 模型 | ⚠ 头条在模型空间，两端有墙钟锚定，中段迁移误差 2–6 pp |
| 规模 | ⚠ 最大 2×8 卡 EP16；EP32–64 与完整 DSv3 深度是外推 |
