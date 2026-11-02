# BASP: Communication-Efficient Batch-Aware Sequence Parallelism for LLM Training

> [arXiv 2609.03151](https://arxiv.org/abs/2609.03151) (v1, 2026-09-02, cs.DC / cs.LG) · **IEEE HPEC 2026 已接收** · Bigyan Ghimire, Jon C. Calhoun（Clemson University）
> 基线：DeepSpeed-Ulysses。实现方式是改开源 DeepSpeed 代码，保留原 API
> 硬件：**2 节点 × 4× A100 40GB（共 8 卡）**，节点内 NVLink，节点间 400 Gb/s InfiniBand。Clemson Palmetto 集群
> 无代码释出声明

---

## TL;DR

**想法一句话说完：Ulysses-SP 不管 micro-batch 多大都做全局 N 路 all-to-all；如果 `N = K·B`，把 N 张卡切成 B 个不相交的 K 卡组、每组包一条序列，就变成 B 个并行的 K 路 all-to-all。每卡 token 数 `BS/N` 和显存都不变。**

关键在于**当 K 恰好等于每节点卡数时，all-to-all 全部落在节点内 NVLink 上，完全不走 InfiniBand**。论文的全部收益都来自这一条。

**⚠ 我的判断：这篇的机制是对的，但基线看起来被错误设定了，"新方法"很可能只是 `DP=B × Ulysses-SP=K` 这个标准组合的重新发现。** 详见 §3——这是读这篇时最该盯住的地方，也是我不建议引用它的主要原因。

数字（8 卡 A100，seq 16K，B=2，K=4=单节点）：端到端 **1.17–1.32×**，all-to-all 时间降 2.23–3.10×，loss 曲线完全重合。序列扩展上 1K/2K/4K 只有 5.7%/3.4%/1.8%，8K 起跳到 13.1%，16K 18.2%，32K 25.9%。

**对我们的价值主要是两条旁证，而不是方法本身：**
1. 序列长度扫描独立佐证了我们 [`mi455x-parallelism-strategy-reference.md`](../knowledge/systems/mi455x-parallelism-strategy-reference.md) 里 **G5「序列 ≥ 8K 才开 CP」**这条阈值——4K 时只有 1.8%，8K 才跳到 13.1%。
2. 它的核心规则「让 K = 每节点卡数，把 a2a 关在快速互联内」就是我们 **G2「把通信关在 scale-up 域内」**换到 CP 维度的说法。**而在 Helios（72 卡单域）上这条规则几乎自动满足，所以这篇的 win region 在我们的目标硬件上基本消失。**

---

## 1. 问题设定

Ulysses-SP（DeepSpeed）沿序列维切分：micro-batch `B`、序列长 `S`、`N` 卡时每卡持 `B·S/N` 个 token。注意力计算需要按 head 并行，于是做一次 all-to-all 把 QKV 投影重分布成"每卡负责若干 head 的完整序列"，算完再做一次 all-to-all 还原回序列切分。**这个 all-to-all 恒定跨全部 N 卡，与 B 无关。**

作者的 profiling（Llama-3.2-3B，8 卡）显示 `NcclSend_Recv`（NCCL 实现 all-to-all 的 kernel）占单次迭代 GPU 时间的 **34%**，且随 micro-batch 线性增长。

动机用 α–β 模型说清楚了：成对交换式 all-to-all 约为 `T = (P−1)(α + m·β)`，`P` 是参与 rank 数。**减小 P 同时省下延迟项和轮数**；在带宽分层的集群上还能把流量整体挪到更快的一层。

---

## 2. 方法

三个模块，都很简单：

1. **Batch-aware group formation**：`K = N/B`，建 `B` 个组。**连续 rank 分配**（GPU `0..K−1` 是组 0，`K..2K−1` 是组 1……），这样 `K` = 每节点卡数时每组恰好落在一个物理节点内。断言 `N mod B == 0`；`B > N` 时退化为普通数据并行。
2. **Batch-aware sequence partitioning**：rank `r` 属于组 `g = ⌊r/K⌋`、组内位置 `ℓ = r mod K`；它只拿序列 `g` 的第 `ℓ` 个 `L/K` 长度分片。
3. **Subgroup all-to-all**：注意力前后的两次 a2a 限制在组内 `K` 卡。

**负载与显存等价性**（论文 §IV-C1）：Ulysses 每卡 `∝ B·H·(S/P)·d`；BASP 每卡处理 `B/G` 个 batch 元素、分片长 `S·G/P`，乘起来还是 `B·H·(S/P)·d`。这一步是对的。

配合 ZeRO：§IV-D 说 ZeRO 在序列并行组与数据并行组两个维度上切分模型状态。实验用 ZeRO-3 + 混合精度。

---

## 3. ⚠ 核心质疑：Table I 的基线

论文 §IV-E 专门论证「把 Ulysses 的 `sequence_parallel_size` 设成 `N/B` 并不等价于 BASP」，靠的是这张表：

| 方法 | Tokens/GPU | a2a 参与者 |
|---|---|---|
| Ulysses `SP=N` | `BS/N` | `N` |
| Ulysses `SP=N/B` | **`B²S/N`** | `N/B` |
| **BASP** `SP=N, K=N/B` | **`BS/N`** | **`N/B`** |

**`B²S/N` 这一格的算法是：SP 组大小 `K=N/B`，于是 DP 度 `= N/K = B`；如果每个 DP 副本仍然拿 `B` 条序列的 micro-batch，那么每副本 `B·S` 个 token 摊到 `K=N/B` 卡上 = `B²S/N`。算术没错——但这意味着全局 batch 变成了 `B` 个副本 × `B` 条序列 = `B²` 条，是 `SP=N` 那一行的 `B` 倍。这不是等 batch 比较。**

**等全局 batch 的比较应该是**：全局 micro-batch `B` 条序列，DP=B 个副本，每副本 1 条序列，摊到 `K=N/B` 卡上 → 每卡 `S/(N/B) = BS/N` 个 token，a2a 跨 `N/B` 卡。**这和 BASP 那一行逐字相同。**

而且论文紧接着写的这句话是直接错的：

> this configuration ... **sequences are no longer split across GPUs**, limiting the maximum trainable sequence length.

`SP = N/B` 下序列**是**被切到 `K` 张卡上的，只是没切到全部 `N` 张——而 BASP 也只切到 `K` 张。两者能训的最大序列长度相同。

**所以我的读法是：BASP ≈ `DP=B × Ulysses-SP=K` 这个标准组合。** 论文没有任何一处讨论 DP 与 SP 如何复合，而 §IV-E 恰恰是该讨论的地方。

**如果 BASP 相对该组合还有真实差异，那只可能在一个地方：连续 rank 分配。** DeepSpeed 默认构造 SP 组时若采用跨节点交错的 rank 顺序，一个 SP 组就会横跨节点，白白把 a2a 送上 IB。**那样的话这篇的贡献本质是一个 rank 映射的拓扑感知修复，而不是一种新的并行策略**——有实用价值，但和摘要里的定位差了一个量级。论文没有报告基线的 rank 排布，无法从文中判定。

---

## 4. 实验与其强度

### 4.1 主结果

单一配置：seq 16K、B=2、8 卡（于是 `K=4` = 每节点卡数）。6 个模型（Llama 3.2-1B/3B、3.1-8B；Qwen 1.5-1.8B、2.5-3B、3-8B，**全部 32 个 attention head**）。30 次迭代平均，**未报误差棒**。

- 端到端 **1.17–1.32×**（Qwen 降 18.2–24.1%，Llama 降 14.7–20.5%）
- all-to-all 时间降 **2.23–3.10×**；a2a 占比 Qwen1.5-1.8B 从 37.7% → 16%
- 通信降幅远大于端到端降幅，符合 Amdahl

论文自己承认了机制归因：

> The communication benefit is also amplified in BASP because when `K=4`, 4 GPUs are confined to a single node, and during the all-to-all, there is no inter-node communication.

### 4.2 ⚠ 两个机制在这套硬件上无法分离

节点只有 4 卡，而 `K = N/B ≤ 4` 在所有测试点都成立，基线 `N=8` 则必然跨节点。**于是每一个数据点都同时开着"组变小"和"流量从 IB 挪到 NVLink"两个机制，全文没有任何一个实验能把它们分开。** 要分开需要在单个 8 卡节点内扫 K，他们的硬件做不到。

**这意味着 1.17–1.32× 这个数字本质是他们那台 2 节点机器上 NVLink 与 400 Gb/s IB 的带宽差，换一套网络层次就得重测。**

### 4.3 Batch 扫描（Llama-3.2-1B，seq 8K）

- `B=1`：`K=8`，BASP ≡ Ulysses，步时基本相同。**这个负对照报得很诚实。**
- `B=2..8`：加速从 1.10× 增到 1.26×
- `B=8` 时号称 all-to-all **85×** 加速，端到端只有 1.25×（Amdahl 上限 1.5×）

**⚠ 那个 85× 不能当加速比读**：`B=8, N=8` → `K=1`，组内只有一张卡，**all-to-all 根本不存在了**。这是把一个 collective 消掉，不是让它变快。而且 `K=1` 时每卡持有一整条 8K 序列，**这个配置已经完全不是序列并行、就是纯数据并行**——论文只在 `B>N` 时才声明退化为 DP，没说 `B=N` 同样退化。

论文诚实地指出 `B=8` 时瓶颈转移到 ZeRO 的通信上。

### 4.4 序列扩展（Llama-3.2-3B，B=2）

| 序列长 | 1K | 2K | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|---|
| 步时降幅 | 5.7% | 3.4% | **1.8%** | **13.1%** | 18.2% | 25.9% |

**这条曲线是全文对我们最有用的东西**（见 §6）。注意 1K→4K 不单调，作者未解释。

### 4.5 精度

Llama-3.2-3B 训 800 步，BASP 与 Ulysses 的 loss 曲线完全重合。**这是数学等价的重排，不重合才是 bug**——作为实现正确性验证合格，作为贡献不算。

### 4.6 缺失的对照

Related work 里讨论了 Ring Attention、Striped Attention、USP、FlexSP，**一个都没做实验对比**。全文唯一基线是 Ulysses-SP。

---

## 5. 局限

**论文自陈（§VI，只有一段）**：要求 `N ≥ B` 且 `N = K·B`、`K` 为整数；不可整除的配置留作 future work。

**未述但更重要的：**

1. **基线可能设错**（§3）。这是最要命的一条。
2. **两个机制不可分离**（§4.2）。
3. **规模**：8 张 A100 40GB，2 节点。任何跨机架、多层网络的结论都无从谈起。
4. **模型全部 32 head**，从未触及 Ulysses 真正的扩展上限——**SP 度不能超过 head 数（GQA 下是 KV head 数）**。有意思的是，缩小组到 `K` 恰好放松了这个约束，**这大概是 BASP 最好的一个论据，而论文完全没提**（当然 `DP×SP` 同样能拿到）。
5. **`85×` 与 `1.17–1.32×` 这两个数被并列放进摘要与结论**，前者是消掉了 collective、后者是端到端，量级差两个数量级却不加区分。
6. **无代码、无误差棒、无 MFU / 无绝对吞吐**，只有归一化步时柱状图。

---

## 6. 与我们的关系

### 6.1 方法本身：不需要引入

**Primus / Megatron 侧这已经是既有实践。** 我们 [`mi455x-parallelism-strategy-reference.md`](../knowledge/systems/mi455x-parallelism-strategy-reference.md) 记的 NVIDIA 官方决策顺序里：

- **G1**「最小化模型并行、最大化数据并行」——TP/EP/PP/**CP** 在不 OOM 前提下尽量小
- **G2**「把通信关在 scale-up 域内」

BASP 就是 G1 + G2 应用到 CP 维度。我们不会先把 CP 拉满再想办法缩小它的通信组，我们一开始就按 G1 把 CP 设到够用为止、剩下给 DP。

### 6.2 值得留下的两条旁证

1. **序列长度阈值的独立佐证。** §4.4 那条曲线在 A100 上给出 `4K → 1.8%、8K → 13.1%` 的跳变，与 G5「序列 ≥ 8K 才开 CP；< 4K 时 CP 开销大于收益」高度一致。这是一个来自不同硬件、不同框架的独立确认，值得记一笔。
2. **Ulysses a2a 的占比量纲**：Llama-3.2-3B、8 卡、跨节点时 `NcclSend_Recv` 占迭代 34%。**只在"SP 组跨慢速互联"这个前提下成立**，引用时必须带上这个限定。

### 6.3 在 Helios 上这篇的 win region 会消失

这是最值得说的一点，而且结构上与 [TEMPO](./tempo.md) 的相图论证同构。

BASP 的全部前提是「NVLink 岛很小（4–8 卡）、出岛就掉到 IB」。MI455X / Helios 是 **72 卡单 scale-up 域**，带宽层次 3.6 TB/s 域内 vs 约 597 GB/s per GPU 跨机架，**6:1**。我们那篇参考里已经用 Megatron-Core MoE 的实测说明了域内扩展几乎是平的——GB200 上 dispatch 从 EP8 到 EP64（8×）只涨 1.73×，而 H100 跨节点同样跨度涨 7.0×。

**同样的逻辑对 CP 成立：只要 CP 组整个落在 72 卡域内，再把它从 8 缩到 4 买不到多少东西。** BASP 赖以生效的那个陡峭带宽悬崖，正是 NVL72 / Helios 这代硬件要消掉的东西。

### 6.4 唯一值得动手的一件小事

**去确认 Primus / Megatron 的 CP（以及 TP/EP）进程组是不是连续 rank 排布，别让一个 CP 组横跨节点。** 如果 BASP 相对 `DP×SP` 真有优势，只可能在这里。成本很低，坏了的话是真问题。

### 6.5 交叉引用

| 笔记 | 关系 |
|---|---|
| [`mi455x-parallelism-strategy-reference.md`](../knowledge/systems/mi455x-parallelism-strategy-reference.md) | G1 / G2 / G5 三条准则已覆盖 BASP 的全部主张；§4.4 的曲线可作为 G5 阈值的外部佐证 |
| [`moe-parallel-folding.md`](./moe-parallel-folding.md) | Parallel Folding 是「不同层用不同并行组映射」的正确一般化框架，BASP 是它在 attention 层 CP 维度上的一个特例 |
| [`tempo.md`](./tempo.md) | 结构同构：都是「收益取决于硬件带宽层次，且该层次正在被新一代 scale-up 域抹平」。TEMPO 把这件事显式做成相图并双向验证，BASP 没有 |

---

## 7. 术语

- **Ulysses-SP / DeepSpeed-Ulysses** → 沿序列维切分、用两次 all-to-all 在"序列切分"与"head 切分"之间来回转换的序列并行。通信量与设备数、序列长成正比地保持恒定。
- **Megatron-SP** → 只把 layernorm / dropout 等非 TP 层沿序列维切，通信量随序列长线性增长而与卡数无关。
- **α–β 模型** → `T = (P−1)(α + m·β)`，`α` 延迟、`β` 带宽倒数、`m` 每轮消息大小、`P` 参与 rank 数。
- **Head divisibility 约束** → Ulysses 的 SP 度不能超过 attention head 数（GQA 下是 KV head 数），这是它的主要扩展上限。

---

## 8. 复现清单

| 项 | 状态 |
|---|---|
| 代码 | ❌ 未声明释出（只说"修改开源 DeepSpeed"） |
| 硬件 | ✅ 明确：2 × 4× A100 40GB，NVLink + 400 Gb/s IB |
| 模型与配置 | ✅ Table II 给了 hidden size / head 数 / 层数；主实验 seq 16K、B=2、ZeRO-3、混合精度、30 次迭代平均 |
| 误差棒 | ❌ 无 |
| 绝对指标 | ❌ 无 MFU、无 tokens/s 绝对值，只有归一化步时 |
| 基线设定 | ⚠ Table I 的 `SP=N/B` 行疑似非等 batch 比较（§3） |
| 机制归因 | ⚠ "组变小"与"不出节点"在其硬件上不可分离 |
| 与其他 SP 方法对比 | ❌ Ring / Striped / USP / FlexSP 均只在 related work 提及 |
