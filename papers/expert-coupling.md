# MoE 预训练中的专家耦合:用相关性放置与 token 洗牌削减 All-to-All 开销
# Expert Coupling in MoE Pretraining: Reducing All-to-All Overhead with Correlated Placement and Token Shuffling

> **arXiv:** [2610.09372](https://arxiv.org/abs/2610.09372) (v1, 2026-10-07) · **PDF:** [pdf](https://arxiv.org/pdf/2610.09372)
> **发表信息:** arXiv preprint(论文未声明投稿会议;IEEE 风格双栏,Section I–VI + Appendix A/B)
> **机构:** Zyphra(Radha Gulhane 通讯 / 实现与全部实验、Quentin Anthony、Beren Millidge)
> **代码:** 未在论文中找到(全文无仓库链接;§A-A2 只说明"在 Megatron-LM 上实现")
> **领域:** MoE 训练系统 · 专家并行 · All-to-All 通信量削减 · 专家放置 · **AMD MI300X**
> **核心贡献:** 在 **8×AMD Instinct MI300X / 节点**的集群上实测 MoE 的 dispatch+combine 占 step 时间:EP8/top-2 **13.5%**、EP32/top-2 **44.8%**、EP8/top-6 **24.0%**、EP32/top-6 **60.4%**。发现预训练早期 router 已形成两类稳定相关性 —— 层内共选(top-2 下 8,128 个专家对中 0.8% 承载 42% 的 token,独立路由只会给 1.6%)与跨层可预测(层 8 的 128 个专家里 112 个把 ≥30% 的 token 送往层 9 同一个专家,中位 48%)。据此提出 **相关性专家放置 + 去重 dispatcher**(最多去掉 58% 的 dispatch 行)与 **token shuffling**(把 token 搬移融进 SP 的 reduce-scatter,单节点本地命中率 12.5%→59%)。合计 A2A 时间降 **1.16–2.63×**、端到端 step 降最多 **1.41×**,且**不改路由决策、不改专家参数、不加专家副本**。

> **与我们的关系:** 这是 2026-09-11~10-28 窗口里**唯一**把基线测在 8×MI300X/节点上的 MoE 通信论文 —— xGMI 节点内全互联 + 每 GPU 一张 100 Gb/s RoCE,硬件画像和我们一致,所以它的 A2A 占比曲线可以直接当参照系。**但对 Primus 而言两个机制里有一个半已经不构成增量:** ① 去重 dispatcher 我们走 `use_turbo_deepep` 时本就有(`is_token_in_rank` 是 `[num_tokens, num_ranks]` 的 bool,`layout.cu:101-103` 写的是 `(is_in_rank[j] > 0)`),论文自己也承认去重是 DeepEP 的已有技术;② token shuffling 硬性要求 `TP == EP`,而 Primus 293 份 recipe 里该条件为真的数量是 **0**(带 `EP>1` 的 109 份全是 `TP=1`),MegaMoE 路径更是直接 assert 禁止 TP。真正新的只有**相关性放置**,好在它的落地形态极便宜:`layout.cu:94` 与 `dispatch_prologue.py:471` 都硬编码 `rank = expert_id / experts_per_rank`,置换全局专家 ID 即可让现有 contiguous kernel 实现相关性放置,零 HIP 改动。**最大的警告是方向性的:** 论文自测在单节点 xGMI 上 top-6 时 token shuffling 的 A2A 收益(1.46×)**低于**只做放置(1.48×),机理是"集合通信时间由最忙的那对 GPU 决定而非聚合字节数" —— 我们的生产 MoE 恰以 EP≤8 单节点为主,论文最漂亮的那批数字(EP16/32/64 跨节点)落在我们主力场景之外。详见 [`../notes/primus-moe/2026-10-28_0546_industry-scan-update.md`](../notes/primus-moe/2026-10-28_0546_industry-scan-update.md) §3。

---

## 一、问题分析

### 1.1 研究背景

MoE 把 dense Transformer 的单个 FFN 换成 `E` 个专家 FFN 加一个轻量 router,每 token 只走 top-`k` 个专家,于是容量随 `E` 增长而每 token FLOPs 只随 `k` 增长。代价是通信:EP 把专家切到多 GPU 后,每个 MoE 层 forward 要跑 dispatch + combine 两次 all-to-all,backward 再跑一次,共四次。通信量同时随 EP 度与 top-`k` 增长,一旦 EP 组跨出单节点,较慢的节点间链路开始支配 step 时间。

§II-A 实测(Megatron-LM 的 `alltoall` dispatcher + contiguous 放置):

| A2A 占 step 比例 | EP8(1 节点) | EP32(4 节点) |
|---|---|---|
| top-2 | 13.5% | 44.8% |
| top-6 | 24.0% | 60.4% |

两个效应叠乘。**在 EP32 + top-6 下,这两个集合通信的耗时超过「专家计算 + 全部非 MoE 计算」之和。** 注意 A2A 时间的定义是 step 花在 dispatch/combine 调用内部的墙钟时间,**包含等待其他 GPU 的时间** —— 它把负载不均造成的等待也算进去了。

传统实现把专家按 contiguous 铺开(专家 `e` 落在 GPU `⌊e/(E/EP)⌋`),完全不管哪些专家被一起用。

### 1.2 与现有手段的区别

| 方法 | 机制 | 降的是什么 | 代价 |
|---|---|---|---|
| 专家复制(DeepSeek-V3) | 热专家多副本 | 峰值单链路流量 | 副本显存 |
| 负载驱动迁移(SmartMoE) | 按负载换专家所在 GPU | 峰值单链路流量 | 不降总量 |
| 动态冗余专家 + 权重预取(MoonEP / Kimi K3) | 动态规划冗余专家 | 峰值单链路流量 | 副本显存 |
| node-limited routing(DeepSeek-V3) | 限制 router 可选专家 | 节点间流量 | **改变路由决策** |
| Occult(ICML'25) | 层内共选放置 + 每 GPU 发一次;另可限制 token 的专家落在少数 GPU | 总量 | 后者**改路由**;只评推理与微调 |
| GRACE-MoE(2509.25041) | 层内共选分组 + 每 GPU 发一次 + 热专家复制 | 总量 | 组大小不等;副本显存;只评推理 |
| ExFlow(IPDPS'24) | 跨层专家亲和性共置 | 总量 | 只适用解码;**不能直接迁到预训练** |
| **本文** | 层内共选放置(等大小组、无副本)+ 去重 dispatcher + 跨层预测驱动的 token 搬移 | 总量 | 不改路由、不加副本、专家显存不变 |

作者对"降峰值"类方法的批评很具体:它们把流量摊到更多 GPU 上,**总流量基本不变**,因为跨 GPU 边界的 token-专家分配比例没变。本文改的正是这个比例 —— 论文称之为 **token-expert locality**:(token, 被选专家) 对中专家就在该 token 所在 GPU 上的那一部分。这些对永远不进网络。

ExFlow 不能迁到预训练的原因讲得很清楚:**自回归解码时 token 可以放在任何方便的 GPU 上;预训练时 token 必须待在持有它那段序列的 GPU 上(attention 需要),搬它需要额外通信。** token shuffling 就是补上这个搬移机制。

### 1.3 关键观测

本节统计量**全部离线计算**:在若干训练后期 step 上记录每 token 在每个 MoE 层的专家选择(每 step 131k token),在一部分 step 上拟合放置与预测器,在另一部分 step 的已记录路由上应用,**不跑 dispatcher**。图 2 用 step 1983–1995(每隔 4 步)共 524k token。

**观测一 —— 层内成对稳定共选**(层 8,top-2 模型):

| 统计量 | 数值 |
|---|---|
| 共选置信度 ≥50% 的专家对 | 14 对,承载全部 token 的 16% |
| 承载 42% token 所需的专家对 | 64 对(8,128 个可能对中的 **0.8%**) |
| 独立路由在同样负载下的对应比例 | 1.6% |
| 有一个"伙伴"占其 token ≥30% 的专家数 | 128 个中的 73 个 |

**观测二 —— 跨层可预测**(层 8→9,top-2 模型):

| 统计量 | 数值 |
|---|---|
| 把 ≥30% token 送往层 9 某一特定专家的层 8 专家数 | 128 个中的 **112** 个 |
| 该"最大占比"的中位数 | **48%** |

**表 I(论文原表,离线统计,held-out iteration 1983/1987,拟合在 1991–1999,除 GPUs 列外单位均为 %)。** `Same GPU` = top-`k` 专家全在同一 GPU 的 token 占比;`GPUs` = 每 token 的不同目的地数;`Dedup` = 被去重掉的行占比;后两列 = 该 token 下一层专家的本地命中率,分别对应"被搬到预测 GPU"与"留在 home GPU(本地率 `1/EP`)"。

| 路由 | EP | 放置 | Same GPU | GPUs | Dedup | 下一层本地率(搬移) | 下一层本地率(不搬) |
|---|---|---|---|---|---|---|---|
| top-2 | 8 | contiguous | 12 | 1.88 | 6 | 33 | 13 |
| top-2 | 8 | **correlated** | 53 | 1.47 | 26 | 58 | 13 |
| top-2 | 16 | contiguous | 6 | 1.95 | 3 | 28 | 6 |
| top-2 | 16 | **correlated** | 45 | 1.55 | 23 | 51 | 6 |
| top-2 | 32 | contiguous | 2 | 1.98 | 1 | 25 | 3 |
| top-2 | 32 | **correlated** | 36 | 1.64 | 18 | 42 | 3 |
| top-6 | 8 | contiguous | 0 | 4.49 | 25 | 25 | 13 |
| top-6 | 8 | **correlated** | 18 | 2.54 | 58 | 60 | 13 |
| top-6 | 16 | contiguous | 0 | 5.24 | 13 | 18 | 6 |
| top-6 | 16 | **correlated** | 8 | 3.08 | 49 | 49 | 6 |
| top-6 | 32 | contiguous | 0 | 5.67 | 6 | 14 | 3 |
| top-6 | 32 | **correlated** | 0 | 3.89 | 35 | 33 | 3 |

正文给了比表格更精确的一个数:EP32 下一个 GPU 只持 4 个专家,top-6 的 token 的专家**不可能**全落在一个 GPU 上(故 `Same GPU`=0),但被去重的行仍从 **5.5% 升到 35.1%**。

放置对 token shuffling 有放大作用:**contiguous 放置下被搬移的 token 在 EP8 只有 33% 的下一层专家本地**(它的下一层专家散在更多 GPU 上),相关性放置下才是 58%。

---

## 二、方法

### 2.1 去重 dispatcher(这一半是已有技术)

常规 top-`k` dispatch 对每个 (token, 专家) 分配发一份 hidden state 拷贝,**即使若干被选专家住在同一个 EP rank 上**。改成每个唯一的 (token, 目的 rank) 对发一行:

- 每行**携带该 token 在该 rank 上的专家 ID 与路由概率**作为元数据
- 接收 rank 用元数据给自己每个被选专家复制一份 token 并按专家分组 —— **因此不需要为元数据单开一个集合通信**
- combine 方向对称:专家 rank 用路由概率给该 token 的各专家输出加权求和,**每个 (token, rank) 对只回一行部分输出**

论文**明确标注去重是已有技术**(引 DeepEP、Zhao et al. 2025),他们只是在 Megatron-LM 里实现它作为相关性放置的基座 —— 因为 Megatron-LM 的 `alltoall` dispatcher 是每个被选专家发一份拷贝。

### 2.2 相关性专家放置(本文新增的部分)

对每个 MoE 层建相关图 `W`,`W[e,e']` = 同时选中专家 `e` 与 `e'` 的 token 数。目标:**找一个让每个 GPU 持有相同数量专家、且最大化「同 GPU 专家对的 `W[e,e']` 总权重」的放置**。

求解器(Algorithm 1):

- **Kernighan–Lin 风格局部搜索**(引 Kernighan & Lin 1970),不是图划分库、不是 ILP
- 从 contiguous 起步,反复取"组内 `W` 增益最大的那次成对交换",直到无正增益或用完 `I_max` 次交换预算
- **因为每一步都是 swap,每个 rank 始终恰好持有 `E/EP` 个专家** —— 专家参数显存一字不增
- 跨多节点时**两级划分**:先把专家划到节点(针对节点间流量),再把每节点专家划到它的 `P/N` 个 GPU(针对 GPU 间流量)

**落地形态就是一个 per-layer 的全局专家 ID 置换**,不改 router 决策也不改专家参数。§A-A2(ii) 说 checkpoint 的专家权重按**全局专家 ID** 索引,所以同一个 checkpoint 可以用不同放置加载。

### 2.3 共选统计的采集:何时、多贵、在线还是离线

- **离线,每个模型算一次**,通过 **router 里的一个 capture hook** 记录路由决策(§A-A2)
- **数据量极小:几千 token 就够。** 用 1,024 token 拟合的表在 top-2 EP8 已能去掉 23.6% 的 dispatch 行、top-6 下 57.9%。用 4,096 token 时拟合矩阵与 held-out 矩阵的 **Pearson 相关 top-2 为 0.93、top-6 为 0.98**,结果与用全部 524k token 相差 1 个百分点内。作者结论:**一个 rank 的一个 microbatch 就够**,因为表刻画的是训练好的 router 的性质
- **结构形成得早且稳定。** 以 step 1999 评估:step 199 的表已给约一半收益(top-2 EP8 去重 14.4% vs 最终 26.5%;top-6 为 42.7% vs 57.7%);step 1000 的表距最终值 2–3 个百分点;step 1400 起几乎等同最终表。相关度从 step 199 的 0.4–0.5 升到 step 1400 起的 >0.93。预测器同步变好:top-2 EP8 命中率在相关性成形前是 12.7%(随机水平 12.5%),step 1000 的表到 54%,step 1800 的表到 57%
- **重算频率:** 2000-step 跑法下在后半段拟合一次就够;对更长的跑法作者**预期**约每 1000 step 重拟合一次足够(原文是"预期",非实测)

### 2.4 Token Shuffling

#### 为什么需要它

专家放置只决定专家住哪儿,不改 token 住哪儿。TP–EP 下 MoE 层跑在 attention 之后 reduce-scatter 分给每 rank 的序列分片上 —— **每个 rank 拥有一段固定连续的序列切片,token 的 rank 完全由位置决定**。于是即便一个 token 需要的专家已被放到同一 GPU 上,**该 token 真的就在那个 GPU 上的概率只有 `1/EP`**。

#### 融进了哪两个集合通信(核心机制)

要求 **TP 组与 EP 组是同一批 rank**。TP–EP 下本来就有两个搬动每个 token 的集合通信:

| 原集合通信 | 位置 | 洗牌后变成 |
|---|---|---|
| SP **reduce-scatter** | attention output projection **之后** | **shuffled reduce-scatter**:把每个 token 发给它的 planned owner rank 而非默认连续布局里持有它的那个 rank;**同时搬该 token 的 residual stream 那一行**,使 residual 加法发生在新 owner 上 |
| SP **all-gather** | 下一个 block 的 QKV projection **之前** | **shuffle-aware all-gather**:恢复原始 token 顺序(attention 依赖顺序) |

关键论证:**reduce-scatter 中"哪个 rank 收哪个 token"由索引决定,改索引不改发送字节数。** 所以置换融进去之后 **token shuffling 不新增任何集合通信**。RS 侧实现为"一个融合的 permute + residual 加法 kernel"。两个融合置换每 step 多花"几毫秒",相比省下的"数百毫秒" A2A。规范 token 顺序在 **attention、输出层、loss 周围**都被恢复(§A-A2 iii)。

#### Owner 预测器(Algorithm 2)

- **输入是前两个 MoE 层的路由** —— token 在 `m-1` 与 `m-2` 用过的专家对是落点的好预测量
- **离线**统计每个专家对把 token 送到各 rank 的频率:`F_m[e,r]`(一阶)、`S_m[e',e,r]`(二阶)、支撑计数 `n_m[e',e]`,可选再混一张按 token id 条件的 `Q_m[tok,r]`
- **back-off 混合**:`d_t = (1/k²) Σ_{e∈S_{m-1}(t)} Σ_{e'∈S_{m-2}(t)} [(1-w) F_m[e,·] + w S_m[e',e,·]]`,其中 `w_{e'e} = min(n_m[e',e]/(n_m[e',e]+β), w_max)`。**某个专家对观测越少,就越把权重让给一阶表**
- **运行时唯一输入是前几层的路由**,planner 不新增集合通信 —— 它需要的 route ID 被塞进 dispatcher 本来就有的 metadata all-gather 里
- **容量约束下的贪心分配**:SP 要求每 rank 恰好 `T/P` 个 token,所以**按置信度降序**(置信度 = 最大项与次大项之差 `d_t^(1) - d_t^(2)`)依次把 token 分给"还有余额的最偏好 rank"。**因此不是每个 token 都能拿到它的预测 GPU**

#### 把 planner 藏到 attention 后面

论文标注为"关键"的一步:**预测 + 配额分配在 TP16 EP16 下约等于 forward all-to-all 时间的三分之一**,串行执行会吃掉大部分收益。但计划只依赖到第 `i` 层的路由、只在第 `i+1` 层的 reduce-scatter 处才需要,所以 planner 跑在**侧流**上,与第 `i` 层的专家计算及随后的 attention 并发,用一次 event wait 汇合。实测主流每 step 只等约 **1 毫秒**(step 的 **0.02%**)。

### 2.5 约束清单

| 约束 | 相关性放置 + 去重 | Token shuffling |
|---|---|---|
| 适用布局 | **任何使用 EP 的布局** | **仅 `TP == EP`** 且用 TP + SP(论文评测 TP8 EP8、TP16 EP16) |
| TP1 布局 | 适用 | **明确不适用**(原文 "It does not apply to layouts without tensor parallelism (TP1)") |
| 专家是否张量并行 | 论文设定里**专家不做 TP** | 同左 |
| sequence parallelism | 不要求 | **必需**,并带来"每 rank 恰好 `T/P` 个 token"的硬配额 |
| 专家参数显存 | **零增量**(每步都是 swap,无副本) | 未在论文中找到(只给了时间开销,未报告预测表显存) |
| 是否改路由决策 / 专家参数 | 否 / 否 | 否 / 否 |
| checkpoint 兼容 | **是**(权重按全局专家 ID 索引) | 未在论文中找到 |
| 额外集合通信 | 无(元数据随激活同行) | **无**(置换融进已有 RS/AG) |
| 逐位等价 | **论文未讨论。** 原文只说 "All variants train the same model and produce the same loss"。但去重改变了求和顺序 —— 基线在 home rank 汇总 token 的 `k` 个专家输出,去重版由专家 rank 先加权求和再每 rank 回一行 —— **BF16 下不应预期逐位一致**。论文未声明逐位等价,也未见任何数值等价性验证 | 同左:residual 加法被搬到新 owner 执行,论文未讨论数值影响 |

---

## 三、实验

### 3.1 模型与预训练配置(§VI + 表 II)

| 项 | 值 |
|---|---|
| 层数 | **12,每层都是 MoE 层** |
| `d_model` | **4096** |
| attention heads / head dim / attention 宽度 | 16 / 32 / **512**(远小于 `d_model`) |
| 每层专家数 | **128,无 shared expert** |
| 每 token 专家数 | **2 或 6**(两个独立训练的模型) |
| 专家 hidden size `d_ff` | **1024**(`4096 → 1024 → 4096`,GELU) |
| 路由 | top-`k` of `E`,**dropless** |
| router | FP32 linear gate;对全部 `E` 个专家 softmax 后取 top-k;gate 权重 = 被选中的 softmax 概率 |
| 负载均衡 | **sequence-level auxiliary loss,系数 `10^-2`** |
| Normalization / 位置编码 | pre-LayerNorm(`ε=10^-5`)/ 学习式绝对位置编码 |
| 词表 | 151,669(Qwen3 tokenizer,pad 到 151,680);输入输出 embedding **tied** |
| 总参 / 专家参 / 每 token 激活参 | **13.6B**(13.0B transformer + 0.6B embedding)/ **12.9B** / top-2 ≈0.31B、top-6 ≈0.71B |
| 数据 | **FineWeb-Edu**,98/2 train/val |
| 序列长度 / global batch | **8192** / **128 序列**(≈1.05M token/step) |
| 训练量 | **2000 step / ≈2.1B token** |
| 优化器 | AdamW,`β=(0.9, 0.95)`,wd 0.1,grad clip 1.0 |
| 学习率 | 峰值 `10^-5`,200 step warm-up,cosine 衰减到 `10^-6` |
| 初始化 / 精度 | `σ=0.008` 正态 / BF16,router 与 optimizer state 为 FP32 |
| **预训练时的并行** | **TP / EP = 1 / 8,单个 8-GPU 节点,micro-batch 2** |

### 3.2 硬件与测量协议

| 项 | 值 |
|---|---|
| 节点 | 每节点 **8× AMD Instinct MI300X** |
| 节点内互联 | **xGMI,节点内全互联**(每对 GPU 一条直连链路) |
| 节点间互联 | **每节点 8 张 100 Gb/s RoCE NIC,每 GPU 一张** |
| EP↔节点映射 | EP8 = 1 节点、EP16 = 2 节点、EP32 = 4 节点、EP64 = 8 节点 |
| 测的是什么权重 | **加载各自模型的 step-2000 checkpoint**,跑同语料 held-out 数据 |
| 测量协议 | 每实验跑两次:第一次 14 个干净 step,**丢前 4 个 warm-up,报后 10 个的中位数**;第二次同样 14 step 但开 region profiler 与 A2A metrics |
| baseline | **Megatron-LM 的 `alltoall` dispatcher + contiguous 放置** |
| 配置标记 | B(baseline)、D(去重)、P(去重+相关性放置)、S(再加 token shuffling,仅 `TP==EP`) |
| 实现 | **Megatron-LM**;router、专家计算、optimizer、训练循环均不改 |

**所有性能数字都是端到端 step 级测量**(step 时间、step 内 A2A 墙钟时间、每 rank A2A 通信量),**不是 per-layer**。表 I 是唯一例外(离线统计),图 2 的两张热图是层 8 与层 8→9 的单层数据。

### 3.3 通信量(§VI-A,仅 EP,TP1)

contiguous 放置下**去重只能去掉碰巧落到同一 rank 的重复**:EP8 下 top-2 去 6%、top-6 去 25%;EP16 下 3% 与 12%。

加上相关性放置后,相对 baseline 的通信量下降:

| 路由 | EP8 | EP16 | EP32 | EP64 |
|---|---|---|---|---|
| top-2 | 1.36× | 1.29× | 1.22× | 1.12× |
| top-6 | 2.36× | 1.95× | 1.54× | 1.24× |

**收益随 top-`k` 增大而变大**(token 的专家中有更多可能共享一个 rank),**随 EP 变宽而衰减**(每 rank 持有的专家变少)。另一个结构性事实:**dispatch 与 combine 按构造携带完全相同的字节数**,所以字节下降同等适用于 backward。

### 3.4 A2A 时间(§VI-A)

| 路由 | EP8(节点内 xGMI) | EP16 | EP32 | EP64 |
|---|---|---|---|---|
| top-2 | **1.16×** | 1.34× | 1.38× | 1.34× |
| top-6 | **1.58×** | 1.95× | 1.48× | 1.35× |

下降出现在全部四个集合通信里。注意 **EP8(节点内)的时间收益明显小于跨节点**,尤其 top-2 只有 1.16×,而其字节下降是 1.36×。

### 3.5 端到端 step 时间(§VI-A,仅 EP,TP1)

非 MoE 计算在所有配置下几乎不变,所以 step 加速完全取决于省了多少 A2A 时间以及 A2A 占 step 多大比例。

| 路由 | A2A 占 step 范围 | EP8 | EP16 | EP32 | EP64 |
|---|---|---|---|---|---|
| top-2 | 13–52% | 1.05× | 1.13× | 1.14× | 1.11× |
| top-6 | 24–60% | 1.14× | **1.41×** | 1.25× | 1.23× |

全文最大的 step 加速 **1.41× 出现在 top-6 EP16(2 节点)**,不是最大 EP。

### 3.6 Token shuffling(§VI-C,仅 TP8 EP8 与 TP16 EP16)

token-expert 本地命中率(无洗牌时为 `1/EP`):

| 路由 | TP8 EP8 | TP16 EP16 |
|---|---|---|
| top-2 | 12.5% → **59%** | 6.3% → **53%** |
| top-6 | 12.5% → **57%** | 6.3% → **46%** |

注意措辞:**dispatcher 仍然把这些行计入,只是它们留在 GPU 上、不过网。**

过网字节下降(相对 baseline):

| 路由 | TP8 EP8 洗牌 | TP16 EP16 洗牌 | TP8 EP8 仅放置 | TP16 EP16 仅放置 |
|---|---|---|---|---|
| top-2 | 2.4× | 2.1× | 1.4× | 1.3× |
| top-6 | 3.1× | 2.4× | 2.4× | 1.9× |

A2A 时间:

| 路由 | TP8 EP8 | TP16 EP16(仅放置 → 加洗牌) |
|---|---|---|
| top-2 | **1.40×** | 1.55× → **1.74×** |
| top-6 | **1.46×**(仅放置为 **1.48×**) | 1.95× → **2.63×** |

端到端 step 时间(`TP==EP` 下 TP 的集合通信抬高了非 MoE 部分,A2A 占 step 更小:top-2 为 9–21%,top-6 为 18–38%):

| 路由 | TP8 EP8 | TP16 EP16 |
|---|---|---|
| top-2 | 1.01–1.06× 区间内 | 1.03×(D) → 1.05×(P) → **1.06×(S)** |
| top-6 | **1.08×** | 1.28×(P) → **1.34×(S)** |

### 3.7 Router 消融(§A-B):三种负载均衡方式

额外训了两个只在均衡方式上不同的 top-6 模型:

| 均衡方式 | max/mean 负载比 | 最终 loss | EP8 去重率(相关性放置) | EP16 去重率 | TP1 EP16 step 加速 |
|---|---|---|---|---|---|
| 主配方:sequence aux loss `10^-2` | 1.52 | 4.46 | **58%** | **49%** | **1.41×** |
| 10× 更强 aux loss `10^-1` | **1.20**(最均衡) | 4.64(最差) | 51% | 40% | 1.30× |
| DeepSeek-V3 式 aux-loss-free(sigmoid 打分 + 每专家 bias,更新率 `10^-3`,加 `10^-4` sequence-level loss) | 1.37 | **4.39**(最好) | 47% | 36% | 1.26× |
| (对照)contiguous 放置 | — | — | 约 25% | 约 13% | — |

三个 router 下相关性都存在但强度不同,在每个 EP16 布局上加速都 ≥1.20×。**关键解读:主模型去重率更高(49% vs 40%)但加速优势相对更小(1.41× vs 1.30×),因为它的负载比更高(1.52 vs 1.20)—— all-to-all 必须等最重的那个 rank。** 因此结论是:**相关性放置 + 去重打的是「通信量」,不是「通信倾斜」;降倾斜的方法(专家迁移/复制、EPLB 一类)打的是另一个来源,两者互补。**

---

## 四、局限与适用边界

逐条记录论文自己说的局限与负面结果:

1. **token shuffling 在 TP1 下完全不适用。** 原文直接写 "It does not apply to layouts without tensor parallelism (TP1)"。
2. **单节点 top-6 下 token shuffling 的 A2A 时间收益低于只做放置:1.46× vs 1.48×。** 机理 —— **在 xGMI 上集合通信时间由最忙的那对 GPU 决定,而不是跟随聚合字节数;把一个 token 的多行集中到同一个 owner 上,在降低总量的同时抬高了最忙链路上的流量。** 这是全文最重要的负面结果,而且正好发生在单节点场景。
3. **单节点 TP8 EP8 上普遍"字节降得多、时间降得少"**(原文:"token shuffling reduces bytes by a similar or larger factor, but reduces the time cost by a smaller factor")。
4. **EP32 + top-6 下 `Same GPU` 恒为 0**:一个 GPU 只持 4 个专家,top-6 token 的专家不可能全本地。放置的收益只能来自"更少的目的地数 + 更多可合并行"。
5. **step 加速被 A2A 占比封顶。** `TP==EP` 布局上 top-2 的 step 加速只有 1.01–1.06× —— **A2A 时间降了 1.74× 也只换来 1.06× 的 step**。
6. **相关性是"训练到一定程度"才有的。** step 199 的表只给约一半收益;预测器在相关性成形前命中率 12.7%,和随机水平 12.5% 几无差别。**对训练最早期(<1B token)没用。**
7. **放置不解决负载倾斜**(见 §3.7),与 EPLB 类方法正交,不能互相替代。
8. **ExFlow 式跨层亲和性不能直接迁到预训练**,这是需要造 token shuffling 的原因,也意味着 token shuffling 的全部复杂度(planner、侧流、配额、residual 搬移、顺序恢复)都是预训练特有的必要代价。
9. **Occult 式"把 token 的专家限制在少数 GPU 上"能换更多节省但会改路由**,本文明确拒绝。
10. **专家不做张量并行。**
11. **评测规模很小**:一个 12 层模型、两种 top-`k`、2.1B token。作者原话是"我们**相信**专家相关性在不同模型族与规模上都成立" —— 这是信念,不是证据。
12. **context parallelism 与 EP 推理留给未来工作**,未评测。
13. **planner 裸开销不小**:TP16 EP16 下约等于 forward A2A 时间的 1/3,必须靠侧流 + 与 attention/专家计算重叠才能藏住。没有这个重叠窗口的实现会亏掉大部分收益。
14. 未在论文中找到:代码链接;预测表显存占用;逐位等价性验证;MI300X 以外硬件的数据;PP/CP 组合下的行为;`I_max`、`β`、`w_max` 的取值。

---

## 五、对 Primus 的可移植性判定

详细代码审计见 [`../notes/primus-moe/2026-10-28_0546_industry-scan-update.md`](../notes/primus-moe/2026-10-28_0546_industry-scan-update.md) §3。摘要:

| 机制 | 对 Primus | 判定依据 |
|---|---|---|
| **去重 dispatcher** | **已经有了**(走 DeepEP 时) | `use_turbo_deepep: true` 下 DeepEP 的 dispatch layout 以 `is_token_in_rank: [num_tokens, num_ranks]` 的 bool 表达"token 是否发往该 rank";kernel 在 `third_party/Turbo/csrc/kernels/deep_ep/layout.cu` 里对每 rank 累加 `(is_in_rank[j] > 0)` —— 一个 token 在同一 rank 选中 3 个专家只计 **1** 行。论文也承认去重是 DeepEP 的技术 |
| **相关性专家放置** | **可用,且是唯一真正的增量** | 适用于任何用 EP 的布局,不要求 TP>1 / SP,专家显存零增量,不改路由。落地极便宜:`layout.cu:94` 的 `rank_idx = expert_idx / num_expert_per_rank` 与 `dispatch_prologue.py:471` 的 `destination_rank = expert_id // experts_per_rank` 都硬编码 contiguous,**按论文做法置换全局专家 ID 即可,零 HIP 改动** |
| **Token shuffling** | **不可用** | 硬性要求 `TP == EP`。Primus 293 份声明 `expert_model_parallel_size` 的 recipe 里 `TP == EP > 1` 为 **0**;带 `EP>1` 的 109 份全是 `TP=1`。MegaMoE 路径更有 `assert config.tensor_model_parallel_size == 1, "MegaMoE is EP-only (TP==1)"` |

**统计采集的落点:** `primus/backends/megatron/core/transformer/moe/router.py:74-77` 已有一个逐 microbatch 的 hook,但累的是 `routing_map.sum(dim=0)` 这个**边际**计数,不是论文要的成对共选矩阵 `routing_map.T @ routing_map`。好在 `routing_map`(bool `[tokens, experts]`)就在那一行的作用域里,差一个 matmul。转储骨架可以克隆 `primus/backends/megatron/patches/kimi_k3_expert_load_probe_patches.py`(config 门控、逐 step 逐层写 JSONL)。

**收益的现实预期(基于论文自己的数字,不外推):**

- 我们的生产 MoE 主力是 **EP8 单节点**。论文在 EP8/TP1 的 step 加速是 **top-2 1.05×、top-6 1.14×**,A2A 时间收益是 **top-2 1.16×、top-6 1.58×**。我们旗舰 recipe 的 top-`k` 多为 4–8,落在偏 top-6 一侧
- 论文最亮眼的 1.41× 出现在 **top-6 EP16(2 节点)**,我们的 recipe 默认全是 EP8 单节点 —— **这个数字我们拿不到**
- **而且我们是从已去重的基线出发**,所以增量只是 6%→26%(top-2 EP8)、25%→58%(top-6 EP8) 这一段,不是 0%→26%/58%
- §4 第 2 条那个单节点负面结果(xGMI 上集合时间由最忙 GPU 对决定)是 Infinity Fabric 的性质而非 CDNA3 特有,**应预期在 MI355X 上同样成立**
- MI355X 还会把比例进一步推向不利:CDNA4 抬高了专家 GEMM 吞吐,我们的 recipe 又常跑 FP8/MXFP8,进一步压缩 step 里的计算侧 —— **A2A 占比更小意味着同样的相对 A2A 削减换来的 step 收益更小**

**诚实的量级估计:** EP8 单节点 MI355X、top-4~8、对已去重的 DeepEP 基线,A2A 时间收益显著低于论文的 1.58×,step 收益在**个位数百分比低位**。这篇论文真正能为我们付钱的配置是**多节点 EP(EP16/EP32)**,而那不是我们今天 recipe 的样子。

---

## 相关文件

- 业界扫描与代码审计:[`../notes/primus-moe/2026-10-28_0546_industry-scan-update.md`](../notes/primus-moe/2026-10-28_0546_industry-scan-update.md)
- EP 负载均衡选型(UltraEP / MoonEP):[`../notes/primus-moe/2026-09-07_0900_moonep-vs-ultraep-port-selection.md`](../notes/primus-moe/2026-09-07_0900_moonep-vs-ultraep-port-selection.md)
- 移植模式四分法:[`../notes/primus-moe/2026-09-11_1541_industry_opt_port_to_rocm_primus.md`](../notes/primus-moe/2026-09-11_1541_industry_opt_port_to_rocm_primus.md)
- 「重叠不免费」的三次独立证伪:同上 §4.3(Tessera / Motif-3 / Instella-MoE),本文 §4 第 2 条是第四次
