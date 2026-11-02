# MegaMoE 论文立项:会议窗口、重构框架、对照实验方案

> **后续**: 用户已定向 —— 不考虑 blog 发布、放弃 MLSys 2027 那个两天窗口,目标改为下一班 MLSys。正式写作计划见 [`2026-10-28_2115_megamoe-paper-mlsys-plan.md`](./2026-10-28_2115_megamoe-paper-mlsys-plan.md)。**本份保留为会议普查 + 三个评审硬伤 + 对照实验方案的底稿**,其中 §0(blog 发布状态)与 §4(两天窗口的取舍)已不再是决策项。

> **When**: 2026-10-28 20:50 UTC+8(21:05 更正 blog 发布状态)
> **Where**: 桌面调研。会议 deadline 当日从官方 CFP 页抓取;代码事实来自本地只读检查 `/perf_apps/xiaoming/3rd/Triton-distributed`(checkout `8260bc3`, 2026-08-06)与 `/perf_apps/xiaoming/rocm-blog/.../primus-mega-moe/README.md`(561 行)
> **Blog 发布状态(已核实,很关键)**: **未发布。** 只存在于内部仓 `AMD-ROCm-Internal/rocm-blogs` 的 feature 分支 `xiaompen_amdeng-primus-mega-moe`(HEAD `a932f51f`,相对 `origin/develop` 领先 43 个 commit),`blogs/software-tools-optimization/primus-mega-moe/` **不在 `origin/develop` 上**。frontmatter 里的 `date: "15 Sep 2026"` 是草稿的计划日期,不是发布日期。**同一结论适用于 `primus-moe-optimization`(分支 `xiaompen_amdeng-primus-moe-optimize`,同样不在 develop 上);只有 `primus-deepseek-v4` 确实已并入 `origin/develop`。**
> **Context**: 评估把 MegaMoE blog 升级成论文。前置判断见本次会话:**blog 自己把 core structure 归给了 MegaScale-MoE(EuroSys'26)与 DeepGEMM(文件级溯源),所以不能按「AMD 上的 MegaMoE 实现」投**

## TL;DR

0. **blog 未发布 ⇒ 双盲完全不是问题,但「先发 blog 还是先投论文」变成了一个必须现在做的排序决定。** 见 §0。
1. **窗口只有两个是活的:MLSys'27 的 deadline 是 2026-10-30 20:00 UTC(后天),OSDI'27 摘要注册 2026-12-01。** EuroSys'27、ASPLOS'27、PPoPP'27 三个的全部 cycle 都已关闭。
2. **MLSys'27 有 Industrial Track,原文明写 "need NOT present new ideas or novel solutions"** —— 这是现有 blog 材料唯一能在两天内合法投出去的地方,但两天内三个评审硬伤一个都补不了。
3. **这其实是两篇不同的论文**,不是一篇论文的两个 deadline:MLSys 工业轨 = 把已建成系统的设计方法学与详细基准写清楚(blog 几乎就是);OSDI = 以「框架决定融合上界」为 thesis 的研究论文,需要补三组实验。
4. **Triton-distributed 的对照实验跑得了,而且结果大概率对我们有利 —— 但它也顺手废掉了一个我们本来以为是新颖点的东西。** 本地 checkout 有 `kernels/amd/ep_all2all_fused.py`(1,267 行)+ `test/amd/test_ep_moe_fused.py`(347 行),`function/amd/ep_moe_fused.py:80-82` 是 `raise NotImplementedError("backward is not implemented")` —— **它只有前向**,于是「前反向共享一套通信原语」从工程细节升格为唯一可比项。但它的前向切分 docstring 写的是 `dispatch + GEMM-1 -> activation -> GEMM-2 + combine`,**和 MegaMoE 的两算子切分完全一样且已在 AMD 上开源**,所以这个切分本身不能当贡献写。

---

## 0. blog 未发布所改变的事

初稿曾按"blog 已于 2026-09-15 发表"来写,**这是错的**。核实结果见文首。影响分三类。

### 0.1 变好的(两条,都不小)

| 原顾虑 | 现状 |
|---|---|
| **双盲自相矛盾** —— 一篇挂 8 位作者名、带 AMD 品牌、搜 "MegaMoE fused grouped GEMM" 必然命中的公开 blog,投双盲会议时匿名化只是走过场;投工业轨尤其尴尬,因为工业轨的价值前提就是"这是某家的真实生产系统" | **不成立。** 没有任何公开物料,匿名化是真实可做的,不需要在"诚实"和"合规"之间选 |
| **自引构成在先技术(prior art by self)** —— 我原以为同系列的 `primus-moe-optimization`(层级/step 级,1.40× BF16 / 1.81× MXFP8)已于 08-21 发表,会压缩 MegaMoE 论文可声称的新颖性边界 | **也不成立。** 它同样未并入 `develop`。**唯一已发布的是 `primus-deepseek-v4`**(四节点 12.0%),而那篇是端到端训练结果,不触及融合算子的机制,基本不构成在先技术 |

### 0.2 没变的(重要 —— 内容侧的判断一个都不动摇)

发布状态只影响"程序",不影响"证据"。以下全部照旧:

- core structure 跟随 MegaScale-MoE(EuroSys'26)、`barrier.py` / `symm_buffer.py` 有 DeepGEMM 的文件级溯源 —— 这是**工作本身**的属性,和 blog 发没发布无关;
- Triton-distributed 的 AMD 路径已开源且前向切分与我们一字不差(§3.1)—— 别人的在先技术不因我们没发 blog 而消失;
- §3 的三个评审硬伤(无同硬件前作对比 / 配置窄且两个设定偏向自己 / liveness 是经验非证明)一条不少。

### 0.3 新出现的问题:发布与投稿的排序冲突

这是之前没有的。blog 一旦并入 `develop` 并上线,§0.1 第一行的顾虑就**原样回来**。所以现在必须决定:

| 排序 | 后果 |
|---|---|
| **先投稿、后发 blog** | 双盲干净。代价是 blog 压在分支上继续等(已经领先 develop 43 个 commit),而 blog 的业务价值本来就有时效性 |
| **先发 blog、后投稿** | blog 该上线上线。代价是之后投任何双盲会议都回到"匿名化走过场"的状态 —— MLSys 明确允许 arXiv 预印,说明它对"内容已公开"本身宽容,但**公开物料带作者名和公司名**是另一回事 |
| **同时** | 最糟。投出去之后几天内 blog 上线,等于主动把双盲作废,还留下时间戳 |

**附带提醒(我无法核实,但必须你去确认):** blog 还没过完内部 review(分支上最近两个 commit 还在 "fix review findings"、"Apply review polish"),意味着这批数据和表述**尚未获得对外发布许可**。把同一批数字写进外投论文,走的大概率是另一条审批路径(AMD 的 external publication approval)。这条不解决,deadline 讨论都是空的。

---

## 1. 会议窗口(2026-10-28 当日核实)

| 会议 | 下一个 deadline | 距今 | 来源可信度 | 适配度 |
|---|---|---|---|---|
| **MLSys 2027** | **2026-10-30 20:00 UTC**(研究轨 + **工业轨**同一天) | **2 天** | 官方 CFP + Dates 页 | **高** —— topics 含 "Efficient model training"、"ML compilers and runtimes"、"Specialized hardware for machine learning" |
| **OSDI 2027** | **2026-12-01 摘要注册** | ~5 周 | 官方(经 CFP tracker 交叉核对) | 中高 —— 需要强 thesis,纯 kernel 工作会被嫌弃 |
| SIGCOMM 2027 | 2027-02-02 主会 | ~3 月 | 官方 | 中 —— **有先例**:AcclEP/EPIC 走的就是这里,通信中心的 EP 框架能被接受 |
| HPDC 2027 | 预计 2027-02 初 | ~3 月 | 从 HPDC'26(2026-02-05)外推,**未核实** | 中 |
| ATC 2027 | 预计 2027-01 中 | ~2.5 月 | 从 ATC'26(2026-01-13)外推,**未核实** | 中 |
| SC 2027 | 预计 2027-04-01 技术论文 | ~5 月 | 从 SC26(摘要 04-01 / 全文 04-08)外推,**未核实** | 中高 —— HPC 口径对 8 卡单节点宽容度更高 |
| EuroSys 2027 | **两个 cycle 全关**(Spring 2026-05-14 / Fall 2026-09-24) | — | 官方 | 下一班车 EuroSys 2028 |
| ASPLOS 2027 | **两个 cycle 全关**(2026-04-15 / 2026-09-09) | — | 官方 | 下一班车 ASPLOS 2028 ≈ 2027-04 |
| PPoPP 2027 | **已关**(2026-08-03) | — | 官方 | PPoPP 2028 预计 2027-08,未核实 |
| ISCA 2027 | 2026-11-09 摘要 | ~2 周 | 官方 | **低** —— 体系结构口径,软件栈工作不对路 |

**MLSys'27 工业轨原文**(值得逐字读):

> MLSys'27 also solicits papers that describe the design and implementation of large-scale ML systems in industry. … We welcome submissions that challenge or reinforce existing solutions, provide deeper insights into known problems, or rigorously validate published techniques at scale. **Industrial track submissions need NOT present new ideas or novel solutions to be accepted, but they should present design methodology and detailed benchmarks for large-scale ML systems that were built.**

两条程序性约束:双盲("good-faith effort to anonymize"),但**允许 arXiv 等公开论坛预印**;格式沿用 `mlsys2025style`,双栏。

**双盲在这里不是障碍**(blog 未发布,见 §0.1),但**前提是 blog 不在投稿前后上线**(§0.3)。工业轨仍需按其要求处理匿名化;真正的卡点不在双盲,而在 §0.3 末尾那条对外发布审批。

---

## 2. 这是两篇论文,不是一篇的两个档期

### 2.1 Paper A — MLSys'27 工业轨(两天,材料已有 ~85%)

**定位:** 把一个已部署的融合 MoE 训练算子的设计方法学与详细基准写清楚。**不声称机制新颖性。**

能直接复用 blog 的部分(561 行里约 85%):

| 论文章节 | 来源 | 需要改什么 |
|---|---|---|
| Motivation | blog Background(MI355X 自测:A2A 占串行时间 46%;前向 dispatch 41% 暴露) | 几乎不动。补一句:为什么低精度不会自动解决(blog 已有) |
| Design:两个融合算子 | blog "Two Fused Operators" + Figure 1 | 不动 |
| Design:三机制 + 一致性 | blog "How the Overlap Happens"(角色特化 / 三张 ready-signal 表 / 无 D2H / relaxed + scope bits)+ Figure 2 | 不动 |
| Design:低精度 | blog "Low Precision Inside a Fused Operator" | 不动 |
| **Methodology** | blog 的 η 定义、双基线、等效 FLOP 稠密 GEMM 反证分母 | **提升为一等贡献**,见 §2.3 |
| Evaluation | blog 的 15 stage 表 + 6 模型训练 step 表 + Figures 3/4 | 不动 |
| Engineering trade-offs | blog 内存盈亏(第 12 层回本)+ 生产者前缀的经验性 + **两阶段 autograd 拆分** + Figure 5 | 不动 |
| **Related work** | blog 致谢段里的 MegaScale-MoE / Triton-distributed / DeepGEMM | **必须重写成正式 Related Work**,并补 MegaFlux(2610.00671, 09-30)、dist_moe、Mixture-of-Kittens |
| Limitations | blog Future Work | 扩写,把三个硬伤主动写成 limitation 而不是等审稿人提 |

**两天内做得到的:** 格式转换、Related Work 重写、把方法学提上来、把三个硬伤写成明确的 limitation。
**两天内做不到的:** Triton-distributed 对照、收敛曲线、liveness 证明。

**我的判断:** 工业轨确实waive了新颖性,但**不 waive 严谨性**。缺同硬件前作对比这一条,即使在工业轨也会被问。两天投是可行但紧张的赌注 —— 赌的是工业轨审稿人接受"我们报告我们建成的东西,并诚实标注未做的对比"。

### 2.2 Paper B — OSDI'27 或之后(研究论文,5 周起)

**Thesis(全文唯一主张):** 融合粒度的上界不由 kernel 决定,而由其上层框架的重叠机制决定;这个上界在算子级测量中不可见。

**为什么这是真命题而非工程教训:** 整个 megakernel 文献(MegaScale-MoE、MegaFlux、Mixture-of-Kittens、dist_moe)默认"融得越多越好"。blog 第 445–451 行记录的事实与之直接冲突 —— 把整层包成一个 autograd Function 后,**算子变快了、训练 step 没变快**,因为 Megatron 分布式优化器的 hook 失去了挂载点,全部 DDP 通信退化成串行。blog 自己写了那句话但把它埋在 "Engineering Trade-Offs" 里。

> **2026-11-02 更正:** 上面把 Mixture-of-Kittens 归进"默认融得越多越好"是错的。MoK 是整层融合,生产端到端 1.41×,并且用 Blackwell Cluster Launch Control 把 SM 让给 FSDP RDMA 流,所以它已经处理过框架通信共存的问题。精读笔记在 `slab/papers/mixture-of-kittens.md`。这一条不能写成"没人讨论过";MoK 解决的是 collective 有没有 SM 可跑,我们碰到的是 collective 能不能按时发出,两者是同一目标的两个必要条件。见 [`2026-10-28_2150_megamoe-paper-skeleton.md`](./2026-10-28_2150_megamoe-paper-skeleton.md) §8.1 / §9.1。

**升格成论文需要的三件事:**

1. **从一个数据点变成一个模型。** 现在是"一个 bug + 一个修复"。需要:形式化"框架 hook 粒度 vs 算子融合粒度"的关系,预测在什么条件下融合会被框架吃掉,然后用至少两个独立框架验证(Megatron 分布式优化器 hook + TorchTitan FSDP2 的 `reshard_after_forward` / post-all-gather hook)。**dist_moe 恰好给了反面样本**:TorchTitan 为它定义了 `DistMoeTransform` + 可选 runtime + GraphPP 显式 slot input,即"先把挂载点设计好再融合",正是我们事后才补上的东西。
2. **把三个硬伤补掉**(§3)。
3. **把 scale-up 域作为第二个实例。** gfx1250/Helios 是 72 卡单 scale-up 域,signal table 要多一级(blog Future Work 第一条)。同一个 thesis 在新拓扑上再验证一次,比在 8 卡上说一遍有力得多。而且业界至今**没有 AMD 侧 scale-up 域的 MoE 成本模型**。

### 2.3 方法学作为第三贡献(两篇都适用,成本接近零)

blog 已经做了但没当成贡献的事,整理成一节即可:

| 做法 | 堵住的攻击 |
|---|---|
| **双基线**:serial(两腿分开测再相加,隔离重叠本身)+ DeepEP 生产路径(真实替换的对象) | "你和谁比" |
| **η 对理想重叠 roofline**,且 `T_gemm` 用**独占全部 256 CU** 的严格分母 | "让 CU 的代价有没有算进去" |
| **用等效 FLOP 的稠密 GEMM 反过来为分母辩护**(grouped 达稠密的 91.5–99.7%) | "GEMM 本来就慢,所以 η 虚高" |
| **两个指标必须合看**:speedup 在两腿等长时最大,η 在一腿压倒时最大,方向相反 | 单指标挑最好看的报 |
| **确定性哈希测试**:固定种子重复跑、顺序敏感哈希。它抓到过一个 wait 常量放松一档的 bug —— **当时精度测试全绿** | "你怎么知道 relaxed 握手是对的" |
| **内存盈亏平衡到层数**:对称堆 8.87 GiB / 每层省 0.766 GiB ⇒ 约 12 层回本,并明说 4 层 proxy 仍是净亏 | "融合省内存"的含糊声称 |

据我所知通算重叠这一类工作**没有标准化的报告口径**,这一节本身可引用。

---

## 3. 三个评审硬伤与堵法

### 3.1 【最致命,成本最低】同硬件前作对比

blog 第 311 行自陈:"does not add an operator-level timing of MegaMoE against DeepEP, and **it does not include a Triton-distributed run on this hardware**"。

**它做得了。** 本地检查(只读):

| 事实 | 位置 |
|---|---|
| AMD EP 融合 kernel 存在,1,267 行 | `python/triton_dist/kernels/amd/ep_all2all_fused.py` |
| AMD 专用测试存在,347 行,带 argparse | `python/triton_dist/test/amd/test_ep_moe_fused.py` |
| AMD autograd 包装存在 | `python/triton_dist/function/amd/ep_moe_fused.py:62` |
| **但反向是 `raise NotImplementedError("backward is not implemented")`** | `python/triton_dist/function/amd/ep_moe_fused.py:80-82` |
| **前向的融合切分与 MegaMoE 相同**:`dispatch + grouped GEMM-1 (up) -> activation -> grouped GEMM-2 (down) + combine` | `ep_moe_fused.py:30`(模块 docstring) |
| 默认激活是 `torch.relu`,不是 gated SwiGLU | `ep_moe_fused.py:55`、`:74` |
| `--capacity` 默认 `world_size`,注释为 "the worst-case safe value" —— 是对称 buffer 容量而非 token drop,**与 MegaMoE 的 `symm_buffer` 容量同语义,可比** | `test_ep_moe_fused.py:109-110` |
| 测试默认形状远小于 DSv3:`hidden_dim=1024`、`ffn_dim=1024`、`num_experts=16`、`topk=4`、`ntokens=4096` | `test_ep_moe_fused.py:103-107` |
| 本地 checkout 是 `8260bc3`(2026-08-06),**早于** PR #208 改 out-of-tree Triton 3.7 plugin | `git log -1` |

**先说一个对我们不利、但必须自己先说出来的发现:** 它那个模块的 docstring 把前向写成 `dispatch + grouped GEMM-1 (up) -> activation -> grouped GEMM-2 (down) + combine` —— **这就是 MegaMoE 的两算子切分,一字不差的同一种切法,而且已经在 AMD 上开源了**。所以「把 MoE 层融成两个算子」在前向方向上**不是我们的新颖点**,这进一步坐实了「不能按『AMD 上的 MegaMoE 实现』投」的判断。真正剩下的差异在反向、在低精度、在框架挂载点,不在前向的切分方式。

**这个结果整体仍对我们有利,但必须诚实地比。** 三个不可比点要在论文里写明:

1. **只有前向。** 所以只能比前向。**这恰恰把 blog delta 第 5 条("前反向共享一套通信原语")从"工程"升格为"唯一可比项"** —— 对手在反向上不存在。
2. **激活函数不同**(relu vs gated SwiGLU),FLOP 与中间张量尺寸都不同。要么把它配成 SwiGLU,要么把 MegaMoE 也跑 relu 做一组,要么只比通信腿。
3. **它的默认形状是小模型。** 必须先验证它能不能跑 DSv3 形状(H=7168, I=2048, E=256, top-8);跑不动本身就是一个要报告的结果,而不是回避对比的理由。

**实验方案(最小可交付):**

| 步骤 | 内容 | 产出 |
|---|---|---|
| E1 | 在 `xiaoming-dev` 容器 + squeue 分配的 MI355X 节点上,按其 README 装 `8260bc3` 这个 checkout(注意它仍是 in-tree Triton 构建,LLVM pin 旧);跑其默认小形状,确认 baseline 能通 | 环境可复现记录 |
| E2 | 把形状推到 DSv3(H=7168, I=2048, E=256, top-8, EP8, 8192 tok/rank)。记录能跑 / OOM / 报错 / 精度不过 | 一张"对手在生产形状上的可用性"表 |
| E3 | **前向、同激活**下的算子级对比:T_comm / T_gemm / T_measured / η,口径与 blog 的表完全一致 | 论文 Table:MegaMoE vs Triton-distributed vs serial |
| E4 | 同时补上 blog 承认缺的那条:**MegaMoE vs DeepEP 的算子级计时**(不只是层级/step 级) | 补齐第二基线的算子级行 |
| E5 | 顺带评估 PR #208 之后的 out-of-tree plugin 版本能否在我们容器里链通(业界扫描里记为"需本地重建验证") | 一条可引用的构建结论 |

**先决条件:** 这件事必须在能跑 MI355X 的节点上做,且要按 `Primus-Turbo/.cursor/rules/iteration_rules.mdc` 的单变量迭代纪律走 —— 对照实验也是实验。

### 3.2 配置窄 + 两个设定偏向自己

blog 自己标注了(这很好,但论文要补):"disabling shared experts removes compute that a non-fused schedule could place next to dispatch, and uniform routing is the balanced case"。

要补三组:

| 实验 | 为什么必须 |
|---|---|
| **开 shared experts** 的配对 | 这是最直接的"不利配置"。非融合调度本可以把 shared expert 的计算摆在 dispatch 旁边,关掉它等于替对手卸掉一条腿 |
| **不均衡路由**下的 step 级(不只是层级) | blog 只说"companion post 的层级比值变化 <1%",step 级没做 |
| **一条真实收敛曲线**,BF16 与 MXFP8 各一条,对 DeepEP 路径做 loss parity | blog 自陈 "These short runs measure throughput, not convergence"。训练系统论文没有收敛证据过不了 |

收敛这条和 roadmap 的 **T1-4(梯度范数回归门)**是同一件事的两个用途:门用来挡回归,曲线用来进论文。做一次,两处用。

### 3.3 liveness 是经验不是证明

原文:"gfx950 has been observed to fill CUs in roughly ascending block-index order… The fill order is **not an architectural guarantee**. We do not pin occupancy, and we do not shrink the grid… The prefix-first layout is an empirical safeguard we have not seen fail on MI355X. **It is not a proof.**"

而 MXFP8 combine 的在线搜索候选集到 **128**(256 CU 的一半都是生产者),那里"the margin rests on fill order alone"。

三条可选处置,按强度:

1. **最强:把它变成可论证的。** 限制 grid 使所有通信块必然驻留后 GEMM 块才被调度(blog 明说"we do not shrink the grid… the GEMM region is deliberately oversubscribed"),代价是放弃超订带来的 latency hiding。需要量化这个代价。
2. **中等:实证边界。** 构造压力实验找出填充顺序会反转的条件(极端 occupancy、多 kernel 竞争、CU masking),报告安全边界。即使没找到反例,"我们试了这些都没反转"也比"我们没见过失败"强得多。
3. **最弱但必须有:** 把 watchdog 的行为写清楚,并明确承认这是设计契约中的一个假设。

**OSDI 这类会议会盯住这一条**,因为它是"你的 kernel 在什么条件下会死锁"。工业轨可以只做第 3 条。

---

## 4. 这件事逼出来的决定

两天的窗口是真的,但补洞的时间是五周起。所以只有两条路:

**路线一:两天冲 MLSys'27 工业轨。** 赌工业轨接受"诚实标注未做对比"的已建成系统报告。好处是 2027-02-28 就有结果,且不影响后续写 Paper B(工业轨论文与研究论文可以共存,但要注意 MLSys 的 per-author 与重复投稿规则 —— **这条我没核实,投之前必须查**)。

**路线二:放掉两天的窗口,按 OSDI'27(12-01 摘要)或 SIGCOMM'27(02-02)做 Paper B。** 五周能补完 §3.1 的对照和 §3.2 的收敛,但 §3.3 的 liveness 和 §2.2 的"两个框架验证 thesis"够呛。更稳的是 SIGCOMM'27(三个月,且 AcclEP/EPIC 有先例)或 SC'27(约五个月)。

我的倾向:**如果两天内能凑出人手做格式转换与 Related Work 重写,路线一值得赌** —— 成本是两天,下行是一次拒稿,上行是 2027-02 就有一篇。blog 未发布进一步抬高了这条路线的性价比(双盲是干净的,见 §0.1)。同时**无论投不投,§3.1 的 E1–E4 都要做**,因为那是 Paper B 的必需品,也是 roadmap T3-3 的内容。

**但这条倾向有一个前置闸门:§0.3 的对外发布审批。** 两天的窗口对这个闸门来说可能本来就不够,这跟论文写得完写不完是两件独立的事,要分别确认。

**还有一个 §0.3 带出来的选择要一起定:blog 的上线时间。** 如果走路线一,blog 最好压到投稿之后;如果 blog 有不能推迟的业务时间点,那就该反过来 —— 先发 blog、放弃对双盲的洁癖,按"内容已公开"的前提去选会议(MLSys 允许 arXiv 预印,说明这条路走得通,只是匿名化变成形式)。这两个不能都要。

---

## 相关文件

- 源材料(**未发布草稿**,分支 `xiaompen_amdeng-primus-mega-moe`):`/perf_apps/xiaoming/rocm-blog/blogs/software-tools-optimization/primus-mega-moe/README.md`(561 行 + 5 图)
- 同系列**同样未发布**:`primus-moe-optimization`(分支 `xiaompen_amdeng-primus-moe-optimize`,层级/step 级,1.40× BF16 / 1.81× MXFP8)
- 同系列**已并入 `origin/develop`**(唯一一篇,但只有端到端结果、不触及融合机制,基本不构成在先技术):`primus-deepseek-v4`(frontmatter `03 Sep 2026`,四节点 12.0%)
- Turbo roadmap(T3-3 即本事项):[`2026-10-28_primus-turbo-roadmap-2026q4-2027h1.md`](./2026-10-28_primus-turbo-roadmap-2026q4-2027h1.md)
- 业界前作定位:[`../primus-moe/2026-10-28_0546_industry-scan-update.md`](../primus-moe/2026-10-28_0546_industry-scan-update.md)
- 迭代纪律:`/perf_apps/xiaoming/Primus-Turbo/.cursor/rules/iteration_rules.mdc`

## 核查溯源

会议 deadline 于 2026-10-28 从各自官方 CFP / Dates 页抓取;标注"未核实"的为从上一 cycle 外推、官方页尚未发布。Triton-distributed 的全部代码事实(文件、行数、`NotImplementedError`、默认参数、checkout 版本)由我本地只读核验。MLSys 工业轨措辞为官方 CFP 原文引用。

blog 发布状态由 `git ls-tree -d origin/develop blogs/software-tools-optimization/<dir>` 对三个目录逐一核验 + `git rev-list --left-right --count origin/develop...HEAD` 得到(注:该仓默认分支是 `develop` 不是 `main`,`origin/HEAD -> origin/develop`)。**初稿误把 frontmatter 的 `date: "15 Sep 2026"` 当成发布日期,由用户指出后更正,并连带发现 `primus-moe-optimization` 也未发布 —— 原先"同系列已发表会压缩新颖性边界"的判断一并撤回。**

**未核实项**:① MLSys 的 per-author 投稿数上限与研究轨/工业轨重复投稿规则 —— 投稿前必须查;② 并入 `origin/develop` 之后是否还有一步才上线公开站点,以及 `primus-deepseek-v4` 当前在公开站的实际可见性 —— 我只核到了 git 层;③ AMD 对外发表审批流程(§0.3)—— 超出我能核查的范围,需你本人确认。
