# MegaMoE 论文骨架(可直接往里写)

> **When**: 2026-10-28 21:50 UTC+8
> **Target**: MLSys 工业轨,双栏 10 页(不含参考文献),`mlsys2025style`
> **约定**: 英文为论文正文骨架;`>` 引用块内的中文是写作提示与出处,成文时删除。`**TODO**` 为必须补的空缺。
> **规则与取舍依据**: [`2026-10-28_2115_megamoe-paper-mlsys-plan.md`](./2026-10-28_2115_megamoe-paper-mlsys-plan.md)(工业轨细则)、[`2026-10-28_2140_megamoe-paper-outline.md`](./2026-10-28_2140_megamoe-paper-outline.md)(页数预算与图表取舍)
> **数字来源**: 全部取自 `primus-mega-moe/README.md`,已逐行核对;新增数字一律标 TODO,不得凭空填

---

## Title

**MegaMoE: Fusing Expert-Parallel Communication into the Grouped GEMM for MoE Training on AMD Instinct MI355X**

> 备选(更突出教训、更贴工业轨口味):
> *Fusing All-to-All into the Grouped GEMM: Building and Deploying MegaMoE on AMD Instinct MI355X*
> **TODO** 定稿时二选一。工业轨可保留公司名/产品名,标题里出现 AMD Instinct 是允许且有利的。

**Authors**: 投稿时匿名化**仅作者名**;AMD、Primus、Primus-Turbo、MI355X 等均可保留(工业轨规则)。
> **TODO** 作者顺序与第一作者归属 —— 工业轨要求第一作者与多数作者来自业界。

---

## Abstract

> 以下为草稿,数字全部已核实,可直接用。

Mixture-of-Experts training on expert-parallel ranks places two all-to-all exchanges around two grouped GEMMs. On an 8-GPU AMD Instinct MI355X node these exchanges account for 46% of the serial routed-expert time, and the forward dispatch alone leaves 41% of its stage exposed ahead of the FC1 GEMM it feeds.

We present MegaMoE, a pair of fused operators that push expert-parallel traffic through the grouped GEMM instead of around it: `dispatch_grouped_gemm` fuses the dispatch all-to-all with FC1, and `grouped_gemm_combine` fuses FC2 with the combine all-to-all and the weighted top-k reduction. Both directions reuse the same communication primitives, because dispatch and combine exchange roles in the backward pass.

On DeepSeek-V3, DeepSeek-V4-Pro and DeepSeek-V4-Flash expert shapes, fusion is worth 1.29–1.61x across 15 operator stages at BF16, reaching 82.7–94.2% of the ideal-overlap roofline. In paired training-step runs over six models, the fused layer delivers 1.16–1.29x at BF16 and 1.17–1.35x at MXFP8.

> **TODO** 加一句收敛结论(§7.5 完成后)。
> **TODO** 加一句部署规模(§3 完成后)。

We report the design, the measurement methodology we developed to keep overlap claims falsifiable, and six lessons from building the system — including the finding that the achievable fusion granularity is bounded not by the kernel but by the training framework above it.

---

## 1 Introduction

> 1.0 页。四段。

**P1 — 问题。** MoE 路由专家路径的结构性问题:两个 A2A 夹两个 grouped GEMM,通信在关键路径上。
> 出处:blog `## Background` 开头。可引 MegaScale-MoE 报告的生产环境 43.6% forward / 32% 端到端(NVIDIA Hopper 上测得,说明非 AMD 特有;该文未披露模型规模与并行配置,只能作数量级参考)。

**P2 — 本节点上的实测。** A2A 占串行时间 46%(11.31 ms / 24.58 ms);前向 dispatch 在自己那个 stage 内暴露 41%(2.265 ms / 5.475 ms)。低精度不解决问题(载荷变小但暴露结构不变)。
> 出处:blog `### The Routed-Expert Forward Path Has No Slack`、`### Low Precision Does Not Make This Go Away`。

**P3 — 我们做了什么。** 两个融合算子一句话各一行 + 共轭性(反向复用同一套通信原语)。给出 §7 的三个头条数字。

**P4 — 贡献列表。** **这一段的写法是全篇成败所在:工业轨不按 novelty 评分,所以贡献不能是加速比。**

1. A complete design-and-implementation report of a fused expert-parallel MoE operator on CDNA4, covering the consistency model, the CU partitioning, and the low-precision data path (§3–§5).
2. A measurement methodology that keeps communication–computation overlap claims falsifiable: two baselines, an ideal-overlap roofline with a deliberately strict denominator, and a dense-GEMM control that defends that denominator (§6).
3. Six lessons from building and deploying the system (§8), several of them negative results, the most consequential being that **the training framework, not the kernel, sets the ceiling on useful fusion granularity**.

> 并在 P4 末尾前向引用 §8 的教训清单,让审稿人第一页就看到落点。

---

## 2 Background: Why the Routed-Expert Path Has No Slack

> 0.75 页。两小节,材料齐全,基本是 blog 直搬。

### 2.1 The forward path has no slack
> 出处:blog 同名小节 + 占比表(46% / 41%)。把 blog 的表折进正文一句话,省一张表。

### 2.2 Low precision does not make this go away
> 出处:blog 同名小节。

> 本节无 TODO。

---

## 3 The Deployed System

> 0.5 页。**整节 TODO —— 这是 blog 完全没有、而工业轨资格条件要求的一节。**
> 工业轨原文:"Submissions must describe a real system: a retrospective evaluation of a deployed system, a planned or upcoming system on a product or deployment roadmap, or a canceled system with lessons learned. Speculative, small, or exploratory projects without a real system do not qualify."

**已由我核实、可直接写进本节的事实:**

- MegaMoE 以**可选层替换**的形式接入 Primus 的 Megatron 后端,由 `use_turbo_mega_moe` 开关控制,**默认关闭**(`primus/configs/modules/megatron/primus_turbo.yaml:93`)。
- 精度由 `turbo_mega_moe_precision` 选择(`bf16` | `mxfp8`),仅在开关打开时读取(同文件 `:94-99`)。
- 适用范围在配置注释里写明为 **EP-only / TP=1**。
- 接入点:`primus/backends/megatron/core/extensions/mega_moe.py` 与 `primus/backends/megatron/core/transformer/moe/v4_moe.py`;**调用的是 `fused_mega_moe_stage1/stage2`(以及 fp8 对应物),即 §8 L1 的两阶段拆分是实际在产路径,不是实验分支**。
- 算子实现在 Primus-Turbo,Primus 通过 `primus_turbo.pytorch.ops.moe.fused_mega_moe{,_fp8}` 导入。
- **集成时间线(5 个 commit,`git log --follow` 核实)** —— 这是本节最有价值的材料,它把"一个真实系统逐步接入"讲成了故事:

| 日期 / PR | 内容 |
|---|---|
| **2026-07-29 · #927** | FlyDSL-based fused MegaMoE layer for Megatron(**EP-only / TP=1 / bf16**)—— 首次落地 |
| #934 | 在 DeepSeek-V4 上启用融合 MegaMoE 专家路径 |
| #983 | 融合 MegaMoE 层的 **MXFP8** 专家精度 |
| #1243 | 接受 MiniMax-M3 的 `swigluoai`(quick_geglu)激活 |
| `311afb9de` | 在 GPT-OSS 上跑通 MegaMoE(expert biases、**released MLP/MoE presets**) |

- **`primus/configs/` 下没有任何配置把它打开**;唯一打开它的是仓库根目录的一个工作配置 `gpt_oss_120B-BF16-pretrain-megamoe-autotune-1node.yaml:187`(不在发布配置树内)。
- **TorchTitan 后端零引用 —— 仅 Megatron 后端接入。**(`grep -ril mega_moe primus/backends/torchtitan/` 为空)

**TODO —— 必须由你提供,我核不到:**

| # | 要回答的 | 为什么论文需要它 | 现状 |
|---|---|---|---|
| T3-1 | **归入三类资格的哪一类?** | 决定整节时态与口径 | **倾向已清楚**:默认关闭 + 发布配置树内无人打开 ⇒ 写"已部署"站不住;但有 3 个月、5 个 commit、四个模型族逐步接入的真实轨迹 ⇒ **"产品/部署 roadmap 上的在建系统"既诚实又足以合规**。待你确认 |
| T3-2 | 有没有**真实训练任务**(非 benchmark / 非 autotune 调参)跑过?哪个模型、多少卡、多少小时或 token | "real system" 的核心证据,也是 impact 评分的主要来源 | **仍全空** |
| T3-3 | 默认关闭的原因(EP-only/TP=1 限制?数值风险?灰度中?) | 审稿人必问;与 §8 L5 的静默回归故事直接相关,诚实回答反而加分 | 空 |
| T3-4 | 何时成为默认路径,前置条件是什么 | 走"在建系统"时这是资格的关键 | 空 |
| T3-5 | 使用者范围:仅本团队 / AMD 内部多团队 / 外部客户 | impact | 空 |
| T3-6 | Primus 与 Primus-Turbo 的发布关系(是否随版本发布、有无版本约束) | 与 roadmap T0 的契约问题同源 | 空 |
| T3-7 | 集群与软件栈版本(ROCm / PyTorch / 镜像 tag) | 可复现性 | 空 |
| T3-8 | `311afb9de` 里的 "released MLP/MoE presets" 指什么?是否意味着有对外发布的预设配置 | 若确有对外发布的 preset,**这是"real system"最硬的一条证据**,应当前置到本节开头 | 空 |

---

## 4 Design: Two Fused Operators

> 1.25 页。材料齐全。

### 4.1 What each operator fuses
> **Table 1** = blog 的算子表:
> - `dispatch_grouped_gemm`:dispatch all-to-all → FC1 grouped GEMM。把本地 token 推进目标 rank 的显存,同时逐 tile 推进 FC1。
> - `grouped_gemm_combine`:FC2 grouped GEMM → combine all-to-all → top-k reduce。算完 FC2 推回源 rank,并把空闲 block 回收做加权 top-k 规约。

### 4.2 Backward is the conjugate of forward
> **Figure 1**(`mega-moe-conjugate-pipeline.png`)。要点:dispatch 与 combine 在反向互换角色,故两个方向复用同一套通信原语;独立的 SwiGLU 在两个方向上都夹在中间;**FC2 的变 K 权重梯度是唯一留在融合对之外的 GEMM**。

> 本节无 TODO。

---

## 5 Implementation on CDNA4 (gfx950)

> 1.25 页。**材料最富余的一节,也是唯一要主动压缩的一节**(§5.5 低精度五个子节压成一段,细节下放附录)。

### 5.1 Role specialization inside a single launch
> **Figure 2**(`mega-moe-block-roles-signals.png`)。通信 block 占最低索引区;每个消费者只自旋在自己拥有的 expert / tile / slot 上;箭头是握手不是全局 barrier。

### 5.2 Tile-level handshake and visibility scope
> **Table 2(新,合并表)** —— 把 blog 的三张表压成一张,这是 10 页预算下的必要合并:
> - 三张 ready-signal 表:`dispatch_flag`(按 expert,通信 block → FC1 GEMM tile)、`combine_flag`(按 GEMM tile,FC2 GEMM tile → combine block)、`reduce_flag`(按 slot,combine block → reduce warp)
> - cache modifier:`18 = sc1|nt` agent 域(跨 XCD、不出 GPU),用于读本地 GEMM 刚产出的结果、FC2 写 C 进本地暂存区;`19 = sc0|sc1|nt` system 域(经 XGMI 到对端 GPU),用于推载荷与发布完成标志
> - CU 划分:`num_dispatch_cu` = 16/32/16(nt/nn/tn),`num_combine_cu` = 64/16

### 5.3 No device-to-host synchronization on the critical path
> 出处:blog 同名小节。

### 5.4 Consistency: relaxed handshakes
> 出处:blog `### Consistency: Relaxed Handshakes, Cache Modifiers for Visibility Scope`。本节与 §8 L6(liveness 是假设)互相引用。

### 5.5 Low precision inside a fused operator
> **压缩成一段**,只留结论:量化融进生产它的 kernel 的 epilogue;`[32,32]` 方形块缩放使量化对转置不变;XGMI 上走 E4M3 + E8M0 scale;**scale 必须与它的 token 一起到达**;combine split 随载荷移动;哪些留高精度。
> 五个子节的细节 → **附录 A**。

> **TODO** 回答"为什么是 FlyDSL 不是 Wave"。Wave 是 AMD 在 MLSys 2026 的 DSL 论文,审稿人会问。放本节末或 §9。

---

## 6 Measurement Methodology

> 0.75 页。**材料已有但散落在 blog 的 Experiments 各处,论文里要集中成一节,这是贡献 2。**

### 6.1 Setup
> **Table 3** = blog `### Setup`:MI355X(gfx950)8 GPU 单节点;EP8,TP=1;XGMI;BF16 与 MXFP8(E4M3,block 32,E8M0 scale);8192 token/rank。模型形状:DeepSeek-V3(7168/2048/256/top-8)、DeepSeek-V4-Pro(7168/3072/384/top-6)、DeepSeek-V4-Flash(4096/2048/256/top-6)。

### 6.2 Two baselines, and why both are needed
> serial(自家两条腿分开测再相加 —— 隔离重叠本身)+ DeepEP 生产路径(真实被替换的对象)。

### 6.3 Two metrics that move in opposite directions
> speedup = (T_comm + T_gemm) / T_measured;η = max(T_comm, T_gemm) / T_measured。
> **speedup 在两腿等长时最大,η 在一腿压倒时最大** —— 必须合看,单取一个就是挑好看的报。实例见 §7.3。

### 6.4 Defending the denominator
> `T_gemm` 用 GEMM **独占全部 256 CU** 的严格分母;再用等效 FLOP 的稠密 GEMM 反过来验证这个分母不虚 —— **Figure 4**(`mega-moe-gemm-roofline.png`),grouped GEMM 自身达稠密的 91.5–99.7%;红条是同一 GEMM 在融合算子内部、保留 256 CU 中的 192–240 个(16 个通信 CU 留约 240,64 个留约 192)。通信受限 stage 上红条下降是因为 GEMM 是较短的那条腿,不是 GEMM 变慢了。

> 本节无 TODO。

---

## 7 Evaluation

> 2.25 页。

### 7.1 Validating a manual correctness argument
> 0.25 页。出处:blog 同名小节。确定性哈希(固定种子重复跑 + 顺序敏感哈希)。结论留此处,教训写进 §8 L3。

### 7.2 Operator-level overlap at BF16
> 0.5 页。**Table 4** = blog 五 stage 表(T_comm / XGMI GB/s / T_gemm / T_measured / speedup / η):
> `dispatch_grouped_gemm` fwd(nt) 1.50× / 87.9%;bwd dgrad(nn) 1.52× / 88.5%;bwd wgrad(tn) 1.48× / 87.5%;`grouped_gemm_combine` fwd(nt) 1.54× / 86.0%;bwd dgrad(nn) 1.42× / 85.6%。
> **Figure 3**(`mega-moe-fusion-speedup.png`)。
> 扩到 V4-Pro 与 V4-Flash 共 15 stage:1.29–1.61×、η 82.7–94.2%、XGMI 有效单向 359–373 GB/s = 理论出口 537.6 GB/s(7 条 IF 链 × 153.6 GB/s 双向)的 67–69%。
> 原文那句定调要保留:这足以说明 η 不是靠慢 all-to-all 撑起来的,但**不是在声称 mesh 已饱和**。

### 7.3 Why the two metrics must be read together
> 0.25 页。**别压缩,这是方法学最有力的实例。** V4-Pro 的 `dispatch_grouped_gemm` bwd dgrad:1.716 ms 通信配 2.075 ms 计算 → 1.61×(两腿接近,能藏的最多);同一算子的 bwd wgrad:1.663 ms 藏进 4.492 ms → η 94.2% 但只有 1.29×。自动调优表是同一事实的另一面:dispatch 前向(计算受限)只留 16 个通信 CU,combine 前向(通信受限)拿 64 个。

### 7.4 Training-step results
> 0.5 页。**Table 5** = blog 六模型配对表。BF16 1.162–1.286×,MXFP8 1.169–1.350×。MiniMax-M2.5 是 BF16 最强(1.286×),Qwen3-235B-A22B 是 MXFP8 最强(1.350×)。
> **blog 的全部限定语必须原样保留,这是工业轨的基准诚实度:** 深度为缩减代理(4/60、8/43、6/78、4/60、2/62、8/94);MBS/GBS 逐模型不同(1/256、1/256、2/512、4/256、3/768、4/128)故**跨行绝对值不可比,只能行内配对比**;seq len 4096;PP=1;recompute 与 shared experts 关闭;均匀路由;mock 输入;20 迭代;每模型实际跑了 5 个 arm,表里只报两个同精度配方的配对。
> 两个早期的 step 级结果作为支撑而非表内行:1.081× BF16 / 1.138× MXFP8(缩减深度 DSv3),以及四节点 12.0%。
> **TODO(引用问题)** 这两个支撑结果在 blog 里引的是 `rocm.blogs.amd.com` 链接,其中 `primus-moe-optimization` **尚未发布**。论文**不能引用不存在的文献** —— 要么把这两个数据作为本文自己的测量重新给出(需要配置与方法说明),要么删除。

### 7.5 Convergence
> 0.25 页。**整节 TODO(A1)。**
> 需要:BF16 与 MXFP8 各一条收敛曲线,对 DeepEP 路径做 loss parity。
> 为什么不能省:blog 自陈 "These short runs measure throughput, not convergence";一篇声称部署了训练系统的论文停在这里过不了。
> 与 roadmap **T1-4(梯度范数回归门)**是同一件工作。

### 7.6 Unfavourable configurations
> 0.25 页。**整节 TODO(A2)。**
> **TODO 7.6-a** 开启 shared experts 的配对。blog 自己标注这是偏向项:关掉它等于移走了非融合调度本可以摆在 dispatch 旁边的计算。
> **TODO 7.6-b** 不均衡路由下的**步级**结果,**必须在本文内测**。现有的"层级比值变化 <1%"论据引自未发布的 companion post,不可引用。

### 7.7 Comparison against alternatives on the same hardware
> 0.25 页。**整节 TODO(A4)。**
> **TODO 7.7-a** Triton-distributed 在 MI355X 上的对比。三个已核实、必须在正文写明的不可比点:①**它只有前向**(`python/triton_dist/function/amd/ep_moe_fused.py:80-82` 为 `raise NotImplementedError`);②默认激活是 `torch.relu` 而非 gated SwiGLU;③其测试默认形状(hidden 1024 / ffn 1024 / 16 experts / top-4)远小于 DSv3,能否跑到生产形状本身就是要报告的结果。
> 还有一个对我方不利、但必须自己先说的事实:**它模块 docstring 里的前向切分是 `dispatch + grouped GEMM-1 (up) -> activation -> grouped GEMM-2 (down) + combine`,与本文的两算子切分相同** —— 因此本文不把这个切分作为新颖性主张。
> **TODO 7.7-b** DeepEP 的**算子级**计时(blog 只有层级与步级)。
> **TODO 7.7-c(新增,来自 MoK)** 朴素 **RCCL all-to-all + PyTorch + grouped GEMM** 的基线。MoK 在 NVL72 上测到,近一半配置里朴素 NCCL+PyTorch 比 DeepEP / HybridEP 都快;而且集中在小 EP、小 token:它的 Table 1 里 EP16、每卡 1024 / 2048 token 的 16 格全是朴素 NCCL 最强,EP64 则主要是 HybridEP。我们的 EP8 比 EP16 还小。本文现有的两个基线(serial、DeepEP 生产路径)都不是朴素 collective,审稿人读过 MoK 一定会问"你是不是在打一个本来就偏弱的 DeepEP"。EP8 单节点 XGMI 上结论可能不同,但必须实测。

---

## 8 Experience and Lessons

> 1.5 页。**全篇差异化所在。** 次序按可迁移性排,不按时间。

### 8.1 L1 — The framework sets the ceiling on fusion granularity
> 0.4 页。材料齐全。
> 把整个 MoE 层包进单个 autograd Function 后,算子更快而训练步没更快:Megatron 分布式优化器的 hook 失去了模块边界,参数 all-gather 与梯度 reduce-scatter 退化成在融合 kernel 之后串行执行。`fused_mega_moe_stage1` / `stage2` 两阶段拆分把边界还回去,同一批 collective 重新与计算重叠。
> **Figure 5**(`mega-moe-ddp-overlap-trace.png`)。**图注原文 "No kernel changed between the two traces." 要在正文直接引用 —— 这是全篇最有力的一句。**
> **(2026-11-02 更正)原稿此处写"整个 megakernel 文献默认融得越多越好,没有一篇讨论过这个上界"—— 这是错的,删掉。** Mixture-of-Kittens(arXiv 2609.36070)是整层训练 megakernel(dispatch + shared/routed FFN + combine),在 512×GB300 生产训练上拿到 1.41× 端到端,并且**正面处理了框架自身通信的共存问题**:用 Blackwell 的 Cluster Launch Control 让 megakernel 把 SM 让给更高优先级的 FSDP RDMA 流。
> **(2026-11-02 精读 MoK 后再次修正)** 上一版写的"两条路解决的是同一个问题"也不准确。融合算子要和框架 collective 重叠,需要两个条件同时满足:
> - **A. 发得出去**:框架要在合适的时间发出 collective,这由 hook / bucket 挂在哪个模块边界决定。这是 MegaMoE 碰到的:整层包成一个 autograd Function 后 Megatron 分布式优化器失去模块边界;`stage1` / `stage2` 拆分把边界还回来(`primus_turbo/pytorch/ops/moe/fused_mega_moe.py` docstring 原文:"restores the DDP comm/compute overlap granularity")。
> - **B. 跑得起来**:collective 发出后要有 SM / CU 给它跑,持久 megakernel 占满所有 SM 时它只能排队。这是 MoK 用 CLC 解决的;它在 FSDP 下替换的是整个 MoE 模块,all-gather 预取挂在模块边界上,条件 A 没有暴露。
> 所以 L1 的写法是:"这个问题有两个必要条件;MoK 在 Blackwell 上处理了 B,我们在 Megatron 上碰到并处理了 A。"
> **TODO 8.1-a** 核实 MegaMoE 在条件 B 上的状况:Figure 5 的 trace 里,RCCL kernel 是和 MegaMoE kernel 本身并发,还是只和非 MoE kernel 并发?MegaMoE 的 grid 是否占满 256 个 CU?
> **TODO 8.1-b** ROCm / CDNA 上有没有与 CLC 等价的"持久 kernel 让出 CU"机制,尚未核实(`knowledge/systems/mi455x-megamoe-ep64-architecture.md` §1.2 也不假设 CDNA5 有)。可能的替代方案(带 CU mask 的 stream、grid 不占满 CU)要先查文档再实测,在此之前不写进正文。
> **必须加的限定句(已核实):MegaMoE 只接入了 Primus 的 Megatron 后端,TorchTitan 后端零引用。** 因此本文只能声称"在 Megatron 分布式优化器的 hook 机制下观测到这个上界",不能声称它是框架无关的普遍现象。把这句写进正文,比等审稿人问"换个框架还成立吗"要好。若将来补了 TorchTitan 侧的对照,那是下一篇(路线 B)的材料,不是本文。

### 8.2 L5 — The current answer to silent numerical regression is vendoring, not testing
> 0.3 页。**blog 里没有,纯写作、零实验成本,性价比最高的一节。**
> 证据原文在 `primus_turbo/flydsl/mega/bf16/__init__.py:14-24`:`#486`(`ed8d7af4`)围绕 `Mfma16x16x32` 与新 S2R loader 重建了**共享**的 FlyDSL BF16 grouped GEMM;该改动对 host 侧 grouped GEMM 路径无害,却**静默**破坏了 MegaMoE BF16 —— it1 grad norm 从 ~1.45 跳到 ~1.88e6。处置是把 tile 钉到 `f6d5ab68` 快照、在 MegaMoE 下私有化。
> 要讲清代价:由此付出三套 mega 栈 / 三套 `symm_buffer` / 三套 `barrier` 的重复。
> 要讲清普遍性:这属于"前向正确、无 NaN、仅权重梯度错"的一类失效,它穿过了常规精度测试。
> **TODO** 决定是否把 roadmap T1-4(梯度范数回归门)作为"我们打算怎么修"写进来。写进来更完整,但要和 §3 的 roadmap 口径一致。

### 8.3 L3 — Accuracy tests do not catch consistency bugs
> 0.2 页。确定性哈希抓到过一个 wait 常量放松一档的 bug,**当时精度测试全绿**。

### 8.4 L4 — Fusion does not always save memory
> 0.2 页。对称堆:BF16 8.87 GiB;MXFP8 5.66 GiB 主缓冲 + 0.88 GiB signal pad = 6.54 GiB(约 BF16 的 3/4;两者的池容量公式不同,只能按分配器总量比)。每层省 0.766 GiB ⇒ 约**第 12 层**才回本,4 层 proxy 仍是净亏。

### 8.5 L6 — Liveness is a design assumption, not a proof
> 0.3 页。gfx950 被观测到大致按 block index 升序填 CU,但这**不是架构保证**;不钉 occupancy、不缩 grid(GEMM 区是故意超订的)。blog 原文:"The prefix-first layout is an empirical safeguard we have not seen fail on MI355X. It is not a proof."
> MXFP8 combine 的在线搜索候选集到 128(256 CU 的一半都是生产者),那里 margin 只剩填充顺序。
> **TODO(A7)** 压力实验的实证边界:极端 occupancy / 多 kernel 竞争 / CU masking 下填充顺序是否反转。即使没找到反例,"我们试了这些条件都没反转"也远强于"我们没见过失败"。
> **因为作者回应只有 3 天,这条必须在正文交代干净,不能留给 rebuttal。**

### 8.6 L2' — A note on reporting
> 0.1 页。一句话指回 §6:为什么我们认为通算重叠这一类工作现有的报告口径不足以支撑其结论。

---

## 9 Related Work

> 0.5 页。**blog 只有致谢式溯源,本节必须正式重写。**

### 9.1 Fused MoE communication
> MegaScale-MoE(EuroSys'26)—— **blog 自陈本文的 overlap structure 跟随它,这个必须在正文明说,不能藏在致谢里**。DeepEP。Triton-distributed。`meta-pytorch/dist_moe`。MegaFlux(arXiv 2610.00671)。
> **Mixture-of-Kittens(arXiv 2609.36070,Stanford + Cursor,开源)—— 与本文最接近的训练 megakernel,必引,而且要放在本节第一位。** 精读笔记已在 `slab/papers/mixture-of-kittens.md`。必须逐条对照的点:
> - 它融合了 **shared expert**(塞进首个 dispatch 窗口),本文的训练步实验关掉了 shared expert ⇒ 7.6-a 从"补强"升为"不做会被直接拿来对比"
> - 它的通信方向是 **pull dispatch + push combine**;本文 dispatch 与 combine **都是 push**(已核实:`primus_turbo/flydsl/mega/bf16/dispatch_grouped_gemm_bf16_kernel.py:67` 的 `_wait_for_expert` 阻塞直到每个 sender 都推完该 expert)。它的论据是 push 的完成信号开销随 EP 扇入增长:EP4 时 39 µs(8%),EP8 时 51 µs(8.9%),EP64 时 135 µs(18%),pull ≤4 µs。但从它的 Figure 4 反推,去掉信号后 push 的纯搬运比 pull 快 5–9%;我们在 RocMoE M1 测到 XGMI 上 pull 每 token 慢约 43%。**正文要把这两个数并排写出来,作为 EP8 / XGMI 上选 push 的依据**
> - 它把 overlap 粒度(minibatch 512–32768 token)做成一等可调参数;同一负载下最好粒度是最差的 1.38–3.53×。本文 dispatch 的等待粒度是"一个 local expert 的全部 token"(同上 `_wait_for_expert`),不可调。要说明这一点,以及它和 `num_dispatch_cu` / `num_combine_cu` 的关系
> - 它没有 CPU–GPU 同步;**本文每次 forward 都有一次 D2H**(`primus_turbo/pytorch/ops/moe/fused_mega_moe.py:26-27`,因此不能 CUDA graph 捕获)。正文要列为限制,或者改掉 ⇒ 见 TODO 表
> - 它是 bitwise 确定性的;本文 §8 L3 的确定性哈希测试要和它区分开(我们是把确定性当测试手段,它是当产品特性)
> - 它把 wgrad 放在所有 dgrad 之后、与最后几次 reverse-dispatch 重叠(博客);本文 FC2 variable-K wgrad 在融合对之外。可以作为两边独立得出的同类设计来写
> - 它把 DeepGEMM 的 MegaMoE 称为"最接近的工作"。**本文的名字会和它撞车**,正文首次出现时要明确区分
> - Helios 在它全文只出现一次,在引言的例子列表里;它没有任何 AMD 数字 ⇒ **本文的平台价值在这里**
> - 一作 Sul 也是 ParallelKittens 的一作。HipKittens 出自同一实验室(Ré、Arora),但和 MoK 只有 Ré 一位共同作者,MoK 也没引用它。博客说 MoK 设计成"借助 agent 可以很快移植到别的平台",加上 Apache 2.0,别人先移植到 AMD 的可能性不能排除,要盯它的仓库
> - 目前还是 arXiv 预印。**TODO** 投稿前确认它有没有中会,尤其是不是 MLSys 2027
> 文件级溯源必须写明:`barrier.py` 改编自 DeepGEMM 的 `barrier.cuh`;`symm_buffer.py` 改编自 `sym_buffer.cuh` / `mega_moe.cuh`。

### 9.2 Prior art at this venue
> **TODO —— 每一篇都必须精读正文后再定位,目前只有标题与摘要级判断:**
> - ParallelKittens(MLSys'26,Stanford)—— 多 GPU kernel 通算重叠的通用原则。**与本文最接近的"原则性"前作**,§8 L1 实际上是对它所在层面的一个上界说明。
> - FarSkip-Collective(MLSys'26,NVIDIA)—— 改 MoE 结构以重叠阻塞 A2A。
> - FP8-Flow-MoE(MLSys'26)—— casting-free FP8 MoE 数据流、消除冗余 quant/dequant。**与本文 §5.5 的"量化融进 epilogue"最接近,重叠程度必须确认。**
> - MoEBlaze(MLSys'26,Meta)—— MoE 训练内存墙,与 §8 L4 相邻。
> - TokenWeave(MLSys'26,MSR India)—— TP 推理的通算重叠。
> - HipKittens(MLSys'26,Stanford & AMD)、Wave(MLSys'26,AMD)—— AMD 上的 kernel 写法与 DSL。**"为什么是 FlyDSL 不是 Wave"在此或 §5.5 回答。**

---

## 10 Conclusion

> 0.25 页。出处:blog `## Summary`。收口回 §1 P4 的三条贡献,结尾落在 L1。

---

## Appendix

- **A** 低精度数据路径的完整细节(blog `### Low Precision Inside a Fused Operator` 五个子节)
- **B** 15 stage 全表(正文只给区间)
- **C** MXFP8 对称堆分配器的容量公式
- **D** 复现:命令与环境(blog `## Reproduce` 改写)
- **TODO** 若投 Artifact Evaluation 则此处放 artifact 说明。**注意 AE 在 MLSys 为录取后的自愿环节、大概率不影响录取决定(待核实),不要为它挤占正文预算。**

> **附录的硬约束:工业轨规则写明"reviewers are not required to read these"。凡支撑核心结论的内容一律不得下放。**

---

## 从 blog 整段删除

- `## Reproduce` → 并入附录 D
- `## Disclaimers` → 删
- `## Future Work` → 压成 §10 最后一句,或并入 §8 各条的 TODO
- **全部指向未发布 companion post 的引用** → 见 §7.4 的 TODO

---

## TODO 汇总(按是否卡住论文排序)

| 优先级 | 编号 | 内容 | 谁来做 |
|---|---|---|---|
| **P0 行政** | — | 对外发表审批(blog 自身内部 review 亦未完结) | 你 |
| **P0 资格** | T3-1 ~ T3-7 | §3 部署刻画七问 —— **只需整理事实,不用跑实验,应当最先做** | 你 |
| **P1** | 7.5 | 收敛曲线(BF16 + MXFP8,对 DeepEP 做 loss parity)= roadmap T1-4 | 实验 |
| **P1** | 7.6-b | 不均衡路由**步级**(原论据引自未发布文章,不可引用) | 实验 |
| **P1** | 7.6-a | 开启 shared experts 的配对(MoK 融了 shared expert,不做会被直接对比) | 实验 |
| **P1** | 7.7-c | 朴素 RCCL A2A + PyTorch 基线(MoK 的核心发现是这个基线经常最强) | 实验 |
| **P1** | 8.1 | 按"两个必要条件"重写 L1;8.1-a 核实 trace 里 RCCL 是否与 MegaMoE kernel 本身并发;8.1-b ROCm 上 CLC 的替代方案 | 写作 + trace + 查文档 |
| **P1** | 9.1 | 每次 forward 一次 D2H:列为限制,或改成固定容量 ring + 设备侧调度表 | 写作或实现 |
| **P2** | 9.1 | push 选择的定量论证(MoK Figure 4 反推 + RocMoE M1 的 43%) | 写作 |
| **P1** | 7.4 | 处理两个指向未发布 blog 的引用 | 写作决策 |
| **P1 零成本** | 8.2 | L5 vendoring 教训成文(证据已在仓里) | 写作 |
| **P2** | 7.7-a/b | Triton-distributed 前向对照 + DeepEP 算子级计时 = roadmap T3-3 | 实验 |
| **P2** | 9.2 | MLSys'26 七篇前作逐篇精读定位 | 阅读 |
| **P2** | 5.5 | "为什么是 FlyDSL 不是 Wave" | 写作 |
| **P2** | 8.5 | liveness 压力实验 | 实验 |
| **P3** | — | gfx1250 第二平台(对冲硬件过时) | 实验 |
| **P3** | — | 标题二选一、作者顺序 | 你 |

---

## 核查溯源

全部数字与图表文件名于 2026-10-28 从 `primus-mega-moe/README.md` 逐行核对。`#486` 的 grad norm 与 `f6d5ab68` 快照来自 `primus_turbo/flydsl/mega/bf16/__init__.py:14-24` 注释原文。§3 中 `use_turbo_mega_moe: false` 默认值、`turbo_mega_moe_precision`、EP-only/TP=1 注释、以及接入文件路径,由我在 `/perf_apps/xiaoming/Primus` 本地只读核验。Triton-distributed 的 `raise NotImplementedError`、默认激活与默认形状来自本地只读检查 `/perf_apps/xiaoming/3rd/Triton-distributed`(checkout `8260bc3`)。

**未核实,不得当作已定**:MLSys 2028 实际 deadline、Authorship Quota 篇数、AI 使用声明是否必需、AE 是否确与录取决定无关、§9.2 七篇前作与本文的实际重叠程度。
