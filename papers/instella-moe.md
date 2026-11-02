# Instella-MoE Technical Report

> [arXiv 2609.00791](https://arxiv.org/abs/2609.00791) (v1, 2026-09-01, cs.CL / cs.AI) · 13 作者（Jiang Liu 等，Emad Barsoum 收尾）
> 单位：论文正文未印单位，但发布组织为 **AMD**（代码 <https://github.com/AMD-AGI/Instella-MoE>，权重 `huggingface.co/amd/Instella-MoE-16B-A3B-*`）
> 硬件：**AMD Instinct MI300X + MI325X**（ROCm）。长上下文阶段单列在 MI325X 上
> 软件栈：**Primus**（<https://github.com/AMD-AGI/Primus>，Megatron-LM 后端）跑到 DPO 为止；RL 用 **Miles**（slime 生态，Megatron-LM learner + SGLang rollout + Ray）
> 模型：16B 总参 / 2.8B 激活 · 27 层（1 dense + 26 MoE）· 64 routed + 2 shared expert · top-6 · 7.1T tokens · 4K→64K 上下文
> 释出内容：**六个阶段的全部 checkpoint + 训练配置 + 数据配比 + 训练代码 + 评测协议**

---

## TL;DR

**这是目前唯一一份"AMD 硬件 + 全流程 + 全开源"的 MoE 训练报告，而且它是用 Primus 训的——对我们来说这不是一篇外部论文，是我们自己框架的第一份 7.1T token 规模的公开验收报告。**

系统侧只有一个核心贡献：**FarSkip-Collective**。它不是 kernel 优化，而是**改架构**——把 MoE 层的数据依赖拆掉，让 Dispatch / Combine 两个 all-to-all 能与 attention 和 shared expert 计算重叠，代价是下游拿到的是**过时或不完整的激活**。预训练吞吐 +12.7%，200B token 对照下质量基本持平（50.33 → 50.38）。

**最该记住的四条：**

1. **AMD 自己选的路线是"改架构 + 纯 PyTorch"，不是"写 megakernel"。** Appendix A 明说**刻意不用 custom kernel**以保证可移植性，重叠靠 `torch.distributed` 的 `async_op=True` + `.wait()`，反向靠 race-safe 异步通信算子加 PyTorch Sequence Number API 重排优先级。12.7% 就是这么拿到的。这对 RocMoE / MonolithEP 的超级核路线既是参照也是压力测试——**我们要论证的是超过这 12.7% 的那部分收益值不值那么多工程量**。

2. **但这个 12.7% 的可扩展性有硬伤，而且论文自己写了线索。** Appendix A 承认重叠"relies on sharing CUs between communication and computation kernels"——这正是 [Tessera](./tessera.md)（EP 通信 kernel 占 ~20 SM 致 10–20% 减速）和 [Motif 3](./motif-3.md)（大规模下"消除"通信优于"隐藏"通信）两篇独立指出的病灶。**Instella-MoE 的 EP=8 正好落在单节点内**，XGMI 域内争抢 CU 的代价最小；跨节点是否还成立，论文没测。

3. **一个可以直接拿去用的 MoE 专属发现：DPO 阶段必须关掉负载均衡。** 开着的时候，54% 的 token 其 top-1 专家与 SFT 不同、top-6 集合的 Jaccard 只有 0.42；关掉后是 5% 和 0.94。**同一套配方在 dense 模型上不出现这个问题。** 原因是 loss-free bias 更新与 aux loss 系数都是按预训练/SFT 的高学习率标定的，而 DPO 的 LR 是 `8e-8`，低了四个数量级。

4. **⚠ 全文没有 MFU、没有绝对吞吐、没有预训练卡数、没有 GPU-hours。** 12.7% 与 39.2% 都出自 Figure 8 的**归一化**柱状图，基线是"standard MoE expert-parallel training baseline"，没有说这个基线调优到什么程度。这与 [Motif 3](./motif-3.md) 是同一种病（见该文 TL;DR）。**不能把 12.7% 当作 FarSkip 的上限或下限引用，只能当作"在他们那套未公开配置下的一次观测"。**

**一个必须打问号的地方**：FarSkip 的质量证据是 200B token、**单次运行**、平均分差 `+0.05`（50.33 → 50.38）。这个差值远在单run噪声之内，所以"质量中性"这个结论**在 200B 尺度上只能说没被证伪，不能说被证实**；而 7.1T 的正式训练是**带着 FarSkip 跑的，没有同规模对照组**。激活过时性会不会在 35 倍 token 量上累积，论文无法回答。

---

## 1. 定位：这是模型报告，但系统含量集中在两处

先说清楚该怎么读。这篇的主张是「一个 2.8B 激活的全开源 MoE 能打过 7B 级全开源 dense 和同级 open-weight 模型」。绝大部分篇幅是数据配比、SFT/DPO/RL 配方与评测，对我们有用的只有：

- **值得精读**：§2.1.3（FarSkip-Collective）、§2.3（训练基础设施）、Appendix A（重叠实现）、§2.8（DPO 路由漂移）、§2.9.1（RL 的 MoE 训推一致性）。
- **值得扫**：§2.1.2（Gated MLA）、§3.3.1（架构消融）、§2.6 训练细节（64K 的显存策略）。
- **可以跳**：§2.4–2.7 的数据配比、§2.9.2–2.9.3 的 RL 目标函数、Appendix B/C。
- **不能当依据的**：任何"FarSkip 在 X 规模上能带来 Y%"的外推——论文只有一个点。

### 1.1 fully open 与 open-weight 的区别是这篇的立论基础

论文反复强调自己是 **fully open**（权重 + 数据 + 配方 + 代码 + 评测协议）而非 open-weight（只有权重）。对比对象里 OLMo-3、SmolLM3、OLMoE 是 fully open，Qwen3.5、Gemma-4、Moonlight、DeepSeek-V2-Lite 是 open-weight。**对我们的实际意义**：这是少数几个可以逐阶段复现、且复现目标平台就是 AMD 的 MoE 训练流程。

---

## 2. 架构

### 2.1 超参一览

| 项 | 值 |
|---|---|
| 总参 / 激活 | 16B / 2.8B |
| 层数 | 27（1 dense + 26 MoE） |
| Hidden size | 2,048 |
| Tokenizer / 词表 | DeepSeek-V3 / 128,896 |
| 上下文（base / 扩展后） | 4,096 / 65,536 |
| Attention | Gated MLA，16 头，KV latent rank 512 |
| QK 维度（content / RoPE） | 96 / 32；value head dim 128 |
| Routed / active expert | 64 / top-6；另有 2 个 shared（融合成一个 FFN，中间维 2,816） |
| Routed expert FFN 中间维 | 1,408 |
| Router top-K 缩放 | 2.5（follow DeepSeek-V3） |
| 归一化 / 激活 / 位置编码 | RMSNorm / SwiGLU / RoPE |
| 辅助目标 | MTP 1 层（仅预训练与 mid-training） |

负载均衡用 DeepSeek-V3 的 **bias-based loss-free balancing**（per-expert bias 只加在 top-K 选择用的分数上、不加进混合权重，步间以 `1e-3` 速率朝目标负载更新），外加系数 `1e-4` 的序列级 aux loss 作次要正则。MTP loss 权重预训练 0.3、mid-training 0.1，长上下文阶段关闭以省激活显存。

### 2.2 Gated MLA：+0.47 的平均分，但胜负项接近对半

在 MLA 的 attention 输出与输出投影之间插一个输入条件的 sigmoid 逐元素门：

`y_t = W_o [ Sigmoid(W_g x_t) ⊙ ô_t ]`

其中 `x_t` 是 RMSNorm 之后的层输入，`ô_t` 是拼接后的多头注意力输出。门让每个 token 能调制每个头内的各个 value 通道。

**⚠ "平均 +0.47" 掩盖了相当混杂的逐项结果。** 200B token 消融（Table 13，单次运行）里 Gated MLA 相对 vanilla MLA：

| 方向 | 项目 |
|---|---|
| 赢 7 项 | MMLU **+4.26**、HumanEval+ **+4.88**、MBPP+ **+2.91**、GPQA +1.78、OBQA +1.60、BoolQ +1.22、ARC-C +0.60 |
| 输 8 项 | BBH **−2.89**、SciQ −1.90、WinoGrande −1.42、GSM8K −1.29、MATH −1.28、HellaSwag −0.75、ARC-E −0.46、PIQA −0.21 |

也就是说**平均分完全由 MMLU 与两个代码基准这三项撑起来**，而数学与常识推理是净负的。论文的表述（"largest gains on MMLU and code generation"）本身没有说谎，但把它读成"Gated MLA 全面更好"是错的。

### 2.3 FarSkip-Collective：本篇唯一的系统贡献

**问题。** 标准 MoE 层四段：router → Dispatch(all-to-all) → expert 计算 → Combine(all-to-all)。Dispatch 依赖上一层输出，Combine 依赖 expert 计算完成。**这条依赖链使得两个 all-to-all 都没有可与之重叠的独立计算**。

**做法。** 改层间连接，让下游用"手头已有的"激活而不是"正确的"激活。记 `h_k` 为 k 层后的输出、`f_i` 为第 i 个子块：

标准 Transformer：`h_k = f_1(h_0) + f_2(h_1) + … + f_k(h_{k-1})`

FarSkip：`h_{k+1} = h_k + f_{k+1}(ĥ_k) = f_1(h_0) + f_2(ĥ_1) + … + f_{k+1}(ĥ_k)`

`ĥ_k` 是 `h_k` **少了若干加数**的版本。两处具体取法：

- **MoE 输入**：`moe-in_k := h_{k-1}`。即 Dispatch 的输入不依赖本层 attention 输出，于是**上一层 Combine 一结束就能发起本层 Dispatch 与路由**，传播延迟最多一层。
- **Attention 输入**：`attn-in_k := h_{k-2} + attn-out_{k-1} + shared-exp-out_{k-1}`。只省掉 routed expert 那一项，于是 routed expert 的 Combine 可以与 shared expert 计算及 attention 的前半段重叠。

**净效果**：在不做流水线的前提下，Dispatch 与 Combine 的重叠窗口被拉长到**整个层的时长**，只除去 router 与 routed-expert 计算本身。

**实现（Appendix A，这段最有工程价值）：**

- **刻意不写 custom kernel**，理由是可移植性。
- 前向：显式排序模型操作，用 `torch.distributed` 的 `async_op=True` 排队 + `.wait()` 保证安全访问。
- 反向：`torch.autograd` 下拿不到显式算子顺序和 `.wait()` 插入点。解法来自 Dukler et al. 2026 的 Appendix C——**race-safe 的异步反向通信算子**，配合 **PyTorch Sequence Number API**（`aten/src/ATen/SequenceNumber.h`）做优先级重排。
- 推理侧另做了一套通信重叠的 EP 生成，同时支持 TP attention 与 DP attention；**并且通过聚合通信把多个张量先求和再发，直接减少通信量**。

**收益**：预训练吞吐 **+12.7%**；8×MI325X 上用 SGLang 服务时 TTFT 吞吐 **+39.2%**。

---

## 3. 训练基础设施（对我们最相关的一节）

| 阶段 | 框架 | 硬件 | 并行与精度 |
|---|---|---|---|
| 预训练 → DPO | **Primus**，Megatron-LM 后端，ROCm | MI300X + MI325X | EP=8（64 个 routed expert 分到 8 卡）、TP、FlashAttention、bf16 混合精度；梯度与参数通信与计算重叠；预训练与 mid-training 额外用 FarSkip 重叠 EP all-to-all |
| 长上下文（64K） | 同上 | MI325X | EP=8 + **context parallelism** 切序列维；**全量重算** + bf16，micro-batch 只能到 1；**不做前反向通信重叠**；优化器状态重新初始化 |
| RL（IF-RL / MOPD） | **Miles**（slime 生态） | — | 3 节点 24 卡：8 卡 learner（EP=8 + distributed optimizer），16 卡跑 2 个 SGLang rollout 引擎（各 TP=8, EP=8, DP attention）；每个 optimizer step 后把权重流回 rollout 引擎 |

**三点值得单独记：**

1. **Primus 被用在了一次 7.1T token 的真实预训练上**，并且是 AMD 官方模型。这是 Primus 目前能拿到的最强背书，也意味着**这条路径上的坑他们已经踩过一遍**——发布的训练配置（Table 3）是可直接对照的基线。

2. **64K 阶段把 FarSkip 关了。** 原文：*"the 64K-context training does not overlap forward and backward communication"*。**这恰恰是通信压力最大的场景。** 论文没解释原因，合理猜测是 CP 与 FarSkip 的激活依赖改写互相冲突，或是全量重算下重叠窗口本就被吃掉了。**这是 FarSkip 适用边界的一个明确证据，值得我们单独验证。**

3. **EP=8 全程不变**，从预训练一路到 RL learner。64 个专家 / 8 卡 = 每卡 8 个专家，all-to-all 完全落在单节点 XGMI 内。**论文从头到尾没有跨节点 EP 的数据。**

---

## 4. 训练流程与关键配置

| 阶段 | 数据量 | 序列长 | Global BS | Peak LR | 调度 | 关键设置 |
|---|---|---|---|---|---|---|
| 预训练 | 7.1T | 4,096 | 4,096 | 4e-4 | WSD | MTP、EOD masking、FarSkip |
| Mid-training | ~100B | 4,096 | 1,024 | 2e-4 | Linear | 3 个 mixture 变体 + 等权 souping |
| 长上下文 S1 | ~194B | 65,536 | 128 | 2e-4 | Linear | YaRN（θ=8e6）、doc masking |
| 长上下文 S2 | ~20B | 65,536 | 128 | 1e-4 | Linear | STEM 定向配比 |
| SFT | ~2.9M 例 | 32,768 | 128 | 1e-4 | Linear | 2 epoch、序列打包、反馈驱动筛选 |
| DPO | 150K 对 | 16,384 | 256 | **8e-8** | Linear | β=5.0、0.75 epoch |
| IF-RL | 29.8K prompt | 16,384 | rollout 512 | 1e-6 | Constant | GRPO + DAPO + R3、1,400 步 |
| MOPD | on-policy | 16,384 | rollout 512 | 1e-6 | Constant | 领域路由的多教师 |

优化器 AdamW，`β=(0.9, 0.95)`，weight decay 0.1，2,000 步 warmup，最低 LR 4e-5；7.1T 结束时训练 CE 约 **1.7**。

几个配方层面值得记的：

- **Mid-training 用 model souping 而不是选最优。** 三个只在少数 STEM/reasoning 子集上不同的变体（79.23 / 79.11 / 79.73），等权平均后是 **80.90**——超过任何单个变体 1.2 分以上。
- **长上下文两阶段是必要的。** Stage 1（194B @64K）把长文能力做上去（HELMET 均值 43.7、RULER 83.9，胜过 OLMo-3-7B），但**短上下文 STEM 严重回退**：GSM8K 77.1 → 62.8，HumanEval+ 59.9 → 55.0。Stage 2 用 20B 的 STEM 定向配比拉回到 81.5 / 65.7，代价是长文回落到 41.5 / 79.4。
- **SFT 末段的 feedback-driven curation 值 1.5 分**（72.85 → 74.37）：让学生做题 → judge 打分并对每个错误产出结构化错因分析 → 更大的 reflection 模型把错因聚合成带权检索 query → 按权重从候选池里做近邻检索。总预算 512K 例。

---

## 5. 两个 MoE 专属的工程发现

### 5.1 DPO 阶段的专家路由漂移（§2.8）

**现象**：直接套 DPO，loss 收敛，但下游准确率**低于** SFT checkpoint。同一套配方在 dense 模型上不出现。

**诊断**：DPO 的 LR 是 `8e-8`，比预训练/SFT 低几个数量级，而 loss-free bias 更新速率（`1e-3`）和 aux loss 系数（`1e-4`）都是按高 LR 标定的。于是**负载均衡机制的更新量相对于策略更新量被放大**，把路由从 SFT 的分布上推走了。

**证据**（Figure 6，26 个 MoE 层，两次仅在是否开负载均衡上不同的 DPO 运行）：

| 负载均衡 | top-1 专家与 SFT 不同的 token 比例 | 与 SFT 的 top-6 专家集合 Jaccard |
|---|---|---|
| 开 | **54%** | **0.42** |
| 关 | 5% | 0.94 |

**处置**：DPO 期间同时冻结 bias 更新并把 aux loss 系数置零。

> **对我们的意义**：这是一条便宜、可复制、且与框架无关的结论——**任何在 MoE 上做低 LR 后训练（DPO / 轻量 SFT / 蒸馏）的流程，都应该先检查负载均衡机制的更新速率是否与主 LR 同量级**。Primus 侧应当把"后训练阶段自动关闭 LB"做成配置项而不是靠人记。

### 5.2 RL 的训推路由一致性（§2.9.1）

SGLang rollout 与 Megatron 训练前向**可能把同一个 token 路由到不同专家**，于是训练时的 `log π^train` 并不是采样时那个分布下的值。两级处置：

1. **Rollout Routing Replay（R3）**：把采样时的专家分配决策记录下来，在训练前向里回放，使 `log π^train` 沿**生成该 token 的那条专家路径**求值。
2. **token 级截断重要性采样（TIS）**：修正推理与训练引擎之间的残余数值失配，逐 token 比值夹在有界区间内。

> 这个问题在 dense 模型上不存在，是 MoE + 分离式 RL 架构特有的。如果 Primus 要做 RL 后训练后端，R3 是必须项而非可选项。

---

## 6. 实验证据与其强度

### 6.1 主结果

| 维度 | Instella-MoE | 最强 fully open 对手 | 最强 open-weight 对手 |
|---|---|---|---|
| Base 平均（13 个预训练基准） | **76.7** | OLMo-3-7B 70.1 / SmolLM3-3B 70.5 / OLMoE-1B-7B 61.9 | Qwen3.5-4B-Base **79.5**（激活更多）；Moonlight-16B-A3B 76.2 |
| 长上下文（HELMET / RULER，≤64K） | 41.5 / 79.4 | OLMo-3-7B 43.1 / 80.2 | — |
| 后训练平均（12 个基准） | **73.22**（Think） | OLMo-3-7B-Think 71.97 | Gemma-4-E4B 70.47 / Qwen3.5-4B 69.73 |

后训练三级递进：SFT 71.58 → DPO 72.67 → Think 73.22。RL 消融（11 基准）：DPO 75.0 → IF-RL 74.0（IFEval 77.1→84.1 但 AIME/GPQA 回退）→ MOPD 75.7（IFEval 保住 83.7，其余回到 DPO 水平）。**MOPD 的价值是"把单领域 RL 的收益并回来而不牺牲其他领域"，这个结论有干净的消融支撑。**

### 6.2 系统结果的证据强度：弱

| 声明 | 证据 | 强度评估 |
|---|---|---|
| FarSkip 预训练吞吐 +12.7% | Figure 8 左，归一化柱状图 | ⚠ **无绝对值、无卡数、无 MFU、无基线调优程度说明**。只能当单点观测 |
| SGLang EP 服务 TTFT +39.2% | Figure 8 右，8×MI325X | 同上；至少给了卡数和卡型 |
| FarSkip 质量中性 | Table 13，200B token，单次运行，50.33 → 50.38 | ⚠ **差值 +0.05 在单 run 噪声内**。只能说"未被证伪" |
| Gated MLA 提升质量 | Table 13，同上，49.86 → 50.33 | 平均 +0.47，但 15 项里输 8 项（见 §2.2） |
| DPO 关负载均衡有效 | Figure 6，两次受控运行 | ✅ **本篇最扎实的系统证据**，机制清楚、对照干净、效应量大（54% vs 5%） |

---

## 7. 局限与风险

**论文自述的**：几乎没有专门的 limitations 小节。仅在 §2.6 承认 Stage 2 用长文能力换短文 STEM 能力，并称这个取舍对通用 base 模型是合适的。

**未述但重要的**：

1. **系统数字不可复算。** 无 MFU / 无绝对吞吐 / 无预训练卡数 / 无 GPU-hours / 无 wall-clock。12.7% 的分母是什么状态的基线，读者无从判断。
2. **EP 全程 =8，从未跨节点。** FarSkip 的收益建立在"通信 kernel 与计算 kernel 共享 CU"上；单节点 XGMI 内争抢代价最小。跨节点 RDMA 下这套还成不成立，本篇无法回答——而 [Perseus](./perseus.md) 在多节点上量到过 megakernel 通信退化最多 10×，[Tessera](./tessera.md) 与 [Motif 3](./motif-3.md) 都指出通信 kernel 吃 SM 会反噬。
3. **FarSkip 在最需要它的场景被关掉了**（64K 长上下文）。这一点论文只用一句陈述带过，没有解释。
4. **激活过时性的规模效应未验证。** 200B token 的对照 vs 7.1T 的正式训练，差 35 倍，且正式训练没有 non-FarSkip 对照组。
5. **架构消融全部 single run**（Table 13 脚注明写 "all results are from a single run"），却用来支撑两个架构决策。
6. **Gated MLA 与 FarSkip 的消融是累加式而非析因式**：只有 Vanilla、+Gated、+Gated+FarSkip 三行，没有"Vanilla+FarSkip"。因此**无法分离 FarSkip 单独作用于标准 MLA 时的影响**。

---

## 8. 与我们的关系

### 8.1 直接可用的三条

1. **把 Table 3 的训练配置当 Primus 的对照基线。** 我们在 MI300X/MI325X 上跑 MoE 时，任何"这个配置合不合理"的争论，现在有了一个真实跑过 7.1T token 的参照点。
2. **后训练阶段自动关闭负载均衡**（§5.1）——建议在 Primus 里做成配置项。这是本篇唯一一条零成本、证据扎实、且我们大概率还没处理的问题。
3. **R3（Rollout Routing Replay）是 MoE + 分离式 RL 的必须项**（§5.2）。如果 Primus 要接 RL 后训练，这条要在设计阶段就进去。

### 8.2 需要我们自己回答的三个问题

1. **FarSkip 的 12.7% 与超级核路线是什么关系？** 两者并不互斥——FarSkip 改的是架构依赖，超级核改的是执行粒度。但如果一个纯 PyTorch 的架构改动就能拿 12.7%，那么 RocMoE / MonolithEP 必须论证自己在**同一基线之上**还能拿多少。**建议把 FarSkip 作为 RocMoE 的一个基线档位复现，而不是只与朴素 EP 对比**——参见 [`.cursor/skills/rocmoe-dev-loop/SKILL.md`](../.cursor/skills/rocmoe-dev-loop/SKILL.md) 里"固定基线"那条规则。
2. **FarSkip 为什么在 64K + CP 下被关掉？** 这是我们能低成本验证的：拿他们发布的代码，在 MI355X 上开 CP + FarSkip 看是正确性问题、性能问题，还是仅仅没做适配。答案会直接影响我们对长上下文 MoE 的方案选择。
3. **跨节点 EP 下 FarSkip 还剩多少？** 与 [`knowledge/systems/mi455x-parallelism-strategy-reference.md`](../knowledge/systems/mi455x-parallelism-strategy-reference.md) 的判断相关——如果 Helios 域内 72 卡确实把通信墙用拓扑消掉，那么 FarSkip 这类"隐藏通信"的架构改动在 MI455X 上的边际价值会下降，而 [TerraceMoE](https://arxiv.org/abs/2608.27874) 的层次化两跳 A2A 反而更值得投。

### 8.3 与已有笔记的交叉

| 已有笔记 | 关系 |
|---|---|
| [`motif-3.md`](./motif-3.md) | 同为 MoE 模型报告 + 系统章节；Motif 3 主张大规模下**消除**通信优于隐藏，Instella-MoE 走的正是**隐藏**。两者规模差一个数量级（314B/12.5T vs 16B/7.1T），**分歧可能只是规模的函数** |
| [`tessera.md`](./tessera.md) | EP 通信 kernel 占 ~20 SM 的实测，是 FarSkip"共享 CU"假设的反面证据 |
| [`megatron-core-moe.md`](./megatron-core-moe.md) | NVL72 上"通信墙被拓扑消掉"的结论，与 FarSkip 的价值前提直接相关 |
| [`knowledge/systems/arxiv-digest-2026-09.md`](../knowledge/systems/arxiv-digest-2026-09.md) | 本篇在该轮扫描里被列为精读第一优先 |

---

## 9. 术语

- **FarSkip-Collective** → 改层间连接、让下游消费过时/不完整激活，从而把 EP 的 all-to-all 与独立计算重叠的架构模式。
- **Gated MLA** → MLA 的注意力输出与输出投影之间插入输入条件 sigmoid 逐元素门。
- **Model souping** → 把多个只在数据配比上不同的 checkpoint 等权平均。
- **MTP** → Multi-Token Prediction，额外预测下一个之后的 token 作为辅助目标。
- **MOPD** → Multi-Teacher On-Policy Distillation，按领域把学生的 on-policy rollout 路由给不同教师，用 token 级 reverse-KL 更新。
- **R3** → Rollout Routing Replay，把采样期的专家分配记录下来在训练前向回放。
- **TIS** → 截断重要性采样，修正推理与训练引擎间残余的数值失配。
- **Delta learning** → DPO 数据构造思路：驱动偏好学习的是 chosen 与 rejected 的**质量差**而非绝对质量，故用强模型轨迹对比明显更弱的模型轨迹。

---

## 10. 复现清单

| 项 | 状态 |
|---|---|
| 权重 | ✅ 六个阶段全部释出（Pretrain / Midtrain / Base / SFT / DPO / Think），`huggingface.co/amd/Instella-MoE-16B-A3B-*` |
| 训练代码 | ✅ <https://github.com/AMD-AGI/Instella-MoE>，含 FarSkip 重叠实现 |
| 训练配置 | ✅ Table 3 逐阶段超参 |
| 数据 | ✅ Table 4 列出全部公开语料来源；配比在发布中 |
| 评测协议 | ✅ OLMES-style，细节在 HF model card |
| 随机种子 | ⚠ 消融称"固定 settings 与 seeds"，未给具体值 |
| 硬件规模 | ❌ **预训练卡数、GPU-hours、wall-clock 全部缺失** |
| 性能基线定义 | ❌ "standard MoE expert-parallel baseline" 未进一步定义 |
