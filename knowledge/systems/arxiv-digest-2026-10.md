# arXiv 速览 2026-10 — megakernel 动态复制、AMD 专家耦合、FP4 格式感知融合

> **检索日期：** 2026-10-12
> **检索范围：** 2026-09-04 ~ 2026-10-08 arXiv 公告；10-09~12 投稿在 10-12 00:55 UTC 查询时尚未进入 mailing
> **方法：** 全量 API 遇 429 后，改扫 `cs.DC/cs.AR/cs.PF/cs.LG/cs.CL` category listing（10 月 2,349 个标题）+ 定向 Atom/Web 查询 + ar5iv 全文抽读；去重对照 `papers/` + 前两轮 digest
> **常设索引：** [`training-optimization-landscape-2026.md`](./training-optimization-landscape-2026.md)
> **上一轮：** [`arxiv-digest-2026-09.md`](./arxiv-digest-2026-09.md)（覆盖到 2026-09-03）
> **复现：** [`scripts/arxiv-scan.py`](../../scripts/arxiv-scan.py)（下次完整窗口仍从 10-01 起，避免把本轮定向检索当成全量覆盖）

---

## 0. 一句话结论

**10 月上旬把主线从“固定放置下怎么 overlap”推进到“如何改变工作放置本身”：MegaFlux 在训练 backward megakernel 内流水化动态专家复制，Expert Coupling 则在 MI300X 上用共激活放置 + token shuffle 直接减少 A2A 总量。** 同期的 Format-Aware Fusion 用 160B-token 训练与开放代码证明，FP4 的优化单位应是“格式 × operand × producer/consumer layout”，不能只比较 E2M1 数值格式。

五条主线：

1. **专家耦合比单纯负载均衡更接近 AMD 当前瓶颈** — [Expert Coupling](https://arxiv.org/abs/2610.09372)（Zyphra）：8×MI300X/node、Megatron-LM、EP8–64；共激活放置 + deduplicating dispatcher + 在 attention RS 中做 token shuffle，A2A **1.16–2.63×**、端到端最高 **1.41×**。
2. **动态复制开始进入完整训练 megakernel** — [MegaFlux](https://arxiv.org/abs/2610.00671)（Princeton + NVIDIA）：on-device planner 决定热专家副本，forward 边传权重边算，backward 边算边归并 replica Wgrad；8×B200 上 F/B geomean **1.45×/1.28×**。但只测单 NVLink 域，且未公开代码。
3. **FP4 配方争论有了长程、可复现的新样本** — [Format-Aware Fusion](https://arxiv.org/abs/2610.00053)：Llama-family 8B 训到 160B tokens，代码/配方/数据 ledger 全公开；MXFP4 + row-gradient SR + fixed-sign H32 达 **37.2K token/s/GPU**，但 raw loss 仍比 BF16 高 **2.11%**。
4. **Scale-up 域上的 MoE 系统设计与 scale-out 不兼容** — [Mixture-of-Kittens](https://arxiv.org/abs/2609.36070)（Stanford + Cursor）：生产 512 GPU 相对 DeepEP 栈 **1.41×**；pull dispatch / push combine、可调 minibatch、设备侧 ring buffer 三点仍是 Helios 基线。
5. **Muon 继续从算法变成超节点系统约束** — [HyperParallel-FSDP](https://arxiv.org/abs/2609.21594)（华为 + 浙大）：505B MoE @768 ranks；layout-driven Muon 应继续对照 Primus。

**负面发现：** 低精度 **wgrad RS / 块缩放梯度通信** 仍无训练侧新玩家；Format-Aware Fusion 优化的是本地 Wgrad operand，不是 collective 线格式。流水线 zero-bubble 仍静默。

---

## 1. 最该先读的十篇

| # | 论文 | arXiv | 投稿 | 为什么排在最前 |
|---|---|---|---|---|
| 1 | **Expert Coupling** | [2610.09372](https://arxiv.org/abs/2610.09372) | 10-07 | 直接在 MI300X + Megatron-LM 验证；端到端最高 1.41×；最适合先做 trace-only 可行性检查 |
| 2 | **MegaFlux** | [2610.00671](https://arxiv.org/abs/2610.00671) | 09-30 | 第一篇把动态专家复制、权重搬运、replica Wgrad 归并一起塞进训练 megakernel；无代码且仅单域 |
| 3 | **Format-Aware Fusion for Fast FP4 Pretraining** | [2610.00053](https://arxiv.org/abs/2610.00053) | 09-04 | 160B-token + 开放代码/recipe ledger；可用于重做 AMD vs NVIDIA FP4 随机性对照 |
| 4 | **HyDra** | [2609.34318](https://arxiv.org/abs/2609.34318) | 09-28 | >11K GPU、256K context 的生产证据；动态 CP 将 2,048 GPU pipeline bubble 36%→14% |
| 5 | **RelayMoE** | [2610.07333](https://arxiv.org/abs/2610.07333) | 10-05 | ring expert/token routing，不物化完整 top-k-expanded dispatch buffer；全模型最高 2.02× |
| 6 | **Mixture-of-Kittens** ✅ 已精读 | [2609.36070](https://arxiv.org/abs/2609.36070) | 09-28 | 专为 NVL72 设计的**全训练** MoE megakernel；点名 Helios → [`papers/mixture-of-kittens.md`](../../papers/mixture-of-kittens.md) |
| 7 | **TopoEP** | [2609.35481](https://arxiv.org/abs/2609.35481) | 09-28 | scale-up/scale-out 分层代价的 GPU-native EP 均衡；已接 Megatron-LM |
| 8 | **HyperParallel-FSDP** | [2609.21594](https://arxiv.org/abs/2609.21594) | 09-18 | 超节点 declarative FSDP + **layout-driven Muon** |
| 9 | **Cobalt** | [2609.36959](https://arxiv.org/abs/2609.36959) | 09-29 | 专家共激活共址；与 Expert Coupling 同线但后者 AMD 证据更直接 |
| 10 | **Weave** | [2609.21483](https://arxiv.org/abs/2609.21483) | 09-18 | 动态切通信/计算 SM；⚠ 推理对照 |

---

## 2. Scale-up MoE / megakernel（本轮重点）

| 论文 | arXiv | 核心 | 与我们的关系 |
|---|---|---|---|
| **Mixture-of-Kittens (MoK)** ✅ 已精读 | [2609.36070](https://arxiv.org/abs/2609.36070) | ① scale-up 上 **pull 与 push 同样便宜** → forward 用 pull dispatch + push combine，消除跨 GPU completion signal（EP64 上 push 信号可占 dispatch 18%）；② overlap **minibatch 粒度**必须可调（最优 512–32768 tokens，错粒度可差 3.53×）；③ NVL72 主机路径比 DGX **慢 1.46–2.97×** → 固定大小 ring buffer + 反向 ring-aware replay，去掉 D2H 分配同步。融合 dispatch / shared+routed FFN / combine 的**确定性训练** megakernel；MXFP8 融合量化、FSDP RDMA overlap、fused router wgrad。形状：Kimi K2.7 / GLM 5.2 / Qwen3.5-397B / DSv4 Pro。vs 最强公开基线最高 **2.37×**（MXFP8 fwd）；Cursor 生产 512×GB300 **1.41×** vs DeepEP 栈 | **Helios / MegaMoE 必读。** 详见 [`papers/mixture-of-kittens.md`](../../papers/mixture-of-kittens.md)。偷三洞察，不搬 CUDA；与 UniEP pull/push 结论相反，须在 MI355X/UALoE 复测 |
| **Weave** | [2609.21483](https://arxiv.org/abs/2609.21483) | 路由结束后才知通信/计算量 → megakernel 内轻量 cost model **按层×卡**切 SM；时间调度压 mid-bubble。4×H100 推理：层 geomean **2.89×**、E2E **1.33×**。基于 ParallelKittens | ⚠ **推理**。与 MoK「可调 SM partition」互补；MegaMoE 若仍 compile-time 固定通信 CU 数，这篇是反例 |
| **mKernel** | [2609.13585](https://arxiv.org/abs/2609.13585) | 多机多卡 fused kernel：tile 级重叠 compute / NVLink / RDMA；GPU command queue + host RDMA proxy；GEMM+AR **1.72×**、Ring Attention **1.88×** | TileLink / Triton-distributed 同族；跨节点 fusion 参考，非 MoE 主线 |

**与 9 月 digest 负面发现的关系：** 「megakernel 断档」在本窗口被 **MoK + Weave** 打破——但热度集中在 **scale-up 域专用设计**，不是 8 月那种通用 AutoMegaKernel 续作。

---

## 3. EP 负载均衡与布局（4 篇）

| 论文 | arXiv | 核心 | 与我们的关系 |
|---|---|---|---|
| **TopoEP** | [2609.35481](https://arxiv.org/abs/2609.35481) | 批评 UltraEP/MoonEP **只看单 scale-up 域**；提出 inter-node 放置 + intra-node 精修的确定性 GPU solver；TMA + **NCCL GIN** 执行；Megatron-LM 集成；32×H800 **+6.2–11.4%**；跨域 token–expert 分配 −96–99% | UltraEP 移植选型的**新对照**。Helios 机架内仍偏 UltraEP；**跨机架 EP / 多 tray 扩展**时 TopoEP 的分层代价模型更对口。仍动梯度归约路径 → **C0-4 之后** |
| **Cobalt** | [2609.36959](https://arxiv.org/abs/2609.36959) | 观察：token 常 **共激活** 多专家对，Ochiai 相似度随训练升高；两阶段布局（周期共址 + 步内节点内再均衡）+ 通信感知 task assignment；32×B200 **1.28–1.89×**，跨节点 token −75–99% | **偷思路**：专家放置目标从「按负载」升级为「共激活 × 负载」。Helios 4 卡托盘 / 8 卡 EP 组内可先做共址启发式，不必整库移植 |
| CascadeEP | [2609.33252](https://arxiv.org/abs/2609.33252) | 推理 prefill 下异步专家执行应对 attention 不均 | 推理侧，低优先 |
| OLED-MoE | [2609.33385](https://arxiv.org/abs/2609.33385) | dLLM 推理专家 offload | 推理侧 |

---

## 4. 分布式训练栈 / Muon / 长序列（4 篇）

| 论文 | arXiv | 核心 | 关键数字 / 关系 |
|---|---|---|---|
| **HyperParallel-FSDP** | [2609.21594](https://arxiv.org/abs/2609.21594) | Dual-mode DTensor（生产零 per-op dispatch + 校验模式）；拓扑感知 FSDP；**layout-driven Muon**（通信域从 sharding 语义推导） | Atlas A3：505B MoE @768 ranks，FSDP 通信仅占步时 **2.9%**；vs FSDP2 −29.7%（16 die）；Muon vs DMuon/MatrixFSDP/TorchTitan **−5.4~16%**。开源 atomgit MindSpore hyper-parallel。**Primus Muon 路径必对照** |
| **NSP** | [2609.22755](https://arxiv.org/abs/2609.22755) | **嵌套序列并行**：长短序列共享 GPU、不同 SP 组大小；树状路由 + inter-level streaming | Qwen3-MoE 至 384K：vs Static SP **1.48×**、vs FlexSP **1.16×**。对标 BASP（9 月 digest）；可进 Projection |
| **Flattening Every Memory Peak** | [2609.14306](https://arxiv.org/abs/2609.14306) | 长上下文 MoE 四类峰值（dispatch / vocab / checkpoint / optim）一并封顶：PipelinedLLEP、Ring-DTP、SCO、OffloadStreamAdamW | 120B–667B、**1M context**；vs FSDP2 吞吐最高 **10.4×**。bitwise 等价调度。长上下文 recipe 待读 |
| DeepSeek-V4-Flash on AMD gfx90a | [2609.15627](https://arxiv.org/abs/2609.15627) | MI200 代上 DSv4-Flash **正确性恢复 + 推理工程** | ⚠ gfx90a / 推理；对 MI355 训练主线旁证弱，但「AMD 上跑通 DSv4」叙事可引用 |

---

## 5. 低精度（本窗口）

| 论文 | arXiv | 核心 | 备注 |
|---|---|---|---|
| **Format-Aware Fusion for Fast FP4 Pretraining** | [2610.00053](https://arxiv.org/abs/2610.00053) | 把 scale completion、packing、双向 layout 与 autograd saved state 作为一个 producer-consumer contract；GB200 上 32 GPU、Llama-family 8B 训到 160B tokens | **代码与 recipe ledger 开放。** MXFP4 + row-gradient SR + fixed-sign H32 为 37.2K token/s/GPU（86.3% BF16 MFU），raw loss endpoint 比 BF16 高 2.11%；说明“随机性是否有害”必须按 operand/格式拆开，不能泛化 |
| FP64 Is All You Want… | [2609.37693](https://arxiv.org/abs/2609.37693) | Ozaki 系用 INT8/FP8/FP4 GEMM **仿真 FP64**；B300 上最高 83× vs native FP64 | 数值/仿真，不是训练配方；与 UE5M3 / MXFP4 预训练线正交 |
| MpFA | [2609.33135](https://arxiv.org/abs/2609.33135) | Blackwell QK4V8 FlashAttention | 推理/attention 低精度 |

**§11 低精度梯度通信：** 定向检索仍 **0 篇训练侧 wgrad RS 压缩新作**。窗口继续开着。

---

## 5.5 10 月上旬新增：放置本身成为优化对象

| 论文 | 核心机制 | 关键数字 | 对 Primus / ROCm 的判断 |
|---|---|---|---|
| **Expert Coupling** [2610.09372](https://arxiv.org/abs/2610.09372) | 训练约 1B tokens 后专家共激活关系稳定；层内把共选专家放到同卡并按目标卡去重 token；层间把下一层 token 迁移折进 attention 后既有 RS | 8×MI300X/node，EP8–64；A2A 1.16–2.63×，step 最高 1.41×；top-2 时 0.8% 专家对承载 42% tokens | **本轮最高 ROI。** 先离线统计现有 checkpoint 的 pair matrix 与目标卡去重率，不改 kernel 即可判断上限。注意 token shuffle 只适用于 TP=EP，Helios EP≤8 主轨先做层内 placement + dedup |
| **MegaFlux** [2610.00671](https://arxiv.org/abs/2610.00671) | on-device planner 按 tile block 复制热专家；Fprop 流水搬权重，Bprop 提前产 replica Wgrad 并在 owner 归并 | 8×B200；vs 固定放置同 kernel，F/B geomean 1.45×/1.28×，峰值 2.14×/2.64×；隐藏复制开销 56–76% / 91–100% | **偷调度，不搬代码。** 给 MegaMoE 加“只有预测收益覆盖搬权重/归并成本才复制”的 gate；论文承认小 backward 最差 0.91×，且没测跨节点 |

这两篇与 9 月的 Cobalt / UltraEP 不重复：Cobalt 优化长期共址，UltraEP 优化单步 exact-load；Expert Coupling 直接减少 A2A 总字节，MegaFlux 则把单步动态复制的额外通信藏进 persistent schedule。合理组合顺序是：**静态共激活放置 → 目标卡 dedup → 剩余长尾再动态复制**。

---

## 5.6 二次扫 category listing 捞回的漏项

| 论文 | 核心与数字 | 优先级 / 边界 |
|---|---|---|
| **HyDra** [2609.34318](https://arxiv.org/abs/2609.34318) | Megatron-Core 上按实际 compute load 动态选 CP；生产 256K job、>11K GPUs 中相近 token 数的 rank time 可差 5×，pipeline bubble 46%。2,048 GPU 将 bubble **36%→14%**，吞吐平均 **1.25×** vs MCore DCP | **P1 长上下文。** Primus 若做 DCP，先按 compute 而非 token 数定 load target，再考虑 ring 内嵌浅 Ulysses |
| **RelayMoE** [2610.07333](https://arxiv.org/abs/2610.07333) | 环形 expert/token routing，避免物化完整 top-k dispatch buffer；30B–57B MoE，单层平均 **2×** vs Megatron-LM，全模型同显存最高 **2.02×** throughput / **2.85×** max sequence | **P1 memory alternate。** activation memory 成为主瓶颈时优先于动态复制；需核查与 MegaMoE persistent layout 是否兼容 |
| **QEffect** [2609.23536](https://arxiv.org/abs/2609.23536) | FP8 pipeline + CUDA Graph 的 state contract；H800 captured layer **1.82–2.79×**，直接 gradient placement 再 **1.132×**，并验证 delayed-scale rollover 的 bitwise parity | **P1 correctness checklist。** 适用于 1F1B/zero-bubble 下 scale generation、cached weight 与 backward buffer lifetime，不是新 schedule |
| **CIERA** [2609.04609](https://arxiv.org/abs/2609.04609) | 复用稳定 exponent 做 sharded MoE weight all-gather 的 lossless compression；16 GPU OLMoE **3.70×** vs lossless baseline，bitwise exact | **P2。** 压的是 param AG，不是 wgrad RS；可作为“先做无损通信压缩”的对照 |
| **RailWave** [2610.03415](https://arxiv.org/abs/2610.03415) | DeepEP rail balance + reusable permutation；GLM-4.5-Air 训练 trace，vs Native 最高 H800 **5.84×** / H20 **4.36×** | **P2 NVIDIA 对照。** 先读 traffic-shaping 思路；不改变“AMD 不 hipify DeepEP V1”的结论 |

仍然没有新的 distributed-Muon 通信系统、FP8/FP4 wgrad RS / block-scaled gradient AllReduce，也没有新的 zero-bubble / 1F1B / DualPipe 调度算法。10 月 Muon 新作集中在算法或 optimizer-state memory；MegaFlux 是本窗口唯一训练 megakernel 命中。

---

## 6. 建议动作（接到 Primus）

按投入产出：

1. **先做 Expert Coupling 的零侵入 trace 检查**：对 Primus 现有 MoE checkpoint 记录每层 top-k，算 expert-pair 共激活矩阵、按 GPU dedup 后的 row reduction、EP8/EP64 与 top-2/top-6 分层结果。达不到论文信号强度就停止，不先改 dispatcher。
2. **用 MegaFlux 重审 MegaMoE 的负载均衡边界**：保留 UltraEP/MoonEP 在 kernel 外做复制的基线，再加一个 cost gate；重点测 backward replica Wgrad 归并能否进入现有 persistent schedule。
3. **复现 Format-Aware Fusion 的最小数值矩阵**：在 MI355X 上只比较 `Wgrad deterministic H16/H32 × Dgrad SR on/off × scale direction`，不要把 SR/RHT 合成一个布尔开关。它与 AMD MLPerf 结论并非直接矛盾：新论文把 SR 只放在 row-facing Dgrad，Wgrad 用 fixed-sign H32。
4. ~~精读 MoK~~ ✅ → [`papers/mixture-of-kittens.md`](../../papers/mixture-of-kittens.md)。派生动作不变：  
   - 默认 **pull dispatch + push combine**（或按算子选方向），重测信号开销  
   - 把 **minibatch / chunk 粒度**做成一等 yaml，扫 512–32K  
   - ring buffer + 反向 replay 对齐 sync-free stage  
   - 用「朴素 RCCL A2A」作 Helios 基线，防止 DeepEP/MegaMoE 在大域上假赢  
5. **TopoEP vs UltraEP 选型表加一列「是否跨域」** — 机架内 UltraEP；跨 tray/机架看 TopoEP。两者都 **C0-4 之后**。  
6. **HyperParallel-FSDP 的 Muon 通信域推导** — 对照 Primus/TorchTitan DistMuon / Megatron Muon，避免慢链路上重复正交化。  
7. **长上下文把 HyDra 升到 P1**；MoE activation memory 先看 RelayMoE。NSP / Flatten peaks / Weave 保持 P2/P3。

---

## 7. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-01 | 初版：09-04~09-30 定向扫描；主线 MoK / TopoEP / Cobalt / HyperParallel-FSDP / Weave；全量 scan 因 429 未完成，已注明 |
| 2026-10-12 | 补 10 月上旬：Expert Coupling / MegaFlux / Format-Aware Fusion；二次 category scan 捞回 HyDra / RelayMoE / QEffect / CIERA / RailWave；修正覆盖终点为 10-08 公告 |
