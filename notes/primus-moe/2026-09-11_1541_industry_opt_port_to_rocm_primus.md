# 业界训练优化 → ROCm / Primus：移植深挖

> **When**: 2026-09-11 15:41 UTC+8
> **Where**: 桌面调研（GitHub / arXiv 当日抓取；无新上机实验）
> **Context**: 把 [`industry-training-optimization-2026.md`](../../knowledge/systems/industry-training-optimization-2026.md) 与 [`2026-09-07_0900_moonep-vs-ultraep-port-selection.md`](./2026-09-07_0900_moonep-vs-ultraep-port-selection.md) 从「搬哪个仓库」推进到「怎么接到 Primus、AMD 卡在哪、该不该搬」
>
> ⚠ **后续扫描见 [`2026-10-28_0546_industry-scan-update.md`](./2026-10-28_0546_industry-scan-update.md)**,其中更正了本文多条判断:**§3.4 的 vendor gap 表有两条是读了已废弃仓库得出的**(`ROCm/rccl` 与 `ROCm/hipBLASLt` 均已迁至 `rocm-systems` / `rocm-libraries`)—— RCCL 2.30.7 其实已带 GIN device API,hipBLASLt 的 FP8 grouped GEMM 2024-11 就合了;**§3.3 顺位 #2 的 RCCLX** 所在的 torchcomms 整库已于 10-27 弃用;**§3.3 顺位 #3 的 UltraEP 已解冻**并把 NVSHMEM 换成 torch SymmetricMemory;**§4.4 的 Triton-distributed LLVM 锁死**已随 out-of-tree plugin 改制失效。四分法本身(偷思路 / 搬代码 / 不要搬 / vendor gap)与 §4.3「重叠不免费」仍然成立,后者在新窗口得到第四次独立观察。

## TL;DR

上一轮排的是 **仓库顺位**（MORI → UniEP 补反向 → UltraEP → RCCLX → MoonEP）。这一轮把同一批材料按 **移植模式** 重切之后，排序变了：

1. **先修自家基线，再谈横向移植。** MegaMoE 在当前容器栈上 XGMI dispatch 从 368–381 GB/s 塌到 50.8 GB/s（正确性全绿），任何「MegaMoE vs MORI vs Triton-distributed」对比都做不了。
2. **最高 ROI 的一批根本不是 HIP-ify。** Motif 3 的「MXFP8 在 EP dispatch 之前量化」、Instella 的「DPO 关负载均衡」、逐层选 CP、linghe 的 routing-map padding、UBEP 的 Sentinel Polling，都是调度 / 配置 / 同步原语，blast radius 远小于再搬一个 EP 库。
3. **值得搬代码的只剩三条，而且都排在 C0-4 之后：** MORI-EP 替换 turbo `deep_ep`（已在同机同口径测到 dispatch **1.21×** / combine **1.58×**）、RCCLX 低精度 collective 评估、UltraEP 的 HIP-ify。MoonEP 官方设备列表写成 **NVIDIA + 真武 PPU**，不是 AMD 候选。
4. **DeepEP V2 把 Primus 现有路径判成死胡同。** 上游 main 已切 NCCL Gin + `ElasticBuffer`，本周还在修 mixed-fabric GIN fence 与 NCCL Device API。继续在 V1 / NVSHMEM 语义上做 ROCm 映射，是在追一个正在被废弃的后端。AMD 侧的 V2 对等物是 **MORI + torch SymmetricMemory**，不是再 hipify 一份 DeepEP。

交互式总表见 [primus-rocm-port-map](/home/xiaompen/.cursor/projects/perf-apps-xiaoming-slab/canvases/primus-rocm-port-map.canvas.tsx)。

---

## 1. 上一轮覆盖了什么，这一轮补哪一层

| 文档 | 粒度 | 留下的洞 |
|---|---|---|
| [`industry-training-optimization-2026.md`](../../knowledge/systems/industry-training-optimization-2026.md) | 大厂在投什么（2026-03~08） | §4 是「该读哪篇」，不是「接到 Primus 的哪一层」 |
| [`training-optimization-landscape-2026.md`](../../knowledge/systems/training-optimization-landscape-2026.md) | arXiv 主题全景 | 优先队列（FlashOverlap / Canzona / CODA）未做可移植性判定 |
| [`2026-09-07_0900_moonep-vs-ultraep-port-selection.md`](./2026-09-07_0900_moonep-vs-ultraep-port-selection.md) | EP 负载均衡仓库选型 + 厂商扫描 | 把工具链、runtime、调度层、vendor gap 压进同一张顺位表 |
| 本篇 | **移植模式 × Primus 插入点 × AMD 就绪度 × roadmap 槽位** | — |

上一轮的仓库排序本身没有被推翻。被推翻的是「排序 = 接下来该做的事」。Primus 现在最大的风险是上游欠债与正确性 CI（roadmap C0），不是缺另一个 fused MoE kernel。

---

## 2. 四种移植模式（先选模式，再选仓库）

| 模式 | 定义 | 典型动作 | 何时用 |
|---|---|---|---|
| **偷思路** | 论文 / 生产报告里的调度、量化位置、同步原语、配置默认值 | 在 Primus / Turbo / Projection 里重写 10–200 行 | blast radius 小、与硬件无关、或我们已有对等 kernel |
| **搬代码** | 开源 runtime / kernel，需要 HIP-ify 或接现成 ROCm 后端 | 接 MORI、HIP-ify UltraEP、评估 RCCLX | 有 MIT/BSD、有 AMD 证据或可测基线、且 Primus 插入点干净 |
| **不要搬** | 闭源、与现有旗舰撞车、生产里被否掉、仓库已休眠、设备列表不含 AMD | 只保留设计约束 | 见 §5 |
| **vendor gap** | 通信库 / ISA / 框架 API 缺面，GitHub 上没有可搬的实现 | 提 RCCL / 上游 PR，或在 MORI 上自建 API 面 | NCCL Gin、Copy Engine collectives、packed FP4 all-gather |

**判定规则：** 如果一项优化不碰参数物理布局、不碰梯度归约序、可以写成 yaml / dispatcher 开关，它就是偷思路，不该进「移植 UltraEP」那种项目。Tessera 把这条写成了原则：只动时序、不动归约序 ⇒ bitwise 等价是构造出来的。

---

## 3. 现在该做的事（按 ROI，不是按论文热度）

### 3.1 硬前置（不做后面的对比都不可信）

| # | 动作 | 为什么现在是第一 | 证据 |
|---|---|---|---|
| G0 | **定位并修 MegaMoE 回归** | 当前栈 grouped GEMM 8.7%、XGMI 50.8 GB/s；正确性 `cos=1.00000` | [`../MegaMoeFlydsl/2026-09-07_1034_mega_bf16_dispatch_regression_vs_0716_baseline_mi355x.md`](../MegaMoeFlydsl/2026-09-07_1034_mega_bf16_dispatch_regression_vs_0716_baseline_mi355x.md) |
| G1 | **C0-4 多机 loss golden**（2 节点 bf16 + fp8，DeepEP 开/关） | UltraEP / MoonEP / 任何 EP 梯度路径都属于 DeepEP 静默错梯度那个 bug 家族 | roadmap C0-4；DeepEP 调查 620 行 |
| G2 | **MORI 评估必须显式传 launch 参数** | 默认参数比调优表低 5–9%；turbo `deep_ep` 的 `num_sms` 默认 20，调到 256 才是对手 | 9-07 同机重测 |

### 3.2 偷思路（本周就能立项，不必等 HIP-ify）

这些是上一轮扫描里被当成「边角」或「论文启示」、但插入 Primus 的成本最低的一批。

| 项 | 来源 | 接到哪 | 预期 | AMD 卡点 |
|---|---|---|---|---|
| **EP dispatch 之前做 MXFP8** | Motif 3 §3 / [`papers/motif-3.md`](../../papers/motif-3.md) §8.4 | Turbo dispatcher：量化从 expert GEMM 前移到 dispatch 前 | 通信量 / top-k，量化次数从每专家一次变一次 | **最便宜的一条**。只要 EP 库能搬 MXFP8 张量。与 C2-6「拆掉 quant↔GEMM 格式契约」同向 |
| **DPO / 低 LR 后训练关负载均衡** | Instella-MoE（**就是用 Primus 训的**） | 配置项：后训练 recipe 关 bias / aux loss | 开着时 54% token 的 top-1 专家与 SFT 不同，关掉后 5% | 零 kernel。roadmap C0-7 还在宣称有 RL workflow，这一项比补 RL 栈更便宜、也更诚实 |
| **逐层选 CP + LPT 重排** | Motif 3 §3.5 / §8.5 | Primus Projection + dataloader | packing 造成 rank 间 attention FLOPs 差 3.54×；纯调度、梯度等价 | 需要文档边界；与硬件无关 |
| **routing-map padding 替代 activation padding** | [linghe](https://github.com/inclusionAI/linghe)（MIT，Triton） | Turbo permute | H800 上 permute 1.69× / unpermute 1.23× | Triton ⇒ ROCm 可跑。仓库 2026-04-28 后未推，**搬 trick 不搬库** |
| **fused linear gate / fp32 router GEMM** | 同上 | Turbo router | gate bwd 7.06×、router fwd 3.93×（H800 microbench） | 与 Primus-Turbo 重叠，只挑这两处对一下现有实现 |
| **Sentinel Polling** | UBEP（SIGCOMM'26）[`papers/ubep.md`](../../papers/ubep.md) | RocMoE / MegaMoE IPC dispatch | 接收侧预填哨兵、发送裸 payload，payload 效率 100%，**不需要 512B 原子写** | 可替换现在的 `atomicAdd_system` + ready flag。数值位型要做一次全局校验 |
| **shadow microbatch / 解耦 `M % K == 0`** | Tessera [`papers/tessera.md`](../../papers/tessera.md) §7.6 | pipeline runtime | ~0.5% 开销换掉超参被系统约束绑架 | 纯调度 |
| **确定性 token 顺序** | UniEP | 任何 overlap kernel 的契约 | overlap 下与串行执行数值一致 | 补反向时必须继承；否则重蹈 DeepEP 错梯度 |
| **FarSkip-Collective 做成可选 recipe** | Instella-MoE | Megatron 后端模型定义，不是 kernel | 他们在 Primus 上拿到预训练 +12.7%（归一化柱、基线未公开） | 依赖通信与计算 **共享 CU**——Tessera / Motif 3 两次指出这是大规模病灶。EP=8 单节点内也许划算，跨节点不要默认开。64K + CP 阶段他们自己关了 |

**FarSkip 的特殊地位：** 这不是外部论文，是 AMD 用 Primus 训 Instella 时已经落地的架构改动。要做的不是移植，是 **产品化成开关**，并写清「EP 出节点 / 长上下文 / CU 争抢」三个关闭条件。

### 3.3 搬代码（C0-4 之后，三条就够）

| 顺位 | 目标 | 层次 | AMD 证据（2026-09-11 复核） | 插入点 | 不要期待的事 |
|---|---|---|---|---|---|
| 1 | **MORI-EP 接进 Primus，替换 turbo `deep_ep`** | 通信底座 | first-party MIT；MI355X 调优表；#544 SymmetricMemory；**当日仍在 push**。同机 DSv3 EP8 bf16 T=8192：dispatch 1.676 vs 2.024 ms（**1.21×**），combine 1.460 vs 2.313 ms（**1.58×**） | `PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND`；已有 `USING_UEP=1` 的 UCCL 槽，加 MORI 槽 | 不要拿它去打 MegaMoE——测量边界不同。跨节点 RDMA 仍是 46–71 GB/s |
| 2 | **RCCLX 低精度 collectives 评估** | 通信优化 | torchcomms **BSD-3**，默认同时编 gfx942/gfx950，wave 级 LP + HIP graph 覆盖；**2026-09-10 仍在 push** | C3-5 wgrad RS / MoE A2A 的线上格式实验床 | 官方 experimental；CTran 未全部开源；公开数字偏推理 |
| 3 | **UltraEP HIP-ify** | runtime（EP 均衡） | 论文点名 Helios；Megatron-LM 参考 fork；~9.6K CUDA；**叠加式、与 grouped GEMM 解耦** | dispatcher **之上**，不替换 DeepEP/MORI | 仓库 **2026-07-17 后再无 push**。配额求解器是 warp 级归约，必须按 wave=64 逐处改。C0-4 硬前置 |

Triton-distributed 的 `ep_all2all_fused.py` **降档**：不是因为思路差，而是工程门关着——本容器 ROCm 7.15 clang 产的 mori bitcode 链不进它 pin 的 LLVM `8957e64a`；autograd `backward` 显式 `NotImplementedError`；AMD EP 公开数字为零。它仍是「编译器独占列」的长期项，不是下周的移植项。

### 3.4 vendor gap（GitHub 上没有可搬实现）

| 缺口 | NVIDIA 现状 | ROCm / Primus 现状 | 含义 |
|---|---|---|---|
| **Device-initiated EP / Gin** | DeepEP V2 默认 NCCL Gin；NCCL 2.28+ Device API（LSA / Multimem / GIN）；DeepEP 本周还在合 `ncclDevCommCreate` 兼容补丁 | RCCL 公开进展停在「4-NIC 拓扑感知 + backport NCCL 2.28 **算法**」 | 跟 V2 不能靠 hipify。跟的是 MORI device primitive + 将来的 RCCL Gin |
| **Copy Engine collectives** | NCCL 2.28 用 copy engine 打 alltoall/allgather，把 SM 从通信里释放 | MORI 已有 SDMA AllGather（DeepSpeed ZeRO-3 声称 e2e +10%，**不是 EP**） | 框架可消费的 API 面仍缺。这就是 Tessera「20 SM 稳定罚 10–20%」在 AMD 上的对应题 |
| **packed FP4/MXFP8 param all-gather** | Megatron GTP `--fp4-param-gather`，分片保持 native packed 4-bit，前向零 per-microbatch 量化 | Primus v26.5 Flux 路径有 FSDP2 **fp8** all-gather，未到 MoE 主线、未到 FP4 | 参数侧是**追平**，不是差异化。差异化在 wgrad RS 的块缩放线上格式（C3-5） |
| **hipBLASLt grouped GEMM FP8** | 有 | ROCm 7.1 穷举后只有 BF16 grouped GEMM | C2-8。一旦可用，单次 API 调用消掉全部 dispatch 开销 |

---

## 4. 深挖后改变判断的几条

### 4.1 MoonEP：代码比想象的小，设备列表把 AMD 排除了

9-07 笔记 §八把「代码规模与 kernel 数量」列为待核实。今日拉 `MoonshotAI/MoonEP` tree：

| | 规模 |
|---|---|
| Python | `moonep/` 约 12 个模块（`planning.py` 58 KB、`api.py` 45 KB、`dispatch.py` 42 KB 为头） |
| CUDA | `csrc/nvl_shared_buffer.cuh` 22 KB + `bindings.cu` 2.8 KB |
| 最后代码 push | **2026-08-13**（今日 star 活动还在，代码已静） |
| License | MIT |
| **Supported Devices（README 原文）** | **NVIDIA GPU；Zhenwu PPU（审核中）。没有 AMD。** |

含义：

- 真正要重写的硬件层是那份 NVLink VMM buffer，不是 9.6K CUDA。Planner / dispatch / combine 在 Python 里，形态上比 UltraEP 更像「接到 MORI SymmetricMemory 上的一套契约」。
- **契约仍然硬：** 每个投影一份连续 `[E+B, H, H']`，group GEMM 按行号寻址；训练还要一份对框架 grad reduce 不可见的 fp32 reduce buffer。这一条 9-07 §四没变，而且是 distributed optimizer 与 MegaMoE / Turbo grouped GEMM 的交火面。
- README 致谢 UltraEP / Echo（即 Megatron-Core MoE 报告 2603.07685）/ DeepEP / **AcclEP**。MoonEP 不是独立发明，是「构造性均衡 + 零拷贝」对 UltraEP 反应式均衡的替代。AcclEP 仓库仍然找不到。
- 设备列表写明 NVIDIA + 真武，等于作者自己没把 ROCm 当目标。gate Q1（symmetric memory）被 mori #544 削掉之后，**剩下的不是「能不能映射 SHMEM」，是「作者根本没为 wave64 / XGMI / gfx950 设计」**。

**结论维持并收窄：** MoonEP 仍是下一代 dispatcher 候选，不是本季度移植项。Phase 2 只回答一个问题：`[E+B, H, H']` 能不能在不拆 MegaMoE 的前提下做一层视图。答不了就停。

### 4.2 DeepEP V2 是活的 mainline，Primus 的 V1 集成在追残骸

DeepSeek 已把库更名为 DeepEveryParallel。V2 要点（[PR #605](https://github.com/deepseek-ai/DeepEP/pull/605)，README 当前正文）：

- NVSHMEM → **NCCL Gin**（header-only，复用已有 NCCL communicator）
- 高吞吐 / 低延迟 API 合成单一 `ElasticBuffer`
- 训练 SM 占用 24 → **4–6**
- 支持到 EP2048；解析式 SM/QP，不再 autotune
- 实验原语：0-SM Engram (RDMA) / 0-SM PP (RDMA) / **0-SM CP (Copy Engine)**
- 要求 SM90 + CUDA 12.3+ + **NCCL 2.30.4+**

今日 commit 仍在修 **GIN barrier 在 NVLink+RDMA 混布时的 system-scope fence**、以及 NCCL Device API 的 `ncclDevCommCreate` 运行时大小。这不是「四月发过一次」的死分支。

对 Primus 的三个具体后果：

1. 现有 turbo `deep_ep`（含 620 行调查根治的 `kUseCheapFence`）是 **V1 语义**。跟 V2 不是 rebase，是换传输层。
2. pip 的 `deep_ep` 0.1.0（UCCL kernels）在 9-07 上 **init SIGSEGV / combine 挂死**。`USING_UEP=1` 这条文档路径目前不能当生产开关。
3. V2 用 4–6 SM 打 EP，直接打 Tessera「20 SM 罚 10–20%」那一枪。AMD 侧若继续用占满 CU 的 host-initiated RCCL A2A，重叠收益会先被自己吃掉。**这是 MORI-EP / megakernel 显式划 COMM WG 的最硬论据，也是「再 hipify 一份 DeepEP V1」的反对论据。**

### 4.3 「重叠是免费的」已经被三次独立证伪

| 来源 | 数字 | 场景 |
|---|---|---|
| Tessera（阿里，OSDI'26，生产 4k–12k GPU） | EP kernel 占 **~20 SM**，与 Attention / MoE-MLP 重叠时 **稳定 10–20% 减速**；因此否掉 Comet 式融合 | 训练 |
| Motif 3（314B，B200） | 大规模下 **消除** 专家权重 All-Gather 优于 **隐藏** | 训练 |
| Instella-MoE（AMD，Primus，EP=8） | FarSkip 明说 overlap「relies on sharing CUs」；64K+CP 阶段把重叠关掉 | 训练 |

推论不是「不要 overlap」，是：

- 任何 overlap 收益估算必须先扣 10–20% CU 税；算出来小于这个数，实际可能为负。
- megakernel / MORI device-initiated 路径的价值是 **COMM WG 数量可调**，不是「融在一起看起来更快」。
- Tessera 否掉融合的运维论点仍然成立：不要把 RCCL / MORI / 未来 Gin 的调用序列内联进 GEMM。通信原语必须是 kernel 内可替换的 device-side 接口。MegaMoE 与 UniEP 都该按这条审。

### 4.4 仓库活性把「长期项」和「死项」分开了

抓取时间 2026-09-11。

| 仓库 | 最后代码 push | 今日状态 | 对选型 |
|---|---|---|---|
| `ROCm/mori` | **2026-09-11** | 周更 | 底座，跟 |
| `deepseek-ai/DeepEP` | **2026-09-11**（Gin / Device API） | V2 主线 | 当协议参考，不当 HIP 源 |
| `meta-pytorch/torchcomms`（RCCLX） | **2026-09-10** | 活跃 | LP collective 首选评估对象 |
| `ByteDance-Seed/Triton-distributed` | 2026-08-12（9-07 核实） | 活着但 LLVM 锁死我们的容器 | 长期工具链 |
| `MoonshotAI/MoonEP` | **2026-08-13** | 代码静、设备列表无 AMD | 下一代，不排期 |
| `Dots-Infra/UltraEP` | **2026-07-17** | OSS 后冻结 | 仍可 fork HIP-ify，但别等上游改 |
| `inclusionAI/linghe` | **2026-04-28** | 休眠 | 只偷 Triton trick |
| `infinigence/FlashOverlap` | **2026-01-20** | 休眠 | landscape 里的 P0 阅读项，**不是移植项** |

FlashOverlap / Canzona / CODA 作为「该读的论文」仍然成立；作为「该搬进 Primus 的代码」不成立。Canzona 仍无开源实现可搬；CODA 是 GEMM-epilogue 训练融合，方向对、但与 Turbo / FlyDSL 重叠，先对一下缺哪些 epilogue 再决定。

### 4.5 9 月 4–11 日没有新的可移植 EP / megakernel 工作

arXiv 高级检索（cs，submitted 2026-09-04 ~ 09-11，关键词 MoE / EP / ROCm / megakernel / FP4 training）52 篇。与移植清单有关的只有：

| 论文 | 跟我们的关系 |
|---|---|
| T1（2609.11042）终端 agent RL，**rollout routing replay** | 与 verl v0.8 的 MoE router replay R2/R3 同族。Primus 若做 RL 一致性，偷这条，不要等自己发明 |
| Hyperparameter Scaling Laws Across MoE Sparsity（2609.08690） | 配置 / 搜参，进 Projection，不是 kernel |
| Job Power Elasticity（2609.11542） | 集群功耗，与移植无关 |

**低精度梯度通信 / wgrad RS 压缩：本窗口仍是 0 篇。** 9-03 立的 C3-5 窗口还开着，没有被抢跑，也没有再拖的理由。对照组仍然是：确定性 Hadamard + FP8 wgrad RS，与 UE5M3「更宽 block scale 让 RHT 冗余」合并成一次 MI355X 实验。

---

## 5. 明确不要搬

| 项 | 原因 |
|---|---|
| Tessera / ChunkFlow | 未开源；深度绑内部 Megatron fork。能借的是 `advance()` 探针 API 与「尾部误差翻转 MILP」方法论 |
| DeepSeek DeepGEMM MegaMoE | SM90/SM100 CUDA-only，与我们 MegaMoE 正面撞车。当参考实现读 |
| Comet 式整层融合当生产默认 | Tessera 在 10k+ GPU 上评估后否掉（后端耦合）。MegaMoE 已经占了这条线，不要再 port 一个 |
| MoonEP 本季度 | 设备列表无 AMD；`[E+B]` 布局穿透 optimizer / grouped GEMM |
| FlashOverlap 代码 | 2026-01 后休眠；tile-signal 思路已被 MegaMoE / UniEP 吸收 |
| pip `deep_ep` 0.1.0 / UCCL 集成包 | 9-07 实测 init SIGSEGV、combine 挂死 |
| 源码自编 mori v1.1.0 | 堆损坏 / SIGSEGV；一律 `amd_mori` wheel |
| nvidia-resiliency-ext | NVML+CUDA+NCCL，只能重写 |
| ThunderKittens / 未改算法的 CUDA tile 库 | CDNA 上 wave specialization 是负优化（HipKittens 已证） |

---

## 6. 映到 Primus roadmap 槽位

| Roadmap | 这一轮建议塞进去的业界项 | 不要塞 |
|---|---|---|
| **C0** 可信度 | MegaMoE 回归（新前置）、C0-4 golden、MORI/DeepEP 评估必须带调优参数与同机口径 | 任何碰梯度路径的 EP runtime |
| **C1** 插件化 | 通信后端做成 `BackendSpecProvider` / 平台能力契约（C1-7：symmetric memory / SDMA / HIP graph 是否存在），为 MORI 槽留官方插入点 | 再 vendor 一份 DeepEP V1 |
| **C2** 存量红利 | MXFP8-before-dispatch、routing-map padding、linghe gate/router、FarSkip 开关、逐层 CP | 新写一个 fused MoE |
| **C3** 规模化 | MORI-EP、RCCLX 评估、UltraEP（C3-2）、RCCL Gin/Copy Engine 诉求（C3-4）、FP4 param gather 追平 + wgrad RS 差异化（C3-5）、sync-free ∪ HIP graph（C3-7） | 把 MoonEP 和 UltraEP 一起开 |
| **C4** 执行层 / 研究 | MoonEP 布局调研；Tessera 式 task 探针若要做，进 Primus-Runtime 而不是 patch；通信感知 top-k | 用研究线代替 C0 |

UltraEP 与 MoonEP **仍然互斥**。先 UltraEP 的理由多了一条仓库证据：UltraEP 是叠加式且作者提供了 Megatron fork；MoonEP 是替换式且官方不支持 AMD。Helios 72 卡域会让 UltraEP 的价值上升，这一条 9-07 没变。

---

## 7. 建议的实验顺序（仍不上新 kernel）

1. **G0** MegaMoE 回归：同镜像、同 `n01-29`、DSv3 EP8 T=8192 bf16，对齐 7-16 的 368 GB/s。不修完不做三方对比。
2. **C0-4** 两条 golden。
3. **MORI-EP → Primus 后端槽**（不是研究原型）：调优参数写入 config_map，禁止默认 `block_num`。验收：同 9-07 表的 1.21× / 1.58× 不回退，loss golden 不变。
4. **偷思路三件套进配置：** DPO 关 LB、MXFP8-before-dispatch（EP 库能搬就开）、FarSkip 开关 + 三个关闭条件。
5. **FP4 / wgrad RS 对照：** 确定性 Hadamard ± UE5M3 宽 scale，同一套基础设施。
6. **RCCLX LP A2A / RS 微基准**，只在跨机架 DP 上看（机架内 HBM:线速低于 DynamiQ 盈亏门槛 9.5）。
7. UltraEP：先在 NVIDIA 参考 fork 上跑通行为基线，再 HIP-ify。不要从零在 MI355X 上「翻译」9.6K CUDA。

---

## 8. 对 9-07 笔记 §八「待核实」的结案

| 当时的问 | 现在的答 |
|---|---|
| MoonEP 代码规模与 kernel 数量 | Python 主体 + 一份 22 KB NVLink VMM header；不是第二个 DeepEP |
| MoonEP e2e 训练曲线的卡数 / 模型 | README 仍只有 H20 EP=8 扫 maxvio 的图，**无文字规模** |
| Kimi K3 2.5× 里 MoonEP 占比 | 官方仍未拆分 |
| UltraEP / MoonEP 第三方复现 | 仍无 |
| AcclEP 是否开源 | 仍未找到仓库 |
| 训练侧 EP gfx942-only | **已过时**：turbo `deep_ep` 在 MI355X 能跑；缺口改成「能跑但默认 `num_sms=20`、且是 V1 语义」 |

---

## 相关文件

- 上一轮选型：[`2026-09-07_0900_moonep-vs-ultraep-port-selection.md`](./2026-09-07_0900_moonep-vs-ultraep-port-selection.md)
- 大厂动向：[`../../knowledge/systems/industry-training-optimization-2026.md`](../../knowledge/systems/industry-training-optimization-2026.md)（本轮回写 §7）
- Roadmap：[`2026-09-03_primus-roadmap-2026q4-2027h2.md`](./2026-09-03_primus-roadmap-2026q4-2027h2.md)
- MegaMoE 回归：[`../MegaMoeFlydsl/2026-09-07_1034_mega_bf16_dispatch_regression_vs_0716_baseline_mi355x.md`](../MegaMoeFlydsl/2026-09-07_1034_mega_bf16_dispatch_regression_vs_0716_baseline_mi355x.md)
- 画布：[`/home/xiaompen/.cursor/projects/perf-apps-xiaoming-slab/canvases/primus-rocm-port-map.canvas.tsx`](/home/xiaompen/.cursor/projects/perf-apps-xiaoming-slab/canvases/primus-rocm-port-map.canvas.tsx)
