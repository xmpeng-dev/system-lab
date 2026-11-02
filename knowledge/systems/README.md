# knowledge/systems — 训练系统/框架知识

## 文件索引

| 文件 | 内容 |
|---|---|
| [`primus-pipeline-runtime-megatron-integration.md`](./primus-pipeline-runtime-megatron-integration.md) | **Primus 流水线 runtime 如何接入 Megatron** — 以 PP=4/1f1b 走通 plan→adapter→executor→handler 五层调用链；执行权移交的绑定点、plan 兼数据流图、延迟 wgrad 机制、接新后端的三样交付物、当前层泄漏 |
| [`mi455x-megamoe-ep64-architecture.md`](./mi455x-megamoe-ep64-architecture.md) | **MI455X / Helios MegaMoE EP64 架构** — EP8/16/32/64 共用协议；64 owner + 可选 8 helper；ring/device planner/persistent worker queue；forward/backward、同步、精度、correctness/perf 门禁与分阶段落地 |
| [`training-optimization-landscape-2026.md`](./training-optimization-landscape-2026.md) | **arXiv 2025–2026 训练/推理优化全景**（更新至 2026-10-12；MoE overlap/放置 · 优化器 · FP4 · 基础设施 · 优先阅读队列） |
| [`arxiv-digest-2026-08.md`](./arxiv-digest-2026-08.md) | **arXiv 2026-08 增量扫描**（2026-05~08 投稿，1157→92→27 篇；执行模型 / EP 通信 / AMD 建模 / 内核 DSL / 训练并行 / 低精度；含最该先读的五篇及其精读笔记入口） |
| [`arxiv-digest-2026-09.md`](./arxiv-digest-2026-09.md) | **arXiv 2026-09 增量扫描**（2026-08-07~09-03 投稿，966→164→23 篇）——三条主线：AMD 自训全开源 MoE（Instella-MoE，MI300X/MI325X）· 汇编级/多 agent kernel 优化（AsmEvo 在 MI300X 对 AITer 二进制 1.09× geomean）· FP4 配方之争（UE5M3 主张去掉 Hadamard，与我们 MI355X 实测冲突）。**含三条负面发现**：低精度梯度通信 0 新增、流水线调度 0 命中、megakernel 断档 |
| [`arxiv-digest-2026-10.md`](./arxiv-digest-2026-10.md) | **arXiv 2026-10 增量扫描**（09-04~10-08 公告；category listing + 定向检索）——新增 **Expert Coupling**（MI300X，E2E 最高 1.41×）· **MegaFlux** · **Format-Aware FP4 Fusion** · **HyDra** · **RelayMoE**；10-09~12 尚未进入 mailing |
| [`industry-training-optimization-2026.md`](./industry-training-optimization-2026.md) | **大厂训练侧优化动向**（2026-03~10）——arXiv + 开源/博客 + 会议。§7 四种移植模式；§8 09 月下旬；**§9（2026-10-12）Primus/Turbo correctness pins、最新 MoE/FP4 与 SOSP/MICRO 跟进次序** |
| [`venue-scan-2026-08.md`](./venue-scan-2026-08.md) | **会议扫描** — 首轮 1,249 篇；**§六（2026-10-12）补 SOSP '26**（CrystalLLM / TensorHub / MeldKernel）并登记 MICRO Meta+AMD 训练 co-design 为 P0-watch |
| [`paper-venues-checklist.md`](./paper-venues-checklist.md) | **论文检索会议清单** — 从 ml-systems-papers（1,101 篇带会议标签）统计出的会议分布；按「arXiv 搜不到、必须直接翻会议」（OSDI/SOSP/EuroSys/ATC/NSDI/ASPLOS/SC/FAST）→「强相关但多在 arXiv」→「ML 会议，arXiv 已覆盖」→「明确排除」四档分级；含 DBLP/USENIX 检索方法与按月的检索日历 |
| [`torchtitan-diff-2025-10-vs-2026-04.md`](./torchtitan-diff-2025-10-vs-2026-04.md) | TorchTitan 半年跨度的 diff 总结(API / 训练循环 / 并行 / 优化器变更) |
| [`training-1024g-stability-interview-notes.md`](./training-1024g-stability-interview-notes.md) | 1024-GPU 训练稳定性面试笔记(checkpoint / failover / 监控 / RCCL 调优) |

## 什么时候来查

- 要做新一轮论文检索 / 想知道该翻哪些会议 → 先看 paper-venues-checklist（**别再只扫 arXiv**）；arXiv 侧直接跑 [`scripts/arxiv-scan.py`](../../scripts/arxiv-scan.py)，起点接上一轮 digest 的终点
- 想知道某家大厂最近在训练侧做什么 / 找对标基线 → 看 industry-training-optimization-2026
- 想把业界训练优化接到 Primus / ROCm（搬不搬、接到哪一层） → 先看 industry-training-optimization-2026 **§7–§9**；当前顺序是 **锁含 DeepEP fence/replay 修复的 commit pin → Expert Coupling trace-only → MegaFlux/FP4 最小对照**
- 准备升级 TorchTitan / 排查跨版本回归 → 看 diff
- 设计大规模训练的容错与监控 → 看 stability 笔记
- 比较 Primus / TorchTitan / Megatron 三个 backend 的工程差异 → 配合 `.cursor/skills/backend-gap-report/SKILL.md`
- 改 Primus 流水线调度 / 加新调度算法 / 给新后端接 runtime → 看 primus-pipeline-runtime-megatron-integration
- 设计 MI455X / Helios 上可扩展到 EP64 的 MegaMoE → 看 mi455x-megamoe-ep64-architecture；并行度选择与硬件依据配合 mi455x-parallelism-strategy-reference

## TODO

- 增补 Megatron-Core MoE 的工程实现要点(已有 paper note `papers/megatron-core-moe.md`,但工程经验单独成文更好)
- Primus 的执行层设计要点已成文(见上表首行);后续补 comm / precision / memory 三块的对应走读
