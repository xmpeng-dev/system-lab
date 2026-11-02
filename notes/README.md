# notes — 项目工作日志

按**项目**组织,每个项目目录下:

- 一份 `README.md`(项目总览: 状态 / 时间线 / 下一步 / 文件索引)
- 多篇按时间顺序的 progress / archive note(`YYYY-MM-DD[_HHMM]_<topic>.md`)

> 论文笔记不在这里,在 [`../papers/`](../papers/README.md);周报不在这里,在 [`../weekly/`](../weekly/README.md);
> 稳定的领域知识不在这里,在 [`../knowledge/`](../knowledge/README.md)。

## 当前项目

| 项目 | 主题 | 状态 |
|---|---|---|
| [`ante/`](./ante/README.md) | Cairn（原 ante）：coding-agent 原生 AMD MoE 训练引擎 | active — MathGraph vs RegionGraph 架构讨论中 |
| [`career-strategy/`](./career-strategy/README.md) | 长期技术定位：边界消解 → 删掉量化↔GEMM 的跨 op 格式契约 | active — 2026-08-12 定落地形态（Pilot 主线已降级、判定层已作废） |
| [`gpt-oss/`](./gpt-oss/README.md) | GPT-OSS-20B MLPerf 调优 (MI355X) | active — best 9963 s |
| [`monolith-moe/`](./monolith-moe/README.md) | MoE super-kernel + CCO (MI355X) | 阶段性收尾 — HIP C++ IPC kernel 待落地 |
| [`monolith-ep/`](./monolith-ep/) | MoE expert parallel super-kernel | placeholder — 待启动 |
| [`rocmoe/`](./rocmoe/README.md) | MoE super-kernel v3：Layout-P + receiver-pull + MonolithEP hot loop (MI355X) | active — 设计冻结 (M0 待启动) |
| [`megaattn/`](./megaattn/README.md) | DSA attention megakernel：indexer + top-k + sparse MLA 三段融合 (MI355X) | active — M0 完成：decode 有条件 GO（B ≤ 16 过 15% 门槛）；训练（CP）NO-GO（128K 时 ≤ 5%） |
| [`hk-attn-bwd/`](./hk-attn-bwd/README.md) | kernel-substrate：自主可控 GEMM 底座（原 attention backward 专项，2026-08-13 重写） | active — M0=GEMM epilogue 融 quant；attention 入口已关闭，见代码核查 |
| [`peer-tiles/`](./peer-tiles/README.md) | 自有 repo：AMD 上融合 MoE 训练 kernel 的 tile 原语库（HK 形状，加第四条原语「所有权与可见性」） | **M0 设计中** — 立项文档已成文，三个待决问题未定 |
| [`MegaMoeFlydsl/`](./MegaMoeFlydsl/README.md) | Mega MoE 通信-计算融合算子 (FlyDSL, 8× MI355X) | ⚠️ blocked — MXFP8 e2e fwd+bwd 14.64 ms / bf16÷fp8 1.40×；但 **2026-09-07 在新栈（torch 2.12+ROCm7.15）上 dispatch 大幅回归**：grouped/dense 8.7%（基准 91.9–100%）、XGMI 50.8 GB/s（基准 368–381），回归点待二分 |
| [`primus-turbo/`](./primus-turbo/README.md) | Primus-Turbo **作为交付物**的 owner 视角规划与审计(库边界 / CI / 六栈治理 / 差异化) | active — 2026-10-28 接任 owner 首版 roadmap 成文:**最严重的是契约不是 kernel**(≥3 个 Primus import 指向 Turbo 零分支存在的模块)、CI 不测 gfx1250、静默数值回归靠 vendoring |
| [`mlperf-llama/`](./mlperf-llama/README.md) | Llama-2-70B LoRA SFT, NeMo vs Primus | active — DataLoader fix + `fp8_param` A/B 待跑 |
| [`pilot/`](./pilot/README.md) | Primus Pilot v2: agentic training-tuning system | active — bootstrap 工具落地,BASELINE 入口待排查 |
| [`MI455x-Perf/`](./MI455x-Perf/README.md) | MI455X 计算 + 通信 overlap 性能分析（4 × MI455X） | active — GEMM + alltoall 根因定位（persistent grid 占满 CU），data-parallel grid E 0 → 0.56 |
| [`moe-train/`](./moe-train/README.md) | Compact MoE training runtime：PithTrain DNA + AMD topology-aware 重写 | **设计** — v1 架构 2026-08-19 成文，repo 未创建 |

## 写 note 的两条 skill

| skill | 用途 | 文件名 |
|---|---|---|
| `progress-note` | 工作过程中即时记一笔(实验跑完 / bug 定位 / 决策点) | `YYYY-MM-DD_HHMM_<topic>.md`(精确到分钟) |
| `archive-notes` | 一段调研或分析结束后做总结归档 | `YYYY-MM-DD_<topic>.md` |

两条 skill 都要求写完后**回写所在项目的 README**(进展时间线 + 下一步)。

## 项目目录命名

`kebab-case`,≤ 3 个词。新项目直接新建目录 + `README.md`,并在本文件的"当前项目"表里加一行。
