# primus-moe — Primus MoE roadmap & planning notes

> Strategic / planning artifacts for the **Primus MoE story** (Primus-LM + Primus-Turbo).
> Hardware-line research notes live under sibling folders: `slab/notes/rocmoe/`, `slab/notes/monolith-moe/`, `slab/notes/monolith-ep/`.

## Status

| 维度 | 值 |
|---|---|
| Owner | AMD AI Brain — Training at Scale (TAS) team |
| Hardware focus | MI300X / MI325X (gfx942), MI350X / MI355X (gfx950) |
| Backends | Megatron-LM, TorchTitan, JAX MaxText |
| Companion repos | `~/workspace/Primus/`, `~/workspace/Primus-Turbo/`, `~/workspace/Primus-dev/` |
| Last roadmap | [2026-09-03 全框架级 2026 Q4 – 2027 H2](./2026-09-03_primus-roadmap-2026q4-2027h2.md)，执行复盘与修订见 [2026-10-12 delta](./2026-10-12_0101_primus-roadmap-update.md) — MoE 细项仍以 [2026-05-29 H2 2026 draft](./2026-05-29_roadmap_h2_2026.md) (mirrors `NVIDIA/Megatron-LM#4815` format) 为准 |

## Index

| 日期 | 文件 | 内容 |
|---|---|---|
| 2026-10-28 | [`2026-10-28_0546_industry-scan-update.md`](./2026-10-28_0546_industry-scan-update.md) | **业界进展扫描 2026-09-11 → 10-28** — 三路只读 subagent（仓库活性 / arXiv / vendor 与上游框架）+ 手工复核。①**9-11 的 vendor gap 表有两条读了已废弃仓库**（`ROCm/rccl`、`ROCm/hipBLASLt` 均已迁走）→ 四条有三条改写，最重要的是 **RCCL 2.30.7 已带 GIN device API**（随 ROCm 10.0.0 发布），卡的只是 `nccl_device.h` 在 host gcc 下的编译卫生；mori 把 `ccoSdma` put/get 暴露给了 Triton/FlyDSL ②**9-03 roadmap 支柱③「护城河」塌一半** — TorchTitan ROCm 立场反转（`main` 有 AMD CI 徽章、`rocm10.0` wheel、MI350X golden 10-20 仍在刷新），且已把我们缺的 C0 设施做齐（`loss_compare.py` / deterministic / SDC replay / fake-backend 干跑）③**Expert Coupling 降级** — 去重我们走 DeepEP 时本就有（`layout.cu` 写 `(is_in_rank[j] > 0)`），token shuffling 要求 `TP == EP` 而 293 份 recipe 里为真的是 0；只剩相关性放置，但落地零 HIP 改动 ④**格式缝窗口被生产关闭** — Meta LP-FA4 把 MXFP8 量化融进前一 kernel 的 epilogue 并给出 `[32,32]` 方块 scaling 转置不变技巧，dist_moe 是第二个独立实现 ⑤**低精度梯度通信仍 0 篇**，Olmo-core 3 反向佐证（梯度仍在 FP32 bucket 归约）⑥仓库变化：torchcomms **整库弃用**（CTran 被删非开源）、**UltraEP 解冻并换成 torch SymmetricMemory**、Triton-distributed 改 out-of-tree plugin（EP 反向仍无）、DeepEP V2.5 删掉 V1+NVSHMEM、新仓库 `meta-pytorch/dist_moe` —— **注意它既没进 `pytorch/pytorch` 也不是 TorchTitan 默认依赖**（独立发版、手动 git 安装、守卫 import，TorchTitan 称其为 "the annex"）；真正的信号是 **TorchTitan 为它定义了厂商可插拔的 routed-expert 契约**（`DistMoeTransform` 替换 `RoutedExperts.Config` 但保留 `GroupedLinear` 参数布局与 FSDP/checkpoint 所有权 + 可选 `dist_moe_runtime` 字段 + GraphPP 把 slot id 作显式 graph input），**即 C1-1/C1-2 的形状 —— 上游示范了厂商 kernel 该怎么接，而我们还在打 patch**。它自身卡在 CUTLASS DSL 无 ROCm 对等物 + 手上没 SM100 机器，判定偷 SyncFree 执行模型/三级容量预算/32×32 方块量化，不搬代码。**更正 10-12 note 的 C2-8 判定**。新增排序 R6/R7 |
| 2026-10-12 | [`2026-10-12_0101_primus-roadmap-update.md`](./2026-10-12_0101_primus-roadmap-update.md) | **对 9-03 roadmap 的执行复盘与修订（delta note）** — 基线 `6e2b114b` → `ad3614d6`，106 commits / +79,556 −16,286，三路只读 subagent 全仓审计。核心判断：①**原分期前提被证伪** —— C2 存量性能跑完一大半（C2-1/4/5/8 完成），而 C0 可信度几乎零位移、C1 插件化七条全未动、C4-3 倒退（SDMA 2 份 → 7 份 1,655 行）；kernel 性能仅占实质投入 ~20%，模型使能 ~30% ②**格式缝先例已由别人写进树里** —— #1208 把 RMSNorm→MXFP4 量化融进 norm epilogue（attention QKV 路径），目标缝（MoE colwise requant）仍空但窗口收窄 ③**C0-4 性价比反转** —— #1229 已建 7.4K 行 convergence harness（但单节点、未接 workflow、不跑 DeepEP 矩阵），同时本地出现活的同类 bug（MegaMoE mxfp8 topk 非 2 幂 + ≥2 µbatch 静默错梯度，grad norm 1.3→1300，无 NaN）④**定时炸弹**：Megatron pin 冻结至 2026-03-06、torchtitan 至 2026-02-20，期间 patch 172→194 而版本门仅 1→2，`patch.py:70-71` 仍静默跳过。修订排序 R1–R5：每条 C0 必须搭着有即时收益的交付走，不再单独立卫生阶段 |
| 2026-09-11 | [`2026-09-11_1541_industry_opt_port_to_rocm_primus.md`](./2026-09-11_1541_industry_opt_port_to_rocm_primus.md) | **业界优化 → ROCm/Primus 移植深挖** — 把 9-07 仓库顺位重切成四种模式（偷思路 / 搬代码 / 不要搬 / vendor gap）。结论：先修 MegaMoE 回归 + C0-4；最高 ROI 是 MXFP8-before-dispatch、DPO 关 LB、逐层 CP，不是再 HIP-ify 一个 EP 库；搬代码只留 MORI-EP（已测 1.21×/1.58×）、RCCLX 评估、UltraEP。MoonEP README 设备列表为 NVIDIA+真武 PPU。DeepEP V2（NCCL Gin）把 Primus V1 路径判成死胡同。9-04~09-11 无新 EP/megakernel 论文。画布：[primus-rocm-port-map](/home/xiaompen/.cursor/projects/perf-apps-xiaoming-slab/canvases/primus-rocm-port-map.canvas.tsx) |
| 2026-09-07 | [`2026-09-07_0900_moonep-vs-ultraep-port-selection.md`](./2026-09-07_0900_moonep-vs-ultraep-port-selection.md) | **EP 负载均衡的移植选型** — 2026 年开源系统扫描筛出两个 MegaMoE 量级候选，均落在 roadmap 的"独占"列，且正好补上 [`2026-07-22`](./2026-07-22_moe_10k_gpu_gap_analysis.md) 识别的**动态负载均衡**空白。结论：**先移植 UltraEP**（MIT，叠加式、与 grouped GEMM 解耦、~9.6K 行、**官方 Megatron-LM 参考 fork**、论文**显式点名 AMD Helios**、256 GPU / 106B–671B 证据），**MoonEP 作为下一代目标架构**（构造性完美均衡 + 零拷贝 + 静态形状消 host 同步，但要求重排专家参数物理布局，穿透 distributed optimizer 与 grouped GEMM）。两者互斥。⚠ **C0-4 多机数值 CI 是硬前置**——两者都碰梯度路径，正是 DeepEP 静默错梯度的同一 bug 家族。另识别独立缺口：训练侧 EP 仍 gfx942-only 而 MORI-EP 已支持 MI355X。**§0 追加（国内厂商全量扫描回来后）**：新增两个候选——**Triton-distributed**（字节，MIT，2026-08 活跃，仓库内已有 `test/amd/` 完整 EP + fused-MoE 测试套件走 **ROCm/mori + rocSHMEM**，给的是 kernel **生成**能力而非又一个手写 kernel，与 runtime 层不冲突可并行推进）与 **UCCL-EP/Zip**（Apache-2.0，已进 AMD TheRock 生态，同时命中 ROCm 可移植 MoE 通信 + 压缩集合通信）；⚠ **DeepEP 已 V2，通信后端从 NVSHMEM 换成 NCCL Gin**，现有基于 V1 的 ROCm 集成路径需重估。**§0之二 追加（国外厂商扫描）**：[`ROCm/mori#544`](https://github.com/ROCm/mori/pull/544) 已把 mori 注册成 **torch SymmetricMemory 后端**，把 MoonEP 的第一个 gate 问题从"自研映射层"降级为"验证现成后端"，MoonEP 成败自此几乎全压在权重布局那一问上；配合 `mori#594` 的 Triton device binding 与字节 [PR #175](https://github.com/ByteDance-Seed/Triton-distributed/pull/175) 的 **1267 行 MIT `ep_all2all_fused.py`**（mori_shmem + wave64 + on-device metadata，切分方式与 MegaMoE 一致，**但 forward-only**），MI355X 上的 ROCm-native MegaMoE 两端已就位。另 **RCCLX**（torchcomms 内，BSD-3，**默认同时构建 gfx942/gfx950**，wave 级 LP + HIP graph 覆盖）取代 UCCL-Zip 成为"压缩集合通信"这一格的首选。**合并后排序：mori 底座 → 补 `ep_all2all_fused` 反向 → UltraEP → RCCLX 评估 → MoonEP**。生态排雷：megablocks / nanotron 实质 dormant、pplx-kernels 已废弃、msccl-tools 已归档、tutel 转型推理 |
| 2026-09-03 | [`2026-09-03_primus-roadmap-2026q4-2027h2.md`](./2026-09-03_primus-roadmap-2026q4-2027h2.md) | **全框架级 roadmap（2026 Q4 – 2027 H2）** — 四路调研合成（slab / 上游 TorchTitan / 上游 Megatron-LM / Primus 现状）。核心判断：①上游欠债是最大风险（Megatron 落后 1391 commits、core 0.16→0.20；TorchTitan 落后 937 commits；156 patch 仅 1 个版本门；`GPTModel` 已废弃、`megatron/legacy/` 已删）②两个上游官方扩展点刚就位（Megatron `BackendSpecProvider`、TorchTitan `overrides/`）是时间敏感窗口 ③护城河经实证（上游 Megatron core 内 ROCm 命中 1 / `torch.cuda` 796；TorchTitan 已停 ROCm CI 并删 golden；zero-bubble 上游 0 命中）。分期 C0 可信度 → C1 插件化 → C2 存量性能 → C3 规模化可靠性 → C4 执行层收拢，附 13 条反证护栏 |
| 2026-08-04 | [`2026-08-04_1049_primus-runtime-design-and-plan.md`](./2026-08-04_1049_primus-runtime-design-and-plan.md) | Primus-Runtime 设计与实施规划 — core/pipeline_parallel 已是 plan/execute 内核且双消费者验证；现存三套并行调度栈、两套 IR（`R` 同名不同义）；P0 收敛 → P1 comm → P2 精度 → P3 HIP graph → P4 对外化 |
| 2026-08-04 | [`2026-08-04_1018_framework-strategy-patch-layer-vs-execution-layer.md`](./2026-08-04_1018_framework-strategy-patch-layer-vs-execution-layer.md) | 框架战略备忘 — patch 只占 megatron 后端代码 17%，已有约 3.3 万行执行层资产散落在 3 个后端命名空间；反对自研 Megatron 替代品，建议归位成独立执行层 Primus-Runtime（R1–R6 六模块 + C0–C3 分期验收） |
| 2026-07-22 | [`2026-07-22_moe_10k_gpu_gap_analysis.md`](./2026-07-22_moe_10k_gpu_gap_analysis.md) | MoE @ 万卡:瓶颈 × Primus 现状 gap 分析 — 把 10k 卡瓶颈逐条对到 DeepEP/1F1B overlap/分层 recompute/projection,识别容错·动态负载均衡·拓扑感知通信三大空白 |
| 2026-05-29 | [`2026-05-29_roadmap_h2_2026.md`](./2026-05-29_roadmap_h2_2026.md) | Primus MoE Roadmap H2 2026 (Q3 + Q4) — full mirror of the Megatron Mcore-MoE Q2 2026 roadmap, AMD-flavored |

## Cross-references

- **Production line**: `slab/notes/monolith-moe/` (4.82 ms / 8× MI355X, DSv3 e2e + loss parity)
- **Research line (super-kernel)**: `slab/notes/rocmoe/` (Layout-P + receiver-pull, BF16 target ≤ 7 ms)
- **Research directions**: `slab/knowledge/moe/research-overview.md` (comm-aware routing, AMD-FSEP, MoE-IR, backward comm scheduling)
- **Megatron MoE comparison reference**: [`NVIDIA/Megatron-LM#4815`](https://github.com/NVIDIA/Megatron-LM/issues/4815)

## Maintenance

- One roadmap doc per release cycle (H1 / H2). Old roadmaps stay in place; the latest is linked from this README.
- Update `Status` row + `Index` table whenever a new roadmap iteration lands.
- Cross-link to the per-feature progress notes in `monolith-moe/` and `rocmoe/` rather than duplicating numbers here.
