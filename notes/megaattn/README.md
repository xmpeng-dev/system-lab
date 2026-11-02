# megaattn — DSA attention megakernel (indexer + top-k + sparse MLA)

> **目标**: 把 DeepSeek Sparse Attention 的 decode 三段 —— lightning indexer (paged MQA logits) → top-k 2048 → sparse MLA —— 融进一个 persistent kernel，复用 MegaMoE v2 已验证的 role-split 骨架
> **平台**: 8x MI355X (gfx950, CDNA4)
> **栈**: ROCm / FlyDSL / Primus
> **目标模型**: DeepSeek-V3.2 → V4 家族。MLA absorbed (QK 576 = 512 latent + 64 rope, V 512, 128 heads)，indexer 64 heads x 128 dim，index_topk = 2048
> **工作仓库**: `~/workspace/FlyDSL/kernels/attention/`
> **参考仓库**:
>   - `~/workspace/FlyDSL/kernels/mega_moe/` — 骨架来源（role-split grid / DispatchSlot 信号表 / 原子工作队列）
>   - `~/workspace/MegaMOE/` — DeepGEMM，indexer 语义与对拍基准（`sm100_fp4_paged_mqa_logits.cuh`）

## 状态

| 维度 | 值 |
|---|---|
| 当前阶段 | **M0 完成 2026-10-29 13:30 — GO（有条件），下一轮开 M1**。MI355X 上 aiter 三独立 kernel 基线（128 MLA heads）的「top-k + logits 往返」占单层 wall：@128K B = 1 / 4 / 16 / 32 / 64 为 45.8 / 33.8 / 21.5 / 14.0 / 11.5%，@8K B = 1 为 23.1%；B ≤ 4 全部 N 过 15% 门槛，B = 16 在 32K / 128K 过，B ≥ 32 不过。logits 写回本身 ≤ 2.4% wall，收益几乎全在 top-k 延迟（9–11 / 15–20 / 29–38 µs @8K / 32K / 128K，几乎不随 batch 变）。M3「@128K 降 ≥ 25%」只在 B ≤ 4 可达，B = 16 需要 M1 indexer 接近 roofline（aiter 2.7–3.9 TB/s）。详见 [M0 分段占比](./2026-10-29_1330_BASELINE_m0_dsa_decode_three_kernel_breakdown.md);**下一步 P0**: M1（FlyDSL FP8 paged MQA logits + k=2048 精确 radix top-k），并用「独立 top-k 极限」重算 M2 门槛 |
| 前置缺口 | `pa_metadata.py:1242` 硬断言 `topk == -1`，FlyDSL 稀疏路径为零；attention 全栈无 backward |
| 已有资产 | MLA m16x8 FP8 tile loop（`mla_fwd_decode_m16x8_fp8_fp8.py`，2096 行）与 split-KV reduce 已就绪，缺的只是「按索引取 KV」 |
| 训练场景（CP） | **M0-train 2026-10-29 14:00 — NO-GO**。128K / CP 8（生产 V4-Flash SFT 配置）下「top-k + logits 写回」只占一个 CSA 层 attention fwd+bwd 的 5.3%（V4-Flash）/ 2.5%（V4-Pro）/ 4.6%（V3.2），占整个 attention 栈 2.3% / 1.3%；占比与 mbs、CP 无关，只随上下文长度增长，V4-Flash 到 1M 才到 16%。训练大头是 sparse MLA 反向（66–73%）。详见 [M0 训练分段占比](./2026-10-29_1400_BASELINE_m0_train_cp_breakdown.md) |
| M0 门槛 | N = 8K / 32K / 128K 三档分段 rocprof，量出 indexer / top-k / sparse MLA 实际 wall 占比 —— **已完成，B ≤ 16 通过**；基线 harness 在 Primus-Turbo 分支 `feat/megaattn/m0-dsa-decode-baseline`：`benchmark/ops/inference/bench_dsa_decode_breakdown_aiter.py` |
| M3 目标 | 单层 attention wall 相对「三独立 kernel」基线降 ≥ 25% @ 128K，精确 top-k 逐位一致 |

## 设计核心（3 个改动 vs 三独立 kernel 基线）

| # | 改动 | 砍掉的开销 | 来源 |
|---|---|---|---|
| 1 | indexer 边算边在 LDS 维护近似阈值，只回写越阈候选 | logits 全量写 + radix 多趟读（N=128K 时约 2.3 MB / 层 / 序列） | LiteTopK ([2607.11976](https://arxiv.org/abs/2607.11976)) |
| 2 | decode 用上一 step 的第 2048 名分数做初始阈值 | 相邻 step top-k 重叠 35-50%，阈值一上来就接近真值 | NVIDIA Guess-Verify-Refine |
| 3 | scorer / selector / attn 三角色共处一个 persistent grid | 3 次 launch → 1 次，2 次全局同步 → 0，段间可流水 | MegaMoE v2 `mega_moe_stage1.py` |

## 阶段计划

| 阶段 | 内容 | 验收 |
|---|---|---|
| M0 | 三独立 kernel 基线 + 分段 profile | 拿到占比表；< 15% 则叫停 |
| M1 | 独立实现 paged MQA logits (FP8) + k=2048 radix top-k | 与 DeepGEMM 逐位对拍；不差于 AITER / ROCm blog |
| M2 | 融合 indexer + 在线剪枝 + top-k | 相对 M1 两段之和降 ≥ 30%；回退率 < 1% |
| M3 | 接 sparse MLA，role-split persistent grid | 相对 M0 降 ≥ 25% @ 128K |
| M4 | prefill 路径、FP4 indexer (V4)、CSA/HCA | 按 V4 落地节奏 |

## 文件索引

| 文件 | 内容 |
|---|---|
| [2026-08-05_1650_megaattn_v1_architecture_design.md](./2026-08-05_1650_megaattn_v1_architecture_design.md) | v1 架构设计：瓶颈分析、role-split 结构、与 MegaMoE 的同构映射、`work_info` 接口改动、分阶段计划与风险表 |
| [2026-10-29_1255_handoff-prompt-for-turbo.md](./2026-10-29_1255_handoff-prompt-for-turbo.md) | M0 交接 prompt：范围、硬约束、交付物、15% 门槛 |
| [2026-10-29_1330_BASELINE_m0_dsa_decode_three_kernel_breakdown.md](./2026-10-29_1330_BASELINE_m0_dsa_decode_three_kernel_breakdown.md) | **M0 BASELINE**：MI355X aiter 三独立 kernel（indexer / top-k / sparse MLA）分段占比，B 1–64 x N 8K / 32K / 128K x 128 / 16 heads；rocprof 读写量；max_split 扫描；GO（有条件）判定；aiter 两个 bug。数据在 [`data/2026-10-29-m0/`](./data/2026-10-29-m0/) |
| [2026-10-29_1400_BASELINE_m0_train_cp_breakdown.md](./2026-10-29_1400_BASELINE_m0_train_cp_breakdown.md) | **M0 BASELINE（训练）**：CP 下单 rank 的 indexer / top-k / sparse MLA fwd+bwd 分段，V4-Flash / V4-Pro / V3.2 x S 4K–1M；mbs / CP 线性检验；整个 attention 栈占比；训练 NO-GO 判定。数据在 [`data/2026-10-29-m0-train/`](./data/2026-10-29-m0-train/) |

## 相关

- [H2 2026 roadmap](../primus-moe/2026-05-29_roadmap_h2_2026.md) — L96 DSA / L120 fused MQA / L121 CSA-HCA / L260 FP8 indexer
- [rocmoe v2 架构设计](../rocmoe/2026-05-21_1252_rocmoe_v2_architecture_design.md) — role-split + scoreboard 同步的先例
- [训练优化 landscape 2026](../../knowledge/systems/training-optimization-landscape-2026.md)
