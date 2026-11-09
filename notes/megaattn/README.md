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
| CP 通信-attention overlap | **M0-overlap 2026-11-08 08:30 — 前向不单独立项**。MI355X ring attention 前向 32K–128K / CP 2–8：overlap 效率 (不 overlap − overlap) / 通信 = 0.73–1.08（CP8 32K 0.42–0.47），多 stream（RCCL P2P + aiter）overlap 相对不 overlap 省 3.7–28.8%，已在有依赖 ring 理想下界的 1.01–1.14× 以内，例外 CP8 32K 1.33×；融合可争取上限 CP2 1–8% / CP4 4–18% / CP8 128K 5–12% / CP8 32K 约 25%（需避开 CU 争用，未证明）。step 内切 chunk 在非融合实现里一律更慢。链路每对 GPU 单向约 56 GB/s 不随 CP 变。详见 [M0 overlap 基线](./2026-11-08_0830_BASELINE_m0_cp_ring_attn_overlap.md);**下一步**：反向同一张表 |
| 现有 ring 实现 overlap（TE / Turbo） | **M0-overlap-impl 2026-11-08 09:15 — 基线已测，未优化**。Primus 训练里只有 TE 路径（`use_turbo_attention: false` + `cp_comm_type: p2p`）有 ring overlap；Turbo 路径 `ring_group` 写死单卡，只剩 Ulysses a2a。单个 attention 层 fwd + bwd，TE overlap 相对不 overlap 省 CP2 2.2–11.0% / CP4 15.1–32.2% / CP8 21.1–36.2%；overlap 效率 TE 0.32–1.14、Turbo 0.55–0.95；反向比前向差（CP2 TE 前向 0.89–1.33、反向 0.41–0.73，原因未定位）；overlap 后 TE / Turbo 耗时 0.48–0.96。详见 [现有 ring 实现 overlap](./2026-11-08_0915_BASELINE_te_turbo_ring_attn_overlap.md) |
| TE ring overlap 端到端（Llama3.1-8B） | **M0-overlap-e2e 2026-11-08 10:10 — 已测，未优化**。Primus 原样 pretrain（TE `p2p`，GQA 32 / 8 heads），overlap 相对不 overlap 每步省 CP2 4.4–5.2% / CP4 19.1–23.8% / CP8 15.3–31.6%（CP8 32K 跨运行 10.7–17.6%），CP2 128K OOM；整步省 ≈ 32 × 单层省（比值 0.93–1.12）；attention 占步时约 44–78%；推到单层 roofline 整步最多再省 5.9–15.2%（上限）。Primus 默认 dataloader 生成 S × S mask 会让长序列步时乱跳，需设 `create_attention_mask_in_dataloader: false`。只适用于 GQA，MLA 另测。详见 [端到端篇](./2026-11-08_1010_BASELINE_llama_e2e_te_ring_overlap.md) |
| 现有 ring 实现 overlap（DSV3 MLA，kernel 级） | **M0-overlap-MLA 2026-11-08 12:20 — 已测，未优化**。128 / 128 heads、d_qk 192、d_v 128、causal：ring 通信量是 GQA 的 20 倍、计算 5.0–5.2 倍，TE 通信 / 计算比是 GQA 的 3.3–4.3 倍。TE（Primus 默认）CP4 / CP8 全部 comm-bound（T_comm / T_comp 1.22–8.49），overlap 已把计算基本藏住，是 roofline（= T_comm）的 1.16–1.30×，省 CP2 18.6–36.3% / CP4 17.4–35.1% / CP8 8.2–24.7%；TE CP > 2 非 batch P2P 只有 33.5–33.7 GB/s（batch 65.5–66.0 GB/s，规格单向 72）；按规格理论下界 CP4 / CP8 也是 comm-bound，TE overlap 是理论 roofline 的 1.81–2.79×。Turbo ring 反向计算慢 8.4–10.6×，CP2 128K dq 错误；128K aiter 全序列参考反向失效，dK / dV 未验证。a2a 未测。详见 [MLA 篇](./2026-11-08_1220_BASELINE_dsv3_mla_ring_attn_overlap.md) |
| TE ring overlap 端到端（DSV3 4 层代理） | **M0-overlap-e2e-MLA 2026-11-08 14:25 — 已测，未优化**。Primus DSV3 BF16 配置切到 4 层 MoE + 1 MTP、EP 8、uniform 路由：overlap 相对不 overlap 每步省 CP2 18.2–20.0% / CP4 13.7–30.3% / CP8 6.5–22.5%（CP8 32K 两次 7.6% / 6.5%），CP2 128K OOM；整步省 ≈ 5 × 单层省（比值 0.94–1.12）；attention 占步时约 44–88%；推到单层 roofline 整步最多再省 6.4–18.6%。32K 下 CP2 → CP8 每卡吞吐 14233 → 2770 tokens/s（MLA CP > 2 comm-bound）。Primus 的 5 个 Turbo MoE / RMSNorm patch 因已装 primus_turbo 缺模块未生效，MoE 走 Megatron alltoall，步时与占比不代表生产配置。详见 [DSV3 端到端篇](./2026-11-08_1425_BASELINE_dsv3_e2e_te_ring_overlap.md) |
| TE vs Turbo attention（单卡 kernel） | **2026-11-08 15:30 — 已测，未优化**。TE 的 CK 融合 attention 底层调的就是 aiter `mha_fwd` / `mha_bwd`，Turbo 固定 AITER 时 GQA / MHA dK / dV 与 TE 逐位相同。Primus 默认设置（atomic fp32 = 0）下 B = 1 sbhd 4K–64K：non-causal 和 MLA 192 / 128 持平（fwd + bwd TE / Turbo 1.00–1.03，MLA Turbo 前向快 4–12%）；GQA / MHA causal Turbo 走 FlyDSL，前向快、反向慢，fwd + bwd 0.79–1.02 / 0.88–0.96（TE 略快，短序列明显）。官方 bench 96 形状各组几何平均 0.96–1.03，d 64 causal B ≥ 2 Turbo 慢约 1.6–1.7 倍。Turbo 的坑：atomic fp32 = 1（Turbo 代码默认）AITER 反向慢 1.29–1.90 倍；V pad 到 192（Turbo ring 反向做法）慢 5.4–6.1 倍且 64K dQ 错误，单卡复现了 Turbo ring CP2 128K 的 dQ 错误。固定 FlyDSL 同形状对比（causal，sbhd）：d 64 FlyDSL fwd + bwd 快 1.44–1.63 倍；d 128 G ≥ 4 持平（0.99–1.09，32 / 8 B = 1 4K 例外 0.79）；MHA TE 快（0.84–0.96）；G 不是 2 的幂（Llama-4、Mixtral-8x22B、Qwen2.5-7B）FlyDSL 不支持。详见 [TE vs Turbo attention](./2026-11-08_1530_BASELINE_te_vs_turbo_attn_kernel.md) |
| FlyDSL 前向 192 / 128（DSV3 MLA 原型） | **2026-11-09 09:42 — 调优到 aiter 的 0.93 倍，单变量调优已停滞**。11 轮单变量迭代（每轮 fp32 对拍、7 个 shape 同卡同进程对比 aiter）：前向几何平均 919.04 → 1058.67 TFLOPS（+15.2%），比值 0.806 → 0.929；4K 达 0.98 / 0.97，32K–64K 停在 0.90。最大一笔是 Q 的 96 宽 fptrunc 被拆成逐元素转换（spill 90 → 0，+6.2%），其次是 8 wave / WG 合并 barrier（+4.8%）和 K_HEAD 4 → 1（+3.5%）；精度不变，D 128 仍与安装版逐位一致。剩余差距在主循环结构：aiter 分块相同，但每 tile 8 个 stage、两组 wave 错开一个 stage；追平需要移植这套流水（重写）。原型阶段（08:40）：拆 D_QK / D_V 只改 QK 一侧，顺带发现并在副本里修了 `block_m 256` causal 的 loop mask 错误；non-causal 错误未定位；反向 192 / 128 未做。详见 [192 前向调优](./2026-11-09_0942_TUNE_flydsl_mla_192_fwd.md)、[192 前向原型](./2026-11-09_0840_PROTO_flydsl_mla_192_fwd.md) |
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
| [2026-11-08_0830_BASELINE_m0_cp_ring_attn_overlap.md](./2026-11-08_0830_BASELINE_m0_cp_ring_attn_overlap.md) | **M0 BASELINE（CP overlap）**：自写 ring pipeline 前向，RCCL P2P + aiter 多 stream overlap，CP 2/4/8 x 32K–128K，chunk 扫描与 knob 对照。数据在 [`data/2026-11-08-overlap/`](./data/2026-11-08-overlap/) |
| [2026-11-08_0915_BASELINE_te_turbo_ring_attn_overlap.md](./2026-11-08_0915_BASELINE_te_turbo_ring_attn_overlap.md) | **M0 BASELINE（现有 ring 实现）**：TE `AttnFuncWithCPAndKVP2P` 与 Turbo `attention_ring` 原样实测，fwd / bwd，不 overlap vs overlap，CP 2/4/8 x 32K–128K；Primus 接线现状；通信 / 计算拆分与 roofline。数据在 [`data/2026-11-08-ring-impl/`](./data/2026-11-08-ring-impl/) |
| [2026-11-08_1010_BASELINE_llama_e2e_te_ring_overlap.md](./2026-11-08_1010_BASELINE_llama_e2e_te_ring_overlap.md) | **M0 BASELINE（端到端）**：Llama3.1-8B Primus pretrain，TE ring overlap vs 运行时串行化 P2P，CP 2/4/8 x 32K–128K；与 32 × 单层的对照；dataloader mask 陷阱。数据在 [`data/2026-11-08-e2e/`](./data/2026-11-08-e2e/) |
| [2026-11-08_1220_BASELINE_dsv3_mla_ring_attn_overlap.md](./2026-11-08_1220_BASELINE_dsv3_mla_ring_attn_overlap.md) | **M0 BASELINE（MLA kernel 级）**：DSV3 MLA 形状下 TE `p2p` 与 Turbo ring 原样实测，CP 2/4/8 x 32K–128K causal；通信 / 计算拆分、实测 roofline、按 MI355X 规格的理论 roofline；现有 MLA CP 实现清单；与 GQA 对比；128K 参考失效与 Turbo dq 错误。数据在 [`data/2026-11-08-ring-impl-mla/`](./data/2026-11-08-ring-impl-mla/) |
| [2026-11-08_1425_BASELINE_dsv3_e2e_te_ring_overlap.md](./2026-11-08_1425_BASELINE_dsv3_e2e_te_ring_overlap.md) | **M0 BASELINE（端到端，MLA）**：DSV3 4 层代理 Primus pretrain，TE ring overlap vs 运行时串行化 P2P，CP 2/4/8 x 32K–128K；与 5 × 单层的对照；每卡吞吐随 CP 的变化；MoE patch 未生效的说明。数据在 [`data/2026-11-08-e2e/`](./data/2026-11-08-e2e/)（`dsv3_*`） |
| [2026-11-08_1530_BASELINE_te_vs_turbo_attn_kernel.md](./2026-11-08_1530_BASELINE_te_vs_turbo_attn_kernel.md) | **BASELINE（单卡 attention kernel）**：TE CK 融合 attention vs Turbo `flash_attn_func`（自动 / 固定 AITER / V pad），GQA / MHA / MLA x S 4K–64K x causal，atomic fp32 = 0 / 1；官方 `bench_attention_{te,turbo}.py` 96 形状；精度对照；Turbo 后端选择规则；V pad 64K dQ 错误。数据在 [`data/2026-11-08-attn-kernel/`](./data/2026-11-08-attn-kernel/) |
| [2026-11-09_0840_PROTO_flydsl_mla_192_fwd.md](./2026-11-09_0840_PROTO_flydsl_mla_192_fwd.md) | **PROTO（FlyDSL 前向 192 / 128）**：gfx950 dual-wave 前向副本拆 D_QK / D_V；改动清单；D 128 逐位回归；DSV3 形状 fp32 / aiter 对拍；同卡 bench vs aiter / TE；VGPR / spill / LDS 占用与 K_HEAD 扫描；现有 FlyDSL 前向 `block_m 256` causal 与 non-causal 错误及前者的修复。数据在 [`data/2026-11-09-flydsl-mla-fwd/`](./data/2026-11-09-flydsl-mla-fwd/) |
| [2026-11-09_0942_TUNE_flydsl_mla_192_fwd.md](./2026-11-09_0942_TUNE_flydsl_mla_192_fwd.md) | **TUNE（FlyDSL 前向 192 / 128）**：11 轮单变量调优记录（4 接受 / 6 回滚）；逐 shape 基线 vs 最终 vs aiter；Q 的 96 宽 fptrunc 标量化导致整段 spill 的定位与修复；barrier 合并；无效方向（读节奏 hint、Q 重载、K_HEAD 2、row-sum 位置、按半组设优先级）；ATT 剩余差距分析与 aiter 8 stage 错相流水对照（实测与推断分开）。数据在 campaign 目录 `Primus-Turbo/agent/workspace/flash_attn_fwd_mla_flydsl_gfx950_20261109/` |

## 相关

- [H2 2026 roadmap](../primus-moe/2026-05-29_roadmap_h2_2026.md) — L96 DSA / L120 fused MQA / L121 CSA-HCA / L260 FP8 indexer
- [rocmoe v2 架构设计](../rocmoe/2026-05-21_1252_rocmoe_v2_architecture_design.md) — role-split + scoreboard 同步的先例
- [训练优化 landscape 2026](../../knowledge/systems/training-optimization-landscape-2026.md)
