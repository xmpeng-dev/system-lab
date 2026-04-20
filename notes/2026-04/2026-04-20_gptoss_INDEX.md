# GPT-OSS-20B MLPerf MI355X 调优思考线索引

**周期**: 2026-04-19 → 2026-04-20
**目标**: 1×8 MI355X 上把 GPT-OSS-20B MLPerf submission 时间压到最低
**起点**: 04-19 凌晨完成首个 ✅ E2E 收敛 run（10375 s）
**终点**: 04-20 晚整理出下次 submission 的最小可执行 yaml diff（估 ~9100 s, −12%）

---

## 阅读顺序（按思考递进）

| # | 文件 | 主题 | 关键产出 |
|---|---|---|---|
| **01** | [`2026-04-19_gptoss_01_mlperf_best_e2e_run.md`](./2026-04-19_gptoss_01_mlperf_best_e2e_run.md) | **起点**：解读首个收敛 run | RESULT 10375 s, val loss 3.3345 ≤ 3.34, 稳态 862 ms / 479 TFLOP/s/GPU |
| **02** | [`2026-04-20_gptoss_02_triton_rmsnorm_optimization.md`](./2026-04-20_gptoss_02_triton_rmsnorm_optimization.md) | **第一个 ablation**：B10 自写 Triton RMSNorm | step 796 → 770 ms (−3.3%), RMSNorm 占比 31% → 4.6% |
| **03** | [`2026-04-20_gptoss_03_triton_rmsnorm_report_raw.md`](./2026-04-20_gptoss_03_triton_rmsnorm_report_raw.md) | 02 的配套原始 trace 数据 | per-kernel 拆分、caller 链、launch 分布 |
| **04** | [`2026-04-20_gptoss_04_grad_reduce_bf16_optimization.md`](./2026-04-20_gptoss_04_grad_reduce_bf16_optimization.md) | **第二个 ablation**：B11 `grad_reduce_in_bf16` | step 770 → 713 ms (−7.45%), 释放 36 GB 显存, 1 行 yaml |
| **05** | [`2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md`](./2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md) | **横向对比**：从 DeepSeek-V3 / Qwen3 ref yaml 找下一批改进 | 推荐 8 行 yaml diff，估 −12% E2E + −25% mem |

---

## 故事线（一句话版）

1. **(01) 看清现状** — best E2E 在哪、瓶颈在哪：稳态 862 ms / step、RMSNorm 31% + elementwise 33% + comm 大头。
2. **(02) 拍最大那块** — 写 Triton RMSNorm 替换 TE 实现：RMSNorm 31% → 4.6%，step −3.3%。
3. **(03) 把 02 的原始数据存档** — kernel/caller/launch 体积分布，方便后续对照。
4. **(04) Profile 暴露第二个大头** — elementwise 里 72% 是 fp32 grad add；改 1 行 `grad_reduce_in_bf16: true` 拿 −10.5% step + −36 GB mem。
5. **(05) 横向找漏项** — DeepSeek/Qwen3 production yaml 都开了 `precision_aware_optimizer + bf16 master/m/v`，当前 mlperf yaml 完全漏掉。再加 `enable_experimental: true`。

---

## 当前已知最佳形态 vs 下次 submission 推荐形态

| 指标 | 当前 best (04-19 run) | 04-20 推荐 (B11 + DeepSeek 借鉴) | Δ |
|---|---:|---:|---:|
| MLPerf RESULT | 10375 s | **~9100 s** | −12% |
| `run_duration` | 10320 s (2h52m) | **~9100 s (2h32m)** | −20 min |
| 稳态 step time | 862 ms | **~760 ms** | −12% |
| 显存 / GPU | 226 GB | **~170 GB** | −25% |
| 收敛性 | val 3.3345 ✅ | 等价（DeepSeek/Qwen3 production 已验证） | — |

**完整 yaml diff** 见 [`05_borrow_from_deepseek_v3_config.md`](./2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md#推荐的下次-mlperf-submission-yaml-完整-diff)。

---

## 还没做、按 ROI 排序的下一步候选

| ROI | 方向 | 来源 note |
|---|---|---|
| ★★★ | 把推荐 yaml diff 落到下次 submission run | [05](./2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md) |
| ★★ | 重测 `use_turbo_rms_norm: true` (DeepSeek 路径) vs 自写 Triton (B10) | [02](./2026-04-20_gptoss_02_triton_rmsnorm_optimization.md) + [05](./2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md) |
| ★★ | DDP `bucket_size_mb` 调优（B11 切 bf16 后默认 25 MB 偏小） | [04](./2026-04-20_gptoss_04_grad_reduce_bf16_optimization.md) |
| ★★ | EP=1 vs EP=8 throughput 对比（当前 sh 脚本覆盖成 EP=1） | [05](./2026-04-20_gptoss_05_borrow_from_deepseek_v3_config.md) |
| ★ | mbs 2→4（应用 04+05 之后释放的 ~50 GB 才够） | [01](./2026-04-19_gptoss_01_mlperf_best_e2e_run.md) + [04](./2026-04-20_gptoss_04_grad_reduce_bf16_optimization.md) |
| ★ | 选择性 GEMM 提到 fp8 (mbs↑ 之后 GEMM 占比上升时再考虑) | [04](./2026-04-20_gptoss_04_grad_reduce_bf16_optimization.md) |

---

## 命名约定

- **`2026-04-DD_gptoss_NN_<topic>.md`**
  - `DD` = note 写作日期
  - `gptoss` = 项目 / topic group（这条思考线都是 GPT-OSS-20B MLPerf）
  - `NN` = 在这条思考线里的递进序号（不是按写作时间，而是按**读起来的逻辑顺序**）
  - `<topic>` = 简短主题
- 同一周/月内 INDEX 命名 `2026-04-DD_<group>_INDEX.md`，DD 取该 group 最后一次更新的日期
