# Tier 1A Plan A — fp8_recipe `tensorwise → delayed` A/B 验证

**日期**: 2026-05-08
**硬件 / 栈**: 8 × MI355X 单机, container `xm-mlperf` (image `tasimage/primus:gpt_oss_20b_training_6.0_20260507_0951`), Primus + Megatron-LM, primus_turbo, FP8 e4m3
**关联**: [note 28](./2026-05-07_gptoss_28_mi355x_new_baseline_trace_breakdown.md) §6 Tier 1A、[note 29](./2026-05-07_gptoss_29_baseline_redo_clean_window.md)、[note 30](./2026-05-07_gptoss_30_fp8_epilogue_fuse_investigation.md) Plan A
**Verdict**: 🟡 **黄灯 / Plan A 不可行 → 转 Plan B**

## TL;DR

把 yaml `fp8_recipe: tensorwise → delayed` 切一行后跑 24-iter trace, 拿到的结论刚好踩中 [note 30 §4.4](./2026-05-07_gptoss_30_fp8_epilogue_fuse_investigation.md#44-决策门) 的黄灯定义:

1. **目标 amax / cast 类 kernel 几乎全部消失** — `primus_turbo::unary_kernel`、`primus_turbo::reduce_row_kernel<AbsMaxOp>`、`transformer_engine::amax_kernel` 三个加起来 baseline 65.16 ms, 在 delayed 下 **0 ms** (n=0); `cast_transpose_optimized_kernel` 反而 +3.13 ms。这一档 **节省 ≈ 62 ms**, 满足 §4.4 "kernel 减少 ≥ 25 ms" 的门槛。
2. **但 step wall 反向暴涨 +157 ms** (938.97 → 1096.29 ms, +16.7%), 同样 iter 10 稳态 trainer log 也是 1093.6 ms (vs baseline 932.8 ms, +161 ms)。回归源 100% 是 MoE 专家路径: `_grouped_persistent_fp8_gemm_kernel` 在 baseline 是主力 fp8 grouped GEMM, **在 delayed 下完全消失**, 取而代之的是 **`_grouped_bf16_persistent_gemm_kernel` (356 ms, n=96)** —— `use_turbo_grouped_mlp` 在 `fp8_recipe=delayed` 下 **静默 fall back 到 BF16**, 不抛 assert, log 也无 warning。
3. **没有触发 [`utils.py:491`](file:///workspace/Primus/primus/modules/trainer/megatron/utils.py) 的 `use_turbo_parallel_linear` assert**, 因为本 yaml 没开 `use_turbo_parallel_linear`。但**另一条静默 fall-back 路径** (`use_turbo_grouped_mlp` × `fp8_recipe ≠ tensorwise`) 等价于把 GroupedGEMM 的 FP8 收益全部退还。

**结论**: 不动源码的 yaml 切换 ≠ "无副作用 A/B"。Plan A 路线作废, 后续只能走 **Plan B (给 primus_turbo Float8Linear *和* GroupedGEMM 都加 delayed scaling)** 或 **Plan D (不动)**。Tier 1A 在当前栈下没有低成本入口被坐实。

## 1. 改动 (临时, 已还原)

| 文件 | 行 | before | after | 状态 |
|---|---:|---|---|---|
| `small_llm_moe_pretraining/primus/gpt_oss_20B-pretrain-fp8.yaml` | 55 | `fp8_recipe: tensorwise` | `fp8_recipe: delayed` (运行时) → `tensorwise` (跑完已还原) | ✓ restored |
| `run.sh` / `config_MI355X*.sh` / 任何 .py / .cu | — | — | **未改** | — |

不关 `use_turbo_parallel_linear` (按 [task 硬约束](#)), 不动 schedule, 不动 turbo flag。

## 2. 跑步流程

```bash
# Step 0: 备份 baseline trace (note 29 的 8-rank 集)
docker exec -w .../primus xm-mlperf bash -c '
  TS=$(date +%H%M%S)  # = 090941
  mkdir -p output/_archive_baseline_${TS}
  mv output/amd/root/gpt_oss_20b/tensorboard/primus-megatron-exp*.pt.trace.json output/_archive_baseline_${TS}/
  mv output/amd/root/gpt_oss_20b/tensorboard/events.out.tfevents.* output/_archive_baseline_${TS}/
  cp run.log run.log.baseline
'

# Step 1: yaml 改 tensorwise→delayed (StrReplace)

# Step 2: 跑 run.sh
docker exec -w .../primus xm-mlperf bash -c 'bash run.sh'
# wall clock 约 154 s (容器内整个 run_duration MLLOG = 29.54s -> 0.49 minutes)

# Step 3: full_breakdown.py rank2 #18
python3 .cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py \
    'small_llm_moe_pretraining/primus/output/amd/root/gpt_oss_20b/tensorboard/primus-megatron-exp[gpt_oss_20b]-rank[2].1778231520479836292.pt.trace.json' \
    ProfilerStep#18
# 输出 → /tmp/breakdown_delayed.txt

# Step 4: 用 ijson 直接抓四个目标 kernel 在 step 18 内的总时
python3 - <<'PY'
import ijson, re
TS0=3152966149847; TS1=TS0+1096290
patterns={'unary_kernel': r'primus_turbo::unary_kernel',
          'reduce_row':   r'primus_turbo::reduce_row_kernel.*AbsMaxOp',
          'te_amax':      r'transformer_engine::amax_kernel',
          'cast_t':       r'cast_transpose_optimized_kernel',
          'gGEMM_bf16':   r'_grouped_bf16_persistent_gemm_kernel',
          'gGEMM_fp8':    r'_grouped_persistent_fp8_gemm_kernel'}
# ... (per-step accumulation, 见正文) ...
PY

# Step 6: yaml 还原 delayed→tensorwise (StrReplace)
```

`run.sh` 没改, profile 窗口仍然 step 16-20, train_iters=24, 跟 note 29 同口径。

## 3. 实测对比 (rank 2)

baseline 用 [note 29 §3](./2026-05-07_gptoss_29_baseline_redo_clean_window.md#3-与-note-28-的-kernel-分类对比rank-2) 的 ProfilerStep#17 = 938.97 ms。delayed 用 ProfilerStep#18 = 1096.29 ms (note 29 §4 已验证同 trace 内 step 间一致性 ≤ 0.1%, 跨 trace #17 ↔ #18 mismatch 可忽略)。

### 3.1 目标 4 项 amax/cast kernel + step wall

| metric | baseline (note 29 #17) | delayed (#18) | Δ ms | Δ % |
|---|---:|---:|---:|---:|
| step wall | **938.97** | **1096.29** | **+157.32** | **+16.7%** |
| `primus_turbo::unary_kernel` (input cast) | 31.27 | 0.00 | -31.27 | -100% |
| `primus_turbo::reduce_row_kernel<AbsMaxOp>` (input amax) | 25.40 | 0.00 | -25.40 | -100% |
| `transformer_engine::amax_kernel` (TE 一支 input amax) | 8.49 | 0.00 | -8.49 | -100% |
| `cast_transpose_optimized_kernel` (TE bwd 一支 cast+transpose) | 10.59 | 13.72 | +3.13 | +29.6% |
| **(amax+cast 4 项总和)** | **75.75** | **13.72** | **-62.03** | **-81.9%** |

→ **§4.4 黄灯条件 1 (kernel ≥ -25 ms) 满足** (-62 ms), **条件 2 (step wall ≤ -15 ms) 严重违反** (反向 +157 ms)。

### 3.2 step wall 暴涨的归属 — MoE GroupedGEMM 退化到 BF16

| GEMM kernel | baseline | delayed | Δ ms | 解读 |
|---|---:|---:|---:|---|
| `_grouped_persistent_fp8_gemm_kernel` | 主力 (note 28 §3 grouped_gemm 326 ms 内) | **0** (n=0) | -326 | **MoE 专家 fp8 路径完全消失** |
| `_grouped_bf16_persistent_gemm_kernel` | 0 (baseline trace 不含此 kernel) | **356.00** (n=96) | +356 | **回退到 bf16 grouped GEMM** |
| `_grouped_variable_k_gemm_kernel` | 含在 326 ms 内 | 196.56 (n=48) | — | wgrad / dgrad-K branch (一直存在) |
| **grouped_gemm 总类** (note 29 §3 口径) | **325.75** | **552.56** | **+226.81** | 累计 +227 ms 全部由 fp8→bf16 退化造成 |

总账:
- amax/cast 节省: **-62 ms**
- MoE GroupedGEMM 退化: **+227 ms**
- 净 step wall: **+157 ms** (+227 - 62 = +165, 跟实测 +157 差 8 ms 的零头来自 cast_transpose 微涨 +3 ms、attn 微涨 -19 / +20 之类的二阶调整, 见 [§3.3](#33-类别级-breakdown))

**核心观察**: `delayed` 在 turbo 栈下不仅吃掉 turbo Float8Linear 的优化, **更糟的是顺手把 turbo GroupedMLP 的 FP8 路径关掉退到 BF16**。这是 [note 30 §2 Plan A](./2026-05-07_gptoss_30_fp8_epilogue_fuse_investigation.md#plan-a-切-fp8_recipe-tensorwise--delayed-最低成本-ab) 当时没估到的另一个静默副作用 — note 30 关注的是 Float8Linear 那一支的损失, 实际暴露在 trace 里的是 GroupedGEMM 这一支爆雷。

### 3.3 类别级 breakdown (sanity check)

| 类别 | baseline (#17) ms | delayed (#18) ms | Δ ms |
|---|---:|---:|---:|
| grouped_gemm | 325.75 | 552.56 | **+226.81** |
| attn_kernel | 190.46 | 171.52 | -18.94 |
| nccl_generic | 106.13 | 133.08 | +26.95 |
| 稠密 GEMM | 125.18 | 127.00 | +1.82 |
| elementwise | 80.07 | 121.73 | +41.66 (delayed 多了 cast/scale 准备的 elementwise 预热) |
| nccl_ag | 86.65 | 89.74 | +3.09 |
| other | 93.14 | 62.24 | -30.90 |
| norm | 25.88 | 26.13 | +0.25 |
| moe_dispatch | 16.46 | 17.75 | +1.29 |
| optimizer | 16.44 | 16.10 | -0.34 |
| nccl_rs | 13.28 | 15.38 | +2.10 |
| reduction | 39.48 | 12.57 | -26.91 (input absmax reduce 没了) |
| memcpy | 4.32 | 5.30 | +0.98 |
| softmax | 3.10 | 3.08 | -0.02 |
| fp8_cast | 12.76 | 0.04 | **-12.72** |
| **合计 (sum across streams)** | **1,139** | **1,354** | **+215** |
| **wall** | **938.97** | **1,096.29** | **+157.32** |
| **oversub** | 121.3% | 123.5% | +2.2 pp |

观察:
- compute 重头 (grouped_gemm + 稠密 GEMM + attn + elementwise) 净 +251 ms, 单 grouped_gemm 一项就吃了 +227 — 主因坐实。
- amax / reduce_row 这一档 (`reduction` -27, `fp8_cast` -13, `other` -31) 总共 -71 ms, 跟 §3.1 的 -62 ms 在 9 ms 之内吻合 (差异来自归类阈值)。
- **§4.4 yellow path: "turbo Linear 损失抵消了 amax 节省"** — 实际是 "turbo **GroupedMLP** 损失盖过 amax 节省", 损失幅度比预想 (note 30 估的 -25 ms 抵消) **大 8×**。

### 3.4 trainer log 稳态 (排除 profile-window cold-cache 偏置)

| iter | baseline (note 29 run.log) | delayed (本次 run.log) | Δ ms |
|---|---:|---:|---:|
| 10 | 932.8 | **1093.6** | +160.8 |
| 20 | 928 ~ 932 (note 29 §3 推断) | **1417.5** | +480+ (这一段在 cosine warmup 区, 有 cosmetic 抖动, 但绝对 > +250 ms) |

→ 稳态 iter 10 的 +161 ms 跟 trace step #18 的 +157 ms 在 ±5 ms 之内, **回归不是 profile-window 偏置, 是真实训练 step 慢了 16-17%**。

### 3.5 8 rank 一致性 (抽查)

新 trace 8 个 rank 大小 305-310 MB / rank, 比 baseline (340 MB) 小 ~10% — 跟 step wall +16.7% (单 trace 总时间长了, 但 active step 数目同 4 个) 反向, 说明 **delayed 下 launch 数变少** (input amax/cast 那批 launch 真的不发了)。这一点跟 §3.1 中三个 kernel n=0 互证。

## 4. 损失 / 数值 sanity (附带)

只是顺手记录, **不作为本次 verdict 的决策项** (Plan A 验证按 [task 说明](#) 只看 step wall + kernel breakdown):

| 指标 | baseline (note 29) | delayed (本次) |
|---|---|---|
| validation lm loss @ iter 24 | (note 29 未显式报, 但 24-iter run 在同位置 ≈ 8.40 区间) | **8.402710** |
| test lm loss @ iter 24 | 同上 | **8.390441** |

24 iter 太短无法判趋势, 但跟 baseline 同区间, 至少**没爆 NaN, 没崩溢出**。delayed 数值合规审到底要做应该是 80-iter loss 曲线, 这次 verdict 结论不依赖它就已经定。

## 5. Verdict + 后续路线

按 [note 30 §4.4](./2026-05-07_gptoss_30_fp8_epilogue_fuse_investigation.md#44-决策门):

- 绿灯 (kernel ≥ -25 ms AND step ≥ -15 ms): ❌ 第二项严重违反
- **🟡 黄灯 (kernel ≥ -25 ms AND step ≤ -5 ms 或反而增加)**: ✅ 命中 — kernel -62, step **+157**
- 红灯 (kernel < -10 ms 或 crash/loss 漂移): ❌ kernel 收益足够, 训练正常完成

→ **Plan A 路线作废**, 转 [note 30 §2 Plan B](./2026-05-07_gptoss_30_fp8_epilogue_fuse_investigation.md#plan-b-给-primus_turbo-float8linear-加-delayed-scaling-中等工程)。

### 5.1 Plan B 修正 — 比 note 30 估的工程量更大

note 30 §2 Plan B 原本只针对 `primus_turbo/pytorch/modules/linear_fp8.py:Float8Linear`, 估 1-2 天。**本次 trace 显示 GroupedGEMM 才是大头**, 所以 Plan B 必须扩到:

1. `primus_turbo/pytorch/modules/linear_fp8.py:Float8Linear` — 加 delayed scaling
2. `primus_turbo/pytorch/modules/grouped_mlp.py` (或路径同名) 里走 turbo grouped MLP 的那一支 — **加 delayed scaling 支持, 且不能掉 fp8 GEMM 路径** (这是新发现的硬阻塞)
3. 配套: `Float8QuantConfig` 加 `recipe: Recipe.DELAYED`, 全部 turbo 入口都得能识别这个 recipe 而不是哑掉
4. 数值合规: amax history 长度选择 + 全局 reduce 的语义对 GBS=32 有影响, MLPerf 数值合规要审

**新工程量估**: 3-5 天 (而不是 note 30 估的 1-2 天), + 80-iter A/B 数值审 (再 1-2 天)。**ROI 仍然只有 -25 ms 量级 (≈ -2.7%)**, 比 [note 28 §6 Tier 0 (grouped GEMM 单 kernel 调优, 估 -3 ~ -5%)](./2026-05-07_gptoss_28_mi355x_new_baseline_trace_breakdown.md) 收益类似但工程量翻倍。

### 5.2 推荐顺序调整 (相对 note 30 §3 决策矩阵)

| Plan | 状态 | 推荐 |
|---|---|---|
| **A. yaml 切 delayed** | ❌ blocked (本 note 验证) | 不再尝试 |
| **B. primus_turbo Float8Linear + GroupedMLP 加 delayed** | ☐ 工程量上修到 3-5 天 | **降级到次优**, 在 Tier 0 / Tier 4 之后做 |
| **C. FP8 chaining** | ☐ 1 周+ | 维持不动 |
| **D. 不动 / Tier 0 (grouped GEMM 单 kernel 调优)** | ✅ 可立刻接 | **接下来主线** |
| **E. Tier 4 (FMHA bwd sweep)** | ✅ 跟 Tier 0 并行 | 备选 |

→ **下一步建议**: 立刻进 **Tier 0 grouped GEMM 单 kernel 调优** ([note 28 §5](./2026-05-07_gptoss_28_mi355x_new_baseline_trace_breakdown.md#5-grouped-gemm-deep-dive) deep-dive 已经把 shape / 时间分布算好), Plan B 留到 Tier 0 / Tier 4 都收割完之后再考虑。

## 6. 这次验证的修正点 (写给将来的自己)

1. **`delayed` 在当前 turbo 栈下是双副作用**, 不只是 note 30 §1.3 说的 "Float8Linear 失去 turbo 优化", 还会**静默关掉 turbo GroupedMLP 的 FP8 路径**回退到 BF16。后者损失 (+227 ms) 是前者预想损失 (-25 ms × 抵消) 的 8 倍。
2. **`use_turbo_parallel_linear` 的 assert ([utils.py:491](file:///workspace/Primus/primus/modules/trainer/megatron/utils.py)) 不是唯一防线** — 当前 yaml 没开 turbo_parallel_linear (line 180-184 只开 `enable_primus_turbo` + `use_turbo_grouped_mlp` + `use_turbo_rms_norm` + `use_turbo_fused_act_with_probs`), 所以 assert 没抓到, 但更深的 GroupedMLP 路径自己 fall-back 到 BF16 也不抛 warning。**未来再切 fp8_recipe 必须先 grep `use_turbo_*` 全套, 估每一个 turbo 入口对当前 recipe 的支持矩阵。**
3. **trace 的 grouped GEMM kernel 名带 dtype 后缀 (`_grouped_bf16_persistent_gemm_kernel` vs `_grouped_persistent_fp8_gemm_kernel`)** —— 这次靠 top-25 一眼看出 BF16 路径, 以后做 FP8 优化看 trace 永远先 grep `bf16.*gemm` / `fp8.*gemm` 两个对照, 确认数据精度路径, 再讨论 amax/cast 那种 ±10 ms 的小账。
4. **Plan A 一个 yaml 行的"低成本 A/B"概念被打破** — 在 turbo 这种多 hook 栈上, 一行 yaml 可以拉起 N 条静默 fall-back, 单纯看 task `support_fp8_recipe = ["tensorwise", "blockwise", "mxfp8"]` assert 不够, 必须实测 trace。下次类似的 "yaml 切 recipe" 类验证, **先做 5-min trace 不做 80-iter loss**, 确认副作用全谱。

## 7. 文件清单

| 路径 | 内容 |
|---|---|
| `small_llm_moe_pretraining/primus/gpt_oss_20B-pretrain-fp8.yaml` | line 55 已恢复 `tensorwise` (临时改动 → 已还原) |
| `small_llm_moe_pretraining/primus/output/amd/root/gpt_oss_20b/tensorboard/primus-megatron-exp[gpt_oss_20b]-rank[*].17782315*.pt.trace.json` | 本次 delayed run 8-rank trace (留作后续复查) |
| `small_llm_moe_pretraining/primus/output/_archive_baseline_090941/` | note 29 baseline trace 备份 (8 rank + tfevents) |
| `small_llm_moe_pretraining/primus/run.log` | 本次 delayed run 完整 log (iter 10 = 1093.6 ms) |
| `small_llm_moe_pretraining/primus/run.log.baseline` | note 29 baseline log 备份 |
| `/tmp/breakdown_delayed.txt` | `full_breakdown.py` 完整输出 (rank 2 #18) |

## 8. 改动总结

只动了 **1 行 yaml** (line 55, `tensorwise` ↔ `delayed`), 跑完已还原。没有动 run.sh、config_MI355X.sh、Python / C++ 源码, 没动 schedule / lr / warmup / train_iters / turbo flag。Submission 路径完全没影响。
