# FlyDSL gfx950 前向 192 / 128 调优续：aiter 流水移植止步，按 XCD 重排 CTA 追平 aiter（TUNE 2）

2026-11-09 14:00（UTC+8），MI355X gfx950 单卡，BF16。接 [TUNE 1](./2026-11-09_0942_TUNE_flydsl_mla_192_fwd.md)（round 7，
1058.67 TFLOPS，对 aiter 0.929）。只改 `megaattention_overlap_bench/flydsl_mla/` 里的副本，不碰 Megatron / Primus / 现有
attention kernel；未提交。两段工作：先按 TUNE 1 的建议移植 aiter 的 8 stage 流水（swp8，7 轮），止步于 0.873；回到 round 7
继续，对照 HipKittens 的 attention 设计，用 PMC 找到 L2 命中率这条线，4 轮（round 12–15）追平 aiter。

## 结论

- **前向几何平均追平 aiter：同卡 1212.71 vs 1208.92 TFLOPS，比值 1.0031（n02-25 节点）。** 同一节点上 round 7 是 1088.92
  （0.9041），提升 11.4%。16K / 32K / 64K 超过 aiter（1.011 / 1.014 / 1.023），B1 4K 1.006，B2 4K 0.996，B1 8K / B2 16K 0.986。
  四轮都只改 CTA 编号到 (head, q-block) 的映射，输出与 round 7 逐位一致。
- **瓶颈不在周期数，而在功耗。** 实测 B1 S16K 持续运行时 round 7、swp8、aiter 都顶在约 1.4 kW（`amd-smi` 显示 PWR_CAP 1400 W），
  频率由功耗决定：aiter 1373–1381 MHz，round 7 1578–1594 MHz。round 7 频率高约 13%，却多用约 13% 的周期，每 FLOP 多耗约 10% 能量。
  所以省周期的改动有一部分会被降频吃掉，省能量的改动会变成频率。
- **找到的那份能量在 L2。** 同为 B1 S16K，round 7 与 aiter 的 L2 请求数相同（3.664e8 vs 3.667e8），round 7 的命中率只有 0.387
  （aiter 0.653）。原因是 grid 以 head 为最快轴：工作组按线性编号轮流派给 8 个 XCD，每个 XCD 同时驻留 16 个 head × 2 个
  q-block，每条 K/V 流只有约 2 个同驻读者。改成每个 XCD 负责每隔 8 个的 head、同一 head 的若干 q-block 相邻派发后，同批次
  测得命中率 0.372 → 0.802，周期反而多 5.6%，但同样 1.4 kW 下频率高 4–5%，墙钟时间下降。
- **aiter 流水移植（swp8）不划算。** 7 轮后 992.33（0.873），仍低于 round 7。stage 拆得再细，周期省下的一部分被降频还回去；
  另有一个只在错相开启时出现的竞争条件没有定位。
- **HipKittens 的思路里，round 7 大部分已经有**（固定参考 max + 偶发 rebase、MFMA 做 rowsum、每 tile 一个 barrier、q-block
  降序派发、无 spill）；QK / exp 流水深度对 round 7 无收益（主循环里没有 MFMA→exp 的 `s_nop`）。真正带来收益的是它的 grid
  映射思路（按 XCD 安排共享 K/V 的 CTA），但 MLA 的 Hkv = Hq，共享者不是同组 head，而是同一 head 的不同 q-block。

## 逐 shape 结果（n02-25，HIP 设备 0 = `amd-smi` GPU 3，同进程背靠背）

DSV3 形状：Hq = Hkv = 128、d_qk 192、d_v 128、causal、sbhd。

| shape | round 7 | round 13 | round 15 | aiter（round 15 同进程） | 比值 |
|---|---|---|---|---|---|
| B1 S4K | 1066.0 | 1079.8 | 1084.6 | 1078.6 | 1.006 |
| B1 S8K | 1109.1 | 1187.5 | 1191.1 | 1208.0 | 0.986 |
| B1 S16K | 1128.9 | 1207.3 | 1284.2 | 1270.0 | 1.011 |
| B1 S32K | 1083.1 | 1247.5 | 1312.9 | 1294.2 | 1.014 |
| B1 S64K | 1113.3 | 1281.7 | 1329.0 | 1298.8 | 1.023 |
| B2 S4K | 1037.8 | 1064.8 | 1066.7 | 1071.1 | 0.996 |
| B2 S16K | 1086.8 | 1205.8 | 1249.3 | 1266.8 | 0.986 |
| 几何平均 | 1088.92 | 1179.59 | 1212.71 | 1208.92 | 1.0031 |

单位 TFLOPS。round 12 / 13 在 n04-25 上完成（下表），中途换节点，round 7 / 13 在 n02-25 同进程重测后再比较。两个节点的
绝对数字差 5–8%（n02-25 上 round 13 的 B1 S16K 持续运行是 1322 W、1927 MHz，没顶到上限；n04-25 是 1400 W、约 1630 MHz）。

## 逐轮记录（round 12 起）

| 轮 | 节点 | 结果 | 改动 | 几何平均 | 比值 |
|---|---|---|---|---|---|
| 7 | n04-25 | 基线 | TUNE 1 终版 | 1058.67 | 0.929 |
| 12 | n04-25 | 接受（取舍） | 每个 XCD 负责每隔 8 个的 head，同一 head 的 8 个 q rank 相邻派发 | 1104.55 | 0.968 |
| 13 | n04-25 | 接受 | 最后 8 个 q rank 恢复按 rank 派发（最长优先） | 1121.20 | 0.983 |
| 14 | n02-25 | 接受（取舍） | 参数扫描：分组 8 → 16 | 1203.93 | 0.9998 |
| 15 | n02-25 | 接受 | 分组按 launch 选：尾部前不足 16 个 rank 时用 8 | 1212.71 | 1.0031 |

- round 12：S16K–64K +6.8% 到 +9.6%，但 B1 4K -3.9%、B2 4K -1.8%。4K 的 PMC：命中率同样上升，周期却多 25%、频率高 17%，
  即 GPU 在空等。推断是负载尾巴：4K 只有 16 个 q rank，一组 8 个占了一半，每个 XCD 最后一批里混着 8 种长度的 causal 任务。
- round 13：最后 8 个 rank 改回 round 7 的顺序（每个 rank 依次跑完该 XCD 的 16 个 head），4K 周期比 round 12 少 15.7%，
  持续吞吐与 round 7 持平；16K 命中率 0.792（round 12 0.802），长序列收益保住。
- round 14：分组 {4, 8, 16} × 尾部 {4, 8, 16} 在两个节点各扫一遍。分组 16 都最好（n04-25 +0.4%，n02-25 +1.6%，相对分组 8），
  分组 4 都比 8 低约 2.5%，分组 16 时尾部长度差别在噪声内。代价是 4K 整段落入尾部、退回 round 7 顺序（-1.5% / -2.8%）。
- round 15：尾部前至少 16 个 rank 用 16，否则用 8。8K 以上映射与 round 14 完全相同，4K 与 round 13 相同（枚举验证：NQ 1–79 与
  256 均为双射）。

映射代码（`attn_helper_mla.py`，round 15；`NUM_XCD = 8`、`XCD_Q_GROUP = 16`、`XCD_Q_GROUP_SHORT = 8`、`XCD_TAIL_Q = 8`）：

```python
_lin = block_idx.y * NH + block_idx.x          # 派发顺序；XCD = _lin % 8
_slot = _lin // NUM_XCD                         # 该 XCD 内的第几个
_nq_head = max(NQ - XCD_TAIL_Q, 0)

def _xcd_map(group):
    q_split = _nq_head // group * group         # 之前按组，之后按 rank
    grp, r = divmod(_slot, group * heads_per_xcd)
    t = _slot - q_split * heads_per_xcd
    if _slot >= q_split * heads_per_xcd:        # 尾部：rank 优先、head 其次
        return t % heads_per_xcd, q_split + t // heads_per_xcd
    return r // group, grp * group + r % group  # 组内：head 优先、rank 其次

h_local, q_rank = _xcd_map(16 if _nq_head >= 16 else 8)
head = h_local * NUM_XCD + _lin % NUM_XCD
q_block = NQ - 1 - q_rank                       # causal 仍是最长优先
```

（kernel 里两个分组都算出来再 `select`，除数全是常量；NH 不是 8 的倍数时用原映射。）

## 实测：功耗、频率与计数器

持续 12 s 循环时用 `amd-smi metric` 采样（B1 S16K，n04-25）：

| kernel | TFLOPS | GFX 频率 (MHz) | 整卡功耗 (W) |
|---|---|---|---|
| round 7 | 1086.6 | 1578–1594 | 1398–1399 |
| round 12 | 1171.2 | 1647–1673 | 1400–1401 |
| round 13 | 1168.6 | 1630–1631 | 1399–1401 |
| swp8 round 6 | 1040.1 | 1594–1609 | 1392–1398 |
| aiter | 1189.2 | 1374–1378 | 1383–1387 |

空闲功耗 239 W。B1 S4K：round 7 999.8（1639–1643 MHz）、round 12 958.0（1919–1935 MHz）、round 13 1000.5（1748 MHz）。

rocprofv3 计数器（B1 S16K，每次 dispatch，n04-25）：

| 计数器 | round 7 | aiter | round 12 | round 13 |
|---|---|---|---|---|
| L2 命中率 HIT / (HIT + MISS) | 0.387 | 0.653 | 0.802 | 0.792 |
| TCC_MISS_sum | 2.245e8 | 1.272e8 | 7.31e7 | 7.61e7 |
| TCC_REQ_sum | 3.664e8 | 3.667e8 | 3.685e8 | 3.664e8 |
| GRBM_GUI_ACTIVE | 1.216e8 | 1.075e8 | 1.291e8 | 1.270e8 |
| SQ_INSTS_MFMA | 3.719e8 | 3.408e8 | | |
| SQ_INSTS_VALU | 9.934e8 | 1.564e9 | | |
| SQ_INSTS_LDS | 4.741e8 | 4.779e8 | | |
| SQ_WAVES | 65536 | 32768 | | |

- round 7 的 MFMA 指令多 9.1%，VALU 反而少 36%，LDS 指令持平。比值与每 tile 44 / 40 条吻合，推断多出的是 rowsum 的 4 条 MFMA
  （aiter 侧未读 ISA 确认）。表中 round 7 / aiter 两列与 round 12 / 13 两列是不同批次，round 12 summary 里同批次的 round 7
  命中率为 0.372。
- `SQ_LDS_BANK_CONFLICT` 在 round 7 和 aiter 上都读到 0，不能用来判断 V 转置读的 bank 冲突。
- round 7 主循环 ISA（每次迭代 2 个 tile）：749 条指令，MFMA 88、`v_exp` 66、VALU 344、DS 112；`s_nop` 共 44 个 wait state，
  30 个在 mask 的 `v_cndmask` / `v_cmp` 之间，10 个在 M0 → DMA，MFMA → exp 一个都没有。

## swp8：aiter 8 stage 流水移植（止步）

| 轮 | 结果 | 改动 | 几何平均 | 比值 |
|---|---|---|---|---|
| 0 | 基线 | 8 stage / tile，两组 wave 错开一个 stage，加调度 hint | 965.66 | 0.847 |
| 1 | 回滚 | QK 用零 seed，平移放到 s5 | 964.14 | 0.846 |
| 2 | 接受（取舍） | i32 `while` 循环计数（SALU 退出判断）+ K 下标 clamp | 978.41 | 0.858 |
| 3 | 回滚 | 零 seed + `v_pk_add_f32` 打包平移 | 949.89 | 0.834 |
| 4 | 接受 | K(t+2) 的 DMA 拆成 s4 一条 + s6 两条 | 987.84 | 0.867 |
| 5 | 回滚 | rowsum 从 s3 挪到 s5（exp / cast 链被 LLVM 下沉进 s4） | 929.23 | 0.816 |
| 6 | 接受 | 同 5，再把 P 包用空 inline asm 钉在 s3 | 992.33 | 0.873 |
| 7 候选 | 未接受 | V buffer 1 的 LDS 读用一个不透明基址（spill 32 → 0） | — | — |

实测发现：
- FlyDSL 的 `scf_range` 总把边界转成 index，i32 的 `range` 仍是 64 位 VALU 退出判断；i32 计数的动态 `while` 才编成
  `s_cmp_lt_u32`（+1.3%）。
- MFMA 交错的代码里，紧跟 `v_mfma_f32_32x32x16_bf16` 的 `v_pk_add_f32` 要等约 40 个 ATT 单位（等 MFMA 结束），标量
  `v_add_f32` / `v_max3_f32` 不等；把每 tile 的平移改成 16 条打包加法反而 -2.9%。
- 后一组 wave 的访存 stage 里第一条 VALU（哪怕是 32 位 `v_add_u32`）在对方 stage 一开头就是 MFMA + exp 时停约 200 个单位，
  拖住后面按序的 LDS 读。这是位置效应，不是 64 位宽度。
- LLVM 会把 exp / cast 链沉到分支后第一次使用处；用带副作用、操作数绑定（`"=v,0"`）的空 inline asm 可以钉住。
- ds_read 的立即数偏移只有 16 位（≤ 65535 B），V buffer 1 的基址（67328 B）超出，LLVM 为每次读单独物化一个地址 VGPR，
  导致 spill，epilogue 里形成 22 段约 650 单位的串行 reload 链（约占每 wave 时间 5%）。
- 7 候选修掉了 spill，但结果不确定：B1 S16K 64 次里 8 次与 round 7 不一致（round 6 88 / 88 一致，关掉错相 64 / 64 一致）。
  推断（未验证）：两组 wave 收尾时有一个顺序漏洞，以前被慢的 epilogue reload 掩盖。
- 计数器（B1 S16K）：swp8 round 6 的 LDS 指令与 aiter 相同，MFMA 多 10%，INT32 VALU 是 aiter 的 7 倍，每 wave 54 条 scratch
  访问，grid 8192 个 CTA vs aiter 4096 个；GRBM_GUI_ACTIVE 多 17%。

## HipKittens 前向（d128，gfx950）对照

读的是 `Primus-Turbo/csrc/kernels/attention/hipkittens/hk_attn_fwd_d128_gfx950.cu`。

| HipKittens 做法 | round 7 / 现状 | 结论 |
|---|---|---|
| 固定零参考 max，不做 running max 与 O 重缩放 | 固定参考 + 越过 margin 96 才 rebase，QK seed 顺带减参考 | 已有，且更安全 |
| rowsum 用 MFMA 乘全 1 操作数 | 同 | 已有 |
| 每 tile 一个 barrier | round 5 起同 | 已有 |
| q-block 降序（最长优先），按 XCD 重排 head 共享 GQA 的 K/V | 有降序；没有 XCD 安排 | 改造后用上（round 12–15），共享者换成同 head 的 q-block |
| QK / exp 流水深度 hint，填 MFMA → VALU 的 8 个 wait state | 主循环里没有这类 `s_nop` | 无收益空间 |
| QK / PV 用 16x16x32 MFMA（矩阵管线寄存器流量少 20%，固定频率下整卡功耗少 6.5%，为该代码注释中的数据） | 32x32x16 | 未做；功耗受限下是下一候选 |
| K / V 三缓冲，V 的等待放到 barrier 之后 | 双缓冲 | 未做 |
| 计数 lgkmcnt 等待用寄存器数据依赖实现（inline asm `"+v"`） | K 部分常驻 + LDS 读分散到计算 cluster | 未做 |
| V 的 LDS swizzle 多一项 XOR，消除 `ds_read_b64_tr_b16` 的 2 路冲突 | 计数器读 0，无法判断 | 待用 ATT 确认 |

## 环境与方法

- 节点：round 12–13 与全部 swp8 在 smci355-ccs-aus-n04-25，round 14–15 在 smci355-ccs-aus-n02-25；容器 `xiaoming-dev`；
  ROCm 10.1.0、torch 2.14.0+rocm10.1.0；HIP 设备 0（两个节点都是 `amd-smi` GPU 3）。换节点后 aiter 的 opus 模块需重新 JIT。
- 精度：每个 batch 256 个随机行加边界行做 fp32 参考，O 相对误差 ≤ 0.003、LSE 最大绝对误差 ≤ 0.01；round 12–15 每轮与上一个
  接受版本逐位对比（4 个 shape × 10 次）。
- 计时：`time.perf_counter` 循环，每次约 3 s、取 5 次均值；aiter 为 Turbo 固定 AITER 的 `flash_attn_func`，每轮同进程重测。
- 功耗 / 频率：`clock_probe.py` 持续跑 12 s，第 6 s 起用 `amd-smi metric -g 3 -c -p` 采样 3 次。
- 计数器：`rocprofv3 --pmc`，每个 kernel 跑 4–5 次取 dispatch 均值。

## 数据

- campaign 目录 `Primus-Turbo/agent/workspace/flash_attn_fwd_mla_flydsl_gfx950_20261109/`：`logs/optimize.md`（含 resume 后的
  PMC 对比、换节点记录）、`logs/performance_trend.md`（含 n02-25 重测行）、`rounds/round-12..15/summary.md`、round 14 的
  `sweep/` 与 `rebase_n02/`、`profiles/pmc_r12/`、`profiles/r_driver.py`。
- swp8 目录 `Primus-Turbo/agent/workspace/flash_attn_fwd_mla_flydsl_swp8_gfx950_20261109/`：`rounds/round-0..6/`、
  `rounds/round-7-candidate/`、`profiles/pmc*/`、`profiles/att_v*/`、`profiles/clock_probe.py`。

## 下一步

1. QK / PV 改用 16x16x32 MFMA：在功耗受限时，矩阵管线的寄存器流量直接对应频率。改动大（S / P 的 lane 布局、mask、
   softmax 归约、rowsum、V 读布局都要跟着变）。
2. B = 2 时 batch 是 grid 最慢的轴，先跑完 batch 0 再跑 batch 1；最长优先与 XCD 分组都没有跨 batch。
3. 用 ATT 量 V 转置读的 bank 冲突，再决定要不要 HipKittens 的 swizzle。
