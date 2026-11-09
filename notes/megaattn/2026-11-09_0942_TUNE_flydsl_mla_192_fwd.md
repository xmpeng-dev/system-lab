# FlyDSL gfx950 前向 192 / 128 调优（TUNE）

2026-11-09 09:42（UTC+8），MI355X gfx950 单卡（GPU 0），BF16。接 [192 前向原型](./2026-11-09_0840_PROTO_flydsl_mla_192_fwd.md)，
只改 `megaattention_overlap_bench/flydsl_mla/` 里的副本，不碰 Megatron / Primus / 现有 attention kernel；未提交。
按单变量逐轮迭代（每轮 fp32 对拍 + 7 个 shape 全量 bench，aiter 每轮同进程重测），共 11 轮。

## 结论

- **前向几何平均 919.04 → 1058.67 TFLOPS（+15.2%），对 aiter 的比值 0.806 → 0.929。** 7 个 shape 全部变快，
  短序列涨得最多（4K +25% / +27%，比值 0.98 / 0.97），32K–64K 涨 9%，比值停在 0.90。精度不变：O 相对 fp32 误差
  0.0016–0.0020、LSE 最大绝对误差 0.0046–0.0051；最终版的输出与 round 5 逐位一致；副本在 D 128 / 128 下仍与安装版
  Turbo FlyDSL 前向逐位一致（12 / 12）。
- **最大的一笔（+6.2%）是一个向量宽度问题，不是调度。** 缩放后的 Q 用一条 96 宽的 `fptrunc`（12 个 k-step × 8）转 bf16；
  96 不是 2 的幂，指令选择把它拆成逐元素转换（`v_cvt_pk_bf16_f32 vD, vF32, s0`），每个 bf16 占一个 VGPR，喂 MFMA 前再用
  `v_perm_b32` 两两拼包。寄存器分配 spill 的正是这 96 个散开的元素，epilogue 每个 Q dword 要两次 scratch 读 + `vmcnt(0)` +
  `v_perm`。改成每 8 个元素一条 `fptrunc` 后 spill 90 → 0、scratch 364 → 0 B、`v_perm` 284 → 0、指令 4026 → 3236。
  D 128（64 宽）天然拆成成对转换，所以 Turbo 现有 kernel 没暴露这个问题。
- **另外两笔：** 8 wave / WG 时把访存 cluster 和 P×V cluster 的 barrier 合并掉（每 tile 4 个 → 1 个，+4.8%）；寄存器吃紧时
  K_HEAD 4 → 1（+3.1% / +0.4%）。
- **单变量调优已停滞，剩下的差距在长序列主循环的结构上。** round 8–11 连续 4 轮无收益（见下表）。aiter 的 d192 kernel
  分块与这里完全相同（每 wave 32 行 q、8 wave、KV tile 64、K / V 双缓冲），差别在流水：每 tile 8 个 stage、每 stage 一个
  barrier、访存 / 计算 stage 交替，两组 wave 错开一个 stage，让一组的 MFMA 与另一组的 LDS 读重叠。这里每 tile 只有 1 个
  barrier，同一 SIMD 上的两个 wave 被锁在同一相位；现有双缓冲协议没有错开一个 barrier 的余量。追平需要移植这套结构，是重写，
  不是调参。

## 逐 shape 结果（GPU 0，同进程背靠背）

DSV3 形状：Hq = Hkv = 128、d_qk 192、d_v 128、causal、sbhd。

| shape | 基线 round 1 | 最终 round 7 | 涨幅 | aiter（round 7 同进程） | 比值 |
|---|---|---|---|---|---|
| B1 S4K | 800.8 | 1004.8 | +25.5% | 1023.6 | 0.982 |
| B1 S8K | 911.4 | 1056.4 | +15.9% | 1137.1 | 0.929 |
| B1 S16K | 987.3 | 1088.5 | +10.3% | 1188.6 | 0.916 |
| B1 S32K | 1002.6 | 1095.2 | +9.2% | 1214.3 | 0.902 |
| B1 S64K | 1020.7 | 1110.1 | +8.8% | 1224.9 | 0.906 |
| B2 S4K | 780.9 | 987.9 | +26.5% | 1013.5 | 0.975 |
| B2 S16K | 961.5 | 1074.1 | +11.7% | 1196.2 | 0.898 |
| 几何平均 | 919.04 | 1058.67 | +15.2% | 1139.49 | 0.929 |

单位 TFLOPS。基线与原型笔记里的数字不同（那篇是另一张卡、CUDA event 中位数）；这里所有比值都来自同一张卡同一进程。

## 逐轮记录

| 轮 | 结果 | 改动 | 几何平均 | 比值 |
|---|---|---|---|---|
| 1 | 基线 | `block_m 256`、K_HEAD 4、loop mask 修复 | 919.04 | 0.806 |
| 2 | 回滚 | QK cluster 用 sched_group_barrier 给尾部 K 读定节奏 | 913.58 | 0.802 |
| 3 | 接受 | K_HEAD 4 → 2（spill 105 → 90） | 947.53 | 0.831 |
| 4 | 接受 | K_HEAD 2 → 1 | 951.43 | 0.835 |
| 5 | 接受 | 8 wave / WG 合并访存与 P×V cluster 的 barrier | 996.81 | 0.875 |
| 6 | 回滚 | epilogue 从全局内存重新加载 Q | 913.70 | 0.802 |
| 7 | 接受 | Q 按 MFMA 包（8 元素）做 fptrunc（spill 90 → 0） | 1058.67 | 0.929 |
| 8 | 回滚 | 无 spill 后 K_HEAD 1 → 2 | 1054.93 | 0.925 |
| 9 | 回滚 | 去掉 row-sum MFMA 与 QK 之间的 fence | 1058.40 | 0.928 |
| 10 | 回滚 | P×V `s_setprio` 只给后派发的半组 wave | 1034.43 | 0.907 |
| 11 | 回滚 | P×V `s_setprio` 只给先派发的半组 wave | 1044.56 | 0.916 |

纯同步类改动（round 5、9、10、11）和数值上应等价的改动（round 6、7）都和上一个接受版本做了逐位对比（4 个 shape × 5 次），
全部一致，用来排除竞争条件。

## 接受的改动

round 7，`attn_helper_mla.py` 的 `scale_all`：

```python
q_f32_vec = Vec(q_all_scaled_f32)
q_packs = []
for ks in range_constexpr(traits.K_STEPS_QK):
    base = ks * traits.MFMA_LANE_K
    q_slice = q_f32_vec.shuffle(q_f32_vec, [base + i for i in range(traits.MFMA_LANE_K)])
    trunc_op = llvm.FPTruncOp(Vec.make_type(traits.MFMA_LANE_K, self.elem_dtype), q_slice.ir_value())
    trunc_op.operation.attributes["fastmathFlags"] = fm_fast_attr
    q_packs.append(trunc_op.result)
# 再按 load_all 的树形 8 -> 16 -> 32 -> 96 拼回原来的向量，接口不变
```

round 5，`flash_attn_fwd_mla.py`（原来只对 `Q_HEADS_PER_WG > 1` 打开）：

```python
fold_mem_barriers = sink_drains and traits.NUM_WAVES > 4
```

依据：每个覆盖 LDS buffer 的 DMA 之前都已有一个计算 cluster 的 barrier 保护读后写，访存 cluster 的那次汇合不保护任何东西；
这条论证对任何共享 K / V DMA 的 8 wave WG 都成立。

round 3 / 4，`ops.py`：`mla_fwd_sbhd` 默认 `k_head=1`（K_HEAD 3 起又出现 spill：3 → 1 个，4 → 11 个）。

## 剩余差距：实测与推断

实测（ATT，round 7，B1 S16K，单 CU）：
- 主循环约占 wave 时间的 93%。按指令类：MFMA 30.5%、`s_waitcnt` 14.5%、`s_barrier` 13.9%、`ds_read` 12.3%、`v_exp` 9.9%。
- 循环里两个 barrier 各占 6.0% / 6.4%，它们前面的 `s_waitcnt vmcnt(0) lgkmcnt(0)` 只各占 0.7%，所以等的是 wave 之间的进度差，
  不是 DMA。
- wave 派发顺序：一个 WG 的 8 个 wave 依次落在 SIMD 0、2、1、3 的 slot 0，再是 slot 1，即 wave w 与 w + 4 共用一个 SIMD
  （假设派发顺序就是 wave 编号）。
- 按 wave 半组拆分 P×V 优先级，两个方向都变慢（-2.3% / -1.3%），说明优先级在这里不是中性的，但靠它造不出相位差。

读 aiter 源码（只读，`profiles/aiter_ref/`）：每 tile 8 个 stage，GEMM0 沿 KV 拆成两个 super unit，每个 super unit 的 K
（48 个 VGPR）在前一个访存 stage 一次读完；GEMM1 沿 D_V 拆；一组 wave 在 prologue 多走一个 barrier、另一组在 epilogue 补一个，
两组全程差一个 stage。LDS 只用 2·K + max(Q, 2·V)。

推断（未验证）：长序列剩下的约 10% 主要来自同一 SIMD 上两个 wave 同相位竞争（访存段和 softmax 段 MFMA 空转），需要 aiter 式的
细粒度 stage 加一组错相才能拿回；在每 tile 1 个 barrier 的现有协议上，错开一个 barrier 会让读后写冲突，必须同时改流水。

## 环境与方法

- 节点 smci355-ccs-aus-n04-25，容器 `xiaoming-dev`；ROCm 10.1.0、torch 2.14.0+rocm10.1.0、flydsl 0.3.4.1；固定 GPU 0。
- 精度门槛：每个 batch 256 个随机行加边界行做 fp32 参考，O 相对误差 ≤ 0.003、LSE 最大绝对误差 ≤ 0.01。
- 计时：`time.perf_counter` 循环，每次约 3 s 预算、取 5 次的均值（各轮单 shape 标准差 0.4–6.5 TFLOPS，约 0.04–0.6%）；aiter 为 Turbo
  固定 AITER 的 `flash_attn_func`，同一份 sbhd 字节，每轮同进程重测。
- 每轮用新的 `FLYDSL_RUNTIME_CACHE_DIR` 编译；ISA 用 `FLYDSL_DUMP_IR=1` 取 `22_final_isa.s` 看 VGPR / spill / scratch。
- 接受规则：几何平均提升且没有 shape 退步 ≥ 5%。

## 数据

campaign 目录 `Primus-Turbo/agent/workspace/flash_attn_fwd_mla_flydsl_gfx950_20261109/`：
`logs/optimize.md`（逐轮记录、停滞复盘、Final Report）、`logs/performance_trend.md`、`rounds/round-N/summary.md`
（每轮假设、ISA、逐 shape 表、kernel 快照）、`profiles/`（ISA dump、ATT trace、`bitwise_check.py`、aiter 源码副本）。

## 下一步（若继续）

1. 在 FlyDSL 里新写一版主循环，移植 aiter 的 8 stage 结构（KV 拆分的 GEMM0、D_V 拆分的 GEMM1、两组 wave 错开一个 stage），
   保留 round 7 的 Q 打包。目标是 16K–64K 的 0.90。
2. 先做小原型：每 tile 4 个 stage 加一组错相，单独量错相能收回多少 barrier 等待，用逐位对比当竞争检测。
3. 新结构跑通后先确认 spill 仍为 0，再调每个 stage 的 MFMA / VALU 交错。
