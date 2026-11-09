# FlyDSL gfx950 前向 192 / 128（DSV3 MLA）原型（PROTO）

2026-11-09 08:40（UTC+8），MI355X gfx950 单卡，BF16。新文件，不改现有 kernel；未提交。

## 结论

- **原型能用，精度和 aiter 同一量级。** DSV3 形状（128 / 128 heads，d_qk 192，d_v 128，causal，sbhd），B 1 / 2、S 4K–64K：
  O 相对 fp32 误差 0.0016–0.0020（aiter 0.0013–0.0017），LSE 最大绝对误差 0.0039–0.0064，原型与 aiter 的 O 相对差 0.0031–0.0033。
- **性能还没追上：** 最好配置（`block_m 256`、`K_HEAD 4`）同卡对比是 aiter 的 0.79–0.84 倍（B = 1 两张卡 10 点几何平均 0.82），
  TE 的 0.88–0.90 倍（几何平均 0.89）；B = 2 是 aiter 的 0.77 / 0.86 倍（4K / 16K）。绝对值 798–1083 TFLOPS。
- **改动集中在 QK 一侧。** V / O 仍是 128，所以 P×V、O 累加器（4 × 16 fp32）和手写流水线不动；改的是 Q / K / V / O 四套
  stride 与 head 偏移、K 的 LDS tile（2 → 3 个 64 列块）、K DMA 次数（2 → 3）、QK k-step（8 → 12）。副本在 D 128 / 128 下
  与安装版 FlyDSL 逐位一致（O 和 LSE，MHA 32 / 32、GQA 32 / 8、64 / 8，B 1 / 2，S 4K / 16K，12 / 12；改完 loop mask 后重跑仍一致）。
- **192 卡在资源上：** 4 wave / WG（Turbo 的 d128 配置 `block_m 128`）编出 411 VGPR、LDS 84736 B，两者都只放得下 1 wave / SIMD，
  只有 aiter 的 0.71–0.84 倍（跨卡）。换 8 wave / WG（`block_m 256`）后 VGPR 被压到 256、spill 105–133 个，但主循环里没有 spill。
- **顺带发现现有 FlyDSL 前向两条 Turbo 不走的路径是错的**（安装版原样复现，Turbo 不受影响）：
  - `block_m 256` 的 causal（也是 builder 对稠密 causal 的默认值 `block_m=None`）：q 块 ≥ 1 的前 64 行（wave 0 / 1）错，
    D 128、S 1K、4 heads 时 O 相对误差 0.080。原因是主循环不做 causal mask，只有 epilogue 的最后 3 个 KV tile 做，
    而 256 行的对角带跨 4 个 tile。副本已修（见下）。
  - non-causal：D 128、`block_m 128`、S 4K 时 O 相对误差 0.155（4064 / 4096 行错），192 原型同样（0.152 / 0.076 @ 4K / 16K）。未定位。

## 改了什么（全部是新文件）

目录 `megaattention_overlap_bench/flydsl_mla/`，从容器里实际加载的安装版复制（`/opt/venv/.../primus_turbo/flydsl/`，
与工作区 `Primus-Turbo/` 的 `attn_helper.py` 只差 buffer 指针构造的 API 写法）：

| 文件 | 来源 | 改动 |
|---|---|---|
| `attn_helper_mla.py` | `flydsl/utils/attn_helper.py` | traits 加 `HEAD_DIM_V`、`SMEM_D_RPT_K` / `_V`、`DEFAULT_STRIDE_V_N` / `_O_N`，`D_CHUNKS` 按 D_V；K / V 各用自己的 DMA 次数、head 偏移、行 stride、batch 基址；`load_all` 拼任意个 Q 包；O 写回用 O 的 stride 与 D_V；`cache_tag` 加 `HEAD_DIM_V`、`K_HEAD` |
| `flash_attn_fwd_mla.py` | `flydsl/attention/flash_attn_fwd.py` | builder 收 `head_dim_v`（只放行 64 / 64、128 / 128、192 / 128）和 `k_head`；`BLOCK_M > 3 × BLOCK_N` 时主循环两处 causal mask 打开（`loop_masks`） |
| `ops.py` | 仿 `attention_flydsl_impl._sbhd_forward_op` | `mla_fwd_sbhd(q, k, v, causal, return_lse, block_m=256, k_head=4, ...)`，返回 O 和 LSE `[B·S, Hq]` |

测试与报告：`flydsl_mla_test.py`（`--mode regress / check / bench`）、`flydsl_mla_debug.py`（构造输入定位错误）、`flydsl_mla_report.py`。

V / O 的 stride 由 Q / K 的 stride 按 D_V / D_QK 换算，不加 kernel 参数，所以只支持连续的 `[S, B, H, D]` / `[T, H, D]`。
softmax scale 固定为 1 / sqrt(D_QK)（`head_dim_runtime`）；Turbo FlyDSL 封装的 `_scale_ok` 也只接受这个值。

vmcnt 等待没改：主循环和 epilogue 的每个等待点只要求“两个 tile 之前发的 DMA”完成，原公式 `(K + V) × reps // 2`
在 K（3 条）≠ V（2 条）时仍不超过这个上限；`VM_DRAIN_V` 只依赖 V。这是读代码的推断，由下面的精度结果佐证。

## 环境与方法

- 节点 smci355-ccs-aus-n04-25（之前几篇是 n05-29），容器 `xiaoming-dev`（镜像 `tasimage/primus:pr-1275`）。
  ROCm 10.1.0，torch 2.14.0+rocm10.1.0，primus_turbo 0.6.1.dev4，flydsl 0.3.4.1，TE 2.17.1+130099ce4。每 CU LDS 163840 B。
- 精度：fp32 参考逐 head 计算，每个 batch 取 256 个随机行加 16 个边界行（0、31、32、63、64、127、128、255、256、末尾等）；
  aiter 为 Turbo 固定 AITER 的 `flash_attn_func`（同一份 sbhd 字节）。
- 计时：CUDA event，热身 3 次后跑满 3 s 取中位数。原型带 LSE 输出；aiter / TE 的输入 `requires_grad`，前向同样保存 LSE。
  FLOPs = B · S(S+1)/2 · Hq · 2 ·（d_qk + d_v）。
- 同一个 kernel 在不同 GPU 上差约 8%（aiter 4K：GPU 0 963、GPU 1 1039 TFLOPS），所以比值只用同一张卡上背靠背的结果；
  不同配置之间的比较只保留差距明显大于 8% 的结论。

## 精度

`block_m 256` / `K_HEAD 4`（`block_m 128` 默认配置的数字逐项相同，见 [`tables.md`](./data/2026-11-09-flydsl-mla-fwd/tables.md)）：

| B | S | O 相对误差 vs fp32 | O 最大绝对误差 | LSE 最大绝对误差 | O 相对差 vs aiter | aiter O 相对误差 vs fp32 |
|---|---|---|---|---|---|---|
| 1 | 4K | 0.0020 | 0.0099 | 0.0053 | 0.0031 | 0.0017 |
| 1 | 8K | 0.0018 | 0.0113 | 0.0055 | 0.0032 | 0.0015 |
| 1 | 16K | 0.0017 | 0.0097 | 0.0048 | 0.0032 | 0.0015 |
| 1 | 32K | 0.0016 | 0.0091 | 0.0039 | 0.0032 | 0.0014 |
| 1 | 64K | 0.0016 | 0.0093 | 0.0045 | 0.0033 | 0.0013 |
| 2 | 4K | 0.0020 | 0.0103 | 0.0064 | 0.0031 | 0.0017 |
| 2 | 16K | 0.0018 | 0.0105 | 0.0041 | 0.0032 | 0.0015 |

原型 O 误差比 aiter 高 13–23%，量级相同。非 causal 见上面的现有路径错误，不列。

## 性能（causal，B = 1 除注明外）

同卡背靠背（`block_m 256`、`K_HEAD 4`）：

| GPU | S | 原型 TFLOPS | aiter TFLOPS | TE TFLOPS | 原型 / aiter | 原型 / TE |
|---|---|---|---|---|---|---|
| 0 | 4K | 798 | 963 | 886 | 0.83 | 0.90 |
| 0 | 8K | 919 | 1123 | 1028 | 0.82 | 0.89 |
| 0 | 16K | 988 | 1193 | 1125 | 0.83 | 0.88 |
| 0 | 32K | 999 | 1212 | 1136 | 0.82 | 0.88 |
| 0 | 64K | 1025 | 1226 | 1142 | 0.84 | 0.90 |
| 1 | 4K | 817 | 1039 | 932 | 0.79 | 0.88 |
| 1 | 8K | 942 | 1168 | 1069 | 0.81 | 0.88 |
| 1 | 16K | 1028 | 1223 | 1146 | 0.84 | 0.90 |
| 1 | 32K | 1076 | 1280 | 1205 | 0.84 | 0.89 |
| 1 | 64K | 1083 | 1323 | 1232 | 0.82 | 0.88 |
| 2（B = 2） | 4K | 821 | 1066 | 961 | 0.77 | 0.85 |
| 2（B = 2） | 16K | 1023 | 1191 | 1106 | 0.86 | 0.93 |

配置对比（各配置在不同 GPU 上，跨卡，只看大差距）：`block_m 128`（4 wave / WG，默认 K_HEAD 10）805–867 TFLOPS，
`block_m 256` / K_HEAD 4 807–1093 TFLOPS；8K 起 `block_m 256` 快 16–26%，4K 两者持平。K_HEAD 在同一张卡上扫（S 16K，各 1 次）：
4 / 6 / 8 / 10 为 1009 / 984 / 822 / 648 TFLOPS，4 和 6 差 2.5%（在噪声内），8 和 10 明显变慢（spill 117 / 133）。

## 资源占用

见 [`resources.md`](./data/2026-11-09-flydsl-mla-fwd/resources.md)。要点：

| D_QK / D_V | block_m | K_HEAD | VGPR | spill | LDS（B） |
|---|---|---|---|---|---|
| 128 / 128 | 128 | 6 | 232 | 0 | 68096 |
| 192 / 128 | 128 | 10 | 411 | 0 | 84736 |
| 192 / 128 | 256 | 4 | 256 | 105 | 84736 |

- `block_m 128` 下 Turbo d128 靠每 CU 2 个 WG（2 × 68096 B）拿到 2 wave / SIMD；192 / 128 两个 WG 要 169472 B，超过 163840 B，
  而且 411 VGPR 本身也只容许 1 wave / SIMD。不加 padding 的 K + V 双缓冲是 81920 B / WG，正好 160 KiB / 2，padding 一加就放不下。
- `block_m 256`（8 wave / WG）每 CU 一个 WG 就有 2 wave / SIMD，LDS 不再是限制；8 wave 的 WG 让编译器把 VGPR 封顶到 256，
  spill 全在 prologue（101 条 scratch 指令）和循环出口 / epilogue（303 条），主循环 0 条。

## `block_m 256` causal 错误与修复

- **现象（实测）：** 安装版 builder，D 128，causal，S 1K，4 heads：`block_m 128` 误差 0.0025、0 坏行；`block_m 256`（stagger 开或关、
  以及 `block_m=None`）误差 0.0803、189 坏行，从第 256 行开始，每个 256 行 q 块里的第 0–63 行。q = 0（纯 V / O 路径）也错，与 QK 无关。
- **原因（读代码）：** 稠密 causal 路径的主循环只在 cross-seqlen 或 SWA 时做 causal mask（“Cluster 6 is before the diagonal band”），
  对角带只在 epilogue 的最后 3 个 KV tile（3 × 64 = 192 行）里 mask。`block_m 256` 的对角带跨 4 个 tile，第一个落进主循环没 mask，
  wave 0 / 1 的行看到了未来的 key；q 块 0 的这个 tile 走 prologue（有 mask），所以没错。
- **修复（副本）：** `BLOCK_M > 3 × BLOCK_N` 时打开主循环两处 `causal_mask_prologue_if_needed`（函数内部按 wave 判断是否需要）。
  修后 D 128 / 128 与 192 / 128 在 `block_m 256` 下都 0 坏行，误差与 `block_m 128` 相同。这处修复多出的分支开销没单独测。
- Turbo 对 d 64 / 128 总是传 `block_m 64 / 128`，所以生产路径不受影响；直接调 builder 而不传 `block_m` 的人会踩到。

## 事实与假设

事实（本次实测）：上面所有误差、TFLOPS、VGPR / spill / LDS、逐位回归、两条现有路径的错误。

假设（未验证），关于与 aiter 剩下的 16–21% 差距：

- QK 每个 tile 的 MFMA 从 16 条变成 24 条，但 `_sched_barrier_pairs` 的交错数（如 `(10, 5)`、`(6, 3)`）是按 d128 手调的，没重调；
- 8 wave / WG 这种配置 Turbo 没用过，流水线是按 4 wave / WG、每 CU 2 WG 调的；
- prologue / epilogue spill 对短的 q 块（causal 前几个块）占比更高；GPU 1 和 B = 2 上 4K 的比值最低（0.79 / 0.77）与此一致，
  GPU 0 上不明显（4K 0.83、8K 0.82），没拆开量；
- `block_m 256` 的对角带更宽（4 个 tile），被 mask 掉的计算更多；主循环加的 mask 分支也有开销。

## 对 MegaAttn（DSV3 CP 通信 + attention tile 级 overlap）的含义

- 前向 192 / 128 在 FlyDSL 上是可写的（本次一轮内跑对）；但要当 MegaAttn 的计算底座，还差三件事：
  前向追平 aiter（现在 0.82）、反向 192 / 128（`flash_attn_bwd.py` 6121 行，D 相关分支多，工作量远大于前向）、
  任意 softmax scale（DSV3 若带 YaRN mscale）。
- 两条现有路径的错误说明 FlyDSL 前向只有 Turbo 实际用的配置被验证过；在副本里动配置，每一步都要配 fp32 对拍。

## 复现

```bash
docker exec xiaoming-dev bash -lc 'cd /perf_apps/xiaoming/megaattention_overlap_bench && \
  python flydsl_mla_test.py --mode regress --batches 1 2 --seqlens 4096 16384 && \
  python flydsl_mla_test.py --mode check --block-m 256 --k-head 4 --batches 1 --seqlens 4096 65536 && \
  python flydsl_mla_test.py --mode bench --block-m 256 --k-head 4 --impls proto aiter te --seqlens 4096 16384 65536'
```

数据在 [`data/2026-11-09-flydsl-mla-fwd/`](./data/2026-11-09-flydsl-mla-fwd/)：`bench_same_gpu*.jsonl`（同卡对比）、`bench.jsonl`（配置对比，跨卡）、
`check*.jsonl`（精度）、`regress.jsonl`（逐位回归）、`tables.md`、`resources.md`；`bench_quick.jsonl` 是有其他任务并行时的首测，仅留档。
