# 编译后资源占用（`FLYDSL_DUMP_IR=1`，`22_final_isa.s` 元数据）

全部为 causal、sbhd、bf16、`waves_per_eu = 2`、stagger 关。`block_m 128` = 4 wave / WG，`block_m 256` = 8 wave / WG。

| D_QK / D_V | block_m | K_HEAD | `.vgpr_count` | `.vgpr_spill_count` | `.private_segment_fixed_size`（B） | `.group_segment_fixed_size`（LDS，B） |
|---|---|---|---|---|---|---|
| 128 / 128 | 128 | 6（默认） | 232 | 0 | 0 | 68096 |
| 128 / 128 | 256 | 6（默认） | 230 | 0 | 0 | 68096 |
| 192 / 128 | 128 | 10（默认） | 411 | 0 | 0 | 84736 |
| 192 / 128 | 256 | 10 | 256 | 133 | 536 | 84736 |
| 192 / 128 | 256 | 8 | 256 | 117 | 472 | 84736 |
| 192 / 128 | 256 | 6 | 256 | 105 | 424 | 84736 |
| 192 / 128 | 256 | 4 | 256 | 105 | 424 | 84736 |

每 CU LDS 163840 B（`torch.cuda.get_device_properties(0).shared_memory_per_multiprocessor`）。

`block_m 256` / K_HEAD 4 的 scratch 指令按基本块分布：主循环（`.LBB0_7`–`.LBB0_14`，回跳在 `.LBB0_14` 末尾）0 条；prologue `.LBB0_5` 101 条；循环出口 `.LBB0_17` 36 条；epilogue `.LBB0_18`–`.LBB0_36` 共 267 条。

K_HEAD 扫描（GPU 4 依次跑，B = 1，S 16K，causal，`block_m 256`，各 1 次）：

| K_HEAD | ms | TFLOPS |
|---|---|---|
| 4 | 10.903 | 1009 |
| 6 | 11.171 | 984 |
| 8 | 13.384 | 822 |
| 10 | 16.975 | 648 |
