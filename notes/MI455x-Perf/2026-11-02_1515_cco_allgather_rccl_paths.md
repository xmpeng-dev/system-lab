# allgather 的 overlap 按 RCCL 实现分三段：Ring 256 channel（≥ 512 MiB）E 0.22–0.32，copy engine（128–256 MiB）0.28–0.81，DDA（≤ 64 MiB）dense GEMM 为负

> 时间: 2026-11-02 15:15 (Asia/Shanghai)
> 项目: MI455x-Perf
> 硬件: 4 × AMD Instinct MI455X（gfx1250，256 CU，每 CU 320 KB LDS），`ctheliosr-rck-g02-e15-01`
> 容器: `xiaompen_ep_overlap`（镜像 `amdprimus/amdprimus:tas-20261019`）
> 软件: torch 2.14.0+rocm10.2.0a20261011，HIP 7.17.26405，RCCL 2.32.3（HEAD:2f2a927），hipBLASLt（`_rocm_sdk_devel`），
>   Primus-Turbo 0.6.1.dev2（FlyDSL 0.3.4.1），aiter 0.1.24.post1，rocprofv3
> 代码: `minimax-poc` main @ 104d48b + 未提交改动（`cco-allgather/run.sh`、README；`common/` 未改）
> 数据: [`data/2026-11-02_cco_allgather/`](./data/2026-11-02_cco_allgather/)

## 1. 上下文

- 上一篇（[14:40](./2026-11-02_1440_cco_alltoall_nogate_flydsl.md)）：`cco_bench.py` 改为顺序 launch 后不再卡死；alltoall 正式
  数据 E：FlyDSL 0.58–0.76，attention 0.50–0.66，dense GEMM 0.30–0.56。下一步 P0 是 allgather / reducescatter 重跑。
- 本篇：`cco-allgather` 按 alltoall 的做法（`gemm` data-parallel grid、`gmm_flydsl`、RCCL 默认）出正式数据；M1 稳态、
  RCCL 在各大小选的实现（`NCCL_DEBUG` + rocprofv3 trace）、telemetry。

## 2. 做了什么

| # | 动作 | 脚本 | 结果 |
|---|---|---|---|
| 1 | `cco-allgather/run.sh` 正式扫描：3 kernel × 6 大小，RCCL 默认，300 ms × 10 轮 | 客户 repo 的 `run.sh` | `results/formal/`，全部通过校验，0 卡死，约 6.5 分钟 |
| 2 | `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING`：各大小 RCCL 选的路径 | `scripts/ag_trace.sh`、`ag_trace.py` | `results/rccl_tuning_allgather.log`（rank 0 的 allgather 相关行） |
| 3 | rocprofv3 kernel + copy trace：allgather 单独（6 个大小），`gemm` / `gmm_flydsl` + 1024 / 64 MiB（计算先发） | 同上，`scripts/ag_trace_sum.py` | `trace/`（2.7 MB，不进 git），下文表 3、表 4 |
| 4 | telemetry 汇总（M1、M2 每个 schedule 的频率 / 功耗中位数） | `scripts/tel_table.py` | 下文表 5、表 6 |
| 5 | README 的单组复现命令（容器默认 `ulimit -n` 1024）跑一次 `gmm_flydsl` + 1024 MiB | — | `results/single_check/`：overlap 1.432 ms、E 0.30，和扫描相同；不需要 `ulimit` |

## 3. 正式数据（`results/formal/`）

客户 README 只放 M2 表。M2 的 E 汇总（N·T_comp 最接近 T_comm 的那组，N 都是 1）：

**表 1：M2 的 E**

| 计算 | 1024 MiB | 512 MiB | 256 MiB | 128 MiB | 64 MiB | 16 MiB |
|---|---|---|---|---|---|---|
| `gemm` | 0.23 | 0.22 | 0.48 | 0.28 | −0.28 | −0.36 |
| `gmm_flydsl` | 0.30 | 0.32 | 0.75 | 0.81 | 0.22 | 0.27 |
| `attn_fwd` | 0.27 | 0.27 | 0.66 | 0.60 | 0.23 | 0.30 |
| RCCL 路径（第 4 节） | Ring | Ring | CE | CE | DDA | DDA |

**表 2：M1 稳态干扰**

| 计算 | 通信 | kernel 单次：单独 → 同时 | 通信单次：单独 → 同时 | 通信 busbw：单独 → 同时 |
|---|---|---|---|---|
| `gemm` | 1024 MiB | 1.244 → 2.703 ms（2.17×） | 0.658 → 0.976 ms（1.48×） | 1224 → 825 GB/s |
| `gemm` | 512 MiB | 1.241 → 2.078 ms（1.67×） | 0.378 → 0.693 ms（1.83×） | 1066 → 581 GB/s |
| `gemm` | 256 MiB | 1.237 → 1.439 ms（1.16×） | 0.243 → 0.497 ms（2.05×） | 830 → 405 GB/s |
| `gemm` | 128 MiB | 1.239 → 1.393 ms（1.12×） | 0.144 → 0.350 ms（2.44×） | 701 → 287 GB/s |
| `gemm` | 64 MiB | 1.241 → 1.481 ms（1.19×） | 0.0737 → 0.251 ms（3.41×） | 683 → 200 GB/s |
| `gemm` | 16 MiB | 1.238 → 1.452 ms（1.17×） | 0.0394 → 0.118 ms（3.00×） | 320 → 107 GB/s |
| `gmm_flydsl` | 1024 MiB | 0.963 → 1.940 ms（2.02×） | 0.658 → 0.971 ms（1.48×） | 1223 → 829 GB/s |
| `gmm_flydsl` | 512 MiB | 0.943 → 1.552 ms（1.65×） | 0.378 → 0.700 ms（1.85×） | 1066 → 575 GB/s |
| `gmm_flydsl` | 256 MiB | 0.943 → 1.097 ms（1.16×） | 0.243 → 0.366 ms（1.51×） | 830 → 550 GB/s |
| `gmm_flydsl` | 128 MiB | 0.943 → 1.065 ms（1.13×） | 0.143 → 0.216 ms（1.50×） | 702 → 467 GB/s |
| `gmm_flydsl` | 64 MiB | 0.945 → 1.111 ms（1.18×） | 0.0719 → 0.249 ms（3.46×） | 700 → 202 GB/s |
| `gmm_flydsl` | 16 MiB | 0.936 → 1.120 ms（1.20×） | 0.0381 → 0.108 ms（2.82×） | 330 → 117 GB/s |
| `attn_fwd` | 1024 MiB | 0.585 → 1.605 ms（2.75×） | 0.661 → 0.848 ms（1.28×） | 1218 → 950 GB/s |
| `attn_fwd` | 512 MiB | 0.584 → 1.122 ms（1.92×） | 0.380 → 0.586 ms（1.54×） | 1061 → 687 GB/s |
| `attn_fwd` | 256 MiB | 0.573 → 0.678 ms（1.18×） | 0.243 → 0.469 ms（1.93×） | 827 → 429 GB/s |
| `attn_fwd` | 128 MiB | 0.572 → 0.647 ms（1.13×） | 0.146 → 0.331 ms（2.27×） | 689 → 304 GB/s |
| `attn_fwd` | 64 MiB | 0.571 → 0.689 ms（1.21×） | 0.0745 → 0.232 ms（3.12×） | 676 → 217 GB/s |
| `attn_fwd` | 16 MiB | 0.570 → 0.652 ms（1.14×） | 0.0391 → 0.122 ms（3.12×） | 322 → 103 GB/s |

- 512 MiB 及以上：计算慢 1.65–2.75×（三个 kernel 都是），通信慢 1.28–1.85×。alltoall 768 MiB 时计算只慢 1.37–1.42×。
- 256 MiB 及以下：计算只慢 1.12–1.21×；通信慢 1.5–3.5×，越小越慢。
- 分界正好在 256 / 512 MiB 之间，和 kernel 无关 —— 是 RCCL 换了实现（第 4 节）。

## 4. 瓶颈分析

### 4.1 RCCL 按大小选三种实现

**表 3：allgather 单独（trace，rank 0 的 kernel；时间为 4 rank × 6 次调用的中位数，含 rocprofv3 开销）**

| 每 rank 输出 | RCCL 路径（`NCCL_DEBUG`） | kernel（每次调用） | WG × 线程 | LDS / WG | VGPR | device 时间 |
|---|---|---|---|---|---|---|
| 1024 MiB | Ring / Simple，channel 0..255 | `ncclDevKernel_Generic` | 256 × 256 | 83.5 KB | 256 | 647 µs |
| 512 MiB | 同上 | 同上 | 256 × 256 | 83.5 KB | 256 | 363 µs |
| 256 MiB | CE collective via DDA scratch（`hipMemcpy`） | `__amd_rocclr_batchMemOp` × 2 + `copyBufferBatch` + `copyBuffer`，另 3 次 copy | 1 × 1 / 256 × 512 / 256 × 512 | 0 | 8 | 290 µs |
| 128 MiB | 同上 | 同上 | 同上 | 0 | 8 | 186 µs |
| 64 MiB | DDA | `ddaAllGatherFabricLL128` | 255 × 512 | 0 | 32 | 59 µs |
| 16 MiB | DDA | 同上 | 255 × 512 | 0 | 32 | 25 µs |

- 阈值（`results/rccl_tuning_allgather.log`）：`AG DDA disqualified: totalBytes > entryThreshold=67108864`（输出 ≤ 64 MiB
  才走 DDA）；`AG CE-scratch disqualified: totalBytes > ddaScratchBytes=268435456`（≤ 256 MiB 才走 CE）；
  `AG CE-registered disqualified: ceAvailable=0 ... ceRegWindow=0`（没注册缓冲区）；剩下的走 Ring，`nc:256`，
  `RCCL Channel Tuning not applied for gfx1250 (pending sweep data, AICOMRCCL-1756)`。
- CE 路径名字叫 copy engine，但 trace 里每次调用有两个 256 WG 的 blit kernel（`copyBuffer*`），不是纯 SDMA；12:30 那篇
  alltoall 64 MiB 也是同样的 kernel。

### 4.2 Ring（≥ 512 MiB）：RCCL 和计算按 CU 切分 GPU

**表 4：计算先发、allgather 后发（trace，µs，从计算 kernel 开始算；4 rank × 6 次的中位数）**

| 计算 | 通信 | 计算 kernel：WG × 线程，LDS，VGPR | 计算单独 | 计算结束 | 通信 kernel 开始 | 通信结束 |
|---|---|---|---|---|---|---|
| `gemm` | 1024 MiB | 3072 × 128，260 KB，512 | 1081 | 1633 | 82 | 1050 |
| `gemm` | 64 MiB | 3072 × 128，260 KB，512 | 1081 | 1091 | 51 | 268 |
| `gmm_flydsl` | 1024 MiB | 2632 × 128，0 KB，416 | 812 | 1332 | 63 | 1017 |
| `gmm_flydsl` | 64 MiB | 2632 × 128，0 KB，416 | 812 | 783 | 43 | 260 |

（"计算单独"取 trace 里背靠背的预热调用；trace 每次之间同步、没到功耗墙，绝对值比 cco_bench 小。）

- Ring kernel 256 个 WG，每个 83.5 KB LDS。每 CU 320 KB LDS：dense GEMM 一个 WG 260 KB，attention 一个 WG 320 KB（12:30
  的 trace），260 + 83.5 > 320，所以**一个 CU 上有 RCCL 的 WG 就放不下 GEMM / attention 的 WG**。两边是在分 CU，不是
  在同一批 CU 上叠加：`gemm` + 1024 MiB 时 GEMM 被推迟约 0.55 ms（1081 → 1633 µs），Ring kernel 自己也从 647 µs 拖到
  968 µs —— 接近串行。M1 里计算慢 2.0–2.75×。
- FlyDSL 在 trace 里报 0 KB LDS（可能没算动态 LDS），也慢 2.02×，所以 LDS 不是唯一的限制。推测 VGPR / wave 槽位也不够：
  FlyDSL 4 wave × 416 VGPR，RCCL 8 wave × 256 VGPR，一个 SIMD 1024 VGPR，旁边有 RCCL 的 WG 时 FlyDSL 每 CU 只能放 1 个
  WG（单独时 2 个）。未验证。
- 不是功耗：同时跑时频率反而**升高**（M1：1735–1936 MHz，计算单独 1208–1450 MHz），功耗都在约 2470 W —— 同样功耗下每拍
  做的计算少了，即计算拿到的 CU 少了。
- 对比 alltoall 768 MiB：两个 kernel 先 192 WG 再 64 WG，计算慢 1.37–1.42×，E 0.56–0.62。allgather Ring 用满 256 个
  channel，占的更多。
- 方向：减 channel 数（`sweep.sh` 的 `nch64`，或 32 / 16），或让 CE 路径覆盖 512 MiB / 1 GiB（scratch 上限 256 MiB；或
  注册缓冲区走 CE-registered），CE 段计算只慢约 1.15×。

### 4.3 CE（128–256 MiB）：通信总是和计算同时结束

- blit kernel 每线程 8 VGPR、不用 LDS，能和计算的 WG 同驻，所以计算只慢 1.12–1.18×（M1），E 是三段里最高的（FlyDSL
  0.75–0.81）。
- 但 M2 里通信**总是在计算结束后 3–6 µs 才结束**（`gemm` 256 MiB：kernel 1.377 / 通信 1.381；FlyDSL 128 MiB：0.982 /
  0.986；attention 256 MiB：0.656 / 0.661），和 12:30 那篇 alltoall 64 MiB 的 `batchMemOp` 一样（1 个线程的 kernel 等到
  GEMM 结束才结束）。所以 E 只取决于计算在 overlap 里慢了多少：`gemm` 128 MiB 慢 0.099 ms（通信本身 0.152），E 0.28；
  FlyDSL 128 MiB 慢 0.019 ms，E 0.81。
- 为什么通信要等计算结束，未查明（trace 只抓了 1024 / 64 MiB 的配对）。

### 4.4 DDA（≤ 64 MiB）：通信被拖慢 3×，计算被功耗墙拖慢 4–10%

- `ddaAllGatherFabricLL128` 255 WG × 512 线程，32 VGPR、不用 LDS，能和计算同驻；trace 里 51 µs 就开始，但跑了 217 µs（单独
  59 µs，3.7×）。LL128 每个 WG 轮询 flag，旁边有计算时拿到的发射槽少。M1 里通信慢 2.8–3.5×。
- M2 里计算慢 4–10%，和降频一致：计算单独 1542–1626 MHz / 2363–2406 W，加上通信后顶到约 2490 W，降到 1318–1412 MHz
  （−8% 到 −17%，表 6）。`gemm` 慢 0.07–0.10 ms，比通信本身（0.06–0.09 ms）还长，所以 E 为负。trace 里没到功耗墙，计算
  基本不慢（1081 → 1091 µs），也说明主要是功耗。推测，未单独验证。

### 4.5 telemetry

**表 5：M1（背靠背）的频率 / 功耗中位数（每卡），以及 TFLOPS / busbw**

| 计算 | 通信 | 纯计算 | 纯通信 | 同时（M1） | TFLOPS：单独 → 同时 | busbw：单独 → 同时 |
|---|---|---|---|---|---|---|
| `gemm` | 1024 MiB | 1437 MHz / 2454 W | 1973 MHz / 1907 W | 1875 MHz / 2473 W | 1989 → 915 | 1224 → 825 GB/s |
| `gemm` | 512 MiB | 1423 MHz / 2452 W | 1968 MHz / 1888 W | 1735 MHz / 2478 W | 1994 → 1190 | 1066 → 581 GB/s |
| `gemm` | 256 MiB | 1420 MHz / 2447 W | 1964 MHz / 1870 W | 1482 MHz / 2391 W | 2000 → 1719 | 830 → 405 GB/s |
| `gemm` | 128 MiB | 1407 MHz / 2454 W | 1953 MHz / 1818 W | 1559 MHz / 2365 W | 1996 → 1776 | 701 → 287 GB/s |
| `gemm` | 64 MiB | 1394 MHz / 2474 W | 1946 MHz / 1839 W | 1641 MHz / 2366 W | 1994 → 1671 | 683 → 200 GB/s |
| `gemm` | 16 MiB | 1398 MHz / 2468 W | 1950 MHz / 1686 W | 1677 MHz / 2316 W | 1998 → 1703 | 320 → 107 GB/s |
| `gmm_flydsl` | 1024 MiB | 1361 MHz / 2502 W | 1986 MHz / 1920 W | 1874 MHz / 2474 W | 1717 → 852 | 1223 → 829 GB/s |
| `gmm_flydsl` | 512 MiB | 1450 MHz / 2448 W | 1967 MHz / 1884 W | 1743 MHz / 2477 W | 1752 → 1065 | 1066 → 575 GB/s |
| `gmm_flydsl` | 256 MiB | 1436 MHz / 2446 W | 1973 MHz / 1863 W | 1369 MHz / 2426 W | 1752 → 1506 | 830 → 550 GB/s |
| `gmm_flydsl` | 128 MiB | 1422 MHz / 2455 W | 1964 MHz / 1808 W | 1399 MHz / 2402 W | 1753 → 1551 | 702 → 467 GB/s |
| `gmm_flydsl` | 64 MiB | 1413 MHz / 2474 W | 1956 MHz / 1846 W | 1647 MHz / 2365 W | 1749 → 1487 | 700 → 202 GB/s |
| `gmm_flydsl` | 16 MiB | 1407 MHz / 2470 W | 1976 MHz / 1670 W | 1718 MHz / 2329 W | 1765 → 1475 | 330 → 117 GB/s |
| `attn_fwd` | 1024 MiB | 1219 MHz / 2502 W | 1934 MHz / 1934 W | 1936 MHz / 2463 W | 1881 → 685 | 1218 → 950 GB/s |
| `attn_fwd` | 512 MiB | 1208 MHz / 2503 W | 1935 MHz / 1894 W | 1772 MHz / 2478 W | 1881 → 980 | 1061 → 687 GB/s |
| `attn_fwd` | 256 MiB | 1308 MHz / 2456 W | 1933 MHz / 1862 W | 1358 MHz / 2420 W | 1920 → 1622 | 827 → 429 GB/s |
| `attn_fwd` | 128 MiB | 1286 MHz / 2470 W | 1931 MHz / 1808 W | 1430 MHz / 2374 W | 1924 → 1700 | 689 → 304 GB/s |
| `attn_fwd` | 64 MiB | 1290 MHz / 2474 W | 1944 MHz / 1808 W | 1511 MHz / 2384 W | 1925 → 1596 | 676 → 217 GB/s |
| `attn_fwd` | 16 MiB | 1286 MHz / 2475 W | 1945 MHz / 1668 W | 1546 MHz / 2341 W | 1931 → 1686 | 322 → 103 GB/s |

**表 6：M2（N·T_comp ≈ T_comm 的那组）的频率 / 功耗，以及 kernel 在 overlap 里慢了多少**

| 计算 | 通信 | 纯计算（M2） | 同时（M2） | kernel：单独 → overlap 中 |
|---|---|---|---|---|
| `gemm` | 1024 MiB | 1355 MHz / 2504 W | 1554 MHz / 2502 W | 1.256 → 1.759 ms（1.40×） |
| `gemm` | 512 MiB | 1383 MHz / 2504 W | 1468 MHz / 2499 W | 1.265 → 1.555 ms（1.23×） |
| `gemm` | 256 MiB | 1496 MHz / 2442 W | 1432 MHz / 2497 W | 1.257 → 1.377 ms（1.10×） |
| `gemm` | 128 MiB | 1489 MHz / 2414 W | 1440 MHz / 2495 W | 1.248 → 1.347 ms（1.08×） |
| `gemm` | 64 MiB | 1542 MHz / 2403 W | 1412 MHz / 2494 W | 1.259 → 1.360 ms（1.08×） |
| `gemm` | 16 MiB | 1586 MHz / 2366 W | 1403 MHz / 2491 W | 1.252 → 1.323 ms（1.06×） |
| `gmm_flydsl` | 1024 MiB | 1387 MHz / 2502 W | 1619 MHz / 2499 W | 0.966 → 1.424 ms（1.47×） |
| `gmm_flydsl` | 512 MiB | 1440 MHz / 2503 W | 1511 MHz / 2500 W | 0.956 → 1.211 ms（1.27×） |
| `gmm_flydsl` | 256 MiB | 1404 MHz / 2482 W | 1372 MHz / 2500 W | 0.963 → 1.015 ms（1.05×） |
| `gmm_flydsl` | 128 MiB | 1437 MHz / 2471 W | 1370 MHz / 2498 W | 0.963 → 0.982 ms（1.02×） |
| `gmm_flydsl` | 64 MiB | 1608 MHz / 2392 W | 1402 MHz / 2492 W | 0.942 → 0.996 ms（1.06×） |
| `gmm_flydsl` | 16 MiB | 1626 MHz / 2374 W | 1387 MHz / 2491 W | 0.954 → 0.988 ms（1.04×） |
| `attn_fwd` | 1024 MiB | 1246 MHz / 2503 W | 1630 MHz / 2499 W | 0.583 → 1.087 ms（1.86×） |
| `attn_fwd` | 512 MiB | 1261 MHz / 2501 W | 1495 MHz / 2500 W | 0.584 → 0.862 ms（1.48×） |
| `attn_fwd` | 256 MiB | 1396 MHz / 2459 W | 1340 MHz / 2497 W | 0.581 → 0.656 ms（1.13×） |
| `attn_fwd` | 128 MiB | 1464 MHz / 2424 W | 1367 MHz / 2491 W | 0.580 → 0.631 ms（1.09×） |
| `attn_fwd` | 64 MiB | 1543 MHz / 2406 W | 1329 MHz / 2491 W | 0.580 → 0.637 ms（1.10×） |
| `attn_fwd` | 16 MiB | 1588 MHz / 2363 W | 1318 MHz / 2488 W | 0.579 → 0.612 ms（1.06×） |

- 纯计算 2446–2503 W（M1）、1208–1450 MHz，和 alltoall 一样顶在功耗墙上；纯通信 1668–1934 W、1931–1986 MHz。
- Ring 段同时跑时频率升高（计算少拿 CU）；CE / DDA 段同时跑时功耗顶到约 2490 W、频率降低。

## 5. 结论

- `cco-allgather` 完成：RCCL 默认，3 kernel × 6 大小全部通过校验，0 卡死；README 只放 M2 表。
- E 由 RCCL 的实现决定，不由计算 kernel 决定：Ring（≥ 512 MiB）0.22–0.32，CE（128–256 MiB）0.28–0.81，DDA（≤ 64 MiB）
  −0.36–0.30。
- 最大的问题在 Ring 段：256 个 channel 的 WG 和计算 WG 放不进同一个 CU，两边按 CU 切分，计算慢 2–2.75×。比 alltoall
  （E 0.56–0.62）差一倍。
- 小包 dense GEMM 的负 E 来自功耗墙降频（计算慢的比通信本身还长）。

## 6. 下一步

| 优先级 | 方向 |
|---|---|
| P0 | reducescatter 用顺序 launch 重跑（`cco-reducescatter`） |
| P1 | allgather ≥ 512 MiB：`NCCL_MAX_NCHANNELS=64 / 32`（`VARIANTS=nch64`），或让 CE 路径覆盖更大的包（scratch 上限 / 注册缓冲区），目标是计算变慢从 2× 降到 CE 段的约 1.15× |
| P1 | CE 路径通信总要等计算结束：trace 128 / 256 MiB 的配对，看 `batchMemOp` / blit kernel 的调度 |
| P1 | 小包（DDA LL128、alltoall）被计算拖慢 3×：channel 数 / CU 预留 / SDMA |
| P1 | 功耗墙确认（单卡 300 ms 窗口 + telemetry） |
| P2 | 开始门卡死最小复现，报 RCCL / HIP |
