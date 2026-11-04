# 跨 tray alltoall 2×2（前端网卡 ionic_0）：RoCE ~27.5 GB/s/节点，TCP 3–5 GB/s（2026-11-03 09:40）

机器：e15-01（node 0，master 10.190.224.68）+ e15-02（node 1），每台 GPU 2、3（e15-02 GPU 1 当时被别人的 vLLM 占用）。
两个 tray 仍是各自的 4 卡 vpod（7 / 3），UALoE scale-up 不能跨 tray，只能走前端 `ionic_0`（400G，RoCE v2，MTU 1500 → RoCE MTU 1024，GID 1）。
容器：e15-01 `xiaompen_ep_overlap`，e15-02 `xiaompen_a2a`（同镜像 tas-20261019，RCCL 2.32.3）。
数据：`data/2026-11-03_cross_tray_a2a/`（`out/*.json` 是结果，`scripts/` 是脚本和 shim）。

## 结论

- RoCE（调好后）：每节点跨机 **~27.5 GB/s（≈220 Gbit/s，线速一半）**，12–384 MiB/卡基本不变；192 MiB/卡 4 卡 alltoall 7.34 ms。
- TCP：每节点 3–5 GB/s，192 MiB/卡 47 ms，比 RoCE 慢 ~6×。
- 推到 2×4（EP8，192 MiB/卡）：每节点要出 384 MiB，按 27.5 GB/s 约 14.6 ms，机内 4 卡是 0.193 ms（~75×）。这条路不代表 Helios。
- RCCL 默认配置在这张前端网卡上跑不起来，要 5 处绕过（见下）；默认 16 channel/peer 时 48 MiB 起丢包重传超时挂死。

## 结果（steady = side stream 连发 20 次的均值，max over ranks；全部校验正确）

| MiB/卡 | RoCE ms | RoCE GB/s/节点 | TCP ms | TCP GB/s/节点 |
|---|---|---|---|---|
| 12 | 0.449 | 28.1 | 2.51 | 5.0 |
| 48 | 1.83 | 27.6 | 16.5 | 3.1 |
| 96 | 3.68 | 27.3 | 26.4 | 3.8 |
| 192 | 7.34 | 27.4 | 46.7 | 4.3 |
| 384 | 14.6 | 27.6 | 84.8 | 4.7 |

GB/s/节点 = 每节点发往另一节点的字节 / 时间。TCP 的单次和连发差别大（48 MiB 8.2 vs 16.5 ms），抖动大。
RoCE 那组机内 2 卡对照在 ≥192 MiB 变成 1.42 ms（`RCCL_SATURATE_P2P_NCHANNELS=0` 让机内 channel 变少），不能用；TCP 组机内是正常的 0.17 ms。

## RoCE 要绕过的问题（逐个排出来的）

1. `ibv_create_qp` EINVAL：RCCL 发送端 / flush QP 用 `max_recv_wr = 0`（`net_ib/connect.cc` 写死），这张 ionic（fw 1.130.6）不接受；
   `ibv_rc_pingpong -r 0` 复现 "Couldn't create QP"，`-r 1` 正常。绕法：`NCCL_IBVERBS_LIB` 指向 shim（`scripts/ibv_rq1.c`，只重写
   `ibv_create_qp@IBVERBS_1.1` 把 0 改 1）。IB-CAST（AINIC 默认路径）每连接 2 QP，qp_num 到 1638 又撞 `max_qp` 2048，改 `NCCL_NET=IB`。
2. GDR 0（网卡挂在 CPU root 下，离 GPU 远）→ 主机 bounce buffer 的 `ibv_reg_mr` EINVAL（ROCm pinned host 内存）。`NCCL_NET_GDR_LEVEL=SYS`
   强制 GDR（dmabuf），另加 `NCCL_IB_PCI_RELAXED_ORDERING=0`（没单独验证是否必须）。
3. RMA proxy 的 `ibv_reg_mr` EINVAL：`NCCL_RMA_DISABLE=1`。
4. 容器 `ulimit -n` 1024 → "Too many open files"：启动前 `ulimit -n $(ulimit -Hn)`。
5. 丢包：默认 channel 时 48 MiB 起 `IBV_WC_RETRY_EXC_ERR`；加大 `NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=7` 只是挂得更久；
   必须 `RCCL_SATURATE_P2P_NCHANNELS=0`（p2pnChannelsPerPeer 16 → 4）才稳定。

最终 RoCE 环境：`NCCL_IBVERBS_LIB=.../libibv_rq1.so NCCL_NET=IB NCCL_IB_HCA=ionic_0 NCCL_IB_GID_INDEX=1 NCCL_IB_PCI_RELAXED_ORDERING=0
NCCL_NET_GDR_LEVEL=SYS NCCL_RMA_DISABLE=1 NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=7 RCCL_SATURATE_P2P_NCHANNELS=0 NCCL_SOCKET_IFNAME=enp4s0np0`。

## 裸网卡

perftest 装不上（apt 源未认证），只有 `ibv_rc_pingpong`（单 QP、一来一回）：1 MiB 61.6 Gbit/s（双向合计），8 MiB 54 Gbit/s，
即单 QP 单向 ~7 GB/s，只作下限参考。

## 为什么和 06-rccl 差这么多：用的是唯一一张前端网卡

06-rccl（tray 内 4 卡）alltoall 128 / 256 MB busbw 1026 / 1247 GB/s，走 UALoE，每卡单向 1.8 TB/s，一个 tray 合计 7.2 TB/s。
这次跨 tray 只有 `ionic_0` 一张 400G（50 GB/s，整个 tray 共用，挂在 CPU 的 PCI 段 0000 上，GDR 距离远），
链路带宽差 ~144×；就算跑满线速，2×2 192 MiB/卡也要 ~4.0 ms（实测 7.34 ms，busbw 20.6 GB/s）。

两台机器上还有 8 张 GPU 侧的 Pensando 网卡 `ionic_1..8`（fw 1.130.10-a-5），每张 GPU 两张，和 GPU 在同一个 PCI 段
（GPU 0 = 0001:01:00.0 对应 0001:44 / 0001:48，以此类推），这才是 scale-out 后端网卡；但两台都是 DOWN / NO-CARRIER，
ethtool 读不到 link mode（没线或没模块），现在用不了。

对照 CDNA5 白皮书（Helios 一节）：`ionic_0` 是前端 "Salina" 400G（lspci `DSC3 Salina`），`ionic_1..8` 应是后端 "Vulcano" 800G，
走每 GPU 两条 PCIe Gen6 x16 的配置（另一种是 UALink128 x8 接 3 张）。即每 GPU 2 × 800G = 200 GB/s 单向、每 tray 800 GB/s 单向，
是现在 `ionic_0` 的 16×，仍是 scale-up（每 GPU 1.8 TB/s 单向）的 1/9。
[CDNA5 白皮书](https://www.amd.com/content/dam/amd/en/documents/products/technologies/cdna/amd-cdna5-whitepaper.pdf)、
[MI455X brochure](https://www.amd.com/content/dam/amd/en/documents/products/accelerators/instinct/amd-instinct-mi455x_brochure.pdf)
（vPod 4 卡到整柜、不要求物理相邻）。

## 下一步

- 2×4（e15-02 GPU 1 空出来后）用同一组 RoCE 环境测 8 卡；预计 ~14.6 ms @192 MiB/卡。
- 前端网卡的丢包 / CRC、`max_recv_wr = 0` 不兼容报给网络 / RCCL 负责人；真正的 Helios 跨 tray 仍要 AFM 把两 tray 配进同一 vpod。
- 问机柜负责人 `ionic_1..8` 后端网卡有没有接线 / 交换机；接上后才能测真正的 scale-out。
