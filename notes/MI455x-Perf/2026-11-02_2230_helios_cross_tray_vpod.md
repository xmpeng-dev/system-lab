# Helios 两个 tray 做 8 卡：跨 tray scale-up 卡在 vpod（2026-11-02 22:30）

目标：`ctheliosr-rck-g02-e15-01`（tray A）+ `ctheliosr-rck-g02-e15-02`（tray B），各 4 × MI455X，做 EP8 测试。
数据：`data/2026-11-02_cross_tray/`（`afm/*_before.json` 是改动前的 `afmctl show device -j`，`scripts/` 是用到的脚本）。

## 结论

- 两个 tray **物理上**在同一个 72 卡 UALoE ppod 里（`a674683c-423e-4398-91fe-4f132ece26fa`），A 的 accel id 0–3，B 是 4–7；
  每卡 36 个 scale-up 端口全部 LINK_UP 400G，两边接同一组 6 台交换机（IFCP peer device 1–6）。
- **逻辑上**每个 tray 是独立的 4 卡 vpod：A 是 vpod 7（0–3），B 是 vpod 3（4–7）。IFoE 网卡侧的 vPOD accelerators /
  whitelist / accelerator MAC 映射也只有本 tray 4 张卡。
- 驱动侧（PSP）可以从主机改 vpod，**网卡侧改不了**：`afmctl update device -l/-v/-w`、端口 accelerator MAC 映射都只能在
  PROVIDER 阶段改，而 ACTIVE → PROVIDER 被拒（`Invalid config phase transition from ACTIVE to PROVIDER`）。
  开机时 amdgpu 加载之前 ppod / vpod 已经配好（B 的 dmesg：ifoe 模块 256 s 加载，amdgpu 939 s 加载时已是 vpod 3），主机上没有
  做这件事的服务 / unit → 应该是机柜级 fabric manager 带外配的。固件 build 名 `fw.mi450-a0.silicon.benchtop`。
- **要走跨 tray scale-up，需要机柜 / fabric 负责人把 accel 0–7 配进同一个 vpod（网卡侧 vpod / whitelist / MAC 映射）。**
- 两边都已恢复原配置；A 上用 `cco_bench` 复测 192 MiB alltoall 0.1927 ms / busbw 783 GB/s（改前 0.1936 / 780），没有退化。

## 过程

1. B 的 amdgpu 开机被 blacklist（`modprobe.blacklist=amdgpu`），13:44 UTC 用不带参数的 `sudo modprobe amdgpu` 加载，和 A 这次开机一致。
2. 两边差异：amdgpu-dkms A `7.1.0.31300009-2419605`、B `-2415424`；MEC 2480 / 2460，RLC 35 / 34，TA_RAS 1B.44.00.09 / .07；
   ifoe 0.3.1 / 0.3.0，afm lib 0.6.0 / 0.5.0。VBIOS 00207988、SMU 125.15.0、2500 W 一致。
3. 前端网卡 `ionic_0` 400G（RoCE v2，MTU 1500）两边可用；8 张后端 ionic 全 DOWN（carrier 0）。走前端网卡 EP8 每次 alltoall 每机要过 384 MiB，
   理论 ~8 ms，不代表 Helios。
4. 容器里的 RCCL（2.32.3）有 "UALoE-enabled (aka MNNVL)" 支持：读 `/sys/class/drm/cardN/device/ualink/*`，tray 内已经走
   `P2P/CUMEMMNNVL`。vpod 覆盖 8 卡时应能直接跨 tray 走 scale-up。
5. 驱动侧改 vpod（`amdgpu_ualink.c`：`config/{vpod_id,vpod_size,vpod_active_accels,addr_mode}` + `commit`，脚本 `scripts/vpod.sh`）：
   - B：vpod_id 0（退出 vpod 3）→ A：vpod 7 扩到 size 8 / 0–7（同 id 扩容走 "membership changed, reconfiguring"，前 3 张卡
     integrity 失败是预期，第 4 张提交后 4 张一起 bounce）→ B：加入 vpod 7。8 张卡 sysfs 都是 active / vpod 7 / 0–7。
   - 但 `afmctl show device` 仍是 vPOD 0–3 / whitelist 0–3，地址表只有本 tray；`afmctl test ping --dst-accelerator 4` → `No such device`。
   - 驱动侧和网卡侧不一致时没敢跑 RCCL（会尝试跨 tray MNNVL），按原样改回：A vpod 7 / 0–3，B vpod 0 → vpod 3 / 4–7。

## 副作用 / 观察

- B 的 accel 4（0001:01:00.1）afmctl ping 引擎一直 `Device or resource busy`（所有端口）：第一次 ping 5,6,7 时被 `timeout` 打断，
  固件里留了会话。数据通路正常（链路 UP，无 failover / paused stream，RCCL 4 卡 alltoall 正确，accel 5 → 4 ping 通）。重载驱动或重启可清。
- A station 0 的 IFOE CRC 错误累计 855 亿，跑 ~1 s alltoall 涨 1100 万；SEQ num errors / Rx decap dropped 23057。B 只有 5.4 万
  （刚开机）。两边 tray 内 alltoall 速度一样，暂不认为影响带宽，但值得报给机柜负责人。
- 自写的 `a2a_local.py`（默认 stream 上同步 `all_to_all_single`）测 192 MiB 是 0.31 ms，比 `cco_bench`（0.193 ms）慢，是测法差异，
  不是机器问题；跨机探测要用 `cco_bench` 的测法。

## 资料调研（只读，未运行）

- 官方定位：vPod 由 AMD Fabric Manager（AFM）创建，AFM 跑在交换机 tray 管理处理器上（3 实例），对外是 northbound REST API，
  由 Cluster Controller / Slurm / K8s 调用；最小单位是一个 compute tray（4 卡），任意 tray 组合可以拼成一个 vPod。
  公开文档里没有 AFM REST / vPod 创建接口（[CDNA5/Helios blog](https://rocm.blogs.amd.com/ecosystems-and-partners/cdna5-helios/README.html)、
  [StorageReview](https://www.storagereview.com/news/amd-mi455x-and-helios-432gb-hbm4-72-gpu-racks-and-a-real-answer-to-vera-rubin)）。
- IFoE 协议（`ifoe-drv-0.3.1.1000/linux/mcdi_pcol.h`）：阶段 System → Provider → Tenant → Showtime 只能前进；Provider 配置
  "is provided from the AI Fabric Manager via the host kernel driver"。回退只能 `MC_CMD_IFOE_RESET_PHASE` / FLR / 实体复位
  （netlink `CFG_CMD_RESET`，即 `afmctl reset device`，不加 `--datapath`）。Provider：local accels、vPOD accels、ppod/vpod、
  端口 accelerator MAC 映射；Tenant：`SET_ENABLED_ACCELERATORS`（白名单，须是 active accels 子集）；Showtime：`SET_PATH_PORT_MAP`。
- amdgpu 侧是被动的：网卡侧配置变化后 PSP 发 `CFG_UPDATE` 中断 → `amdgpu_ualink_config_update_handler` 重新查询 vpod。
  `amd-smi set --fabric-vpod/--fabric-ppod`（`amdsmi_set_gpu_fabric_vpod_config`）在裸机上就是写同一组 sysfs + commit，等价于
  `scripts/vpod.sh`，不会改网卡侧。跨机导入显存走 fabric 上的 NPA-REQ/RSP 消息（`amdgpu_ualink.c`），所以网卡侧必须认识对端。
- 已有人在同一 vPOD 的两个 tray 上跑通：[ROCm/mori#466](https://github.com/ROCm/mori/pull/466)（MI455X slot2+slot14，各 4 卡，
  vPOD 4）cross-node EP8 dispatch/combine，bf16 dispatch ~200 GB/s、combine ~173–192 GB/s @8192 tok（tray 内 xGMI ~335）。
  vPOD 识别用 sysfs `ppod_id` + `vpod_id`（和 RCCL 一致），`MORI_VPOD_ID` 只是声明分组，网卡侧不通照样失败。
- 各库：
  - RCCL：`ppod_id` = clusterUuid、`vpod_id` = cliqueId，同 clique 内 `P2P/CUMEMMNNVL`；`NCCL_CUMEM_ENABLE` 在 gfx1250 默认开；
    [rocm-systems#6152](https://github.com/ROCm/rocm-systems/pull/6152) 给 gfx1250 MNNVL 放开 64 channel 上限；
    `RCCL_FABRIC_TELEMETRY_ENABLE=1`（需 `RCCL_USE_AMD_SMI_LIB=1`）可看 UALoE 计数器。RCCL 文档里 MNNVL 段仍是 NVIDIA IMEX 原文。
    2026-10-26 `19a27dca6e`（#13802）才让对称 kernel 在跨节点 MNNVL comm 上可用（之前 `isAllDirectNVlink` 恒 false）；
    容器镜像 tas-20261019 大概率不含，跨 tray 跑 RCCL 时要确认版本。
  - HIP/HSA：`hipMemHandleTypeFabric`(0x8)、`hipDeviceAttributeHandleTypeFabricSupported` →
    `hsa_amd_vmem_export/import_fabric_handle` → `DRM_IOCTL_AMDGPU_UALINK_HANDLE`。
  - PyTorch SymmMem：[pytorch#194989](https://github.com/pytorch/pytorch/pull/194989) 在 ROCm 上加 fabric handle rendezvous，
    要求 accel 已 provision 到 ready（导出时检查）。
  - mori-io：`--backend fabric`，要求两机 `amd-smi fabric` 的 `PPOD_ID`/`VPOD_ID` 相同且 `READY`，否则建 session 直接报错。
  - [ROCm/cvs#192](https://github.com/ROCm/cvs/pull/192)：IFoE TransferBench 多机预检要求所有节点同一 `vpod_id`。
  - aiter：没有跨 tray 内容，custom allreduce / quick allreduce 用 hipIpc（只限同机）。FlyDSL：只是 kernel DSL，跨 tray 逻辑在 mori。
    hipBLASLt：纯 GEMM，无关。Primus-Turbo DeepEP：只有上游带来的 `allow_mnnvl`（NVSHMEM 开关），无 UALoE 适配。

## 下一步

- 找机柜 / fabric 负责人：把 e15-01（accel 0–3）和 e15-02（accel 4–7）配进同一个 vpod（size 8），包括网卡侧。
- 配好后：RCCL 日志应出现 `MNNVL ... cliqueSize 8`、跨机 `P2P/CUMEMMNNVL`；先测 8 卡 alltoall，再做 EP8 的 cco-alltoall（每卡 16 个 expert）。
