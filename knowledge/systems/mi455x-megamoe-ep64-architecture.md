# MI455X / Helios MegaMoE EP64 架构设计

> **状态：** 设计提案，2026-10-15；MI455X / UALoE 上尚未实测。硬件数字来自 [`../hardware/gpu-comparison.md`](../hardware/gpu-comparison.md) 与三篇 AMD ROCm 博客（[CDNA 5 + Helios](https://rocm.blogs.amd.com/ecosystems-and-partners/cdna5-helios/README.html) 2026-08-04、[MI455X vs MI355X](https://rocm.blogs.amd.com/software-tools-optimization/mi455x-vs-mi355x/README.html) 2026-10-08、[MI450 LDS](https://rocm.blogs.amd.com/software-tools-optimization/mi450-lds-optimization/README.html) 2026-08-28），并行策略来自 [`mi455x-parallelism-strategy-reference.md`](./mi455x-parallelism-strategy-reference.md)，执行模型主要参考 [Mixture-of-Kittens](../../papers/mixture-of-kittens.md)、MegaFlux `2610.00671`、Expert Coupling `2610.09372` 与现有 Primus MegaMoE。

## 0. 结论

目标不是把现有 EP8 kernel 的 rank 常量改成 64，而是构造一个 **EP64-native、EP8/16/32/64 共用协议**的训练执行层：

1. **EP 只在 Helios 72 卡单 scale-up 域内扩展，最大 64。** 跨机架只走 DP / PP / FSDP，不让 expert token 跨 rack。
2. **逻辑 ownership 与物理 execution 解耦。** `EP_owner ∈ {8,16,32,64}`；EP64 首版用 64 owner，后续可把剩余 8 卡作为 helper pool，动态复制热专家，但不改变 Megatron 可见的 EP=64。
3. **数据面采用固定容量 ring + device schedule + persistent worker queue。** 不允许按路由结果 D2H、host malloc 或每 chunk launch。
4. **通信方向不是常量。** 首轮验证 `pull dispatch + push combine`，但 RCCL A2A 与反向方向组合必须保留为基线；UALoE 实测决定默认值。
5. **先 BF16 打通完整 forward/backward，再开 MXFP8，最后评估 MXFP4。** helper replica 的 Wgrad 必须按固定顺序回收到 owner，不能使用非确定浮点原子归约。
6. **Helios 机架内是单层、多平面的交换式单跳 fabric（AMD 官方：12 台 UALoE 交换机分布在 6 个交换机 tray），既不是 MI355X 的点对点 full mesh，也不是多级树。** 因此 EP8→64 的带宽成本几乎不变，新增成本来自 per-peer 开销、incast（多发一收）与多平面分流；协议围绕这三点设计，见 §2.4。

EP64 的价值是让 256 专家模型每卡保有 4 个专家、expert TP=1，并把全部 expert 通信留在单跳域内。它不是默认一定最快：EP8/16/32/64 必须由运行形状和实测代价模型选择。

---

## 1. 目标、边界与硬约束

### 1.1 目标 workload

| 维度 | 首要目标 | 必须覆盖 |
|---|---|---|
| 模型 | DeepSeek-V3/V4 类 routed + shared expert MoE | `E=256/384/512`，`top-k=2/6/8/10` |
| 精度 | BF16 → MXFP8 | MXFP4 作为后续独立里程碑 |
| EP | 8、16、32、64 | 同一 API、同一 metadata contract |
| 序列 | 4K 起步 | 长上下文允许 CP，但 EP 仍不跨 rack |
| 训练 | 完整 forward + backward | DGrad、WGrad、router grad、shared expert |
| 路由 | balanced、真实 skew、极端热点 | 不 drop token；结果确定 |

### 1.2 非目标

- 首版不做跨机架 EP。
- 不以 HIP-ify DeepEP V1 为实现路径。
- 不假设 MI455X 有与 Blackwell TMA / CLC 完全等价的指令。CDNA5 有 TDM（Tensor Data Movement），但它在 HBM→LDS 拷贝中不能改变连续维，语义与 TMA 不同（§8.1）。
- 不把 MI455X 推理或单 kernel 峰值当训练性能结论。
- 不在第一版引入 helper pool、MXFP4 与低精度 wgrad collective；它们分别设门禁。

### 1.3 四条硬约束

1. **正确性先于融合：** 任一 EP 度数都要能切回 RCCL A2A + unfused grouped GEMM。
2. **热路径复杂度不得是 `O(EP²)`：** metadata 随 routed rows 增长，不能维护 64×64 的逐 tile 状态矩阵。
3. **没有 host 参与的动态路径：** route、capacity、ring slot、helper mapping 全在设备侧产生和消费。
4. **所有远端 publish 都有明确 release/acquire 与 payload lifetime：** 不能只靠 atomic 值“看起来更新了”。

---

## 2. Rack 与 process-group 设计

### 2.1 EP64 的两种物理模式

Helios 是 18 个 4-GPU tray，共 72 GPU。64 正好对应 16 个完整 tray。

| 模式 | 64 owner GPU | 剩余 8 GPU | 用途 |
|---|---:|---:|---|
| **V1 owner-only** | 参与完整模型与 EP64 | 不属于该 replica | 建立最小可信 EP64 基线；避免 helper 语义污染 |
| **V2 owner+helper** | 保持参数 owner 与 optimizer shard | 2 个 tray 组成 helper pool | 热专家 replica、shared expert、尾部工作；partial Wgrad 回收到 owner |

V2 中 Megatron 仍看到 `expert_model_parallel_size=64`。8 个 helper 不拥有长期 optimizer state，因此不能把物理 72 卡误报成 EP72。

**vPod 约束：** Helios 可以按 node 切成相互隔离的 Virtual Pod（vPod），GPU 故障只重启所在 vPod 的作业。EP 组必须完整落在一个 vPod 内：EP64 / V2 需要占满整机架（一个 72 卡 vPod），任一 GPU 故障会重启整个作业；另一种布局是两个 36 卡 vPod（按 9 个 tray 推断），各跑 EP32 + 4 张备用 / helper 卡，故障影响面减半。是否采用取决于集群的 vPod 切分策略与 checkpoint 间隔，需与运维确认。

### 2.2 Process groups

```text
ep_owner_group      8 / 16 / 32 / 64 ranks，专家参数长期 ownership
ep_exec_group       = owner_group（V1）或 owner + 8 helpers（V2）
helper_group        仅 V2，最多 8 ranks
dp_group            跨 rack，相同 owner index 组成；不与 helper 混用
pp_group            仅模型仍需跨 rack 时启用
cp/tp_group          按序列与 attention 形状决定；不得把 EP 推出 rack
```

`ep_owner_rank` 使用 6 bit 即可覆盖 0–63。所有 hot-path owner 集合统一用 `uint64_t` bitmap；helper 用独立 `uint8_t` bitmap。禁止保留 EP8 时代的 3-bit rank、8 元素数组或 `uint8_t ready_mask`。

### 2.3 EP 选择

对于 256 routed experts：

| EP | 每 owner 专家数 | 主要适用区 |
|---:|---:|---|
| 8 | 32 | 大 local batch；最大 grouped-GEMM 聚合度；完整使用 72 卡时最自然 |
| 16 | 16 | 通信与 GEMM 的中间点 |
| 32 | 8 | 多数 scale-up 训练的候选主轨 |
| 64 | 4 | 大模型、小 per-rank token、需要把 expert TP 固定为 1；必须处理小 M 与热点 |

运行时不在单 step 内改变 EP；EP 是 job-level 配置。单 step 内只允许改变 chunk、CU/WG 配额与 helper replica mapping。

### 2.4 通信拓扑：交换式单跳，不是点对点 full mesh，也不是树

MI355X 节点和 Helios 机架在逻辑上都是“任意两卡一跳可达”，但物理形态不同，这决定了数据面该怎么写。

| 维度 | MI355X 8 卡节点 | MI455X / Helios 72 卡机架 |
|---|---|---|
| 物理形态 | XGMI 点对点 full mesh：每卡 7 条链路直连 7 个 peer | 单层多平面交换：每卡 36×400 GbE 接入 12 台 UALoE 交换机（6 个交换机 tray，每 tray 2 台） |
| 任意两卡跳数 | 1（无交换机） | 1（经过一台交换机），任意两卡之间有多条路径 |
| 每卡总带宽 | ≈1.075 TB/s 双向（7×≈153.6 GB/s） | 3.6 TB/s 双向（每方向 ≈1.8 TB/s） |
| 单 pair 上限 | 一条链路，约总量的 1/7 | 可把整卡带宽打向一个 peer（AMD 实测 single-pair 27×） |
| all-to-all 聚合 | 必须同时向 7 个 peer 发送才能吃满 | 发送对象数量不影响上限，上限是本卡端口（AMD 实测 aggregate 3.8×） |
| 节点 / tray 局部性 | 节点内 8 卡等距 | 无：AMD 称两卡无论是否在同一 node，都经交换机、带宽相同 |
| 竞争点 | 每条链路独占，没有交换机排队 | 目标卡接收端口与交换机出口队列（incast） |
| 故障降级 | 断一条链路即失去该 pair 的直连 | 断 1 条链路 ≈ −3%、断 1 台交换机 ≈ −8%、断 1 个交换机 tray ≈ −17%，自动重路由 |
| 远端语义 | XGMI P2P load/store，fence 行为有 MI300X/MI355X 实测经验 | UALink over Ethernet；CU 发起与 DMA 发起的流量都走该 fabric；原子、顺序与 coherence 语义未公开 |

**官方确认不是树：** AMD 08-04 博客原文是 “single-hop, multi-plane UALoE scale-up fabric: all 72 GPUs reach each other through 12 UALoE switches across 6 switch trays”。规格本身也自洽：12 × 21.6 TB/s ≈ 259 TB/s，正好等于 72 卡 × 3.6 TB/s，交换机容量全部面向 GPU，没有上联层。故障降级数字进一步说明连接是均匀的：−3% ≈ 1/36（每卡 36 条链路），−8% ≈ 1/12，−17% ≈ 2/12，即每卡的 36 条链路均匀分到全部 12 台交换机（每台 3 条）。交换机或交换机 tray 故障会让**每张卡**等比例掉带宽，而不是只影响部分 pair，所以 planner 不需要拓扑感知的重路由。跨机架走 Pensando Vulcano 800 scale-out 网络（≈597 GB/s/卡，域内 : 跨机架 ≈ 6 : 1），只承载 DP / PP。RCCL 的 ring / tree 是集合通信的逻辑算法，与物理拓扑不是同一个概念。

**带宽倍数用聚合数建模：** AMD 10-08 博客用 TransferBench（16 GB 传输）对比 4×MI455X bring-up 节点与 8×MI355X 节点：每卡聚合带宽单向 3.8×、双向 3.7×；单 pair 单向 27×、双向 26×。博客明确建议 all-to-all 用 3.8×，27× 只说明交换式拓扑下单 pair 能用满整卡带宽。注意这组数是在 4 卡节点上测的，63 个 peer 并发和 incast 行为没有公开数据，E0 必须在 64/72 卡上重测。

对 MegaMoE 的直接后果：

1. **EP 扩展成本来自 per-peer 开销与路由偏斜，不是带宽。** 每卡跨卡字节占 routed bytes 的 `(EP−1)/EP`：EP8 为 87.5%，EP64 为 98.4%，几乎不变；变化的是 peer 数 7→63、每 peer 消息缩小 8×、完成信号与 metadata 宽度。每 peer 每 chunk 字节数为 `rows_chunk × top_k × H × bytes / EP`；例如 `H=7168`、BF16、`top_k=8`、1024 token 的 chunk，在 EP64 下每 peer 约 1.8 MB。理想下界参考：`T/rank=4096` 时每卡 dispatch 约 470 MB，按每方向 1.8 TB/s 计约 0.26 ms。
2. **incast 是 Helios 上新出现的主要风险。** MI355X 上 7 个 source 写同一 owner 走 7 条独立链路；Helios 上 63 个 source 共享同一 owner 的接收带宽（36 个端口合计每方向 ≈1.8 TB/s）和对应的交换机出口队列。热专家 owner 同时是 dispatch 的接收瓶颈和 combine 的发送瓶颈。Ethernet 承载下拥塞反压是否会波及其他流，目前未公开。因此 dispatch 优先由 owner 拉取（owner 自己限制在途读请求，相当于接收端流控），push 路径必须带 credit / 在途字节上限；helper pool 的首要价值是削热点 owner 的端口压力，其次才是算力。
3. **机架内不做两级（树形）dispatch。** DeepEP 式“节点内聚合 + 跨节点 RDMA”只在有快慢两级链路时有意义。Helios 机架内所有 pair 等距（AMD 明确说同 node 两卡也经交换机、带宽相同），先把 token 聚到 tray leader 再转发只会多一次 HBM 读写和一跳。expert placement 也不需要 tray 亲和；helper 放在哪 8 张卡对通信距离没有影响。
4. **多平面分流决定单个 worker 能否吃满带宽。** 一个 peer 的流量如何分到 36 个端口 / 12 个平面（逐包喷洒还是按流哈希）未公开。AMD 博客提到断链后 DMA 自动重路由，而 CU 流量由 fabric manager 在约 1 s 内重新均衡，由此推测 CU 发起流量的平面分配由 fabric manager 配置（推断），具体粒度仍需实测。若按流哈希，少量大流会撞同一端口，需要每 peer 多个并发流或足够多的在途请求。MI355X 上“每 peer 一个 WG 吃满一条链路”的经验在这里不成立：瓶颈从单条链路变成本卡端口加平面均衡。
5. **单次远端操作更贵，信号必须更粗。** 经交换机和 Ethernet 封装的往返延迟预计高于直连 XGMI（具体数值待测）。MoK 在 NVL72 EP64 上测到 push 完成信号可占 dispatch 的 18%。publish 粒度保持在 non-empty peer segment / macrobatch；远端写本地 flag、本地自旋，禁止跨卡轮询和逐 token / 逐 tile 远端原子。
6. **不规划交换机内归约。** NVL72 有 NVLS 交换机内 reduce；slab 中没有 UALoE 交换机侧归约的记录。top-k combine 与 helper WGrad 归约都在 GPU 上完成。
7. **通信相对 HBM 更便宜，overlap 的收益上限更低。** HBM : fabric 带宽比 MI355X ≈7.4，Helios ≈6.5（双向口径；HBM 按 AMD 公布的 2.91× MI355X ≈ 23.3 TB/s 计，方法见 [`mi455x-parallelism-strategy-reference.md`](./mi455x-parallelism-strategy-reference.md) §5.6）。AMD 实测也一致：HBM 读带宽 3×，scale-up 聚合 3.8×，fabric 涨得比 HBM 多。结合 GB200 上 EP8→64 dispatch 只增 1.73×、专用 dispatcher 只领先朴素 A2A 1.38× 的经验，首要瓶颈更可能是 host gap、小 M GEMM 效率与偏斜，这也是 RCCL 基线必须保留的原因。
8. **fabric 降级是常态事件，协议超时要区分“慢”和“坏”。** 断链、断交换机会自动重路由，带宽掉 3–17%，CU 流量最长约 1 s 才重新均衡。ring 协议的 fail-fast 超时必须明显长于这个窗口，否则一次链路抖动会被误判成 peer 丢失；性能门禁也要在降级 fabric 下看 rank-max 是否仍可接受。

从现有 MI355X MegaMoE 迁移时必须删除的假设：每 peer 绑定一个 resident WG、靠“同时打满 7 条链路”达到带宽、8-bit ready mask、tray / node 内两级聚合、wave64 + MFMA 的 tile 与 lane 映射、160 KB LDS 预算，以及直接沿用 gfx950 的 XGMI fence 语义。

---

## 3. 软件分层

```text
Primus / Megatron backend
  └─ MegaMoELayerAdapter
       ├─ ProcessGroupContract
       ├─ PrecisionContract
       ├─ TransportPolicy (RCCL | UALoE fused)
       └─ GraphSafeWorkspace
            ↓
Device Route Planner
  ├─ deterministic top-k normalization
  ├─ owner / helper assignment
  ├─ source segment construction
  └─ chunk + ring-slot schedule
            ↓
Persistent MegaMoE Executor
  ├─ COMM workers
  ├─ FC1 / activation / FC2 workers
  ├─ COMBINE / tail workers
  └─ backward DGrad / WGrad / replica-reduce workers
            ↓
UALoE peer memory or RCCL fallback
```

### 3.1 Adapter 层

Adapter 对 Megatron 暴露普通 MoE layer 语义，不让 schedule、ring 或 helper 概念泄漏到 optimizer：

- 输入输出保持 `[tokens, hidden]`；
- router 选择与权重不因 helper 改变；
- owner rank 是 checkpoint / optimizer 的唯一权威；
- activation checkpoint 重放必须复用保存的 route/schedule，或保证 planner bitwise deterministic；
- 图捕获时 workspace 地址、ring 数量与 metadata 上界固定。

### 3.2 TransportPolicy

同一 executor 支持两条数据面：

1. **`rccl_a2a`：** correctness 与性能底线；独立 kernels 可以被 graph 捕获。
2. **`ualoe_fused`：** persistent kernel 内 peer load/store + device flags。

融合路径没有权利删除基线。每个生产 shape 都必须记录两者的 rank-max latency。

---

## 4. Metadata 与内存布局

### 4.1 Route plan

每个 routed row 只保留一次紧凑记录：

```text
RouteEntry {
  token_id
  topk_lane
  global_expert
  owner_rank
  owner_local_expert
  source_rank
  source_ordinal
  helper_rank_or_none
}
```

planner 生成以下索引：

```text
src_segment[owner][macrobatch] = {offset, rows}
expert_segment[owner_local_expert][macrobatch] = {offset, rows}
combine_map[routed_row] = {source_rank, token_id, topk_lane}
expected_owner_mask[source_rank][macrobatch] : uint64
```

热路径只遍历 non-empty segment。禁止为每个 `(src,dst,expert,tile)` 分配固定状态。

### 4.2 Producer-side pre-pack

为满足跨 row 连续访问，source 端按以下键稳定排序：

```text
(owner_rank, owner_local_expert, macrobatch, source_ordinal)
```

然后一次写入固定 ring。这样：

- pull dispatch 从每个 source 读取连续 segment；
- owner 本地 expert rows 连续，可直接喂 grouped GEMM；
- backward 可复用 forward schedule，避免第二次 sort；
- combine_map 保留回原 token 的确定顺序。

这一步可以与量化融合，但第一版 BF16 必须先验证纯 layout。

### 4.3 Ring

每个 rank 预分配：

```text
dispatch_ring[slots][max_rows][hidden_packed]
fc1_state_ring[slots][max_rows][intermediate_or_replay_state]
combine_ring[slots][max_rows][hidden_packed]
route_metadata[slots][max_routed_rows]
epoch[slots]
```

容量来自静态上界，不从 CPU 读取本 step token count。slot 只有在 forward 与对应 backward 都释放后才能复用；activation checkpoint replay 也必须持有相同 generation。

---

## 5. Forward 数据流

### 5.1 五阶段

1. **Router / plan：** GPU 上完成 top-k、stable ordering、owner/helper mapping。
2. **Pre-pack：** source 将 token 按 owner/expert 顺序写入本地 dispatch ring；可同时产 MX scales。
3. **Dispatch：** 默认实验为 owner pull；每个 owner 从 source 的连续 segment 拉取，不做逐 token 远端原子。
4. **Expert compute：** 对 ready macrobatch 深度优先执行 FC1 → SwiGLU → FC2；shared expert 填 dispatch 首尾空窗。
5. **Combine：** 默认实验为 owner push；输出写回 source combine ring，source 按固定 top-k lane 顺序求和。

### 5.2 Persistent worker 角色

grid 不随 EP 线性增大。所有 WG 从 device queue 取 descriptor：

```text
COMM      : dispatch / combine payload movement
COMPUTE   : FC1 / SwiGLU / FC2 tile
REDUCE    : top-k combine、helper partial reduction
CONTROL   : queue refill、epoch、tail completion
```

角色比例是运行时参数，不在源码里固定“通信占 N 个 CU”。最少扫：

- comm CU fraction；
- macrobatch rows；
- compute M-tile；
- ring slot 数；
- helper replica budget。

队列优先级：阻塞最老 compute 的 dispatch > 已完成 FC2 的 combine > 预取下一 macrobatch > helper 权重搬运。

### 5.3 通信方向

首轮矩阵必须包含：

| 阶段 | 候选 A | 候选 B | 选择依据 |
|---|---|---|---|
| forward dispatch | pull | push | 远端 signal、read RTT、source pack 连续性 |
| forward combine | push | pull | N→1 contention 与 top-k reduction locality |
| backward dispatch-grad | forward 的镜像 | RCCL | payload 与 schedule 是否可复用 |
| backward combine-grad | forward 的镜像 | RCCL | DGrad 返回与 WGrad ownership |

MoK 的 `pull dispatch + push combine` 是优先验证假设，不是 MI455X 既定事实。UALoE 的读写非对称和 coherence 必须由 microbenchmark 给结论。

交换式 fabric 给 pull dispatch 多了一条理由：owner 发出读请求、自己限制在途数量，相当于接收端流控，能直接压住 63→1 的 incast（§2.4）。combine 方向解决不了热 owner 的发送瓶颈：无论 push 还是 pull，热 owner 都要送出更多字节，这只能靠 helper replica 或路由均衡处理。

---

## 6. Backward 数据流

Backward 是架构成立与否的门槛，不能在 forward 做完后临时补。

### 6.1 两条并行分支

每个 expert macrobatch 在 FC2 backward 后分裂：

```text
DGrad: dY → FC2 dgrad → SwiGLU bwd → FC1 dgrad → source token grad
WGrad: (activation, dY) → dW2；(input, dZ) → dW1
```

DGrad 优先级更高，因为它解锁上游层；WGrad 在剩余 compute slots 中运行。两者共享 forward 保存的 route plan 与 ring generation。

### 6.2 Helper replica

V2 中 helper 只计算 planner 指派的完整 tile：

1. owner 或 helper 从权威 owner 取得 expert weight generation；
2. helper 产生 DGrad 与 partial WGrad；
3. DGrad 按原 source/token 返回；
4. partial WGrad 按 `(helper_rank ascending)` 固定顺序归约到 owner；
5. optimizer 只读取 owner 的 `main_grad`。

禁止 helper 对 owner `main_grad` 做非确定 atomic add。小 workload 下，planner 必须允许 replica budget=0；MegaFlux 已观察到小 backward 动态复制可回退到 0.91×。

### 6.3 Router 与 shared expert

- router Wgrad 可在 routed expert 尾部融合，但 reduction order 固定；
- shared expert 与 routed expert 使用独立 weight ownership；
- shared expert 可以由 helper pool 承担，但只有在额外 token 传输低于释放 owner compute 的收益时开启。

---

## 7. 同步与一致性协议

### 7.1 Epoch

每个 ring slot 带 64-bit generation：

```text
FREE → PACKING → DISPATCH_READY → COMPUTE_READY
     → COMBINE_READY → FWD_HELD → BWD_ACTIVE → FREE
```

状态迁移只能单向。wrap-around 前必须确认 forward、backward、checkpoint replay 三方都释放。

### 7.2 Publish

- payload 先写，随后 system-scope release publish；
- consumer 用 system-scope acquire 读 epoch / mask；
- publish 粒度是 non-empty peer segment 或 macrobatch，绝不是 tile/token；
- combine completion 使用 `uint64_t owner_done_mask`，恰好覆盖 EP64；
- helper completion 使用独立 8-bit mask。

具体 UALoE 原子与 cache-coherence 语义未公开前，不能从 gfx950 的 XGMI fence 直接平移；M0 必须做 litmus test。

### 7.3 确定性

以下顺序写进 contract：

```text
route: expert id → source rank → source ordinal → top-k lane
combine: top-k lane ascending
replica WGrad: owner local partial → helper rank ascending
router grad: token id → top-k lane
```

相同输入、相同 route 与相同 EP 下，graph capture、activation checkpoint replay、helper on/off 都必须给出相同 BF16 reference 语义；低精度按既定容差验证。

---

## 8. Kernel 设计约束

### 8.1 矩阵指令与 LDS（CDNA5 已公开部分）

AMD 的 [MI450 LDS 博客](https://rocm.blogs.amd.com/software-tools-optimization/mi450-lds-optimization/README.html) 以 gfx1250 为对象，[CDNA 5 + Helios 博客](https://rocm.blogs.amd.com/ecosystems-and-partners/cdna5-helios/README.html)列出了架构特性；slab 现有记录把 MI455X 训练 build 对应到 gfx1250。与 MI355X（gfx950）相比，已公开的差异：

| 项 | MI355X（gfx950） | MI450 系列（gfx1250） |
|---|---|---|
| wave 宽度 | 64 | **32**（Wave32） |
| 矩阵指令 | MFMA | **WMMA** |
| 计算单元 | CU | **WGP**（博客称等同于以前的 CU 层级），4 个 SIMD 分成 2 对（SIMD0/2、SIMD1/3），warp `N` 落在 SIMD `N mod 4` |
| LDS | 160 KB / CU | **320 KiB / WGP**：6 个 64 KiB 物理分区，5 个给 LDS、1 个给 L0 向量缓存 |
| LDS 带宽 | — | 2 个端口各 256 B/cycle；两个 SIMD 对同周期访问同一分区会串行化，带宽减半（partition conflict） |
| HBM→LDS | direct-to-LDS / async copy | **TDM**（Tensor Data Movement）+ async copy；两者都不能在拷贝中改变连续维；每条 TDM 指令每 warp 最多搬一个 piece |
| LDS 转置读 | — | `ds_load_tr8_b64` / `ds_load_tr16_b128`，宽读后硬件跨 lane 重排 |
| 其他 | — | 更大 L2、更大寄存器文件；E5M3 scale 格式；VALU 上的 4-bit tensor LUT 反量化指令 |

对 MegaMoE kernel 的约束：

- **tile API 必须携带 `arch_traits`**：wave 宽度、WMMA 形状、LDS 容量与分区数都不同，MI355X 的 MFMA tile 和 wave64 lane 映射不能直接复用；具体 WMMA 形状以 CDNA5 ISA 指南为准。
- **pre-pack 直接产出 WMMA 需要的 K 连续布局。** TDM 不能在拷贝中转置，所以布局要么在 pre-pack 时一次排好，要么在 LDS 读取时用 `ds_load_tr` 补救；不要指望拷贝引擎顺带重排。
- **wave specialization 要考虑 SIMD 对。** 角色按 warp id 分配就决定了它落在哪个 SIMD 对上。通信 warp 的 staging buffer 与 GEMM operand buffer 应放在不同 LDS 分区（Gluon 的 `PartitionedSharedLayout` / partition-aware allocator），否则两类 warp 同周期访问同一分区会让 LDS 带宽减半。
- **320 KiB LDS 留出了 staging 余量**：FC1/FC2 至少双缓冲，再考虑在 LDS 中暂存一个 dispatch chunk；具体分配要和 occupancy 一起扫。
- LDS padding / swizzle 按 gfx1250 的 bank model（64 bank，`ds_load_b128` 按 16 lane 一组）重新验证。

### 8.2 五问落地

| 问题 | 本设计答案 |
|---|---|
| Q1 跨 row read 连续？ | 是；source 端按 owner/expert/macrobatch pre-pack |
| Q2 跨 row write 连续？ | dispatch ring 与 owner expert segment 连续；combine 通过 combine ring 避免直接散写 token output |
| Q3 VGPR 同时背 read/write？ | 默认 HBM ring staging；本地 HBM→LDS 用 TDM / async copy，不经 VGPR；远端 payload 是否能直接落 LDS 待 E0 验证 |
| Q4 wave 锁步还是独立？ | payload burst 写用锁步（注意 wave32 下每 wave 一次搬运宽度减半）；远端 pull 延迟隐藏用 wave-independent；WMMA tile 内按 LDS exchange 要求锁步 |
| Q5 push/pull 匹配硬件？ | policy 可切；以 UALoE microbenchmark 而不是 XGMI 经验决定；交换式 fabric 下优先 receiver-driven，push 必须带 credit（§2.4） |

### 8.3 不允许的实现

- 一个 peer 固定占一个 resident WG；
- 每个 expert tile 做 system fence；
- route count 回 CPU 后再分配；
- `if (EP == 8)` 的快路径成为唯一正确路径；
- helper 与 owner 对同一 gradient 做无序浮点原子；
- 图重放时重新运行 nondeterministic dispatch planner。

---

## 9. 精度路线

### M0/M1：BF16

BF16 是协议与同步的 ground truth。所有 EP、top-k、ring wrap、skew、checkpoint replay 测试先在 BF16 通过。

### M2：MXFP8

- activation quantization 融入 pre-pack；
- owner/helper 使用同一 scale contract；
- scale metadata 与 payload 同 ring slot、同 epoch；
- WGrad accumulation 保持宽精度；
- 对比 unfused Transformer Engine / Primus-Turbo 路径。

### M3：MXFP4

单独立项，不从 MXFP8 自动打开：

- deterministic Hadamard、row-gradient SR、scale direction 分开做消融；
- 评估 CDNA5 新增的 E5M3 scale 格式与 4-bit tensor LUT 反量化指令，它们可能改变 MXFP4 的 scale 选择与反量化开销；
- AMD 的 MXFP4 GEMM 20.0 PFLOPS（MI355X 6.0）是近常数输入下的峰值交付值，不能当训练 GEMM 的预期；
- router、norm、敏感输出层允许保持 BF16/FP8；
- 必须跑训练收敛，不接受只跑 kernel correctness。

---

## 10. EP64 扩展机制

EP64 不能通过增加 resident comm WG 来扩展。必须同时满足：

1. **metadata rank-width 到 64；**
2. **worker queue 与 EP 解耦；**
3. **64-bit completion bitmap；**
4. **只为 non-empty peer 建 segment；**
5. **chunk 粒度随 token/rank 调整；**
6. **每卡 4 experts 时仍能合并足够 M tiles 喂满 WMMA；**
7. **真实 skew 下允许 helper 或 owner 内重排，但 router 语义不变；**
8. **接收端在途字节有上限（credit），防止 63→1 incast 在交换机出口排队；**
9. **每 peer 消息不小于 E0 测得的饱和点，否则加大 chunk 或合并 peer segment。**

EP8→64 的扩展测试固定 per-rank tokens，分别报告：

- route/plan 时间；
- 每 peer 消息大小，以及热点 owner 的接收 / 发送带宽；
- dispatch、FC1、FC2、combine；
- exposed comm；
- rank-max / rank-median；
- UALoE payload GB/s；
- compute WGP 的 WMMA busy 与 LDS partition conflict；
- host gap；
- ring occupancy 与 wrap 次数。

参考目标而非既成事实：GB200 同形状 dispatch EP8→64 为 1.73×、combine 为 2.11×。MI455X V1 的 stretch goal 是 rank-max MoE layer latency 扩展不劣于该量级；最终门禁以本机 RCCL A2A 为准。

---

## 11. Correctness 门禁

| 测试 | 必须覆盖 |
|---|---|
| EP sweep | 8/16/32/64 |
| top-k | 2/6/8/10，含非 2 次幂 |
| microbatch | 1、2、≥4 |
| route | force-balanced、真实 checkpoint、单热点专家、空专家 |
| ring | 正常、满载、至少两次 wrap |
| backward | input grad、dW1、dW2、router grad、shared expert grad |
| replay | activation checkpoint on/off；graph capture on/off |
| transport | RCCL vs fused |
| helper | off、1/4/8 ranks；replica budget=0 |
| fault pattern | 缺失 publish、stale epoch 必须 fail-fast；delayed peer 在 fabric 重路由窗口（约 1 s）内只能告警，超过后才 fail-fast |

特别检查三类已知 silent-error 家族：

- dispatch backward payload fence；
- checkpoint 重放时 route row 顺序变化；
- combine backward replay payload 丢失。

只比较 loss 不够；必须逐层比较 routed rows、route map、input grad 与 expert main_grad。

---

## 12. 性能门禁

每个优化都要过同工况三基线：

1. PyTorch + RCCL A2A + unfused grouped GEMM；
2. Primus 当前 MegaMoE；
3. 新 EP64 executor 的 unfused / fused 两档。

接受条件：

- correctness 全通过；
- 生产 shape 的 geomean 胜 RCCL 基线；
- 任一主力 shape 不允许因默认融合显著回退，回退时自动选择 RCCL；
- helper 模式必须扣除 8 GPU 成本，报告 rack-level tokens/s，而不只报 owner tokens/s/GPU；
- 端到端训练收益必须复现 layer-level 收益，不能只报 isolated kernel。

---

## 13. 实施里程碑

| 阶段 | 交付 | 退出条件 |
|---|---|---|
| **A0 contract** | rank-width、PG、workspace、route plan API | EP64 metadata 单测；无 8-rank 常量 |
| **A1 RCCL baseline** | BF16 EP8/16/32/64 forward/backward | 全 correctness matrix；rank-max breakdown |
| **A2 fused forward** | pre-pack + ring + persistent FWD | 胜或自动回退 RCCL；无 D2H |
| **A3 fused backward** | DGrad/WGrad 双分支 + replay | main_grad 与 reference 对齐 |
| **A4 graph** | 固定 workspace + partial HIP graph | graph on/off 等价；host gap 可测 |
| **A5 MXFP8** | fused quant + scales + grouped GEMM | kernel + short-train 数值门禁 |
| **A6 64+8 helper** | dynamic replica + deterministic grad return | rack-level 净收益；小 shape 自动关 |
| **A7 MXFP4** | 独立配方 | 收敛与 E2E 收益同时成立 |

不要交换 A3 与 A6：没有可信 backward，helper replica 没有训练语义。

---

## 14. 首批实验

### E0：UALoE 原语

- peer read vs peer write；
- contiguous rows vs routed rows；
- message 4 KB → 64 MB；
- peers 8/16/32/64；
- system release/acquire litmus；
- 单 WG / 单流到单 peer 的带宽，以及打满本卡端口需要的并发流数与在途请求数（判断多平面是逐包喷洒还是按流哈希）；
- incast：1/8/32/63 个 source 同时写或被读同一 owner，测 owner 接收带宽与对其他流的干扰；
- 远端 flag 往返延迟：同 tray 与跨 tray 各测一组（AMD 只说明带宽相同，延迟未公开）；
- 以上全部在 64/72 卡上测：AMD 公开的 TransferBench 数据来自 4 卡 bring-up 节点，不含 63 peer 并发；
- 若平台允许，在单链路 / 单交换机降级下复测 rank-max 与重路由窗口；
- 每 peer 消息大小饱和曲线，用于设定 EP64 下的最小 chunk；
- RCCL A2A 对照。

### E1：EP 扩展基线

`H=7168, E=256, top-k=8, T/rank=1024/2048/4096`，扫 EP8/16/32/64；BF16，force-balanced 与真实 route 各一组。

### E2：方向与 chunk

四种 dispatch/combine 方向 × macrobatch 512/1K/2K/4K/8K/16K/32K，记录 rank-max 而非平均。

### E3：小 M

EP64 每卡 4 experts，扫热点与空专家；确认 grouped GEMM、wave specialization 与 helper gate。

### E4：训练正确性

至少 100 step，microbatch≥2、FullAC on/off、graph on/off；逐层 gradient parity + loss。

---

## 15. 最终推荐配置

在上机数据出现前，建议默认配置是：

```yaml
expert_model_parallel_size: 32
expert_tensor_parallel_size: 1
transport: auto              # rccl_a2a | ualoe_fused
dispatch_direction: auto
combine_direction: auto
graph_safe_workspace: true
helper_pool_size: 0
precision: bf16
```

EP64 是必须通过的一级配置，不是未经测量的默认：

```yaml
expert_model_parallel_size: 64
expert_tensor_parallel_size: 1
transport: auto
helper_pool_size: 0          # A6 后才允许 8
precision: mxfp8             # A5 通过后
```

这保留了两条真实可能性：若 UALoE 已把通信墙消掉，EP32 可能因更大的 local grouped GEMM 获胜；若每卡显存、expert TP=1 与小 per-rank token 更重要，EP64 才成为主轨。架构必须支持两者，而不是提前替硬件做决定。

---

## 16. 相关资料

- 并行策略与硬件推断：[`mi455x-parallelism-strategy-reference.md`](./mi455x-parallelism-strategy-reference.md)
- Helios 硬件规格：[`../hardware/gpu-comparison.md`](../hardware/gpu-comparison.md)
- AMD ROCm 博客：[CDNA 5 + Helios](https://rocm.blogs.amd.com/ecosystems-and-partners/cdna5-helios/README.html)（fabric、故障降级、vPod）、[MI455X vs MI355X](https://rocm.blogs.amd.com/software-tools-optimization/mi455x-vs-mi355x/README.html)（实测倍数与方法）、[MI450 LDS](https://rocm.blogs.amd.com/software-tools-optimization/mi450-lds-optimization/README.html)（Wave32、WGP、LDS 分区、TDM）
- Scale-up megakernel：[`../../papers/mixture-of-kittens.md`](../../papers/mixture-of-kittens.md)
- 数据搬运五问：[`../kernels/memory-access-patterns.md`](../kernels/memory-access-patterns.md)
- 通算融合实现模式：[`../kernels/cco-overlap.md`](../kernels/cco-overlap.md)
- 最新训练系统扫描：[`arxiv-digest-2026-10.md`](./arxiv-digest-2026-10.md)
