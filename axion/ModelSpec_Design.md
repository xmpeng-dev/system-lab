# Axion Model Spec 设计

> **文档类型:** 详细设计文档  
> **核心问题:** Model Spec 如何定义才能兼顾直观性、可扩展性，同时为编译器提供足够的静态信息？  
> **设计原则:** 分层描述、渐进精化、关注点分离

---

## 0. 设计思路

### 0.1 Model Spec 的三个消费者

```
Model Spec 被三个消费者读取，它们的需求不同：

  消费者 1：Axion Compiler
    需要：并行策略、Expert 拓扑、通信组定义
    关心：编译期静态信息（shape、并行度、拓扑）

  消费者 2：FSEP 运行时（Slow Planner）
    需要：Expert 数量、初始分布、负载均衡约束
    关心：Expert 物理分布的搜索空间

  消费者 3：用户 / 框架集成者
    需要：直观、少写、易改
    关心：不想关心底层分片细节，声明意图即可

→ Model Spec 必须满足三者，但不能为任何一者过度设计
```

### 0.2 分层设计原则

```
Model Spec 分三层，用户可以选择写到哪一层：

  Layer 1：Model Shape（必填）
    描述模型的"骨架"：层数、hidden size、Expert 数量等
    → 对应：ModelShapeSpec

  Layer 2：Parallelism Strategy（必填）
    描述如何把模型分布到多 GPU：DP/TP/EP/FSDP 的配置
    → 对应：ParallelismSpec

  Layer 3：Expert Placement Hint（可选）
    描述 Expert 的初始物理分布 hint 和迁移约束
    → 对应：ExpertPlacementSpec

用户只写 Layer 1+2，Layer 3 由 Slow Planner 自动决定（推荐）
或者用户手动指定 Layer 3 覆盖自动决策（高级用法）
```

---

## 1. 顶层结构

```python
@dataclass
class AxionModelSpec:
    """
    Axion Model Spec 顶层结构
    
    设计原则：
      - 每个字段职责单一
      - 可选字段有合理默认值
      - 扩展点通过 extensions 字典开放
    """
    
    # ── Layer 1：模型骨架（必填）──────────────────────────────
    model:       ModelShapeSpec
    
    # ── Layer 2：并行策略（必填）──────────────────────────────
    parallelism: ParallelismSpec
    
    # ── Layer 3：Expert 布局 Hint（可选，不填则自动规划）──────
    expert_placement: ExpertPlacementSpec | None = None
    
    # ── 数值精度配置（可选）────────────────────────────────────
    dtype:       DTypeSpec = field(default_factory=DTypeSpec.default)
    
    # ── 编译器行为控制（可选）──────────────────────────────────
    compiler:    CompilerSpec = field(default_factory=CompilerSpec.default)
    
    # ── 扩展点（保留，供未来 feature 使用）────────────────────
    extensions:  dict[str, Any] = field(default_factory=dict)
```

---

## 2. Layer 1：ModelShapeSpec（模型骨架）

### 2.1 顶层 ModelShapeSpec

```python
@dataclass
class ModelShapeSpec:
    """
    描述模型的静态结构，与并行策略无关。
    这是模型的"数学描述"，不涉及任何分布式概念。
    """
    
    arch:         ModelArch           # 模型架构类型（见 2.2）
    num_layers:   int                 # Transformer 层数
    hidden_size:  int                 # hidden dimension H
    
    # Attention 相关
    num_heads:    int                 # attention head 数
    head_dim:     int | None = None   # head dim（None 则自动推导为 H/num_heads）
    num_kv_heads: int | None = None   # GQA/MQA 的 KV head 数（None = MHA）
    
    # FFN 相关
    ffn_spec:     FFNSpec = field(default_factory=FFNSpec.default)
    
    # 序列相关
    max_seq_len:  int = 4096
    vocab_size:   int = 32000
    
    # MoE 相关（仅当 arch 包含 MoE 时有效）
    moe_spec:     MoESpec | None = None
```

### 2.2 ModelArch 枚举

```python
class ModelArch(Enum):
    """
    模型架构枚举。
    使用 Flag 支持组合（如 DENSE | ROPE 表示带 RoPE 的 Dense 模型）
    """
    # 基础架构
    DENSE       = auto()    # 纯 Dense Transformer
    MOE         = auto()    # Mixture of Experts（稀疏 FFN）
    MOE_DENSE   = auto()    # 混合：部分层 MoE，部分层 Dense（如 DeepSeek-V3）
    
    # 位置编码
    ROPE        = auto()    # Rotary Position Embedding
    ALIBI       = auto()    # ALiBi
    LEARNED_PE  = auto()    # Learned Position Embedding
    
    # Attention 变体
    GQA         = auto()    # Grouped Query Attention
    MLA         = auto()    # Multi-head Latent Attention（DeepSeek 特有）
    
    # 组合示例（实际存储为 set[ModelArch]）：
    # DSv3 = {MOE_DENSE, ROPE, GQA, MLA}
```

### 2.3 FFNSpec

```python
@dataclass
class FFNSpec:
    """FFN 结构描述，与 MoE/Dense 无关的公共部分"""
    
    # FFN 类型
    activation:   Activation = Activation.SWIGLU  # GELU | RELU | SWIGLU | GEGLU
    
    # 维度
    intermediate_size: int | None = None   # FFN 中间层维度（None = 4 * hidden）
    multiple_of:  int = 256                # intermediate_size 对齐粒度
    
    @staticmethod
    def default() -> 'FFNSpec':
        return FFNSpec()

class Activation(Enum):
    GELU   = "gelu"
    RELU   = "relu"
    SWIGLU = "swiglu"   # SiLU(gate) * up，DeepSeek/Llama 系常用
    GEGLU  = "geglu"
```

### 2.4 MoESpec（MoE 专用描述）

```python
@dataclass
class MoESpec:
    """
    MoE 结构描述。
    关注点：Expert 的"数学属性"，不涉及分布式分配。
    """
    
    # Expert 基本属性
    num_experts:        int             # 总 Expert 数，e.g. 256
    num_experts_per_tok: int            # Top-K，e.g. 2（每 token 激活几个 Expert）
    expert_ffn_size:    int | None = None  # Expert 的 FFN 中间维度（None = 同 Dense FFN）
    
    # 路由属性
    router_type:        RouterType = RouterType.TOP_K
    router_aux_loss:    bool = True      # 是否使用 auxiliary load balance loss
    router_z_loss:      bool = False     # z-loss 正则化
    router_norm:        bool = True      # softmax before topk（vs. sigmoid）
    
    # MoE 层分布（对 MOE_DENSE 架构）
    moe_layer_freq:     int = 1          # 每几层放一个 MoE 层（1=全 MoE，2=隔层 MoE）
    first_k_dense:      int = 0          # 前 K 层强制 Dense（如 DSv3 的前 3 层）
    
    # Shared Expert（DSv3 特有：有几个 Expert 是共享的，每个 token 必选）
    num_shared_experts: int = 0

class RouterType(Enum):
    TOP_K          = "top_k"           # 标准 Top-K 路由
    EXPERT_CHOICE  = "expert_choice"   # Expert 主动选 token（负载天然均衡）
    HASH           = "hash"            # 确定性哈希路由（无学习，用于 baseline）
    SOFT           = "soft"            # Soft MoE（可微分）
```

---

## 3. Layer 2：ParallelismSpec（并行策略）

### 3.1 ParallelismSpec 顶层

```python
@dataclass
class ParallelismSpec:
    """
    并行策略描述。
    核心设计：用"并行度"而非"设备分配"描述策略，
    由 Axion Compiler 自动生成设备分配方案。
    """
    
    # 并行度配置
    data_parallel_size:   int = 1    # DP 副本数
    tensor_parallel_size: int = 1    # TP 切分数（Attention + Dense FFN）
    expert_parallel_size: int = 1    # EP 切分数（Expert FFN）
    pipeline_parallel_size: int = 1  # PP 流水线段数（v1 暂不支持，预留）
    
    # FSDP 配置（ZeRO-style 参数分片）
    fsdp:         FSDPSpec = field(default_factory=FSDPSpec.default)
    
    # 集群拓扑（用于通信路径优化）
    topology:     ClusterTopologySpec = field(
                      default_factory=ClusterTopologySpec.auto_detect
                  )
    
    # 约束检查（编译器会验证）
    def validate(self, model: ModelShapeSpec) -> None:
        total = (self.data_parallel_size
                 * self.tensor_parallel_size
                 * self.expert_parallel_size
                 * self.pipeline_parallel_size)
        assert total == self.topology.total_gpus, \
            f"并行度乘积 {total} != 总 GPU 数 {self.topology.total_gpus}"
        
        if self.expert_parallel_size > 1:
            assert model.moe_spec is not None, "EP > 1 要求模型有 MoE 层"
            assert model.moe_spec.num_experts % self.expert_parallel_size == 0, \
                f"num_experts={model.moe_spec.num_experts} 必须整除 EP={self.expert_parallel_size}"
```

### 3.2 FSDPSpec

```python
@dataclass
class FSDPSpec:
    """
    FSDP 参数分片策略。
    借鉴 veScale-FSDP 的 RaggedShard 理念：分片粒度由计算语义决定。
    """
    
    # 分片策略
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    
    # RaggedShard 配置（Axion 的核心扩展，对应 veScale-FSDP 理念）
    ragged_shard: RaggedShardConfig = field(
                      default_factory=RaggedShardConfig.default
                  )
    
    # 通信策略
    reduce_scatter_hier: bool = True    # 分层 RS（节点内 NVLink → 节点间 IB）
    lazy_all_gather:     bool = True    # 懒惰 AG（真正需要时才触发）
    
    # 重计算策略（内存 vs 计算权衡）
    activation_checkpointing: CheckpointPolicy = CheckpointPolicy.SELECTIVE
    
    @staticmethod
    def default() -> 'FSDPSpec':
        return FSDPSpec()

class ShardingStrategy(Enum):
    FULL_SHARD    = "full_shard"    # ZeRO-3：参数+梯度+优化器状态全分片
    SHARD_GRAD_OP = "shard_grad_op" # ZeRO-2：仅分片梯度和优化器状态
    NO_SHARD      = "no_shard"      # 不分片（纯 DP）

@dataclass
class RaggedShardConfig:
    """
    RaggedShard 配置：按计算语义而非按行均匀切分（veScale-FSDP 核心）
    """
    enabled:      bool = True
    block_size:   int  = 128        # 量化 block 大小（影响分片粒度）
    optimizer_aware: bool = True    # 是否感知优化器需求（Shampoo/Muon 支持）
    
    @staticmethod
    def default() -> 'RaggedShardConfig':
        return RaggedShardConfig()

class CheckpointPolicy(Enum):
    NONE       = "none"        # 不重计算，保留所有激活
    FULL       = "full"        # 重计算所有层
    SELECTIVE  = "selective"   # 只重计算内存密集层（Attention）
```

### 3.3 ClusterTopologySpec

```python
@dataclass
class ClusterTopologySpec:
    """
    集群拓扑描述。
    用于 CommFabric 的网络感知路由和分层通信优化。
    """
    
    num_nodes:         int            # 节点数
    gpus_per_node:     int            # 每节点 GPU 数
    
    # 带宽描述（用于通信成本估算）
    intra_node_bw_gbps: float = 600.0  # 节点内 NVLink 带宽
    inter_node_bw_gbps: float = 200.0  # 节点间 IB 带宽
    
    # 网络类型
    intra_node_network: NetworkType = NetworkType.NVLINK
    inter_node_network: NetworkType = NetworkType.INFINIBAND
    
    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node
    
    @staticmethod
    def auto_detect() -> 'ClusterTopologySpec':
        """运行时自动探测集群拓扑（通过 NCCL / torch.distributed）"""
        return ClusterTopologySpec._from_env()

class NetworkType(Enum):
    NVLINK      = "nvlink"
    INFINIBAND  = "infiniband"
    ETHERNET    = "ethernet"
    NVLINK_RDMA = "nvlink_rdma"   # NVLink + GPUDirect RDMA 混合
```

---

## 4. Layer 3：ExpertPlacementSpec（Expert 布局 Hint）

```python
@dataclass
class ExpertPlacementSpec:
    """
    Expert 物理布局 Hint。
    
    通常不需要手动填写——Axion Slow Planner 会自动规划。
    高级用法：用户可以提供 hint 或约束，指导 Planner 搜索方向。
    """
    
    # 初始布局策略
    init_strategy: ExpertInitStrategy = ExpertInitStrategy.ROUND_ROBIN
    
    # FSEP 分片配置（借鉴 LAER-MoE）
    fsep: FSEPConfig = field(default_factory=FSEPConfig.default)
    
    # 迁移约束
    migration: MigrationConfig = field(default_factory=MigrationConfig.default)
    
    # 手动 pin（优先级最高，覆盖自动规划）
    pinned_experts: dict[int, int] = field(default_factory=dict)
    # {expert_id: gpu_id}，被 pin 的 expert 不参与自动迁移

class ExpertInitStrategy(Enum):
    ROUND_ROBIN    = "round_robin"     # Expert i → GPU (i % num_gpus)，简单均匀
    PROFILE_GUIDED = "profile_guided"  # 从历史 profile 数据初始化（需提供 profile_path）
    RANDOM         = "random"          # 随机分配（消融实验用）

@dataclass
class FSEPConfig:
    """
    FSEP（Fully Sharded Expert Parallel）配置。
    核心：热点 Expert 可以跨多个 GPU 分片存储。
    """
    enabled:             bool  = True
    
    # 热点 Expert 触发阈值
    # 当 expert 接收的 token 数 > avg_tokens * overload_threshold 时
    # 触发该 expert 的 shard 扩展（分裂到更多 GPU）
    overload_threshold:  float = 2.0   # 2x 平均负载触发分片
    
    # shard 数量约束
    min_shards_per_expert: int = 1     # 每个 expert 至少 1 个 shard（不分片）
    max_shards_per_expert: int = 4     # 每个 expert 最多分成 4 个 shard
    
    @staticmethod
    def default() -> 'FSEPConfig':
        return FSEPConfig()

@dataclass
class MigrationConfig:
    """Expert 迁移行为配置"""
    
    enabled:          bool  = True
    
    # 触发频率：每隔多少 step 检查一次是否需要迁移
    check_interval:   int   = 50       # 借鉴 LAER-MoE 的 K step 间隔
    
    # 迁移触发条件：负载不均衡度超过阈值才触发
    # imbalance = max_gpu_load / avg_gpu_load
    imbalance_threshold: float = 1.3   # 最忙 GPU 比平均忙 30% 才迁移
    
    # 迁移开销约束：预期节省 > 迁移成本才执行
    min_roi:          float = 1.1      # 收益/成本 > 1.1 才迁移
    
    # 单次迁移上限（避免大规模迁移冲击训练稳定性）
    max_experts_per_migration: int = 8
    
    @staticmethod
    def default() -> 'MigrationConfig':
        return MigrationConfig()
```

---

## 5. DTypeSpec（数值精度）

```python
@dataclass
class DTypeSpec:
    """
    数值精度配置。
    关注点分离：dtype 配置独立于模型结构和并行策略。
    """
    
    # 参数精度
    param_dtype:      Dtype = Dtype.BF16
    
    # 计算精度
    compute_dtype:    Dtype = Dtype.BF16
    
    # 优化器状态精度（通常比参数精度高）
    optimizer_dtype:  Dtype = Dtype.FP32
    
    # 量化配置（可选，启用 veScale-FSDP RaggedShard 的量化感知分片）
    quantization:     QuantizationSpec | None = None
    
    @staticmethod
    def default() -> 'DTypeSpec':
        return DTypeSpec()

@dataclass
class QuantizationSpec:
    """Block-wise 量化配置（启用后 RaggedShard 按 block 边界分片）"""
    
    weight_quant:     QuantType = QuantType.INT8   # 权重量化精度
    act_quant:        QuantType | None = None       # 激活量化（None=不量化激活）
    block_size:       int = 128                     # 量化 block 大小
    
    # 与 RaggedShardConfig.block_size 必须一致（编译器会检查）

class QuantType(Enum):
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8     = "int8"
    INT4     = "int4"

class Dtype(Enum):
    FP32  = "float32"
    BF16  = "bfloat16"
    FP16  = "float16"
    FP8   = "fp8"
```

---

## 6. CompilerSpec（编译器行为控制）

```python
@dataclass
class CompilerSpec:
    """
    编译器行为控制。
    通常用默认值，高级用户可以微调编译器策略。
    """
    
    # Overlap 调度
    overlap_strategy: OverlapStrategy = OverlapStrategy.AGGRESSIVE
    chunk_size:       int | None = None  # A2A chunk 大小（None=自动推导）
    
    # FSEP Sharding Plan 算法
    sharding_algo:    ShardingAlgo = ShardingAlgo.GREEDY
    
    # 编译缓存
    enable_cache:     bool = True
    cache_dir:        str  = ".axion_cache"
    
    # 调试选项
    dump_ir:          bool = False       # 输出各 Pass 的 IR 到文件
    dump_comm_graph:  bool = False       # 输出通信依赖图
    profile_passes:   bool = False       # 统计各 Pass 的编译时间
    
    @staticmethod
    def default() -> 'CompilerSpec':
        return CompilerSpec()

class OverlapStrategy(Enum):
    CONSERVATIVE = "conservative"  # 只做有把握的 overlap（安全优先）
    AGGRESSIVE   = "aggressive"    # 尽可能多 overlap（性能优先，默认）
    DISABLED     = "disabled"      # 关闭 overlap（调试用）

class ShardingAlgo(Enum):
    GREEDY   = "greedy"    # 贪心，快，次优（默认，Phase 1 实现）
    ILP      = "ilp"       # 整数线性规划，最优，慢（Phase 3 实现）
    RL       = "rl"        # 强化学习，离线训练策略（未来版本）
```

---

## 7. 使用示例

### 7.1 最简用法（DeepSeek-V3-like 模型）

```python
spec = AxionModelSpec(
    model = ModelShapeSpec(
        arch         = {ModelArch.MOE_DENSE, ModelArch.ROPE, ModelArch.GQA},
        num_layers   = 61,
        hidden_size  = 7168,
        num_heads    = 128,
        num_kv_heads = 128,
        moe_spec = MoESpec(
            num_experts          = 256,
            num_experts_per_tok  = 8,
            num_shared_experts   = 1,
            first_k_dense        = 3,
            moe_layer_freq       = 1,
        ),
    ),
    parallelism = ParallelismSpec(
        data_parallel_size    = 8,
        tensor_parallel_size  = 4,
        expert_parallel_size  = 64,
        topology = ClusterTopologySpec(
            num_nodes     = 64,
            gpus_per_node = 8,
        ),
    ),
    # expert_placement、dtype、compiler 全部使用默认值
)
```

### 7.2 完整用法（开启量化 + 自定义 Expert 布局）

```python
spec = AxionModelSpec(
    model = ModelShapeSpec(
        arch        = {ModelArch.MOE_DENSE, ModelArch.ROPE},
        num_layers  = 32,
        hidden_size = 4096,
        num_heads   = 32,
        moe_spec = MoESpec(
            num_experts         = 64,
            num_experts_per_tok = 2,
            router_type         = RouterType.TOP_K,
            router_aux_loss     = True,
        ),
    ),
    parallelism = ParallelismSpec(
        data_parallel_size    = 4,
        tensor_parallel_size  = 2,
        expert_parallel_size  = 8,
        fsdp = FSDPSpec(
            sharding_strategy = ShardingStrategy.FULL_SHARD,
            ragged_shard = RaggedShardConfig(
                enabled    = True,
                block_size = 128,
                optimizer_aware = True,
            ),
            reduce_scatter_hier = True,
        ),
        topology = ClusterTopologySpec(
            num_nodes     = 8,
            gpus_per_node = 8,
            intra_node_bw_gbps = 600.0,
            inter_node_bw_gbps = 200.0,
        ),
    ),
    expert_placement = ExpertPlacementSpec(
        init_strategy = ExpertInitStrategy.PROFILE_GUIDED,
        fsep = FSEPConfig(
            enabled              = True,
            overload_threshold   = 2.0,
            max_shards_per_expert = 4,
        ),
        migration = MigrationConfig(
            check_interval        = 50,
            imbalance_threshold   = 1.3,
            max_experts_per_migration = 8,
        ),
        # pin 前 4 个 Expert 到 GPU 0（它们是 shared experts，负载极高）
        pinned_experts = {0: 0, 1: 0, 2: 1, 3: 1},
    ),
    dtype = DTypeSpec(
        param_dtype   = Dtype.BF16,
        compute_dtype = Dtype.BF16,
        quantization  = QuantizationSpec(
            weight_quant = QuantType.INT8,
            block_size   = 128,
        ),
    ),
    compiler = CompilerSpec(
        overlap_strategy = OverlapStrategy.AGGRESSIVE,
        sharding_algo    = ShardingAlgo.GREEDY,
        dump_ir          = True,   # 开发阶段开启
    ),
)
```

### 7.3 从 YAML/JSON 加载（生产环境推荐）

```yaml
# axion_config.yaml
model:
  arch: [moe_dense, rope, gqa]
  num_layers: 61
  hidden_size: 7168
  num_heads: 128
  num_kv_heads: 128
  moe_spec:
    num_experts: 256
    num_experts_per_tok: 8
    num_shared_experts: 1
    first_k_dense: 3

parallelism:
  data_parallel_size: 8
  tensor_parallel_size: 4
  expert_parallel_size: 64
  topology:
    num_nodes: 64
    gpus_per_node: 8

# expert_placement、dtype、compiler 省略 → 使用默认值
```

```python
# 加载方式
spec = AxionModelSpec.from_yaml("axion_config.yaml")

# 或从 dict（方便程序化生成）
spec = AxionModelSpec.from_dict({...})
```

---

## 8. 可扩展性设计

### 8.1 新架构的接入

```
添加新模型架构（如 Mamba、RWKV）的步骤：

1. 在 ModelArch 枚举添加新值
     ModelArch.MAMBA = auto()

2. 添加对应的 ArchSpec（如 MambaSpec）
     @dataclass
     class MambaSpec:
         d_state:      int = 16
         d_conv:       int = 4
         expand_factor: int = 2

3. 在 ModelShapeSpec 添加可选字段
     mamba_spec: MambaSpec | None = None

4. 在 Axion IR 指令集添加对应 Op
     Recurrent.SelectiveScan(...)

→ 不需要修改 ParallelismSpec 和 ExpertPlacementSpec
```

### 8.2 新并行策略的接入

```
添加新并行策略（如 Context Parallelism）的步骤：

1. 在 ParallelismSpec 添加新字段
     context_parallel_size: int = 1

2. 在 validate() 添加约束检查

3. 在 Axion Compiler 添加对应 Pass

→ 不需要修改 ModelShapeSpec 和 ExpertPlacementSpec
```

### 8.3 extensions 字典的使用

```python
# 实验性 feature 通过 extensions 传递，不污染主 Spec

spec = AxionModelSpec(
    model = ...,
    parallelism = ...,
    extensions = {
        # 实验性：Expert 负载预测模型（未正式支持）
        "load_predictor": {
            "type": "lstm",
            "lookback_steps": 10,
        },
        # 实验性：梯度压缩
        "grad_compression": {
            "type": "powersgd",
            "rank": 4,
        },
    }
)
```

---

## 9. Spec 与 Axion 各组件的关系

```
AxionModelSpec
    │
    ├── ModelShapeSpec ──────────────────→ Axion IR 生成
    │   ├── MoESpec                          Expert.Gate Op 参数
    │   └── FFNSpec                          Expert.FFN 维度
    │
    ├── ParallelismSpec ─────────────────→ Compiler Pass 3（Sharding Plan）
    │   ├── FSDPSpec（RaggedShard）           CommTensor layout 决策
    │   └── ClusterTopologySpec              CommFabric 网络路由
    │
    ├── ExpertPlacementSpec ─────────────→ FSEP 运行时（Slow Planner）
    │   ├── FSEPConfig                       ExpertShard 分裂阈值
    │   └── MigrationConfig                  迁移触发参数
    │
    ├── DTypeSpec ───────────────────────→ Compiler Pass 5（Layout Lowering）
    │   └── QuantizationSpec                 RaggedShard block 边界对齐
    │
    └── CompilerSpec ────────────────────→ 各 Pass 行为控制
        ├── OverlapStrategy                  Pass 4（Overlap Insertion）
        └── ShardingAlgo                     Pass 3（Sharding Plan）
```

---

## 10. 已知设计取舍（Trade-offs）

| 决策 | 选择 | 放弃 | 理由 |
|------|------|------|------|
| 并行度用"数量"而非"设备列表" | `expert_parallel_size=64` | `expert_gpus=[0,1,...,63]` | 更直观，设备分配交给 Compiler |
| arch 用 `set[Enum]` 而非单一 Enum | `{MOE_DENSE, ROPE, GQA}` | `Arch.MOE_DENSE_ROPE_GQA` | 组合爆炸问题，set 更可扩展 |
| ExpertPlacementSpec 可选 | 默认 `None`，Planner 自动规划 | 强制用户填写 | 降低使用门槛，自动规划是首选 |
| dtype 独立于 model/parallelism | 单独的 `DTypeSpec` | 嵌入在各处 | 关注点分离，量化配置集中管理 |
| extensions 字典开放扩展点 | `extensions: dict` | 每次加 feature 改 Spec | 实验性 feature 不污染主 Spec |

---

*文档版本 v0.1 | Axion Model Spec 设计 | 2026-03-08*  
*参考：LAER-MoE (arXiv:2602.11686) | veScale-FSDP (arXiv:2602.22437)*
