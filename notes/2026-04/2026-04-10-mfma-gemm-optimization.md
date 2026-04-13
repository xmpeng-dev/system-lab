# MI355X MFMA GEMM 优化总结

## 目标
在 AMD MI355X (gfx950) 上实现高性能 hand-written Grouped GEMM，用于 MoE persistent kernel 内部的 Expert GEMM 计算。

## 基准配置 (DeepSeek-V3)
- FC1: M=128, K=7168, N=4096, E=32 experts
- FC2: M=128, K=2048, N=7168, E=32 experts
- 数据类型: BF16 (输入) → FP32 (累加) → BF16 (输出)

## 性能演进

| 版本 | FC1 TFLOPS | FC2 TFLOPS | vs CK | 关键优化 |
|------|-----------|-----------|-------|---------|
| v2 baseline | 247T | 239T | 60% | 128×128 tile, K8-VGPR MFMA |
| v4a | 250T | 253T | 60% | Pipelined global loads |
| v4b | 368T | 370T | 89% | + ds_read for LDS |
| **v4c** | **432T** | **427T** | **104%** | + buffer_load |

**最终结果: 超越 CK 参考实现 (415T)**

---

## 优化路径详解

### Phase 1: 基础 GEMM 框架 (v2)
- **Tile 配置**: 128×128×32 (M×N×K)
- **MFMA 指令**: `v_mfma_f32_32x32x8_bf16` (K=8, VGPR mode)
- **LDS 布局**: Double-buffered, 36KB total
- **瓶颈**: 全局内存加载效率低 (~48% HBM 带宽利用率)

### Phase 2: Global Load 流水线 (v4a → v4b)

**问题诊断**: Assembly 分析发现 `global_load_dwordx4` 后立即跟随 `s_waitcnt vmcnt(0)`，导致每次只有 1 个 in-flight load。

**解决方案**:
```
┌─────────────────────────────────────────────────────────┐
│  Issue (Phase 1)  →  Compute (Phase 2)  →  Commit (Phase 3)  │
│  发射 4 个 loads      MFMA 计算当前 buffer    提交到 LDS       │
└─────────────────────────────────────────────────────────┘
```

1. **Prologue**: 批量发射 4 个 global_load (无 wait)
2. **Main Loop**: 
   - Phase 1: 发射下一 K-tile 的 4 个 loads
   - Phase 2: 在当前 LDS buffer 上执行 MFMA (loads in-flight)
   - Phase 3: 提交 loads 到 LDS
3. **关键**: 使用 branch-free 代码避免编译器插入 wait

**LDS 读取优化 (v4b)**:
- 问题: 编译器生成 `flat_load_dwordx2` 而非 `ds_read_b64`
- 解决: 使用 `__attribute__((address_space(3)))` 强制 LDS 地址空间
```cpp
auto* p = reinterpret_cast<const __attribute__((address_space(3))) T*>(
    (const __attribute__((address_space(3))) char*)nullptr + byte_off);
return *p;
```

### Phase 3: Buffer Load 硬件优化 (v4c)

**global_load vs buffer_load 对比**:

| 特性 | global_load | buffer_load |
|------|-------------|-------------|
| 地址模式 | Flat (64-bit per lane) | Buffer Resource Descriptor |
| TLB 访问 | 每 lane 独立查询 | 单次查询 (base in SGPRs) |
| 硬件 Prefetch | 无 | 有 |
| Cache 控制 | 有限 | GLC/SLC/DLC hints |

**实现**:
```cpp
// Buffer Resource Descriptor (128-bit, stored in SGPRs)
__amdgpu_buffer_rsrc_t rsrc = __builtin_amdgcn_make_buffer_rsrc(
    ptr, 0 /*stride*/, 0x7fffffff /*num_records*/, 0 /*flags*/);

// Buffer load with hardware prefetch
int4v r = __builtin_amdgcn_raw_buffer_load_b128(rsrc, voff, 0, 0);
```

**Assembly 验证**:
```asm
; Prologue: 4 consecutive buffer_loads, no wait between them
buffer_load_dwordx4 v[4:7], v4, s[4:7], 0 offen
buffer_load_dwordx4 v[8:11], v9, s[4:7], 0 offen
buffer_load_dwordx4 v[12:15], v12, s[8:11], 0 offen
buffer_load_dwordx4 v[16:19], v3, s[8:11], 0 offen
; ... compute happens here while loads in-flight ...
```

---

## 寄存器与 Occupancy

| 版本 | VGPRs | SGPRs | WGs/CU | Occupancy |
|------|-------|-------|--------|-----------|
| v2 | 126 | 48 | 4 | 100% |
| v4b | 128 | 50 | 4 | 100% |
| v4c | 120 | 50 | 4 | 100% |

所有版本均保持 4 WGs/CU 的高 occupancy。

---

## 与 CK 剩余差距分析

虽然 v4c 已超越 CK 单点性能，但 CK 还有以下优势可借鉴:

1. **L2-aware Tile Partitioning**: 根据 L2 cache 容量优化 tile 调度
2. **Persistent Kernel + Work Stealing**: 减少 kernel launch 开销
3. **Multi-stage Pipeline**: 3-4 stage software pipeline vs 我们的 2-stage

---

## 代码位置
- 优化后的 GEMM: `/home/xiaompen/AIInfra-Book/3rd/MMOE/benchmarks/mfma_gemm_bench.hip`
- 关键函数: `gemm_core_v4<kM, kN, kK, mode, kDS=true, kBuf=true>`

## 下一步
将 v4c GEMM 集成到 MoE super-kernel (`fused_moe_super_kernel.hip`) 的 Expert Compute Phase。
