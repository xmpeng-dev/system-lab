# Llama2-70B LoRA — 191 ms idle 真因定位 · Layer 1：单进程 DataLoader + 主线程 collate

| 字段 | 值 |
| --- | --- |
| 日期 | 2026-04-29 |
| 模型 / 配方 | Llama-2-70B · LoRA SFT · `bf16_with_fp8_hybrid` · 8 × MI355X |
| 分析 trace | `/results/torch_profiler_traces/smci355-...145319.1777433707462544573.pt.trace.json` |
| 分析 step | `ProfilerStep#82`（窗口 ts=2355147800116, dur=1626.45 ms） |
| 配套 | [`2026-04-29_llama2_70b_lora_baseline_trace_breakdown.md`](./2026-04-29_llama2_70b_lora_baseline_trace_breakdown.md) §3 / §6 |
| 核心结论 | **每个 step 头部 ~191 ms idle = `_SingleProcessDataLoaderIter.__next__` 在主线程上同步跑 collate**，其中 ~125 ms 花在为 8192×8192 causal mask 调 `aten::ones` + `aten::tril` + `aten::lt`；GPU 完全空闲等数据。 |

---

## 1. 症状回顾

`full_breakdown.py` 在 `ProfilerStep#82` 给出的关键量：

```
=== Compute / NCCL overlap (slot=50us) ===
  compute-only              1380.35 ms
  nccl-only                   54.40 ms
  overlap(compute & nccl)      0.20 ms
  idle                       191.55 ms      ← 这一行
```

时间分箱（每 bin = 20.3 ms）显示 **bins 0-8（0 → 162.6 ms）GPU 几乎完全空闲**，第 9 个 bin（t=183 ms）才出现首批 GEMM/attn kernel：

```
bin   t(ms)    gemm  gGEMM   attn   norm  ...   total
  0     0.0    0.00   0.00   0.00   0.00         0.02
  1    20.3    0.00   0.00   0.00   0.00         0.00
  ...                      （8 个全零 bin）
  8   162.6    0.00   0.00   0.00   0.00         0.00
  9   183.0    0.36   0.00   2.76   0.26        16.50  ← 首批 kernel
```

— 所以 191 ms idle **不是均匀洒在 step 里**，而是**集中在 step 头部**。Layer 1 假设：**单进程 DataLoader 在主线程同步执行 `collate_fn`**，期间 GPU 没事可干。

## 2. Layer 1 假设的证据链

需要分别证明 4 件事，缺一不可：

| # | 命题 | 证据来源 |
| ---: | --- | --- |
| 2.1 | recipe 真把 `num_workers=0` 喂给了 PyTorch DataLoader | recipe.py + run.log |
| 2.2 | DataLoader 在 `num_workers=0` 时会在主线程跑 collate | PyTorch 源码（已知行为） + trace 中的 `_SingleProcessDataLoaderIter.__next__` user_annotation |
| 2.3 | 这个 collate_fn 包含 8192×8192 因果 mask 构建 | sft.py + recipe 的 `dataset_kwargs` |
| 2.4 | 这条主线程路径在每个 step 头部消耗 ~180 ms | trace 里同步排列的 `aten::ones` / `aten::tril` / `aten::lt` / `aten::stack` 时间戳 |

### 2.1 num_workers=0 真到了 DataLoader

**配方默认值**（`Megatron-Bridge/.../recipes/primus/recipes/llama2_custom.py:534-545`）：

```541:545:Primus-dev/third_party/Megatron-Bridge/src/megatron/bridge/recipes/primus/recipes/llama2_custom.py
            num_workers=1,
            do_test=False,
            do_validation=True,
            dataset_kwargs={"return_cu_seqlen": False},
        )
```

> 注意：配方原始默认 `num_workers=1`，但下方还有一段从 env 读 `PRIMUS_DL_NUM_WORKERS` 覆盖的逻辑。baseline 跑时 env 没设置 → 默认 `0`（不是 1）：

```555:559:Primus-dev/third_party/Megatron-Bridge/src/megatron/bridge/recipes/primus/recipes/llama2_custom.py
    import os as _os
    _dl_workers = int(_os.environ.get("PRIMUS_DL_NUM_WORKERS", "0"))
    _dl_prefetch = int(_os.environ.get("PRIMUS_DL_PREFETCH_FACTOR", "2"))
    _dl_persistent = _os.environ.get("PRIMUS_DL_PERSISTENT_WORKERS", "0") == "1"
    dataset_cfg.num_workers = _dl_workers
```

**run.log 反查**（`baseline_trace_run.log:917, 925`）：

```
module_utils.py:227] :   dataset.num_workers          : 0 (int)
module_utils.py:227] :   dataset.persistent_workers   : False (bool)
```

`yaml dump` 段也再确认一遍（`baseline_trace_run.log:2258-2272`）：

```
memmap_workers: 1
num_workers: 0
persistent_workers: false
```

✅ 命题 2.1 成立。

### 2.2 DataLoader 在主线程跑 collate

DataLoader 构造（`Megatron-Bridge/.../data/samplers.py:126-134`）：

```126:134:Primus-dev/third_party/Megatron-Bridge/src/megatron/bridge/data/samplers.py
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
    )
```

PyTorch 行为（已知 / `torch/utils/data/dataloader.py`）：

- `num_workers=0` → `_SingleProcessDataLoaderIter` 接管。
- `next(iter)` 在**调用线程**里依次调用 `dataset.__getitem__()` 和 `collate_fn(batch)`，**没有 prefetch、没有 worker 进程**。
- `pin_memory=True` 此时只是把结果搬到 pinned memory，时间相对小。

trace 直接给了证据 — 每个 step 的开头都有一个跨度极大的 user_annotation：

| step | ts (us) | dur (us) | dur (ms) |
| --- | ---: | ---: | ---: |
| 80 | 2355144546028 | 179,434 | **179.4** |
| 81 | 2355146174980 | 180,790 | **180.8** |
| **82** | **2355147800440** | **180,588** | **180.6** |
| 83 | 2355149426900 | 179,765 | **179.8** |
| 84 | 2355151052982 | 179,412 | **179.4** |

这些事件的 `cat: user_annotation` / `name: enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__` 由 PyTorch profiler 自动注入，**只在 `num_workers=0` 时出现这个名字**（多进程模式是 `_MultiProcessingDataLoaderIter`）。

✅ 命题 2.2 成立。

### 2.3 collate_fn 真的会构建 8192×8192 mask

recipe 端的关键开关（`llama2_custom.py:544`）：

```544:544:Primus-dev/third_party/Megatron-Bridge/src/megatron/bridge/recipes/primus/recipes/llama2_custom.py
            dataset_kwargs={"return_cu_seqlen": False},
```

`packed_sequence=True` + `dataset_kwargs={"return_cu_seqlen": False}` ⇒ 实际跑的是
`GPTSFTPackedDataset`（`sft.py:739`），其 `collate_fn` 在 `return_cu_seqlen` 分支选择上**走了 else 分支**（`sft.py:996-1002`）：

```996:1002:Primus-dev/third_party/Megatron-Bridge/src/megatron/bridge/data/datasets/sft.py
        else:
            attention_mask = [self._create_attention_mask(max_length) for _ in batch]
            processed_batch.update(
                {
                    "attention_mask": torch.stack(attention_mask),
                }
            )
```

`_create_attention_mask` 是经典的 fp32 dense 因果 mask 构造（`sft.py:662-672`）：

```662:672:Primus-dev/third_party/Megatron-Bridge/src/megatron/bridge/data/datasets/sft.py
    @torch.no_grad()
    def _create_attention_mask(self, max_length):
        """Creates an upper-triangular causal attention mask.
        Args:
            input_ids: A 1D tensor that holds the indices of tokens.
        """
        # seq_length = len(input_ids)
        # `attention_mask` has the shape of [1, seq_length, seq_length]
        attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
        attention_mask = attention_mask < 0.5
        return attention_mask
```

`max_length = self.max_seq_length = 8192`（packed），所以每个 sample 都要：

1. **`torch.ones((8192, 8192))`** — 默认 `dtype=float32`，**256 MB CPU 内存分配 + 256 MB 写零**
2. **`torch.tril(...)`** — 256 MB tensor 上把上三角清零
3. **`.unsqueeze(0)`** — view，免费
4. **`< 0.5`** — 256 MB fp32 → 64 MB bool 比较
5. **`torch.stack([...])`** — 把 batch 里的 mask 拼起来

> 班级歧视：`GPTSFTPackedDataset` 自己的 docstring（`sft.py:754-756`）就警告过："This flag should be True unless you have a specific use case." — recipe 把它显式设成 False，等于刻意走了**反直觉的 fallback**。

✅ 命题 2.3 成立。

### 2.4 这条主线程路径吃掉了 step 头部 180 ms

把 step 82 的主线程 cpu_op 按时间戳排出来（绝对 `ts` 减去 `ProfilerStep#82` 的起点 `2355147800116`）：

| 偏移（ms 起算 step 起点） | 事件 (`pid=tid=145319`，主线程) | dur (ms) | 累积 (ms) |
| ---: | --- | ---: | ---: |
| **+0.32** | `enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__` 起 | **180.6** (整个跨度) | — |
| +2.28 | `aten::ones` （8192×8192 fp32） | **37.8** | 40.1 |
| +40.10 | `aten::tril` | **58.4** | 98.5 |
| +112.69 | `aten::lt` （`< 0.5`） | **29.1** | 141.6 |
| +156.14 | `aten::stack` | **15.3** | 171.4 |
| +180.92 | DataLoader.__next__ 返回；首个 GPU kernel ~ +183 ms | — | — |

总计 ~141 ms 是 mask 构建（ones+tril+lt），剩余 ~40 ms 是 dataloader prefetch / `_collate_item` / token & label 列表打包 + pin memory 拷贝。

把上面五条事件的 dur 加起来：37.8 + 58.4 + 29.1 + 15.3 = **140.6 ms** — 仅 mask 4 步就吃掉 step 的 8.6%；外加另外 ~40 ms 的非 mask collate，共 180.6 ms = `_SingleProcessDataLoaderIter.__next__` 整段的 dur。

每个 ProfilerStep 都重复同样的图（见 2.2 表），所以 idle 不是抖动，是结构性的。

```
trace 中的关键 grep（直接复现，不依赖工具链）：
  grep -n '"name": "aten::ones"'   trace.pt.trace.json | head
  grep -n '"name": "aten::tril"'   trace.pt.trace.json | head
  grep -n '"name": "aten::lt"'     trace.pt.trace.json | head
  grep -n '_SingleProcessDataLoaderIter.__next__' trace.pt.trace.json | head
```

✅ 命题 2.4 成立。

## 3. 把 191 ms 拆开

| 来源 | ms | 解释 |
| --- | ---: | --- |
| `aten::ones((8192,8192))` × 1 | 37.8 | fp32 256 MB 分配 + 写零（malloc + memset） |
| `aten::tril` × 1 | 58.4 | 串行扫 256 MB 写零（main-thread C++ 实现，单线程） |
| `aten::lt(< 0.5)` × 1 | 29.1 | 256 MB fp32 → 64 MB bool 比较 |
| `aten::stack` × 1 | 15.3 | mbs=1 但仍走 stack（list-of-tensor → tensor） |
| 其他 collate（_collate_item / pin / hostalloc / RNG） | ~40 | 包含 `enumerate.__next__` 的剩余 dur |
| **合计** | **180.6** | DataLoader.__next__ 跨度 |
| Trace breakdown 报告的 idle | 191.6 | 多出来的 ~11 ms ≈ 启动首个 GEMM 之前的 dispatcher / launch / 第一次 kernel JIT |

吻合精度优于 6%。

## 4. 为什么这是结构性问题（不是抖动 / 不是噪声）

1. **5 个 ProfilerStep 全一致**（179.4 / 180.8 / 180.6 / 179.8 / 179.4 ms）— 不是 GC、不是磁盘 I/O 抖动。
2. **CPU 全部跑在主线程**（`pid=145319 tid=145319`）— `num_workers=0` 没有 worker 进程，没有 prefetch。
3. **GPU 完全干等**（bins 0-8 总 GPU work = 0.02 ms）— 不是 H2D 拷贝阻塞、不是 cudaMalloc 卡顿。
4. **mask 内容每个 step 完全相同**（max_length=8192, mbs=1）— 这 140 ms 是**纯重复**的浪费；真正只需要算一次然后 cache。

## 5. 这个设计为什么会出现 — intent vs. 实际

`_create_attention_mask` 的设计 intent 是给**普通（非 packed）short-seq SFT** 用的：mbs 几个、seq 几百，每 step 几毫秒可以接受。问题在于 `GPTSFTPackedDataset` 继承了它的 `collate_fn`，而 packed 模式下 `max_length` 永远等于 `self.max_seq_length`（这里 8192），且**实际不需要这个 mask** — packed 训练正确做法是用 `cu_seqlens` 喂进 THD attention，让 attention kernel 自己决定哪些 token 跨段不能看。

recipe 里 `dataset_kwargs={"return_cu_seqlen": False}` 是**把这条正确路径主动关了**，等于让一个 packed dataset 用 non-packed 的 mask 路径 —— 既慢，又有 packed 段间互相 attention 的语义 bug（不同 sample 拼到同一个 packed seq 里，dense causal mask 会让它们互相可见）。

类自己的 docstring（`sft.py:754-756`）已经写得很清楚：

> `return_cu_seqlen: ... This flag should be True unless you have a specific use case.`

这条 recipe 没有"specific use case" — 只是**配错了**。

## 6. 修复路径

| 路径 | 改动 | 预期收益 | 风险 / 代价 |
| ---: | --- | --- | --- |
| **A** | `num_workers ≥ 2` + `persistent_workers=true` + `prefetch_factor=2` | 把整段 ~180 ms 挪到 worker 进程里跟 fwd/bwd 重叠 → idle ≈ 0 | **fork-after-CUDA 死锁**（已实测踩过）— 70B base weights 已在父进程建立 CUDA context，`fork` 出来的 worker 一旦碰 CUDA 就死锁。需要切 `multiprocessing.set_start_method("spawn")`，但 spawn 又要重新加载 dataset / 重 import，启动慢；并且 spawn 与 megatron-bridge 的 distributed 初始化路径冲突需另查 |
| **B** ✅ 推荐 | `dataset_kwargs={"return_cu_seqlen": True}` | 直接走 `if self.return_cu_seqlen:` 分支（`sft.py:955-995`），**完全跳过 ones/tril/lt/stack**；attention 用 THD + cu_seqlens；不仅省 idle，还**修了 packed 段间 cross-attention 的语义 bug** | 模型侧需要支持 `cu_seqlens` 输入（Megatron-Bridge 的 GPTModel 已支持，这是 packed 训练默认推荐）；要做一次 1:1 A/B 验证 loss 曲线一致 |
| **C** | 在 dataset 上 cache mask（`functools.lru_cache` 或 `__init__` 时算一次） | mask 形状只有一种 (8192,8192) → 缓存命中率 100%，省 140 ms | 256 MB fp32 常驻 CPU 内存（每 worker 一份），但 mbs=1 这成本可接受；治标不治本，没修语义 bug |
| **D** | 把 mask 用 `torch.ones((s,s), dtype=torch.bool)` + `tril` 直接生成 bool | dur 减半（128 MB 而不是 256 MB） | 同上，治标不治本 |

**结论**：路径 B 是正解，零额外内存、零进程拓扑变化、还顺手修语义 bug。本 note 落盘后下一步直接做 B。

## 7. Layer 2 / Layer 3 的存在性（占位，留给后续 note）

为了完整性，把"如果 Layer 1 修了之后还可能剩什么"列出来作为 follow-up 索引：

- **Layer 2 — 首个 GPU kernel launch 延迟**：即使 collate 在 worker 跑完了，主线程从 `iter()` 拿到 batch → H2D 拷贝 → 第一个 GEMM dispatch 之间还有 ~5-15 ms 的 launch/JIT。修法：CUDA Graph capture / pre-warm 第一批 kernel。
- **Layer 3 — `Sampler.__next__` 自己**：Megatron `MegatronPretrainingBatchSampler` 在每 step 算 indices，正常很快但若 packed metadata 解析复杂可能 1-3 ms。需要单独抓 `cpu_op` 里的 sampler 调用确认。

这两层只有在 Layer 1 修完、idle 仍 > 30 ms 时才需要看。

## 8. 复现 / 重测命令

```bash
# 1) 抓 baseline trace（已存档于 results/baseline_trace_run）
docker exec xiaoming-mlperf-llama bash -c '
  cd /home/xiaompen/mlperf-training-llama/llama2_sft/primus &&
  TRACE=1 PRIMUS_TRAIN_ITERS=100 PRIMUS_EVAL_INTERVAL=9999 \
  PRIMUS_PROFILE_STEP_START=80 PRIMUS_PROFILE_STEP_END=85 \
  RUN_LOG_FILE=/results/baseline_trace_run/baseline_trace_run.log \
  bash run.sh'

# 2) 验证 num_workers / collate 路径
grep -E "num_workers|persistent_workers" \
  /results/baseline_trace_run/baseline_trace_run.log | head

# 3) 直接 grep trace JSON 复现 §2.4 表格（无需任何 Python 工具）
TRACE=/results/torch_profiler_traces/smci355-...145319.....pt.trace.json
for op in '_SingleProcessDataLoaderIter.__next__' \
          'aten::ones' 'aten::tril' 'aten::lt' 'aten::stack'; do
  echo "--- $op ---"
  grep -n "\"name\": \"$op\"" "$TRACE" -A1 | head -10
done

# 4) 验证 §1 的 idle 数字
python3 .cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py \
  "$TRACE" ProfilerStep#82 | grep -A6 "Compute / NCCL overlap"
```

## 9. 相关文件

- `Primus-dev/third_party/Megatron-Bridge/src/megatron/bridge/recipes/primus/recipes/llama2_custom.py` — recipe（`num_workers` env 覆盖逻辑、`return_cu_seqlen=False`）
- `Primus-dev/third_party/Megatron-Bridge/src/megatron/bridge/data/samplers.py:126-134` — `DataLoader` 构造
- `Primus-dev/third_party/Megatron-Bridge/src/megatron/bridge/data/datasets/sft.py:662-672` — `_create_attention_mask`
- `Primus-dev/third_party/Megatron-Bridge/src/megatron/bridge/data/datasets/sft.py:739-1004` — `GPTSFTPackedDataset.collate_fn`，含 `return_cu_seqlen` 分支
- `llama2_sft/primus/results/baseline_trace_run/baseline_trace_run.log` — 配置 dump
- `llama2_sft/primus/results/baseline_trace_run/breakdown_step82.txt` — overlap / 时间分箱
- `llama2_sft/primus/results/torch_profiler_traces/smci355-...145319.....pt.trace.json` — 5 个 ProfilerStep 的 cpu_op 时间戳

## 10. 接下来

下一篇 note：执行路径 B（`return_cu_seqlen=True`），跑 100-iter A/B，验证：

1. step time：期望从 1626 ms → ~1450-1500 ms（-130~-180 ms）
2. idle 分箱：期望 bins 0-8 不再全零，idle 总量 < 20 ms
3. loss 曲线：与 baseline 1:1 对齐（packed 语义其实更对，loss 可能略好）
4. VRAM：期望 Pmax 略降（少一份 256 MB fp32 mask × 8 rank = -2 GB 节奏）
