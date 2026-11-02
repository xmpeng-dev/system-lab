# Turns

这里保存 A → B → C → A 环形讨论的逐次发言。

## 排序

文件名：

```text
YYYY-MM-DD-HHMMSS-<run-slug>-NNNN-<role>.md
```

- `run-slug` 区分并发 Cursor 会话；
- `NNNN` 在一个 run 内单调递增；
- `<role>` 为 `a`、`b` 或 `c`；
- 按文件名排序就是发言顺序；
- 每个文件只包含一个角色的一次发言。

## 角色

- `a`：GPT-5.6 Sol，主持与提案；
- `b`：Compiler Advocate；
- `c`：Complexity Skeptic。

调度状态见同级目录的 `state-<run-id>.json`，完整协议见
[`../protocol.md`](../protocol.md)。历史文件只追加、不改写。启用 per-run state 前的
文件没有 `run-slug`，按 frontmatter 和 `responds_to` 链归属。
