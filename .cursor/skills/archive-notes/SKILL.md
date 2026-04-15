---
name: archive-notes
description: >-
  Archive analysis, research notes, and technical findings into the notes/
  directory organized by date. Use when the user asks to save, archive,
  record, or write up analysis results, or says "归档", "记录", "保存笔记",
  "写到 notes".
---

# Archive Notes

将当前对话中的分析、调研、实验结果等内容归档到 `notes/` 目录。

## 目录结构

```
notes/
└── YYYY-MM/
    └── YYYY-MM-DD_topic_slug.md
```

- 按 **年-月** 分子目录：`notes/2026-04/`
- 文件名格式：`YYYY-MM-DD_简短英文主题_slug.md`（下划线分隔，全小写）
- 日期取**今天的日期**

## 归档流程

1. **确定内容**：从当前对话中提取要归档的分析内容。如果对话中有多个独立主题，分别归档为不同文件。
2. **生成文件名**：`YYYY-MM-DD_topic.md`，topic 用简短英文，下划线分隔，例如 `2026-04-14_moe_e2e_performance_benchmark.md`
3. **创建目录**：确保 `notes/YYYY-MM/` 目录存在
4. **写入内容**：按下方模板组织内容
5. **确认**：告知用户文件路径

## 内容模板

```markdown
# 标题（中文）

**日期**: YYYY-MM-DD

## 背景 / 目标

简要说明分析的背景和目标。

## 主要发现 / 结论

核心结论，用表格或列表呈现关键数据。

## 详细分析

分节展开，包含：
- 实验配置（硬件、软件、参数）
- 数据/表格/代码片段
- 原因分析

## 下一步 / 建议

（可选）后续行动建议。

## 相关文件

（可选）关联的代码、文档路径。
```

## 内容规范

- **语言**：正文用中文，代码/命令/技术术语保留英文原文
- **数据优先**：尽量用表格呈现定量数据，避免纯文字描述性能数字
- **精简**：去除对话中的试探、纠错、重复内容，只保留最终结论和关键推导
- **可追溯**：保留实验命令、脚本路径、环境信息，方便日后复现
- **自包含**：读者无需回溯对话即可理解全部内容

## 示例

用户说："把今天的 MoE overlap 分析归档一下"

→ 创建 `notes/2026-04/2026-04-14_moe_comm_overlap_analysis.md`，内容从对话中提取整理。
