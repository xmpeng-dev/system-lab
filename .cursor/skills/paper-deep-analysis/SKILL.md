---
name: paper-deep-analysis
description: >-
  Deep analysis of research papers with detailed problem background, quantitative 
  performance metrics, industry solution comparison, and full Chinese translation.
  Use when the user asks to deeply analyze a paper, compare with industry solutions, 
  translate a paper, or wants detailed problem analysis with specific performance numbers.
---

# Paper Deep Analysis

## When to Apply

Use this skill when the user needs any of the following:
- Deep understanding of paper's problem background and solution
- Specific performance data and experimental results
- Comparison with similar industry solutions
- Full Chinese translation of the paper

## Inputs

1. **Source**: PDF path, arXiv URL, paper webpage link, or pasted paper content
2. **Language**: Output in Chinese by default, unless user specifies otherwise

## Workflow

1. **Obtain paper content**: Read PDF or fetch webpage content
2. **Structured analysis**: Extract key information following the template
3. **External search**: Use WebSearch to find similar industry solutions
4. **Full translation**: Translate paragraph by paragraph, maintaining academic accuracy
5. **Integrated output**: Output complete analysis following the template

## Output Template (Must Follow Strictly)

Output as a single markdown document with four main sections:

```markdown
# [论文标题]
# [Paper Title (Original)]

> **arXiv/DOI:** [论文编号](链接) | **PDF:** [链接]  
> **发表信息:** [会议/期刊名称], [年份]  
> **机构:** [作者单位]  
> **代码:** [GitHub 链接，如有]  
> **领域:** [研究领域标签，用 · 分隔]  
> **核心贡献:** [一句话总结核心贡献和关键数据]

---

## 一、问题分析 (Problem Analysis)

### 1.1 研究背景 (Research Background)

**领域现状** (Current State of the Field):
- [Current technical development in this field, 3-5 points]

**核心挑战** (Core Challenges):
- [Main challenges and bottlenecks in this field, 2-4 points]

**研究动机** (Research Motivation):
- [Why this research is needed, limitations of existing methods]

### 1.2 问题定义 (Problem Definition)

**具体问题** (Specific Problem):
[Clearly describe the specific problem the paper addresses, including:]
- What is the input
- What is the output
- What are the constraints
- What are the evaluation metrics

**问题形式化** (Problem Formalization, if applicable):
[Mathematical formulation in LaTeX format]

### 1.3 解决方案 (Solution)

**核心思路** (Core Idea):
[One paragraph summarizing the paper's key innovation and approach]

**方法概述** (Method Overview):
1. [Method step 1 - detailed description]
2. [Method step 2 - detailed description]
3. [Continue...]

**技术细节** (Technical Details):

*[Key Component 1 Name]*:
- Function:
- Implementation:
- Innovation:

*[Key Component 2 Name]*:
- Function:
- Implementation:
- Innovation:

**算法/架构描述** (Algorithm/Architecture Description):
[Detailed text description of the core algorithm flow or architecture diagram]

---

## 二、实验效果 (Experimental Results)

### 2.1 实验设置 (Experimental Setup)

| Item | Details |
|------|---------|
| Datasets | [Dataset names, scale, characteristics] |
| Baselines | [List of comparison baseline methods] |
| Metrics | [Evaluation metrics used] |
| Hardware | [GPU/TPU model, quantity, training time] |

### 2.2 主要结果 (Main Results)

**核心性能指标** (Core Performance Metrics):

| Method | Metric 1 | Metric 2 | Metric 3 | ... |
|--------|----------|----------|----------|-----|
| Baseline 1 | xx.x | xx.x | xx.x | |
| Baseline 2 | xx.x | xx.x | xx.x | |
| **This Paper** | **xx.x** | **xx.x** | **xx.x** | |
| Improvement | +x.x% | +x.x% | +x.x% | |

**关键发现** (Key Findings):
- [Most important finding 1, with specific numbers]
- [Most important finding 2, with specific numbers]
- [Most important finding 3, with specific numbers]

### 2.3 消融实验 (Ablation Study)

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| Full method | xx.x | - |
| w/o Component A | xx.x | Performance drops x.x% |
| w/o Component B | xx.x | Performance drops x.x% |

**消融结论** (Ablation Conclusions):
- [Which component is most important, how much it contributes]

---

## 三、业界类似方案 (Industry Similar Solutions)

### 3.1 方案对比表 (Solution Comparison Table)

| Solution | Year | Core Idea | Advantages | Disadvantages | Performance |
|----------|------|-----------|------------|---------------|-------------|
| [Solution 1] | 20xx | [Brief] | | | |
| [Solution 2] | 20xx | [Brief] | | | |
| [Solution 3] | 20xx | [Brief] | | | |
| **This Paper** | 20xx | [Brief] | | | |

### 3.2 技术路线对比 (Technical Approach Comparison)

**路线A (Approach A): [Technical approach name]**
- Representative works: [List representative papers/methods]
- Core idea:
- Pros and cons:

**路线B (Approach B): [Technical approach name]**
- Representative works:
- Core idea:
- Pros and cons:

### 3.3 本文定位 (This Paper's Position)

- **Improvement over Approach A**:
- **Improvement over Approach B**:
- **Unique contributions**:

### 3.4 推荐进一步阅读 (Recommended Further Reading)

| Paper | Reason |
|-------|--------|
| [Paper 1 title] | [Why worth reading] |
| [Paper 2 title] | [Why worth reading] |
| [Paper 3 title] | [Why worth reading] |

---

## 四、全文翻译 (Full Translation)

> Below is the full Chinese translation of the paper, maintaining original structure and paragraph divisions.
> Technical terms are annotated with English original on first occurrence.

### 摘要 (Abstract)

[Translation content]

### 1. 引言 (Introduction)

[Translation content, maintaining original paragraph structure]

### 2. 相关工作 (Related Work)

[Translation content]

### 3. 方法 (Method)

[Translation content, including all subsections]

### 4. 实验 (Experiments)

[Translation content]

### 5. 结论 (Conclusion)

[Translation content]

### 参考文献 (References)

[Selectively translate key reference titles, or note "References omitted"]

---

## 附录 (Appendix)

### A. 术语表 (Glossary)

| English Term | Chinese Translation | Explanation |
|--------------|---------------------|-------------|
| [term] | [translation] | [brief explanation] |

### B. 复现检查清单 (Reproducibility Checklist)

- [ ] Code open-sourced: [Yes/No, link]
- [ ] Data available: [Yes/No, notes]
- [ ] Hyperparameters complete: [Yes/No]
- [ ] Random seeds: [Yes/No, value]
- [ ] Hardware requirements: [description]
```

## Translation Guidelines

### Term Handling
- First occurrence of technical terms: Chinese translation (English Term)
- Subsequent occurrences: Use Chinese directly or common abbreviations
- Terms without established translation: Keep English with annotation

### Translation Style
- Maintain rigor of academic language
- Sentence structure can be adjusted for Chinese conventions without changing meaning
- Keep formulas and symbols unchanged
- Figure/table titles should be translated; content may remain in English

### Special Handling
- Code blocks: Keep original, may add Chinese comments
- Algorithm pseudocode: Keep keywords in English, translate comments
- Abbreviations: Give full form and Chinese translation on first occurrence

## Quality Requirements

1. **Problem Analysis**: Must be detailed enough for readers to understand innovations without reading original
2. **Performance Data**: All numbers must come from the original paper, no fabrication
3. **Industry Comparison**: Use WebSearch to ensure accuracy and timeliness
4. **Translation**: Accurate technical terms, fluent sentences

## Notes

- If paper lacks certain information, explicitly mark "Not provided in original"
- For preprints, note version number (e.g., arXiv v1, v2)
- If full text unavailable, note in translation section "Based on abstract/available portions only"
