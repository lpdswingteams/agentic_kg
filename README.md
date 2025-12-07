# Agentic Knowledge Governance (LangChain ≥1.0)

一个最小可运行的示例，用于基于 LangChain 1.0.x 构建“agentic”知识治理智能体：

- 通过规划型 Agent 自主决策：根据文件与问题列表选择切片策略与评估关注点（使用 `langchain.agents.create_agent` 的最新接口）。
- 自动检测文档结构，生成多种切片方案（字符、递归、语义）。
- 以向量检索为核心，根据给定问题做召回并计算覆盖度。
- 自动对比策略、挑选最佳方案并生成 Markdown 报告。
- Agent 可调度的工具：PDF 文本提取、OCR 解析、版面分析、通用预览等，便于在规划阶段快速检查原始文件。

## 安装依赖

```bash
pip install -r requirements.txt
```

> 核心依赖已锁定在 LangChain 1.0.x 版本（`<1.1`），并配套使用与之兼容的 text-splitters 版本；如需语义切分可额外安装 `langchain-experimental` 与其兼容版本，请勿升级到 1.1+ 以免破坏接口兼容性。
> 默认示例使用 `FakeEmbeddings`，生产环境请替换为 OpenAI、bge、GTE 等真实 embedding 模型。
> 工具基于 `langchain_core.tools.tool` 声明，并通过 `create_agent` 自动挂载，可根据业务问题列表动态决策策略。

## 快速开始

```bash
python example_usage.py
```

- 将 `example_usage.py` 中的 `files` 修改为用户上传的文件列表。
- 将 `questions` 修改为用户可能提出的问题列表。
- 替换示例中的 `FakeListChatModel` 为你的聊天模型（如 `ChatOpenAI`）。
- 运行后会打印一份对比报告，展示 Agent 规划、策略覆盖度和推荐方案。

## 核心类

- `PipelineConfig`：配置 embedding 模型与检索 top_k 等。
- `AgenticGovernanceAgent`：规划要执行的治理步骤并调用内部管线完成评估，输出 `GovernanceReport`。
- `AgenticGovernancePipeline`：被 Agent 调用以加载文件、生成策略、构建向量库、评估并输出 `GovernanceReport`。
- `GovernanceReport`：提供 `to_markdown()` 方法生成易读的汇总报告。

## 扩展点

- **自定义 Loader**：在 `AgenticGovernancePipeline` 初始化时通过 `loaders` 参数传入文件后缀到 Loader 的映射。
- **自定义切片策略**：传入自定义的 `ChunkingStrategyConfig` 列表，或修改 `build_default_strategies` 以添加新的切割方式（如章节解析、表格拆分等）。
- **评估指标**：默认使用基于词覆盖度的快速打分，亦可扩展为基于 LLM 的答案匹配或基准问答评测。
