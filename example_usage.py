"""Demo: run agentic governance agent over local files and question list."""
from __future__ import annotations

from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agentic_kg import AgenticGovernanceAgent, PipelineConfig


class ToolAwareFakeListChatModel(FakeListChatModel):
    """Fake model that supports tool binding for demos without external APIs."""

    def bind_tools(self, tools, *, tool_choice=None, parallel_tool_calls=None, **kwargs):  # type: ignore[override]
        return self


def main():
    # Replace FakeEmbeddings and ToolAwareFakeListChatModel with production models (OpenAI, bge, etc.).
    config = PipelineConfig(embedding_model=FakeEmbeddings(size=1536))
    # Tool-aware FakeListChatModel simulates a planner reply; swap in ChatOpenAI or your chat model.
    planner = ToolAwareFakeListChatModel(responses=[
        '{"reasoning":"README 提到切片与检索，优先用递归切分以保留结构","chosen_strategy":"recursive","secondary_strategies":["semantic"],"evaluation_focus":["章节边界","覆盖度"]}'
    ])

    agent = AgenticGovernanceAgent(planner_model=planner, pipeline_config=config)
    # Agent will call built-in tools (PDF 提取、OCR、版面分析等) 在规划时检查文件。

    files = ["README.md"]  # replace with uploaded files
    questions = [
        "这个项目的目标是什么？",
        "如何配置切片策略？",
        "我可以用什么方式做召回评估？",
    ]

    report = agent.run(files, questions)
    print(report.to_markdown())


if __name__ == "__main__":
    main()
