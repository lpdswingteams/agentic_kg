"""Agentic governance orchestrator using LangChain agents and planning."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document

from .pipeline import AgenticGovernancePipeline, PipelineConfig
from .report import GovernanceReport, RetrievalEvaluation
from .strategies import ChunkingStrategyConfig, build_default_strategies


class PlanningResponse(BaseModel):
    """Structured output for the planner agent."""

    reasoning: str = Field(..., description="Why the chosen actions make sense")
    chosen_strategy: str = Field(..., description="Name of the chunking strategy to run")
    secondary_strategies: List[str] = Field(
        default_factory=list,
        description="Optional extra strategies to compare",
    )
    evaluation_focus: List[str] = Field(
        default_factory=list,
        description="What to pay attention to when evaluating",
    )


@dataclass
class AgenticGovernanceAgent:
    """Plan-then-act agent that chooses chunking and evaluation steps autonomously."""

    planner_model: BaseChatModel
    pipeline_config: PipelineConfig
    top_comparisons: int = 3

    def _build_system_prompt(self, strategies: Sequence[ChunkingStrategyConfig]) -> str:
        choices = "\n".join(
            f"- {s.name}: {s.description} (hints: {'; '.join(s.hints) if s.hints else '无'})"
            for s in strategies
        )
        return (
            "你是知识治理智能体，负责自主规划。\n"
            "根据文件结构与问题列表，选择合适的切片策略和评估关注点，"
            "使用 JSON 严格返回。\n"
            "可选策略列表:\n"
            f"{choices}"
        )

    def _build_user_message(
        self, strategies: Sequence[ChunkingStrategyConfig], doc_summary: str, questions: List[str]
    ) -> str:
        return (
            f"文档统计: {doc_summary}\n"
            f"预期问题: {questions}\n"
            "请基于可选策略生成规划，并返回结构化 JSON（包含 reasoning, chosen_strategy, secondary_strategies, evaluation_focus）。"
        )

    def _summarize_documents(self, docs: Sequence[Document]) -> str:
        lengths = [len(d.page_content) for d in docs]
        heading_density = 0.0
        if docs:
            from .strategies import _estimate_heading_density

            heading_density = _estimate_heading_density(docs)
        summary = "无文档"
        if lengths:
            summary = (
                f"{len(lengths)} 份文档，平均长度 {statistics.mean(lengths):.0f}，"
                f"最大 {max(lengths)}, 标题密度 {heading_density:.3f}"
            )
        return summary

    def plan(
        self,
        documents: Sequence[Document],
        questions: List[str],
        strategies: Sequence[ChunkingStrategyConfig],
    ) -> PlanningResponse:
        doc_summary = self._summarize_documents(documents)
        agent_graph = create_agent(
            self.planner_model,
            tools=None,
            system_prompt=self._build_system_prompt(strategies),
            response_format=PlanningResponse,
        )
        user_message = self._build_user_message(strategies, doc_summary, questions)
        result = agent_graph.invoke({"messages": [{"role": "user", "content": user_message}]})
        if "structured_response" in result and result["structured_response"] is not None:
            return result["structured_response"]
        messages = result.get("messages", [])
        if not messages:
            raise ValueError("Agent did not return any messages")
        content = getattr(messages[-1], "content", "")
        return PlanningResponse.model_validate_json(content)

    def _select_strategies(
        self,
        plan: PlanningResponse,
        strategies: List[ChunkingStrategyConfig],
    ) -> List[ChunkingStrategyConfig]:
        name_to_strategy: Dict[str, ChunkingStrategyConfig] = {s.name: s for s in strategies}
        selected: List[ChunkingStrategyConfig] = []
        if plan.chosen_strategy in name_to_strategy:
            selected.append(name_to_strategy[plan.chosen_strategy])
        for alt in plan.secondary_strategies:
            if alt in name_to_strategy and name_to_strategy[alt] not in selected:
                selected.append(name_to_strategy[alt])
        if not selected:
            selected.append(strategies[0])
        return selected[: self.top_comparisons]

    def run(self, files: List[str], questions: List[str]) -> GovernanceReport:
        base_pipeline = AgenticGovernancePipeline(self.pipeline_config)
        documents = base_pipeline.load_documents(files)
        strategies = build_default_strategies(
            documents,
            embedding_model=self.pipeline_config.embedding_model,
            embedding_dimension=self.pipeline_config.semantic_dimension,
        )
        plan = self.plan(documents, questions, strategies)
        chosen = self._select_strategies(plan, strategies)

        # Reuse the evaluation logic from the pipeline while keeping the agent in control.
        evaluations: List[RetrievalEvaluation] = []
        for strategy in chosen:
            evaluations.append(
                base_pipeline._evaluate_strategy(strategy, documents, questions)
            )

        selected = max(evaluations, key=lambda ev: ev.weight * ev.mean_coverage)
        recommendations = plan.evaluation_focus if plan.evaluation_focus else None

        return GovernanceReport(
            source_files=files,
            questions=questions,
            evaluations=evaluations,
            selected_strategy=selected,
            plan_summary=plan.reasoning,
            plan_recommendations=recommendations,
        )

