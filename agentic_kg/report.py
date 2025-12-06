"""Reporting utilities for the governance pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from langchain_core.documents import Document


@dataclass
class RetrievalEvaluation:
    strategy_name: str
    description: str
    question_results: Sequence[Dict]
    chunk_stats: Dict[str, float]
    mean_coverage: float
    weight: float
    hints: List[str]

    def score(self) -> float:
        return self.mean_coverage * self.weight


@dataclass
class GovernanceReport:
    source_files: List[str]
    questions: List[str]
    evaluations: List[RetrievalEvaluation]
    selected_strategy: RetrievalEvaluation
    plan_summary: str | None = None
    plan_recommendations: List[str] | None = None

    def to_markdown(self) -> str:
        lines = ["# 知识治理策略评估报告", ""]
        lines.append("## 输入")
        lines.append(f"- 文件: {', '.join(self.source_files)}")
        lines.append(f"- 预期问题: {len(self.questions)} 个")
        lines.append("")

        if self.plan_summary:
            lines.append("## 规划概览")
            lines.append(self.plan_summary)
            if self.plan_recommendations:
                lines.append("- 附加建议: " + "; ".join(self.plan_recommendations))
            lines.append("")

        lines.append("## 策略对比")
        for ev in self.evaluations:
            lines.append(f"### {ev.strategy_name} ({ev.score():.3f})")
            lines.append(f"描述: {ev.description}")
            if ev.hints:
                lines.append(f"提示: {'; '.join(ev.hints)}")
            lines.append(
                f"切片统计: {int(ev.chunk_stats['count'])} 块, 平均长度 {ev.chunk_stats['avg_length']:.0f}, std {ev.chunk_stats['std_length']:.0f}"
            )
            lines.append(f"平均覆盖度: {ev.mean_coverage:.3f}")
            lines.append("逐问题评估:")
            for q in ev.question_results:
                best_snippet = _extract_snippet(q["hits"])
                lines.append(f"- 问题: {q['question']}")
                lines.append(f"  覆盖度: {q['coverage']:.3f}")
                if best_snippet:
                    lines.append(f"  片段: {best_snippet}")
            lines.append("")

        lines.append("## 推荐策略")
        best = self.selected_strategy
        lines.append(f"> 推荐使用 **{best.strategy_name}**，综合评分 {best.score():.3f}。")
        lines.append(
            f"> 原因：覆盖度 {best.mean_coverage:.3f}，切片数 {int(best.chunk_stats['count'])}，平均长度 {best.chunk_stats['avg_length']:.0f}。"
        )
        if best.hints:
            lines.append(f"> 应用场景：{'；'.join(best.hints)}。")

        return "\n".join(lines)


def _extract_snippet(hits: Sequence[Document], max_len: int = 120) -> str:
    if not hits:
        return ""
    snippet = hits[0].page_content.strip().replace("\n", " ")
    return snippet[:max_len] + ("…" if len(snippet) > max_len else "")
