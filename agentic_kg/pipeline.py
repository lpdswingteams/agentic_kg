"""Agentic knowledge governance pipeline using LangChain 1.0.x primitives."""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pypdf import PdfReader

from .report import GovernanceReport, RetrievalEvaluation
from .strategies import ChunkingStrategyConfig, build_default_strategies, summarize_chunks


@dataclass
class PipelineConfig:
    """Configuration for the governance pipeline."""

    embedding_model: Embeddings
    top_k: int = 4


class AgenticGovernancePipeline:
    """End-to-end pipeline for analyzing documents and proposing governance strategies."""

    def __init__(
        self,
        config: PipelineConfig,
        loaders: Optional[Dict[str, Callable[[str], List[Document]]]] = None,
        strategies: Optional[List[ChunkingStrategyConfig]] = None,
    ) -> None:
        self.config = config
        self.loaders = loaders or {}
        self._strategies = strategies

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        documents: List[Document] = []
        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            loader = self.loaders.get(ext)
            if loader:
                documents.extend(loader(path))
                continue

            if ext == ".pdf":
                documents.extend(_load_pdf(path))
            elif ext in {".md", ".txt"}:
                documents.extend(_load_text(path))
            else:
                documents.extend(_load_unstructured(path))
        return documents

    def _ensure_strategies(self, documents: Iterable[Document]) -> List[ChunkingStrategyConfig]:
        if self._strategies is None:
            self._strategies = build_default_strategies(
                documents,
                embedding_model=self.config.embedding_model,
            )
        return self._strategies

    def _evaluate_strategy(
        self,
        strategy: ChunkingStrategyConfig,
        documents: List[Document],
        questions: List[str],
    ) -> RetrievalEvaluation:
        chunks = strategy.split(documents)
        chunk_embeddings = self.config.embedding_model.embed_documents(
            [c.page_content for c in chunks]
        )

        question_results = []
        for question in questions:
            hits = _top_k_similar(
                chunks,
                chunk_embeddings,
                self.config.embedding_model.embed_query(question),
                k=self.config.top_k,
            )
            coverage = _lexical_coverage(question, hits)
            question_results.append({
                "question": question,
                "hits": hits,
                "coverage": coverage,
            })

        chunk_stats = summarize_chunks(chunks)
        mean_coverage = sum(r["coverage"] for r in question_results) / max(1, len(question_results))
        return RetrievalEvaluation(
            strategy_name=strategy.name,
            description=strategy.description,
            question_results=question_results,
            chunk_stats=chunk_stats,
            mean_coverage=mean_coverage,
            weight=strategy.weight,
            hints=strategy.hints,
        )

    def run(self, file_paths: List[str], questions: List[str]) -> GovernanceReport:
        documents = self.load_documents(file_paths)
        strategies = self._ensure_strategies(documents)
        evaluations = [
            self._evaluate_strategy(strategy, documents, questions)
            for strategy in strategies
        ]
        best = max(evaluations, key=lambda ev: ev.weight * ev.mean_coverage)
        return GovernanceReport(
            source_files=file_paths,
            questions=questions,
            evaluations=evaluations,
            selected_strategy=best,
        )


def _lexical_coverage(question: str, hits: List[Document]) -> float:
    import re

    tokens = {
        token.lower()
        for token in re.findall(r"[\w]+|[\u4e00-\u9fff]", question)
        if len(token) > 0
    }
    if not tokens:
        return 0.0
    hit_tokens = set(
        itertools.chain.from_iterable(
            chunk.page_content.lower().split() for chunk in hits
        )
    )
    overlap = tokens.intersection(hit_tokens)
    return len(overlap) / len(tokens)


def _load_pdf(path: str) -> List[Document]:
    reader = PdfReader(path)
    documents: List[Document] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        documents.append(
            Document(page_content=text, metadata={"source": path, "page": idx})
        )
    return documents


def _load_text(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": path})]


def _load_unstructured(path: str) -> List[Document]:
    try:
        from unstructured.partition.auto import partition

        elements = partition(filename=path)
    except Exception:  # noqa: BLE001
        return [
            Document(
                page_content=f"无法解析文件 {path}",
                metadata={"source": path, "error": "parse_failed"},
            )
        ]

    documents: List[Document] = []
    for idx, element in enumerate(elements, start=1):
        text = getattr(element, "text", "")
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": path, "chunk": idx},
                )
            )
    if not documents:
        documents.append(
            Document(page_content="未从文件中提取到文本", metadata={"source": path})
        )
    return documents


def _top_k_similar(
    chunks: List[Document],
    chunk_embeddings: List[List[float]],
    query_embedding: List[float],
    k: int,
) -> List[Document]:
    if not chunks:
        return []
    vectors = np.array(chunk_embeddings, dtype=float)
    query_vec = np.array(query_embedding, dtype=float)
    if vectors.ndim != 2:
        return []
    denom = np.linalg.norm(vectors, axis=1) * (np.linalg.norm(query_vec) + 1e-8)
    scores = vectors.dot(query_vec) / (denom + 1e-8)
    top_indices = scores.argsort()[-k:][::-1]
    return [chunks[i] for i in top_indices]
