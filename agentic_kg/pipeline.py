"""Agentic knowledge governance pipeline using LangChain 0.3+ primitives."""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_core.document_loaders import BaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

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
        loaders: Optional[Dict[str, Callable[[str], BaseLoader]]] = None,
        strategies: Optional[List[ChunkingStrategyConfig]] = None,
    ) -> None:
        self.config = config
        self.loaders = loaders or {}
        self._strategies = strategies

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        documents: List[Document] = []
        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            loader_factory = self.loaders.get(ext)
            if loader_factory:
                docs = loader_factory(path).load()
            elif ext == ".pdf":
                docs = PyPDFLoader(path).load()
            elif ext in {".md", ".txt"}:
                docs = TextLoader(path, encoding="utf-8").load()
            else:
                docs = UnstructuredFileLoader(path).load()
            documents.extend(docs)
        return documents

    def _ensure_strategies(self, documents: Iterable[Document]) -> List[ChunkingStrategyConfig]:
        if self._strategies is None:
            self._strategies = build_default_strategies(
                documents,
                embedding_model=self.config.embedding_model,
            )
        return self._strategies

    def _build_vector_store(self, chunks: List[Document]) -> VectorStore:
        return FAISS.from_documents(chunks, self.config.embedding_model)

    def _evaluate_strategy(
        self,
        strategy: ChunkingStrategyConfig,
        documents: List[Document],
        questions: List[str],
    ) -> RetrievalEvaluation:
        chunks = strategy.split(documents)
        store = self._build_vector_store(chunks)
        retriever: BaseRetriever = store.as_retriever(search_kwargs={"k": self.config.top_k})

        question_results = []
        for question in questions:
            hits = retriever.invoke(question)
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
