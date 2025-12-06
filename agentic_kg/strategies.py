"""Chunking and parsing strategies for the governance pipeline."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker


@dataclass
class ChunkingStrategyConfig:
    """Represents a single chunking strategy and how to apply it."""

    name: str
    splitter_factory: Callable[[Iterable[Document]], "BaseTextSplitter"]
    description: str
    weight: float = 1.0
    hints: List[str] = field(default_factory=list)

    def split(self, docs: Iterable[Document]) -> List[Document]:
        splitter = self.splitter_factory(docs)
        return splitter.split_documents(list(docs))


def _estimate_heading_density(documents: Iterable[Document]) -> float:
    """Approximate how often Markdown/section headings appear."""
    heading_pattern = re.compile(r"^#{1,6}\\s|\\n\\s*\\d+\\.\\s")
    content = "\n".join(doc.page_content for doc in documents)
    if not content:
        return 0.0
    headings = len(heading_pattern.findall(content))
    return headings / max(1, content.count("\n"))


def build_default_strategies(
    documents: Iterable[Document],
    embedding_model: Embeddings,
    embedding_dimension: Optional[int] = None,
) -> List[ChunkingStrategyConfig]:
    """Create a set of candidate chunking strategies.

    Args:
        documents: Loaded documents.
        embedding_dimension: Dimensionality hint for semantic chunking.
    """

    docs_list = list(documents)
    heading_density = _estimate_heading_density(docs_list)

    def char_splitter(_: Iterable[Document]):
        return CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=100,
        )

    def recursive_splitter(_: Iterable[Document]):
        return RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n## ", "\n# ", "\n\n", "\n", " "]
        )

    def semantic_splitter(_: Iterable[Document]):
        return SemanticChunker(
            embedding_model,
            breakpoint_threshold_type="standard_deviation",
            buffer_size=2,
            embedding_dimension=embedding_dimension,
        )

    strategies: List[ChunkingStrategyConfig] = [
        ChunkingStrategyConfig(
            name="character",
            splitter_factory=char_splitter,
            description="Fixed-size character windows with small overlap.",
            hints=["适合通用文本或平均分布的问题"],
        ),
        ChunkingStrategyConfig(
            name="recursive",
            splitter_factory=recursive_splitter,
            description="Respect headings and paragraphs before falling back to characters.",
            weight=1.2 if heading_density > 0.05 else 1.0,
            hints=["含有章节标题/列表的文档"],
        ),
        ChunkingStrategyConfig(
            name="semantic",
            splitter_factory=semantic_splitter,
            description="Use embeddings to find semantic breakpoints.",
            weight=1.3,
            hints=["长段落或主题跨度较大的文本"],
        ),
    ]

    return strategies


def summarize_chunks(chunks: List[Document]) -> Dict[str, float]:
    """Provide quick stats for generated chunks."""
    if not chunks:
        return {"count": 0, "avg_length": 0.0, "std_length": 0.0}

    lengths = [len(doc.page_content) for doc in chunks]
    count = len(lengths)
    avg_length = sum(lengths) / count
    variance = sum((l - avg_length) ** 2 for l in lengths) / count
    return {
        "count": count,
        "avg_length": avg_length,
        "std_length": math.sqrt(variance),
    }
