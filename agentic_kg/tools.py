"""Utility tools the agent can call for parsing and inspection."""
from __future__ import annotations

import os
from typing import List

from langchain_core.tools import tool


@tool
def extract_pdf_text(file_path: str) -> str:
    """Read a PDF file and return a compact text preview for planning."""
    if not os.path.exists(file_path):
        return f"文件不存在: {file_path}"
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    combined = "\n".join((page.extract_text() or "") for page in reader.pages)
    snippet = combined[:1200]
    if len(combined) > len(snippet):
        snippet += "…"
    return snippet


@tool
def ocr_image_to_text(file_path: str) -> str:
    """Use OCR to extract text from an image or scanned PDF page."""
    if not os.path.exists(file_path):
        return f"文件不存在: {file_path}"
    from unstructured.partition.image import partition_image

    try:
        elements = partition_image(filename=file_path)
    except Exception as exc:  # noqa: BLE001
        return f"OCR 失败: {exc}"

    texts: List[str] = []
    for el in elements:
        text = getattr(el, "text", "")
        if text:
            texts.append(text)
    if not texts:
        return "OCR 未提取到文本"
    return "\n".join(texts)


@tool
def analyze_layout(file_path: str) -> str:
    """Perform lightweight layout analysis to count elements per page."""
    if not os.path.exists(file_path):
        return f"文件不存在: {file_path}"
    from unstructured.partition.pdf import partition_pdf

    try:
        elements = partition_pdf(filename=file_path, strategy="hi_res", include_page_breaks=True)
    except Exception as exc:  # noqa: BLE001
        return f"版面分析失败: {exc}"

    page_counts = {}
    page_index = 1
    for el in elements:
        if getattr(el, "category", "") == "PageBreak":
            page_index += 1
            continue
        page_counts[page_index] = page_counts.get(page_index, 0) + 1
    summary = ", ".join(f"第{idx}页: {count}段" for idx, count in sorted(page_counts.items()))
    if not summary:
        summary = "未检测到版面元素"
    return f"检测到 {len(page_counts)} 页；元素统计：{summary}"


@tool
def sniff_unstructured_text(file_path: str) -> str:
    """Parse a general file with Unstructured loader to preview text."""
    if not os.path.exists(file_path):
        return f"文件不存在: {file_path}"
    try:
        from unstructured.partition.auto import partition

        elements = partition(filename=file_path)
    except Exception as exc:  # noqa: BLE001
        return f"预览失败: {exc}"

    preview_parts = []
    for element in elements:
        text = getattr(element, "text", "")
        if text:
            preview_parts.append(text)

    preview = "\n".join(preview_parts)
    return preview[:1200] + ("…" if len(preview) > 1200 else "")
