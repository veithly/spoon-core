from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests


@dataclass
class LoadedDoc:
    id: str
    text: str
    source: str


def _strip_html(html: str) -> str:
    # naive removal of script/style and tags
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    # collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def _try_convert_github_url(url: str) -> str:
    """
    Convert GitHub blob URLs to raw URLs to extract clean content without HTML UI.
    Example: https://github.com/user/repo/blob/main/README.md 
    -> https://raw.githubusercontent.com/user/repo/main/README.md
    """
    # Pattern matches: github.com/{user}/{repo}/blob/{branch}/{path}
    pattern = r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$"
    match = re.match(pattern, url)
    if match:
        user, repo, branch, path = match.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url


def _load_file(path: Path) -> Optional[LoadedDoc]:
    suffix = path.suffix.lower()
    try:
        if suffix in (".txt", ".md"):
            content = path.read_text(encoding="utf-8", errors="ignore")
            return LoadedDoc(id=path.stem, text=content, source=str(path))
        if suffix in (".html", ".htm"):
            html = path.read_text(encoding="utf-8", errors="ignore")
            return LoadedDoc(id=path.stem, text=_strip_html(html), source=str(path))
        if suffix == ".pdf":
            try:
                import PyPDF2  # type: ignore
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    texts = []
                    for page in reader.pages:
                        texts.append(page.extract_text() or "")
                return LoadedDoc(id=path.stem, text="\n".join(texts), source=str(path))
            except Exception:
                return None
    except Exception:
        return None
    return None


def _load_url(url: str) -> Optional[LoadedDoc]:
    try:
        # 1. GitHub 转换: 尝试将 GitHub Blob URL 转为 Raw URL，以便获取纯内容
        target_url = _try_convert_github_url(url)
        
        # 2. 策略判断: 
        # 如果是 Github Raw 链接或常见的纯文本/代码文件后缀，直接下载更高效且精准。
        # 否则 (通用网页)，尝试使用 Jina Reader 将 HTML 转换为高质量 Markdown。
        
        # 常见纯文本/代码后缀，不需要 LLM Reader 进行清理
        raw_extensions = (
            ".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".xml", ".ini", ".conf",
            ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".cs", ".php", ".rb", ".sh"
        )
        
        is_github_raw = "raw.githubusercontent.com" in target_url
        is_pure_text = target_url.lower().endswith(raw_extensions)
        
        should_use_jina = not (is_github_raw or is_pure_text)

        if should_use_jina:
            # 3. 尝试 Jina Reader (https://jina.ai/reader)
            # 它可以将杂乱的网页转换为干净的 Markdown，非常适合 RAG
            jina_api_key = os.getenv("JINA_API_KEY")
            headers = {"X-Retain-Images": "none"}
            if jina_api_key:
                headers["Authorization"] = f"Bearer {jina_api_key}"
            
            try:
                jina_url = f"https://r.jina.ai/{target_url}"
                r_jina = requests.get(jina_url, headers=headers, timeout=20)
                if r_jina.status_code == 200:
                    return LoadedDoc(id=url, text=r_jina.text, source=url)
            except Exception:
                # 如果 Jina 服务超时或失败，静默回退到普通下载
                pass

        # 4. 回退/默认路径: 直接请求目标 URL
        # 适用于 Jina 失败、或者是直接下载路径 (GitHub Raw/Text files)
        r = requests.get(target_url, timeout=20)
        r.raise_for_status()
        
        content_type = r.headers.get("content-type", "").lower()
        if "html" in content_type:
            # 使用简易方式去除标签作为保底
            text = _strip_html(r.text)
        else:
            text = r.text
            
        return LoadedDoc(id=url, text=text, source=url)
    except Exception:
        return None


def load_inputs(paths_or_urls: Iterable[str]) -> List[LoadedDoc]:
    docs: List[LoadedDoc] = []
    for item in paths_or_urls:
        if item.startswith("http://") or item.startswith("https://"):
            d = _load_url(item)
            if d:
                docs.append(d)
            continue
        p = Path(item)
        if p.is_dir():
            for child in p.rglob("*"):
                if child.is_file():
                    d = _load_file(child)
                    if d and d.text.strip():
                        docs.append(d)
        elif p.is_file():
            d = _load_file(p)
            if d and d.text.strip():
                docs.append(d)
    return docs


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    if chunk_size <= 0:
        return [text]
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks

