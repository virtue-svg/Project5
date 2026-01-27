# -*- coding: utf-8 -*-
from __future__ import annotations
# 作用: 文本读取与清洗工具。
# 流程: 去除 URL/@/重复字符/emoji 等噪声。
# 输出: 清洗后的文本字符串。

import re
from pathlib import Path


_URL_RE = re.compile(r"https?://\\S+|www\\.\\S+")
_MENTION_RE = re.compile(r"@\\w+")
_WHITESPACE_RE = re.compile(r"\\s+")
_REPEAT_RE = re.compile(r"(.)\\1{2,}")
_EMOJI_RE = re.compile(
    "["
    "\\U0001F300-\\U0001F5FF"
    "\\U0001F600-\\U0001F64F"
    "\\U0001F680-\\U0001F6FF"
    "\\U0001F700-\\U0001F77F"
    "\\U0001F780-\\U0001F7FF"
    "\\U0001F800-\\U0001F8FF"
    "\\U0001F900-\\U0001F9FF"
    "\\U0001FA00-\\U0001FAFF"
    "\\u2600-\\u26FF"
    "\\u2700-\\u27BF"
    "]+",
    flags=re.UNICODE,
)

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "this",
    "that",
    "it",
    "be",
    "as",
    "at",
}


def read_text(path: Path) -> str:
    # 读取文本文件，兼容编码异常
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def clean_text(text: str) -> str:
    # 基础清洗：URL/@/空白/小写
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip().lower()


def clean_text_advanced(
    text: str,
    replace_emoji: bool = True,
    collapse_repeats: bool = True,
    remove_stopwords: bool = False,
) -> str:
    # 进阶清洗：emoji 替换、重复字符折叠、可选停用词
    text = clean_text(text)
    if replace_emoji:
        text = _EMOJI_RE.sub(" <emoji> ", text)
    if collapse_repeats:
        text = _REPEAT_RE.sub(r"\\1\\1", text)
    if remove_stopwords:
        tokens = [t for t in text.split() if t not in _STOPWORDS]
        text = " ".join(tokens)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
