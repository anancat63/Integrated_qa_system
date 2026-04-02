# -*- coding:utf-8 -*-
"""
Query Normalizer（轻量标准化模块）
目标：不改变语义，只把 query 变得“更规整、更可分类、更可检索”
放置位置：意图识别 / Router 之前
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from base import logger, Config

conf = Config()


@dataclass
class NormalizedQuery:
    raw_query: str
    normalized_query: str
    changes: List[str]          # 记录做过哪些标准化动作
    tokens_hint: List[str]      # 可选：给后续做一些轻量特征（可不使用）


class QueryNormalizer:
    """
    轻量标准化：规则 + 词典
    - 规则：全半角/大小写/空白/标点统一/重复压缩
    - 词典：同义词、术语归一（不做生成式改写）
    """

    def __init__(
        self,
        synonym_map: Optional[Dict[str, str]] = None,
        keep_case: bool = False
    ):
        self.keep_case = keep_case
        # 同义词/术语归一表：key -> value
        self.synonym_map = synonym_map or self._default_synonym_map()

        # 预编译正则（性能 + 统一）
        self._re_multi_space = re.compile(r"\s+")
        self._re_multi_punct = re.compile(r"([，。,\.!?！？；;:：、])\1+")
        self._re_space_punct = re.compile(r"\s*([，。,\.!?！？；;:：、])\s*")

    def normalize(self, query: str) -> NormalizedQuery:
        raw = (query or "").strip()
        if not raw:
            return NormalizedQuery(raw_query=query, normalized_query="", changes=["empty"], tokens_hint=[])

        changes: List[str] = []
        q = raw

        # 1) Unicode 归一化：全半角、兼容字符统一
        q2 = unicodedata.normalize("NFKC", q)
        if q2 != q:
            changes.append("unicode_nfkc")
            q = q2

        # 2) 大小写策略
        if not self.keep_case:
            q2 = q.lower()
            if q2 != q:
                changes.append("lowercase")
                q = q2

        # 3) 空白压缩：多空格 / 换行 / tab → 单空格
        q2 = self._re_multi_space.sub(" ", q).strip()
        if q2 != q:
            changes.append("collapse_spaces")
            q = q2

        # 4) 标点清洗：重复标点压缩；标点两侧空白统一
        q2 = self._re_multi_punct.sub(r"\1", q)
        q2 = self._re_space_punct.sub(r"\1", q2)
        if q2 != q:
            changes.append("normalize_punct")
            q = q2

        # 5) 词典归一：仅做“术语/别名 → 标准词”，不扩写、不改语义
        q_before = q
        q = self._apply_synonyms(q, self.synonym_map, changes)
        if q != q_before:
            changes.append("synonym_map")

        # 6) 轻量 token hint（可选，不依赖分词器；仅给 Router/规则判断用）
        tokens_hint = self._simple_tokens_hint(q)

        if not changes:
            changes = ["no_change"]

        logger.info(f"[QueryNormalizer] raw='{raw}' -> norm='{q}', changes={changes}")
        return NormalizedQuery(raw_query=raw, normalized_query=q, changes=changes, tokens_hint=tokens_hint)

    def _apply_synonyms(self, text: str, mapping: Dict[str, str], changes: List[str]) -> str:
        # 简单替换：按 key 长度从长到短，避免短词先替换造成误替换
        items = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)
        out = text
        for k, v in items:
            if k in out and v:
                out = out.replace(k, v)
        return out

    def _simple_tokens_hint(self, text: str) -> List[str]:
        # 非严格分词：用于“有多个点/比较/多实体”的轻量提示，不影响主逻辑
        # 中文按标点切片，英文按空格切片
        parts = re.split(r"[，。！？;；:：、,/]\s*", text)
        parts = [p.strip() for p in parts if p.strip()]
        return parts[:10]

    def _default_synonym_map(self) -> Dict[str, str]:
        # 你可以把这些放到 config.ini 或单独 JSON 文件里加载
        return {
            # 英文/缩写 → 中文标准词
            "ai": "人工智能",
            "a.i.": "人工智能",
            "nlp": "自然语言处理",
            "llm": "大语言模型",
            "rag": "检索增强生成",

            # 课程/术语常见别名归一
            "java基础": "java入门",
            "java核心": "java高级",
            "机器学习": "机器学习",
            "深度学习": "深度学习",

            # 常见口语归一（尽量“短替换”，不要扩写）
            "啥": "什么",
            "咋": "怎么",
            "区别是啥": "区别是什么",
        }


if __name__ == "__main__":
    qn = QueryNormalizer()

    # # 例子1：全角/空白/口语词归一（更规整）
    # query1 = "ＡＩ　是　啥？  "
    # out1 = qn.normalize(query1)
    # print("\n" + "=" * 80)
    # print("例子1 原始:", repr(query1))
    # print("例子1 标准:", repr(out1.normalized_query))
    # print("例子1 changes:", out1.changes)
    # print("例子1 tokens_hint:", out1.tokens_hint)
    #
    # # 例子2：缩写/大小写/重复标点（更可检索、更可分类）
    # query2 = "NLP 和 LLM 的区别是啥？？？？"
    # out2 = qn.normalize(query2)
    # print("\n" + "=" * 80)
    # print("例子2 原始:", repr(query2))
    # print("例子2 标准:", repr(out2.normalized_query))
    # print("例子2 changes:", out2.changes)
    # print("例子2 tokens_hint:", out2.tokens_hint)
    #
    # # 例子3：多分句输入（tokens_hint 可给 Router/规则做“多点问题”提示）
    # query3 = "我想问：RAG 是啥，怎么用？和传统检索有什么区别？"
    # out3 = qn.normalize(query3)
    # print("\n" + "=" * 80)
    # print("例子3 原始:", repr(query3))
    # print("例子3 标准:", repr(out3.normalized_query))
    # print("例子3 changes:", out3.changes)
    # print("例子3 tokens_hint:", out3.tokens_hint)

    # 更抽象/口语化、包含多点诉求（适合体现“更规整、更可分类、更可检索”）
    query4 = "我想了解下 A.I. / RAG 在教育行业咋落地呀？？跟传统 NLP 有啥不一样；最好顺便说下 LLM 的作用～"
    out4 = qn.normalize(query4)
    print("\n" + "=" * 80)
    print("例子4 原始:", repr(query4))
    print("例子4 标准:", repr(out4.normalized_query))
    print("例子4 changes:", out4.changes)
    print("例子4 tokens_hint:", out4.tokens_hint)

    # 更抽象：目标/方向型表达 + 多诉求 + 口语缩写混杂
    # query5 = "想把这套东西做得更智能点：AI/LLM+RAG 怎么组合才靠谱？哪些环节最关键、怎么评估效果？？"
    # out5 = qn.normalize(query5)
    # print("\n" + "=" * 80)
    # print("例子5 原始:", repr(query5))
    # print("例子5 标准:", repr(out5.normalized_query))
    # print("例子5 changes:", out5.changes)
    # print("例子5 tokens_hint:", out5.tokens_hint)
