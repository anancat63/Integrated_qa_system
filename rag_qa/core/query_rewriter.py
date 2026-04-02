# -*- coding:utf-8 -*-
"""
core/query_rewriter.py
功能：基于 few-shot 的 Query 改写（标准化/多查询扩展/子查询拆分）
特点：中文日志 + 明确标记“是否调用大模型”
"""

import json
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from base import logger, Config
from openai import OpenAI

conf = Config()


@dataclass
class RewriteResult:
    rewrite_type: str                 # "none" | "single" | "multi" | "decompose"
    queries: List[str]                # 最终用于检索的 query 列表（1~N）
    rewritten_query: str              # 主 query
    reason: str                       # 原因说明（用于日志/可观测）
    confidence: float                 # 0~1
    raw: Optional[Dict[str, Any]] = None


class QueryRewriter:
    """
    Few-shot Query Rewriter（中文日志）
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=Config().DASHSCOPE_API_KEY,
            base_url=Config().DASHSCOPE_BASE_URL
        )
        self.model = getattr(Config(), "LLM_MODEL", "qwen-plus")
        logger.info(
            f"【Query改写器】初始化完成 | base_url={Config().DASHSCOPE_BASE_URL} | model={self.model}"
        )

    # --------------------------
    # 1) 门控：哪些情况不必改写
    # --------------------------
    def _should_skip(self, query: str) -> (bool, str):
        q = (query or "").strip()
        if not q:
            return True, "空查询"
        if len(q) <= 6:
            return True, "查询过短"
        faq_like = ["多少钱", "价格", "退费", "退款", "电话", "地址", "报名", "开班", "课程表", "怎么联系"]
        if any(k in q for k in faq_like):
            return True, "疑似FAQ/强关键词问题（优先直检/BM25）"
        return False, "允许改写"

    # --------------------------
    # 2) Few-shot Prompt
    # --------------------------
    def _fewshot_messages(self, query: str) -> List[Dict[str, str]]:
        system = (
            "你是一个“Query 改写器”，只负责把用户问题改写成更适合知识库检索的文本表示。\n"
            "严格规则：\n"
            "1) 不改变原问题语义，不新增事实，不输出答案。\n"
            "2) 必须保留关键实体（课程名/学科/产品/时间/地点/对比对象等）。\n"
            "3) 输出必须是 JSON，且只能输出 JSON（禁止多余解释）。\n"
            "4) rewrite_type 只能是：none / single / multi / decompose。\n"
            "5) candidates 最多 5 条，每条不超过 60 字。\n"
            "6) confidence 为 0~1 小数。\n"
            "\n"
            "JSON schema：\n"
            "{\n"
            '  "rewrite_type": "none|single|multi|decompose",\n'
            '  "rewritten_query": "string",\n'
            '  "candidates": ["string", ...],\n'
            '  "confidence": 0.0,\n'
            '  "reason": "string"\n'
            "}\n"
        )

        examples = [
            (
                "我想问下AI班都教啥啊？",
                {
                    "rewrite_type": "single",
                    "rewritten_query": "AI学科课程包含哪些教学内容？",
                    "candidates": [
                        "AI学科课程包含哪些教学内容？",
                        "AI课程大纲包括哪些模块？"
                    ],
                    "confidence": 0.86,
                    "reason": "口语化表达，改写为标准问法，便于命中知识库条目"
                }
            ),
            (
                "Java就业班学多久？学费多少？能不能分期？",
                {
                    "rewrite_type": "decompose",
                    "rewritten_query": "Java就业班学多久，学费多少，是否支持分期？",
                    "candidates": [
                        "Java就业班学习周期是多久？",
                        "Java就业班学费是多少？",
                        "Java就业班是否支持分期付款？"
                    ],
                    "confidence": 0.83,
                    "reason": "包含多个独立问点，拆分为子查询提高召回"
                }
            ),
            (
                "那个大模型的微调怎么搞比较好？",
                {
                    "rewrite_type": "multi",
                    "rewritten_query": "大模型微调有哪些常用方法与流程？",
                    "candidates": [
                        "大模型微调有哪些常用方法与流程？",
                        "SFT 与 LoRA/QLoRA 的区别是什么？",
                        "大模型微调的步骤与注意事项有哪些？"
                    ],
                    "confidence": 0.72,
                    "reason": "问题抽象，生成多个等价/补全表达增强检索信号"
                }
            ),
            (
                "AI学科的教学内容是什么？",
                {
                    "rewrite_type": "none",
                    "rewritten_query": "AI学科的教学内容是什么？",
                    "candidates": [],
                    "confidence": 0.9,
                    "reason": "问题表达清晰，直接检索即可"
                }
            ),
        ]

        msgs = [{"role": "system", "content": system}]
        for uq, out in examples:
            msgs.append({"role": "user", "content": uq})
            msgs.append({"role": "assistant", "content": json.dumps(out, ensure_ascii=False)})
        msgs.append({"role": "user", "content": query})
        return msgs

    # --------------------------
    # 3) 解析与回退
    # --------------------------
    def _safe_json_loads(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except Exception:
            return None

    def rewrite(self, query: str, trace_id: Optional[str] = None) -> RewriteResult:
        """
        trace_id：可选，用于把一次请求链路的日志串起来（例如 session_id / request_id）
        """
        q = (query or "").strip()
        tid = trace_id or "-"

        skip, skip_reason = self._should_skip(q)
        if skip:
            logger.info(f"【Query改写器】跳过改写 | 原因={skip_reason} | 原query='{q}'")
            return RewriteResult(
                rewrite_type="none",
                queries=[q],
                rewritten_query=q,
                reason=f"跳过改写:{skip_reason}",
                confidence=1.0,
                raw=None
            )

        messages = self._fewshot_messages(q)

        # 明确标记：这里要调用大模型
        logger.info(f"【Query改写器】开始调用大模型进行改写 | model={self.model} | 输入query='{q}'")
        t0 = time.time()

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
            )
            dt = time.time() - t0

            content = resp.choices[0].message.content
            logger.info(f"【Query改写器】大模型返回成功 | 耗时={dt:.2f}s | 返回长度={len(content or '')}")

            data = self._safe_json_loads(content)
            if not data:
                logger.warning(f"【Query改写器】返回结果JSON解析失败，回退为不改写 | 原始返回='{content}'")
                return RewriteResult(
                    rewrite_type="none",
                    queries=[q],
                    rewritten_query=q,
                    reason="回退:JSON解析失败",
                    confidence=0.0,
                    raw=None
                )

            rewrite_type = data.get("rewrite_type", "none")
            rewritten_query = (data.get("rewritten_query") or q).strip()
            candidates = data.get("candidates") or []
            confidence = float(data.get("confidence") or 0.0)
            reason = (data.get("reason") or "").strip()

            # 校验与规整
            if rewrite_type not in {"none", "single", "multi", "decompose"}:
                logger.warning(f"【Query改写器】rewrite_type 非法('{rewrite_type}')，强制回退为 none")
                rewrite_type = "none"

            candidates = [str(x).strip() for x in candidates if str(x).strip()]
            candidates = candidates[:5]

            # 回退：置信度过低或改写为空
            if not rewritten_query or confidence < 0.35:
                logger.info(
                    f"【Query改写器】置信度过低，回退为不改写 | conf={confidence:.2f} | 原因='{reason}' | 改写query='{rewritten_query}'"
                )
                return RewriteResult(
                    rewrite_type="none",
                    queries=[q],
                    rewritten_query=q,
                    reason=f"回退:低置信度:{reason}",
                    confidence=confidence,
                    raw=data
                )

            # 组织最终 queries
            if rewrite_type == "none":
                final_queries = [q]
            elif rewrite_type == "single":
                final_queries = [rewritten_query]
            elif rewrite_type == "multi":
                final_queries = [rewritten_query] + candidates
            else:  # decompose
                final_queries = candidates if candidates else [rewritten_query]

            # 去重（保持顺序）
            seen = set()
            uniq = []
            for x in final_queries:
                if x not in seen:
                    uniq.append(x)
                    seen.add(x)

            logger.info(
                f"【Query改写器】改写完成 | 类型={rewrite_type} | conf={confidence:.2f} | 主改写='{rewritten_query}' | 输出queries={uniq} | 说明='{reason}'"
            )

            return RewriteResult(
                rewrite_type=rewrite_type,
                queries=uniq,
                rewritten_query=rewritten_query,
                reason=reason,
                confidence=confidence,
                raw=data
            )

        except Exception as e:
            dt = time.time() - t0
            logger.error(f"【Query改写器】调用大模型失败，回退为不改写 | 耗时={dt:.2f}s | 错误={e}")
            return RewriteResult(
                rewrite_type="none",
                queries=[q],
                rewritten_query=q,
                reason=f"回退:异常:{e}",
                confidence=0.0,
                raw=None
            )


if __name__ == "__main__":
    qr = QueryRewriter()
    tests = [
        "我想问下AI班都教啥啊？",
        "Java就业班学多久？学费多少？能不能分期？",
        "那个大模型的微调怎么搞比较好？",
        "AI学科的教学内容是什么？",
    ]
    for t in tests:
        r = qr.rewrite(t, trace_id="local_test")
        print("\nQ:", t)
        print("改写类型:", r.rewrite_type)
        print("用于检索的queries:", r.queries)
        print("置信度:", r.confidence)
        print("原因:", r.reason)
