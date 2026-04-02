#
# # debug_paths.py （也可以直接粘进 bm25_search.py 顶部）
# import os, sys
# from pprint import pprint
#
# print("\n===== 基本信息 =====")
# print("sys.executable =", sys.executable)          # 当前 Python 解释器路径
# print("sys.version   =", sys.version)
# print("cwd           =", os.getcwd())             # 当前工作目录（很关键）
# print("__file__      =", __file__)
#
# print("\n===== 计算路径（你代码里的逻辑） =====")
# current_dir = os.path.dirname(os.path.abspath(__file__))
# module_dir  = os.path.dirname(current_dir)
# project_root = os.path.dirname(module_dir)
# print("current_dir  =", current_dir)
# print("module_dir   =", module_dir)
# print("project_root =", project_root)
#
# print("\n===== sys.path（按顺序） =====")
# pprint(sys.path)
#
# print("\n===== 环境变量（可选） =====")
# for k in ["PYTHONPATH", "CONDA_PREFIX", "VIRTUAL_ENV"]:
#     print(f"{k} =", os.environ.get(k))
#


# -*- coding:utf-8 -*-
# 说明：基于 BM25 的传统文本检索模块（FAQ / 精确问答检索）
# 系统角色：
# - 作为检索系统中的“高精度直达通道”
# - 适用于固定问答、标准问题、强关键词匹配场景
# - 命中即返回；未命中则交由后续语义检索（如向量 RAG）

# 导入 BM25（Okapi BM25）算法实现
from rank_bm25 import BM25Okapi
# 导入数值计算库，用于 softmax 等数学计算
import numpy as np

# ========== 路径处理：确保项目内模块可被正确导入 ==========
import sys, os
# 当前文件所在目录（例如 .../mysql_qa/retrieval）
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f'current_dir--》{current_dir}')
# retrieval 目录的父目录（例如 .../mysql_qa）
module_dir = os.path.dirname(current_dir)
print(f'module_dir--》{module_dir}')
# 将 mysql_qa 加入模块搜索路径
sys.path.insert(0, module_dir)
# 项目根目录
project_root = os.path.dirname(module_dir)
# 将项目根目录加入模块搜索路径，保证可 import base / utils / db / cache
sys.path.insert(0, project_root)

# 文本预处理函数：通常包含分词、去停用词、规范化等
# from utils.preprocess import preprocess_text
from utils.preprocess import preprocess_text
# MySQL 数据访问层：问题与答案的持久化存储
# from db.mysql_client import MySQLClient
from db.mysql_client import MySQLClient
# Redis 缓存层：缓存问题列表、分词结果与命中答案
# from cache.redis_client import RedisClient
from cache.redis_client import RedisClient
# 全局日志器
from base import logger





class BM25Search:
    """
    BM25 搜索封装类。

    职责：
    - 加载问题数据（Redis 优先，MySQL 兜底）
    - 构建 BM25 索引
    - 对用户 query 进行 BM25 相似度计算
    - 在高置信度时直接返回答案并缓存

    设计取向：
    - 高精度、低延迟
    - 不追求语义泛化能力
    - 常作为检索系统的第一道或兜底策略
    """

    def __init__(self, redis_client, mysql_client):
        # 使用统一的全局日志器
        # Redis 客户端：负责缓存问题列表与答案
        self.redis_client = redis_client
        # MySQL 客户端：负责持久化问答数据
        self.mysql_client = mysql_client

        # BM25 模型实例（在加载数据后初始化）
        self.bm25 = None
        # 分词后的问题列表（BM25 实际使用的语料）
        self.questions = None
        # 原始问题列表（用于索引反查与数据库查询）
        self.original_questions = None

        # 初始化并加载 BM25 所需数据
        self._load_data()

    def _load_data(self):
        """
        加载并初始化 BM25 语料与模型。

        加载策略：
        1. 优先从 Redis 读取（启动快）
        2. Redis 未命中则从 MySQL 加载
        3. 分词结果写回 Redis，供后续快速复用
        """
        """
        BM25 初始化时，会“约定”两个 Redis 键：
一个存「原始问题列表」
一个存「分词后的问题列表」
启动时优先去 Redis 看这两个键在不在：
都在 → 直接用，BM25 立刻可用
任意一个不在 → 回源 MySQL
重新拉问题
重新分词
写回 Redis
再初始化 BM25
        """
        # Redis 中缓存原始问题的 key「原始问题列表」
        original_key = "qa_original_questions"
        # Redis 中缓存分词后问题的 key「分词后的问题列表」
        tokenized_key = "qa_tokenized_questions"

        # 从 Redis 读取原始问题列表（最快路径）

        self.original_questions = self.redis_client.get_data(original_key)
        logger.info(f"尝试读取redis问题列表...")

        # 从 Redis 读取分词后的问题列表
        tokenized_questions = self.redis_client.get_data(tokenized_key)
        logger.info(f"尝试读取redis分词后问题列表...")


#
        # 若 Redis 未缓存，则回源 MySQL
        if not self.original_questions or not tokenized_questions:
            # 从 MySQL 获取所有问题（返回 tuple 列表）
            logger.info(" Redis 未缓存，回源 MySQL")
            self.original_questions = self.mysql_client.fetch_questions()
            # MySQL 中也未获取到数据，直接告警并终止初始化
            if not self.original_questions:
                logger.info("MySQL 中也未获取到数据，直接告警并终止初始化")
                logger.warning("未加载问题")
                return
            #original_questions为数据库中所有问题列表
            # 对每个问题进行文本预处理（分词）
            tokenized_questions = [
                preprocess_text(q[0]) for q in self.original_questions
            ]

            # 将原始问题列表的文本写入 Redis（便于下次快速加载）
            self.redis_client.set_data(
                original_key,
                [(q[0]) for q in self.original_questions]
            )
            # 将分词后的问题写入 Redis
            self.redis_client.set_data(
                tokenized_key,
                tokenized_questions
            )

        # 设置 BM25 的语料库（token list）
        self.questions = tokenized_questions
        # 初始化 BM25 模型
        self.bm25 = BM25Okapi(self.questions)
        # 记录初始化完成日志
        logger.info("BM25 模型初始化完成")

    # def _softmax(self, scores):
    #     """
    #     对 BM25 原始得分进行 softmax 归一化。
    #
    #     目的：
    #     - 将不同 query 的 BM25 分数映射到 (0,1)
    #     - 便于使用统一阈值进行置信度判断
    #
    #     实现细节：
    #     - scores 减去最大值，避免 exp 计算时数值溢出
    #     """
    #     exp_scores = np.exp(scores - np.max(scores))
    #     return exp_scores / exp_scores.sum()

    def _softmax(self, scores):
        """
        对 BM25 原始得分做 softmax，但先对“分数值去重”，避免重复项把 top1 概率平均稀释。
        """
        scores = np.asarray(scores, dtype=np.float64)

        # 关键：按“分数值”去重（不是按文档去重）
        unique_scores = np.unique(scores)

        # 标准 softmax（数值稳定）
        x = unique_scores - np.max(unique_scores)
        exp_x = np.exp(x)
        probs = exp_x / (exp_x.sum() + 1e-12)

        # 建立：score_value -> prob 的映射
        score2prob = dict(zip(unique_scores.tolist(), probs.tolist()))

        # 返回每个原始score对应的prob（同分的prob相同，不再被平均稀释）
        return np.array([score2prob[float(s)] for s in scores], dtype=np.float64)

    def search(self, query, threshold=0.85):
        """
        使用 BM25 检索最相似问题，并在高置信度时返回答案。

        参数：
        - query: 用户输入的问题文本
        - threshold: softmax 相似度阈值，超过才认为是“可靠命中”

        返回值：
        - (answer, False)：成功命中，无需进入后续检索流程
        - (None, True)：未命中或不可靠，建议交由语义检索（RAG）
        """
        # 校验查询合法性
        if not query or not isinstance(query, str):
            logger.error("无效查询")
            return None, False

        # ---------- 1. 查询 Redis 缓存 ----------
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            # 命中缓存，直接返回结果
            logger.info("直接命中")
            return cached_answer, False
        logger.info("未直接命中")
        try:
            # ---------- 2. 查询文本分词 ----------
            query_tokens = preprocess_text(query)
            print(f'分词处理后：query_tokens:{query_tokens}')
            # ---------- 3. BM25 打分 ----------
            scores = self.bm25.get_scores(query_tokens)

            # ---------- 4. 分数归一化 ----------
            softmax_score = self._softmax(scores)

            # ---------- 5. 选取最高分 ----------
            best_idx = softmax_score.argmax()
            best_score = softmax_score[best_idx]
            #
            # query_tokens = preprocess_text(query)
            # print("query =", query)
            # print("query_tokens =", query_tokens, "len =", len(query_tokens))
            #
            # scores = self.bm25.get_scores(query_tokens)
            # print("scores stats:", float(np.max(scores)), float(np.min(scores)), float(np.mean(scores)))
            # top_idx = np.argsort(scores)[-5:][::-1]
            # print("top5:")
            # for i in top_idx:
            #     print(i, float(scores[i]), self.original_questions[i])

            # ---------- 6. 阈值判断 ----------
            if best_score >= threshold:
                # 根据索引找到对应的原始问题
                original_question = self.original_questions[best_idx]

                # 从 MySQL 查询该问题对应的答案
                answer = self.mysql_client.fetch_answer(original_question)
                if answer:
                    # 将命中结果写入 Redis 缓存
                    self.redis_client.set_data(f'answer:{query}', answer)
                    logger.info(
                        f'搜索成功，Softmax 相似度：{best_score:.3f}'
                    )
                    return answer, False

            # 分数不足以认为是可靠答案
            logger.info(
                f"未找到可靠答案，最高 Softmax 相似度: {best_score:.3f}"
            )
            return None, True

        except Exception as e:
            # 捕获任意异常，避免 BM25 阶段影响整体系统
            logger.error(f'搜索查询失败：{e}')
            return None, True


if __name__ == "__main__":
    # 自测入口：
    # 用于验证 Redis / MySQL 连接、BM25 初始化与搜索流程
    redis_client = RedisClient()
    mysql_client = MySQLClient()

    bm25_search = BM25Search(redis_client, mysql_client)
    a= bm25_search.search(
        query="VMware安装VMware Tools时显示灰色如何解决"
        # query="user2"
        # query="黑马程序员"
    )

    print(a)













