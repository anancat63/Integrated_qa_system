# -*-coding:utf-8-*-
# 说明：MySQL 问答系统入口（MySQL + Redis + BM25）
# 系统定位：
# - 这是一个“结构化问答检索”子系统的整合层（Orchestrator）
# - 负责把 MySQL 数据源、Redis 缓存、BM25 检索器串起来形成可交互的 QA 服务
# - 命中则直接返回答案；未命中则提示需要交给后续 RAG 系统（语义检索/生成）

# 导入 MySQL 数据访问层：负责问题/答案的持久化读取
from db.mysql_client import MySQLClient
# 导入 Redis 缓存层：负责缓存问题列表、分词结果、命中答案等
from cache.redis_client import RedisClient
# 导入 BM25 检索器：负责基于关键词相关性的检索与置信度判断
from retrieval.bm25_search import BM25Search
# 导入全局日志器：统一输出格式、输出位置（文件 + 控制台）
from base import logger
# 导入时间库：用于统计处理耗时
import time


class MySQLQASystem:
    """
    MySQL 问答系统整合类。

    职责：
    - 初始化依赖组件（MySQLClient / RedisClient / BM25Search）
    - 提供对外的 query(query_text) 接口
    - 对一次查询的执行过程进行日志记录与耗时统计

    设计取向：
    - 轻量编排层，不在这里实现分词/检索细节
    - BM25Search 负责“能否可靠命中”的判断
    - 若 BM25 不能可靠命中，返回默认提示，交给上层（RAG）继续处理
    """

    def __init__(self):
        # 初始化 MySQL 客户端：连接数据库、提供问题/答案查询接口
        self.mysql_client = MySQLClient()
        # 初始化 Redis 客户端：提供缓存读写接口
        self.redis_client = RedisClient()
        # 初始化 BM25 检索器：依赖 Redis + MySQL（Redis 优先缓存，MySQL 兜底）
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

    def query(self, query):
        """
        处理一次用户查询并返回答案。

        流程：
        1) 记录开始时间与输入 query
        2) 调用 BM25Search.search 进行检索（带阈值判断）
        3) 命中则返回答案；未命中则提示需要进入 RAG
        4) 记录耗时并返回最终 answer

        返回：
        - answer：字符串答案（命中则为真实答案，未命中则为默认提示）
        """
        # 记录本次查询开始时间，用于耗时统计
        start_time = time.time()

        # 记录查询内容（便于排查与分析日志）
        logger.info(f'处理查询：{query}')

        # 执行 BM25 检索：
        # search 返回 (answer, need_fallback)
        # 这里用 _ 忽略第二个返回值，默认由 answer 是否为空来判断
        answer, _ = self.bm25_search.search(query, threshold=0.85)

        # 若命中答案：记录并返回
        if answer:
            logger.info(f'Mysql答案：{answer}')
        else:
            # 未命中：提示需要进入后续 RAG 系统（语义检索/生成）
            logger.info('SQL中未找到答案，需要调用RAG系统')
            # 设置兜底回复（避免返回 None）
            answer = 'SQL中未找到答案'

        # 计算并记录处理耗时
        process_time = time.time() - start_time
        logger.info(f'查询处理的耗时：{process_time:.2f}秒')

        return answer


def main():
    """
    CLI 交互入口（命令行问答）。

    功能：
    - 创建 MySQLQASystem 实例
    - 循环读取用户输入
    - 输入 exit 退出
    - 发生异常时记录日志
    - 最终确保关闭 MySQL 连接（释放资源）
    """
    mysql_qa = MySQLQASystem()
    try:
        print('\n欢迎使用 MYSQL 问答系统')
        print("输入查询进行回答，输入 'exit' 退出。")

        while True:
            # 读取用户输入，并去掉首尾空格
            query = input("\n请输入查询：").strip()

            # 约定：输入 exit 退出系统
            if query.lower() == 'exit':
                logger.info("退出Mysql系统")
                print("再见")
                break

            # 调用系统查询接口并打印答案
            answer = mysql_qa.query(query)
            print(f'\n答案：{answer}')

    except Exception as e:
        # 捕获系统运行异常并记录日志
        logger.error(f'系统错误：{e}')

    finally:
        # 无论是否异常，最终都尝试关闭 MySQL 连接
        mysql_qa.mysql_client.close()


if __name__ == '__main__':
    # 脚本直接运行时进入 CLI 主程序
    main()
