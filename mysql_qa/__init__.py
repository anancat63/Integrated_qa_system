# 说明：运行时动态调整模块搜索路径，并导入 MySQL / Redis / BM25 相关组件
# 使用场景：
# - 当该脚本不是以“项目根目录”为启动入口时
# - 通过手动修改 sys.path，确保可以正常导入项目内部模块

import os, sys

# 获取当前脚本文件的绝对路径（包含文件名）
current_dir = os.path.abspath(__file__)

# 获取当前脚本所在目录（例如 mysql_qa 或其子目录）
# 这里假设该目录就是需要加入 Python 模块搜索路径的“模块根”
mysql_qa_path = os.path.dirname(current_dir)
# print(f'mysql_qa_path--》{mysql_qa_path}')

# 将 mysql_qa_path 插入到 sys.path 的最前面
# 目的：
# - 确保 Python 在导入模块时优先从该路径查找
# - 避免因运行目录不同导致的 ImportError
sys.path.insert(0, mysql_qa_path)

# 导入 MySQL 数据访问层
# 负责从数据库中读取问题与答案
from db.mysql_client import MySQLClient

# 导入 Redis 缓存层
# 负责缓存问题列表、分词结果和命中答案
from cache.redis_client import RedisClient

# 导入 BM25 检索模块
# 负责基于关键词相似度的高精度问答匹配
from retrieval.bm25_search import BM25Search
