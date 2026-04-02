# cache/redis_client.py
# 说明：封装 Redis 访问逻辑，作为项目的缓存层（Cache Layer）
# 设计目标：
# - 提供简单、统一的 Redis 读写接口
# - 支持 JSON 数据的序列化/反序列化
# - 用于缓存问答结果或中间计算结果，减少重复计算与数据库访问

# 导入 Redis 官方 Python 客户端
import redis
# 导入 JSON 序列化/反序列化工具
import json
import os, sys

# ========== 路径处理：确保能导入项目公共模块 ==========
# 获取当前文件所在目录（例如 .../cache）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 cache 目录的父目录（例如 .../mysql_qa）
module_dir = os.path.dirname(current_dir)
# 获取项目根目录（假设 mysql_qa 位于项目根目录下）
project_root = os.path.dirname(module_dir)

# 将项目根目录加入 sys.path
# 目的：保证可以直接 import base.Config / base.logger
# 注意：这是运行时路径修正，IDE 静态分析可能仍然需要 Sources Root 配置
sys.path.insert(0, project_root)

# 导入全局配置与日志器
from base import Config, logger


class RedisClient:
    """
    Redis 客户端封装类。

    职责：
    - 管理 Redis 连接
    - 提供通用的 set / get JSON 数据接口
    - 提供特定业务语义的方法（如 get_answer）

    使用方式：
        redcli = RedisClient()
        redcli.set_data("key", value)
        value = redcli.get_data("key")
    """

    def __init__(self):
        try:
            # 创建 Redis 连接客户端
            # decode_responses=True：
            # - 返回 str 而不是 bytes
            # - 便于直接进行 json.loads / 字符串拼接
            self.client = redis.StrictRedis(
                host=Config().REDIS_HOST,
                port=Config().REDIS_PORT,
                password=Config().REDIS_PASSWORD,
                db=Config().REDIS_DB,
                decode_responses=True
            )
            # 记录 Redis 连接成功
            logger.info("Redis 连接成功")
        except redis.RedisError as e:
            # 捕获 Redis 连接相关异常
            # 记录错误日志后继续抛出异常，避免系统在“假连接成功”状态下运行
            logger.error(f"Redis 连接失败: {e}")
            raise


# 数据写入 Redis
    def set_data(self, key, value):
        """
        将任意可 JSON 序列化的数据写入 Redis。

        参数：
        - key: Redis 键
        - value: Python 对象（dict / list / str / int 等）

        实现说明：
        - 使用 json.dumps 序列化
        - ensure_ascii=False，保证中文不被转义为 \\uXXXX
        """
        try:
            # 将 Python 对象序列化为 JSON 字符串并写入 Redis
            self.client.set(key, json.dumps(value, ensure_ascii=False))
            # 记录写入成功
            logger.info(f"存储数据到 Redis: {key}")
        except redis.RedisError as e:
            # 捕获 Redis 写入异常
            logger.error(f"Redis 存储失败: {e}")

# 数据读取 Redis
    def get_data(self, key):
        """
        从 Redis 中读取 JSON 数据并反序列化。

        参数：
        - key: Redis 键

        返回：
        - 若 key 存在：反序列化后的 Python 对象
        - 若 key 不存在或异常：None
        """
        try:
            # 从 Redis 获取原始字符串
            data = self.client.get(key)
            # 若存在数据则进行 JSON 反序列化，否则返回 None
            return json.loads(data) if data else None
        except redis.RedisError as e:
            # 捕获 Redis 读取异常
            logger.error(f"Redis 获取失败: {e}")
            return None


# 获取“问答结果”的缓存答案
    def get_answer(self, query):
        """
        获取“问答结果”的缓存答案。

        业务约定：
        - 使用固定前缀 "answer:" 作为命名空间
        - 一个 query 对应一个缓存答案

        参数：
        - query: 用户查询文本

        返回：
        - 若缓存命中：答案字符串
        - 未命中或异常：None
        """
        try:
            # 构造业务约定的缓存 key
            answer = self.client.get(f"answer:{query}")
            if answer:
                # 记录缓存命中
                logger.info(f"从 Redis 获取答案: {query}")
                return answer
            # 未命中缓存
            return None
        except redis.RedisError as e:
            # 捕获 Redis 查询异常
            logger.error(f"Redis 查询失败: {e}")
            return None




import json
import redis

def dump_all(redis_client, match="*", count=200, max_items=2000):
    """
    遍历 Redis 中的 key 并打印内容（安全用 SCAN，不用 KEYS）。
    - match: key pattern，比如 "answer:*"
    - count: 每次 scan 建议条数（不是严格）
    - max_items: 防止打印过多
    """
    printed = 0
    for key in redis_client.scan_iter(match=match, count=count):
        key_type = redis_client.type(key)
        ttl = redis_client.ttl(key)

        if key_type == "string":
            val = redis_client.get(key)
            # 你 set_data 存的是 JSON 字符串，这里尝试解析一下
            try:
                val_json = json.loads(val)
                val = val_json
            except Exception:
                pass

        elif key_type == "hash":
            val = redis_client.hgetall(key)

        elif key_type == "list":
            val = redis_client.lrange(key, 0, -1)

        elif key_type == "set":
            val = list(redis_client.smembers(key))

        elif key_type == "zset":
            val = redis_client.zrange(key, 0, -1, withscores=True)

        else:
            val = f"<unsupported type: {key_type}>"

        print(f"\nKEY={key} TYPE={key_type} TTL={ttl}\nVALUE={val}")

        printed += 1
        if printed >= max_items:
            print(f"\n[STOP] reached max_items={max_items}")
            break


if __name__ == '__main__':
    # 自测入口：
    # 用于验证 Redis 连接、基本读写功能是否正常
    redcli = RedisClient()
    dump_all(redcli.client)
    # redcli.set_data("user2",'qhd')
    # redcli.set_data("黑马程序员",'相当垃圾')
    # redcli.set_data("answer:黑马程序员",'相当垃圾')
    # logger.info("存入成功")
    # 示例：读取普通 key
    # print(redcli.get_data(key="user2"))
    # 示例：读取问答缓存
    # print(redcli.get_answer(query="黑马程序员"))
