# -*- coding:utf-8 -*-
# 说明：源代码文件本身按 UTF-8 保存（对 Python2 有意义；对现代 Python 主要是声明习惯）
# 目标：定义 Config 类，从项目根目录下的 config.ini 读取配置，并把常用配置项暴露为对象属性

# 导入配置文件 ini 的解析库（标准库）
import configparser
# 导入路径操作库（标准库）
import os

# ========== 路径推导：定位 config.ini ==========
# 获取当前脚本文件的绝对路径（例如 .../base/config.py）
current_file_path = os.path.abspath(__file__)
# 获取当前脚本所在目录（例如 .../base）
current_dir_path = os.path.dirname(current_file_path)
# 获取“项目根目录”：这里假设 base 目录在项目根目录下，因此再上一级就是根目录
project_root = os.path.dirname(current_dir_path)

# 拼出 config.ini 的绝对路径：默认认为 config.ini 位于项目根目录
config_file_path = os.path.join(project_root, 'config.ini')


class Config():
    def __init__(self, config_file=config_file_path):
        """
        读取 ini 配置并将配置项挂到实例属性上。

        参数：
        - config_file: 配置文件路径。默认使用项目根目录下的 config.ini

        注意点：
        - 这里显式用 open(..., encoding="utf-8") 读取配置文件，避免 Windows 默认编码（GBK）导致的解码报错。
        - 如果你的 config.ini 是 UTF-8 with BOM，可能需要用 utf-8-sig 才更稳（但本代码当前使用 utf-8）。
        """

        # 1) 创建 ini 配置解析器（支持 [section] + key=value 的配置格式）
        self.config = configparser.ConfigParser()

        # 2) 读取配置文件
        # 不直接使用 self.config.read(config_file) 的原因：
        # - read() 在不同 Python/不同平台下默认编码可能不同（Windows 常见为 GBK），会引发 UnicodeDecodeError
        # - open 指定 encoding 可以确保用 UTF-8 解码配置文件
        with open(config_file, "r", encoding="utf-8") as f:
            self.config.read_file(f)

        # ========== 3) 逐类读取配置并写入对象属性 ==========
        # 设计意图：将 ini 里的配置“展平”为 Python 属性，后续模块直接 conf.XYZ 使用

        # ---------------- MySQL 配置 ----------------
        # 使用 get(section, option, fallback=默认值)：
        # - ini 中缺失该项时，不会抛异常，而是使用默认值（提高健壮性，便于快速启动）
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='123456')
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='subjects_kg')

        # ---------------- Redis 配置 ----------------
        # getint 用于把字符串转 int，避免后续使用时类型不对
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        self.REDIS_PORT = self.config.getint('redis', 'port', fallback=6379)
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='1234')
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)

        # ---------------- Milvus 配置 ----------------
        # 注意：MILVUS_PORT 这里使用 get（字符串），而不是 getint（整数）
        # 如果下游代码拼接 URI（host:port）通常用字符串也没问题，但若要数值比较或校验，可能更适合 int。
        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='localhost')
        self.MILVUS_PORT = self.config.get('milvus', 'port', fallback='19530')
        self.MILVUS_DATABASE_NAME = self.config.get('milvus', 'database_name', fallback='itcast')
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name', fallback='edurag_final')

        # ---------------- LLM 配置 ----------------
        # LLM_MODEL：模型名称（例如 qwen-plus）
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen-plus')

        # DASHSCOPE_API_KEY：这里没有 fallback，如果 llm 段或该 key 缺失会抛异常
        # 这属于“关键配置必须提供”的设计（合理）
        # 例如：self.DASHSCOPE_API_KEY = self.config.get('llm', 'dashscope_api_key')

        # DASHSCOPE_BASE_URL：提供 fallback，方便不同环境切换
        self.DASHSCOPE_BASE_URL = self.config.get(
            'llm', 'dashscope_base_url',
            fallback='https://dashscope.aliyuncs.com/compatible-mode/v1'
        )
        self.DASHSCOPE_API_KEY = self.config.get(
            'llm',
            'dashscope_api_key',
            fallback=None
        )


        # ---------------- 检索参数（RAG） ----------------
        # 这些参数用于文档切分（父块/子块/重叠）与检索数量控制
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)#父块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)#子块大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)#块重叠大小
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)#检索返回数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)#最终候选数量

        # ---------------- 应用配置 ----------------
        # CUSTOMER_SERVICE_PHONE：这里没有 fallback，如果缺失会抛异常（同样属于关键配置）
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone')

        # VALID_SOURCES：从 ini 读取字符串后用 eval 转为 Python 对象（期望是 list）
        # 例如 ini 写：valid_sources=["ai","java","ops"]
        # 风险提示（注释说明用）：eval 会执行任意代码，ini 来源若不可信存在安全风险。
        # 但若 ini 完全由自己维护，使用 eval 是“方便但有风险”的取舍。
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback=["ai", "java", "test", "ops", "bigdata"])
        )#支持某些学科搜索

        # ---------------- 日志配置 ----------------
        # LOG_FILE：日志文件相对路径（通常相对于项目根目录或运行目录，取决于 logger 实现）
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')


if __name__ == '__main__':
    # 自测入口：实例化 Config 并打印几个字段，确认类型与读取是否正确
    conf = Config()
    print(conf.CHUNK_OVERLAP)        # int
    print(conf.VALID_SOURCES)        # 期望为 list
    print(type(conf.VALID_SOURCES))  # 用于确认 eval 后的类型
