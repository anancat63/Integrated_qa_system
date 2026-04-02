# -*- coding:utf-8 -*-
# 说明：MySQL 数据访问层封装（DAO / Repository）
# 角色定位：
# - 负责连接 MySQL、执行 SQL、管理事务（commit/rollback）
# - 为上层 BM25 / RAG / 业务逻辑提供“问答表 jpkb”的读写接口
# - 将数据库连接参数集中从 Config 读取，避免硬编码

import pymysql                 # PyMySQL：MySQL 的 Python 客户端实现
import pandas as pd            # pandas：用于 CSV 批量导入（DataFrame 读写）

# ========== 路径处理：确保能导入项目公共模块 ==========
import sys, os
# 获取当前文件所在目录（例如 .../db）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 db 目录的父目录（例如 .../mysql_qa）
module_dir = os.path.dirname(current_dir)
# 获取项目根目录（假设 mysql_qa 位于根目录下，因此再上一级为根）
project_root = os.path.dirname(module_dir)
# 将项目根目录加入模块搜索路径，以便 import base.Config / base.logger
sys.path.insert(0, project_root)

# 导入全局配置与日志器（配置从 config.ini 读取，logger 统一管理输出）
from base import Config, logger


class MySQLClient:
    """
    MySQL 客户端封装类。

    职责：
    - 建立数据库连接并创建 cursor
    - 创建 jpkb 表（若不存在）
    - 从 CSV 导入问答数据到 jpkb
    - 查询问题列表 / 根据问题查询答案
    - 关闭连接

    使用场景：
    - 初始化数据（create_table + insert_data）
    - 检索侧（BM25Search 等）从数据库读取问题与答案
    """

    def __init__(self):
        try:
            # 建立 MySQL 连接：连接参数统一从 Config 获取
            self.connection = pymysql.connect(
                host=Config().MYSQL_HOST,
                user=Config().MYSQL_USER,
                password=Config().MYSQL_PASSWORD,
                database=Config().MYSQL_DATABASE
            )
            # 创建游标：用于执行 SQL 查询与写入
            self.cursor = self.connection.cursor()
            # 连接成功日志
            logger.info("MySQL 连接成功")
        except pymysql.MySQLError as e:
            # 捕获连接异常：记录日志并抛出，避免系统在无数据库状态下继续
            logger.error(f"MySQL 连接失败: {e}")
            raise

# ========== 数据访问层：创建 jpkb 表（若不存在） ==========
    def create_table(self):
        """
        创建问答表 jpkb（若不存在）。

        表字段设计：
        - id：自增主键
        - subject_name：学科名称
        - question：问题文本
        - answer：答案文本

        使用场景：
        - 首次部署 / 初始化环境时执行
        """
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS jpkb (
            id INT AUTO_INCREMENT PRIMARY KEY,
            subject_name VARCHAR(20),
            question VARCHAR(1000),
            answer VARCHAR(1000))
        '''
        try:
            # 执行建表语句并提交事务
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("表创建成功")
        except pymysql.MySQLError as e:
            # 建表失败：记录日志并抛出异常
            logger.error(f"表创建失败: {e}")
            raise

# ========== 数据访问层：批量导入数据 ==========
    def insert_data(self, csv_path):
        """
        从 CSV 文件批量插入数据到 jpkb 表。

        参数：
        - csv_path：CSV 文件路径

        CSV 字段约定（列名必须匹配）：
        - 学科名称
        - 问题
        - 答案

        实现策略：
        - pandas 读取 CSV → DataFrame
        - 逐行 INSERT（简单直观，但数据量大时性能一般）
        - 任意异常则 rollback，防止插入半截导致数据不一致
        """
        try:
            # 读取 CSV 为 DataFrame
            data = pd.read_csv(csv_path)
            # 打印头部数据：用于人工确认字段与内容是否正确
            print(data.head())

            # 遍历每行插入数据库
            for _, row in data.iterrows():
                insert_query = "INSERT INTO jpkb (subject_name, question, answer) VALUES (%s, %s, %s)"
                self.cursor.execute(insert_query, (row["学科名称"], row["问题"], row["答案"]))

            # 提交事务
            self.connection.commit()
            logger.info("Mysql数据插入成功")
        except Exception as e:
            # 插入过程任意异常：记录日志并回滚事务
            logger.error(f'Mysql数据插入失败:{e}')
            # rollback：撤销当前事务内所有操作，恢复到执行前状态
            self.connection.rollback()
            raise


# ========== 数据访问层：获取 jpkb 表中所有问题列表。 ==========
    def fetch_questions(self):
        """
        获取 jpkb 表中所有问题列表。

        返回值：
        - cursor.fetchall() 的结果通常为 tuple 列表：
          例如： (('问题1',), ('问题2',), ...)
        用途：
        - 构建 BM25 语料库 / 索引
        - 批量分析问答库覆盖范围
        """
        try:
            # 执行查询语句
            self.cursor.execute("SELECT question FROM jpkb")
            # 获取全部结果
            results = self.cursor.fetchall()
            logger.info("成功获取问题")
            return results
        except pymysql.MySQLError as e:
            # 查询失败：记录日志并返回空列表（上层可做兜底）
            logger.error(f"查询失败: {e}")
            return []


# ========== 数据访问层：根据问题文本获取答案。 ==========
    def fetch_answer(self, question):
        """
        根据问题文本获取对应答案。

        参数：
        - question：问题字符串

        返回值：
        - 若命中：答案字符串
        - 若未命中或异常：None

        安全性：
        - 使用参数化查询 (question,) 避免 SQL 注入
        """
        try:
            # 按问题精确匹配查询答案
            self.cursor.execute("SELECT answer FROM jpkb WHERE question=%s", (question,))
            result = self.cursor.fetchone()
            # result 形如：('答案文本',)；因此取 result[0]
            return result[0] if result else None
        except pymysql.MySQLError as e:
            # 查询失败：记录日志并返回 None
            logger.error(f"答案获取失败: {e}")
            return None

    def close(self):
        """
        关闭数据库连接。

        使用建议：
        - 脚本或服务退出前调用
        - 避免连接泄漏，释放资源
        """
        try:
            self.connection.close()
            logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            logger.error(f"关闭连接失败: {e}")


if __name__ == '__main__':
    # 自测入口：用于快速验证连接、查询功能是否正常
    mysql_client = MySQLClient()

    # 初始化表结构（首次使用时打开）
    # mysql_client.create_table()

    # 从 CSV 导入数据（需要 CSV 列名匹配：学科名称/问题/答案）
    # mysql_client.insert_data(csv_path='../data/JP学科知识问答.csv')

    # 获取全部问题（常用于 BM25 建索引）
    # results = mysql_client.fetch_questions()
    # print(f'results--》{results}')


    # 查询指定问题对应的答案（精确匹配）
    # a = mysql_client.fetch_answer(question="黑马程序员机构")
    # print(f'a--》{a}')

    # 关闭连接
    mysql_client.close()
