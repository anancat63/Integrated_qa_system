# base/logger_db.py
# -*- coding:utf-8 -*-
# 说明：定义并初始化项目的全局日志器（logger）
#
# 设计目标：
# 1. 统一整个项目的日志出口（所有模块共用同一个 logger）
# 2. 支持日志写入文件（用于问题回溯、线上排错）
# 3. 可选是否将日志输出到控制台（用于本地调试）
# 4. 多次 import logger 时，不重复添加 handler，避免日志重复打印

import logging
import os
from .config import Config

# ========== 路径推导：定位项目根目录 ==========
# 当前文件 logger_db.py 的绝对路径
current_file_path = os.path.abspath(__file__)

# 当前文件所在目录（通常是 base/）
current_dir_path = os.path.dirname(current_file_path)

# 项目根目录（假设 base/ 位于项目根目录下）
project_root = os.path.dirname(current_dir_path)

# 从 Config 中读取日志文件路径（相对路径），并拼成绝对路径
# 例如：logs/app.log → <project_root>/logs/app.log
log_file_path = os.path.join(project_root, Config().LOG_FILE)


def setup_logging(log_file=log_file_path):
    """
    初始化并返回项目级 logger。

    日志体系说明：
    - logger：日志入口（logger.info / logger.error 等）
    - handler：日志输出位置（文件 / 控制台）
    - formatter：日志显示格式

    ⚠️ 控制“是否在控制台输出日志”的关键位置：
    - StreamHandler（控制台输出）
    - FileHandler（文件输出）
    👉 是否添加某个 handler，决定日志会不会输出到对应位置
    """

    # 确保日志文件所在目录存在（如 logs/）
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 获取（或创建）名为 "IntegratedQA" 的 logger
    # logging.getLogger 是单例机制：同名 logger 全局唯一
    logger = logging.getLogger("IntegratedQA")

    # 设置 logger 的最低日志级别
    # 低于 INFO 的日志（如 DEBUG）将被整体过滤
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler：
    # 如果 logger.handlers 非空，说明已经初始化过
    if not logger.handlers:

        # ---------- 文件日志处理器（核心、建议保留） ----------
        # 功能：将日志写入文件，用于长期保存和问题追溯
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # ---------- 控制台日志处理器（可选，用于调试） ----------
        # 功能：将日志打印到终端 / 控制台
        # ⚠️ 如果你“不想在控制台看到任何日志”，
        #    只需要不创建或不添加这个 handler 即可
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # ---------- 日志格式 ----------
        # 定义日志在文件 / 控制台中的显示样式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 为 handler 绑定格式
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将 handler 挂载到 logger
        # 👉 添加哪个 handler，就往哪个地方输出日志
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# ========== 全局 logger 实例 ==========
# 模块加载时立即初始化 logger
#
# 其他模块使用方式：
#   from base import logger
#   logger.info("xxx")
#
# 无需关心 handler / formatter / 文件路径等细节
logger = setup_logging()
