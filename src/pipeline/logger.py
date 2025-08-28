# logger_config.py
import logging
import sys
from datetime import datetime
from pathlib import Path
from concurrent_log_handler import ConcurrentRotatingFileHandler  # 核心：解决文件占用问题

def create_logger(
    name: str,
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,   # 10MB
    backup_count: int = 10,              # 最多保留 10 个备份文件
    log_dir: str = None                  # 可选：自定义日志目录
) -> logging.Logger:
    """
    创建一个安全的日志记录器，日志文件按大小分割（最大10MB），支持多进程/线程安全写入。
    文件命名格式：app_2025-08-19.log, app_2025-08-19-1.log, app_2025-08-19-2.log ...

    :param name: Logger 名称
    :param log_level: 日志级别
    :param max_bytes: 单个日志文件最大字节数
    :param backup_count: 保留的旧日志文件数量
    :param log_dir: 自定义日志目录路径（可选）
    :return: logging.Logger 实例
    """
    # 1. 确定日志目录
    if log_dir is None:
        current_dir = Path(__file__).parent.absolute()
        logs_dir = (current_dir / '../log').resolve()
    else:
        logs_dir = Path(log_dir).resolve()

    logs_dir.mkdir(parents=True, exist_ok=True)

    # 2. 日志文件名：app_YYYY-MM-DD.log
    log_filename = f'app_{_get_date_prefix()}.log'
    log_file = logs_dir / log_filename

    # 3. 获取 logger 实例
    logger = logging.getLogger(name)

    # 防止重复添加处理器（避免日志重复输出）
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(log_level)

    # 4. 使用 ConcurrentRotatingFileHandler（关键：解决 Windows 文件占用）
    file_handler = ConcurrentRotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)

    # 5. 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # 6. 统一日志格式
    formatter = logging.Formatter(
        '%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 7. 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def _get_date_prefix():
    """获取当前日期前缀，如 2025-08-19"""
    return datetime.now().strftime('%Y-%m-%d')