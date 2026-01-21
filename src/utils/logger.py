#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
通用日志工具：提供 get_logger(name, log_file=None, level='INFO')。

特点：
- 同时输出到控制台与文件（可选）；
- 防止重复添加 handler；
- 简洁格式，包含时间、等级、模块与消息；
- 在 macOS/conda 环境下无需额外依赖。
"""
import logging
import os
from typing import Optional


_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}


def get_logger(name: str,
               log_file: Optional[str] = None,
               level: str = 'INFO') -> logging.Logger:
    """创建或获取一个带有控制台与可选文件输出的 Logger。

    参数：
    - name: 日志器名称（模块/脚本名）。
    - log_file: 可选日志文件路径，若提供则写入文件；若父目录不存在则自动创建。
    - level: 字符串等级，默认 'INFO'（可选 'DEBUG' 等）。

    返回：logging.Logger 实例。
    """
    lvl = _LEVELS.get(str(level).upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(lvl)

    # 若已存在 handler，避免重复添加
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')

        # 控制台输出
        sh = logging.StreamHandler()
        sh.setLevel(lvl)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        # 文件输出（可选）
        if log_file:
            try:
                log_dir = os.path.dirname(os.path.abspath(log_file))
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                fh.setLevel(lvl)
                fh.setFormatter(fmt)
                logger.addHandler(fh)
            except Exception as e:
                logger.warning(f"Failed to initialize file handler for {log_file}: {e}")

        # 避免向上冒泡到 root logger
        logger.propagate = False

    else:
        # 更新现有 handler 的级别，确保与传入的 level 一致
        for h in logger.handlers:
            h.setLevel(lvl)
        logger.setLevel(lvl)

    return logger