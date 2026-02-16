"""
日志通知处理器 - 捕获错误日志并发送通知

功能：
- 继承 logging.Handler，捕获指定级别日志
- 发送到 NotificationManager
- 支持日志去重（避免重复通知）
- 支持堆栈跟踪包含
- 支持配置热更新
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set


@dataclass
class ErrorLogConfig:
    """错误日志通知配置"""

    # 是否启用错误日志通知
    enabled: bool = True
    # 日志级别: "error" | "critical"
    level: str = "error"
    # 是否包含堆栈跟踪
    include_stacktrace: bool = False
    # 去重窗口（秒）
    deduplication_window: int = 600

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "enabled": self.enabled,
            "level": self.level,
            "include_stacktrace": self.include_stacktrace,
            "deduplication_window": self.deduplication_window,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorLogConfig":
        """从字典创建"""
        return cls(
            enabled=data.get("enabled", True),
            level=data.get("level", "error"),
            include_stacktrace=data.get("include_stacktrace", False),
            deduplication_window=data.get("deduplication_window", 600),
        )


class NotificationLogHandler(logging.Handler):
    """日志通知处理器

    继承 logging.Handler，捕获 ERROR 及以上级别日志并发送通知。

    使用方式：
        handler = NotificationLogHandler(config)
        logging.getLogger().addHandler(handler)
    """

    def __init__(self, config: Optional[ErrorLogConfig] = None):
        """初始化日志处理器

        Args:
            config: 错误日志配置
        """
        super().__init__()

        self._config = config or ErrorLogConfig()
        self._lock = threading.RLock()

        # 去重缓存：存储已发送的错误签名
        self._dedup_cache: Dict[str, float] = {}

        # 日志级别映射
        self._level_map = {
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }

        # 设置处理器级别
        self._update_level()

        # 通知管理器引用（延迟设置）
        self._notification_manager: Optional[Any] = None

        # 统计信息
        self._stats = {
            "total_captured": 0,
            "total_sent": 0,
            "total_deduped": 0,
            "total_failed": 0,
        }

    def _update_level(self):
        """更新日志级别"""
        level = self._level_map.get(self._config.level.lower(), logging.ERROR)
        self.setLevel(level)

    def set_config(self, config: ErrorLogConfig):
        """设置配置

        Args:
            config: 错误日志配置
        """
        with self._lock:
            self._config = config
            self._update_level()

    def set_notification_manager(self, manager: Any):
        """设置通知管理器

        Args:
            manager: NotificationManager 实例
        """
        with self._lock:
            self._notification_manager = manager

    def emit(self, record: logging.LogRecord):
        """处理日志记录

        Args:
            record: 日志记录
        """
        if not self._config.enabled:
            return

        with self._lock:
            self._stats["total_captured"] += 1

            try:
                # 生成错误签名用于去重
                error_signature = self._generate_signature(record)

                # 检查是否重复
                if self._is_duplicated(error_signature):
                    self._stats["total_deduped"] += 1
                    return

                # 构建通知内容
                content = self._build_content(record)

                # 发送通知
                if self._notification_manager:
                    self._send_notification(record, content)
                    self._stats["total_sent"] += 1
                else:
                    self._stats["total_failed"] += 1

            except Exception:
                # 处理器内部错误不应影响主程序
                self.handleError(record)

    def _generate_signature(self, record: logging.LogRecord) -> str:
        """生成错误签名用于去重

        Args:
            record: 日志记录

        Returns:
            错误签名（MD5哈希）
        """
        # 使用文件名、行号、函数名和错误消息作为签名基础
        signature_base = f"{record.pathname}:{record.lineno}:{record.funcName}:{record.getMessage()}"
        return hashlib.md5(signature_base.encode()).hexdigest()

    def _is_duplicated(self, signature: str) -> bool:
        """检查是否重复

        Args:
            signature: 错误签名

        Returns:
            是否重复
        """
        now = time.time()

        # 清理过期的去重缓存
        expired_keys = [
            k
            for k, v in self._dedup_cache.items()
            if now - v > self._config.deduplication_window
        ]
        for k in expired_keys:
            del self._dedup_cache[k]

        # 检查是否存在
        if signature in self._dedup_cache:
            return True

        # 记录新的签名
        self._dedup_cache[signature] = now
        return False

    def _build_content(self, record: logging.LogRecord) -> str:
        """构建通知内容

        Args:
            record: 日志记录

        Returns:
            通知内容字符串
        """
        lines = []

        # 基本信息
        lines.append(f"【{record.levelname}】{record.name}")
        lines.append(f"位置: {record.pathname}:{record.lineno}")
        lines.append(f"函数: {record.funcName}")
        lines.append(f"消息: {record.getMessage()}")

        # 堆栈跟踪
        if self._config.include_stacktrace and record.exc_info:
            exc_text = "".join(traceback.format_exception(*record.exc_info))
            lines.append(f"\n堆栈跟踪:\n{exc_text}")

        return "\n".join(lines)

    def _send_notification(self, record: logging.LogRecord, content: str):
        """发送通知

        Args:
            record: 日志记录
            content: 通知内容
        """
        if not self._notification_manager:
            return

        try:
            # 确定通知级别
            level = "error" if record.levelno == logging.ERROR else "critical"

            # 发送通知
            self._notification_manager.send_notification(
                template_key="error_log",
                level=level,
                variables={
                    "error_type": record.levelname,
                    "error_message": record.getMessage(),
                    "error_location": f"{record.pathname}:{record.lineno}",
                    "error_function": record.funcName,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                content=content,
            )
        except Exception:
            # 通知发送失败不应影响主程序
            pass

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                **self._stats,
                "config": self._config.to_dict(),
                "dedup_cache_size": len(self._dedup_cache),
            }

    def clear_dedup_cache(self):
        """清空去重缓存"""
        with self._lock:
            self._dedup_cache.clear()

    def reset_stats(self):
        """重置统计信息"""
        with self._lock:
            self._stats = {
                "total_captured": 0,
                "total_sent": 0,
                "total_deduped": 0,
                "total_failed": 0,
            }


# 全局日志处理器实例
_log_handler: Optional[NotificationLogHandler] = None
_handler_lock = threading.Lock()


def get_log_handler() -> NotificationLogHandler:
    """获取全局日志处理器实例

    Returns:
        NotificationLogHandler 实例
    """
    global _log_handler
    if _log_handler is None:
        with _handler_lock:
            if _log_handler is None:
                _log_handler = NotificationLogHandler()
    return _log_handler


def init_log_handler(
    config: Optional[ErrorLogConfig] = None,
    notification_manager: Optional[Any] = None,
    attach_to_root: bool = True,
) -> NotificationLogHandler:
    """初始化日志处理器

    Args:
        config: 错误日志配置
        notification_manager: 通知管理器实例
        attach_to_root: 是否附加到根日志器

    Returns:
        NotificationLogHandler 实例
    """
    global _log_handler

    with _handler_lock:
        if _log_handler is None:
            _log_handler = NotificationLogHandler(config)
        elif config:
            _log_handler.set_config(config)

        if notification_manager:
            _log_handler.set_notification_manager(notification_manager)

        if attach_to_root:
            # 检查是否已附加
            root_logger = logging.getLogger()
            if _log_handler not in root_logger.handlers:
                root_logger.addHandler(_log_handler)

        return _log_handler


def shutdown_log_handler():
    """关闭日志处理器"""
    global _log_handler

    with _handler_lock:
        if _log_handler is not None:
            try:
                root_logger = logging.getLogger()
                if _log_handler in root_logger.handlers:
                    root_logger.removeHandler(_log_handler)
                _log_handler.close()
            except Exception:
                pass
            _log_handler = None
