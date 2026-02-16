"""
控制台通知渠道 - 降级输出��日志

提供：
- 将通知输出到日志系统
- 支持不同日志级别映射
- 无外部依赖的降级方案
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .base import NotificationChannel, NotificationMessage, NotificationLevel

try:
    from src.common.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")


class ConsoleNotificationChannel(NotificationChannel):
    """控制台通知渠道

    将通知消息输出到日志系统，作为QQ渠道的降级方案。
    """

    # 通知级别到日志级别的映射
    LEVEL_MAP = {
        NotificationLevel.DEBUG: "debug",
        NotificationLevel.INFO: "info",
        NotificationLevel.WARNING: "warning",
        NotificationLevel.ERROR: "error",
        NotificationLevel.CRITICAL: "critical",
    }

    def __init__(
        self,
        name: str = "console",
        min_level: NotificationLevel = NotificationLevel.INFO,
        include_metadata: bool = False,
    ):
        """初始化控制台通知渠道

        Args:
            name: 渠道名称
            min_level: 最低输出级别
            include_metadata: 是否包含元数据
        """
        super().__init__(name)
        self._min_level = min_level
        self._include_metadata = include_metadata

        # 统计信息
        self._total_sent: int = 0
        self._last_send_time: float = 0

    @property
    def min_level(self) -> NotificationLevel:
        """获取最低输出级别"""
        return self._min_level

    @min_level.setter
    def min_level(self, value: NotificationLevel):
        """设置最低输出级别"""
        self._min_level = value

    @property
    def include_metadata(self) -> bool:
        """是否包含元数据"""
        return self._include_metadata

    @include_metadata.setter
    def include_metadata(self, value: bool):
        """设置是否包含元数据"""
        self._include_metadata = value

    def is_available(self) -> bool:
        """检查渠道是否可用

        控制台渠道始终可用。

        Returns:
            始终返回True
        """
        return self._enabled

    async def send(self, message: NotificationMessage) -> bool:
        """发送通知消息到日志

        Args:
            message: 通知消息

        Returns:
            是否发送成功（始终返回True）
        """
        if not self._enabled:
            return False

        # 检查级别过滤
        if not message.level.should_send(self._min_level):
            return False

        try:
            # 格式化消息
            log_message = self._format_log_message(message)

            # 获取对应的日志级别
            log_level = self.LEVEL_MAP.get(message.level, "info")

            # 输出到日志
            log_func = getattr(logger, log_level, logger.info)
            log_func(log_message)

            # 更新统计
            self._total_sent += 1
            self._last_send_time = time.time()

            return True

        except Exception as e:
            # 降级到print，确保消息不丢失
            try:
                print(f"[ConsoleChannel] 日志输出失败: {e}")
                print(f"[ConsoleChannel] 原始消息: {message.format_full_message()}")
            except Exception:
                pass
            return True  # 即使失败也返回True，避免影响其他渠道

    async def send_batch(self, messages: list[NotificationMessage]) -> Dict[str, bool]:
        """批量发送通知消息

        Args:
            messages: 通知消息列表

        Returns:
            消息去重键到发送结果的映射
        """
        results: Dict[str, bool] = {}

        for message in messages:
            success = await self.send(message)
            results[message.dedup_key or message.template_key] = success

        return results

    def _format_log_message(self, message: NotificationMessage) -> str:
        """格式化日志消息

        Args:
            message: 通知消息

        Returns:
            格式化后的日志消息
        """
        parts = [f"[通知] {message.title}"]

        # 添加内容
        if message.content:
            parts.append(message.content)

        # 添加元数据
        if self._include_metadata and message.metadata:
            parts.append(f"元数据: {message.metadata}")

        # 添加时间戳
        parts.append(f"时间: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """获取渠道统计信息

        Returns:
            统计信息字典
        """
        return {
            "name": self._name,
            "enabled": self._enabled,
            "min_level": self._min_level.value,
            "include_metadata": self._include_metadata,
            "total_sent": self._total_sent,
            "last_send_time": self._last_send_time,
        }

    def reset_stats(self):
        """重置统计信息"""
        self._total_sent = 0
        self._last_send_time = 0
