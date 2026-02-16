"""
通知渠道基类 - 定义通知渠道的抽象接口

提供：
- NotificationLevel 枚举 - 通知级别
- NotificationChannel 抽象基类 - 通知渠道接口
- NotificationMessage 数据类 - 通知消息结构
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class NotificationLevel(Enum):
    """通知级别枚举"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @classmethod
    def from_string(cls, level_str: str) -> "NotificationLevel":
        """从字符串创建枚举值

        Args:
            level_str: 级别字符串

        Returns:
            NotificationLevel 枚举值
        """
        level_map = {
            "debug": cls.DEBUG,
            "info": cls.INFO,
            "warning": cls.WARNING,
            "error": cls.ERROR,
            "critical": cls.CRITICAL,
            "all": cls.DEBUG,  # "all" 等同于最低级别
        }
        return level_map.get(level_str.lower(), cls.INFO)

    def should_send(self, min_level: "NotificationLevel") -> bool:
        """检查是否应该发送此级别的通知

        Args:
            min_level: 最低发送级别

        Returns:
            是否应该发送
        """
        level_order = [self.DEBUG, self.INFO, self.WARNING, self.ERROR, self.CRITICAL]
        return level_order.index(self) >= level_order.index(min_level)


@dataclass
class NotificationMessage:
    """通知消息数据类"""

    title: str  # 通知标题
    content: str  # 通知内容
    level: NotificationLevel  # 通知级别
    template_key: str  # 模板键名
    timestamp: datetime = field(default_factory=datetime.now)  # 时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    # 用于去重的唯一标识
    dedup_key: Optional[str] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.dedup_key is None:
            # 默认使用模板键和级别生成去重键
            self.dedup_key = f"{self.template_key}:{self.level.value}"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            字典表示
        """
        return {
            "title": self.title,
            "content": self.content,
            "level": self.level.value,
            "template_key": self.template_key,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "dedup_key": self.dedup_key,
        }

    def format_full_message(self) -> str:
        """格式化完整消息

        Returns:
            格式化后的完整消息字符串
        """
        level_emoji = {
            NotificationLevel.DEBUG: "🔍",
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨",
        }
        emoji = level_emoji.get(self.level, "📢")
        return f"{emoji} {self.title}\n\n{self.content}"


class NotificationChannel(ABC):
    """通知渠道抽象基类

    定义通知渠道的标准接口，所有具体渠道实现都需要继承此类。
    """

    def __init__(self, name: str):
        """初始化通知渠道

        Args:
            name: 渠道名称
        """
        self._name = name
        self._enabled = True

    @property
    def name(self) -> str:
        """获取渠道名称"""
        return self._name

    @property
    def enabled(self) -> bool:
        """获取是否启用"""
        return self._enabled

    def enable(self):
        """启用渠道"""
        self._enabled = True

    def disable(self):
        """禁用渠道"""
        self._enabled = False

    @abstractmethod
    async def send(self, message: NotificationMessage) -> bool:
        """发送通知消息

        Args:
            message: 通知消息

        Returns:
            是否发送成功
        """
        pass

    @abstractmethod
    async def send_batch(self, messages: list[NotificationMessage]) -> Dict[str, bool]:
        """批量发送通知消息

        Args:
            messages: 通知消息列表

        Returns:
            消息去重键到发送结果的映射
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查渠道是否可用

        Returns:
            渠道是否可用
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name}, enabled={self._enabled})"
