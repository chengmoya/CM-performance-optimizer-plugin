"""
通知渠道包 - 提供多种通知发送渠道

渠道类型：
- QQNotificationChannel - QQ私聊消息渠道
- ConsoleNotificationChannel - 控制台日志渠道
"""

from .base import NotificationChannel, NotificationLevel, NotificationMessage
from .console_channel import ConsoleNotificationChannel
from .qq_channel import QQNotificationChannel

__all__ = [
    "NotificationChannel",
    "NotificationLevel",
    "NotificationMessage",
    "QQNotificationChannel",
    "ConsoleNotificationChannel",
]
