"""
通知管理器 - 管理通知的发送、冷却、去重

功能：
- 多渠道通知发送（QQ、控制台）
- 冷却控制（防止频繁发送）
- 去重检测（避免重复通知）
- 频率限制（每日/每小时限制）
- 通知级别过滤
- 通知模板管理
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

try:
    from src.common.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")


# 通知模板定义
NOTIFICATION_TEMPLATES = {
    "memory_warning": {
        "title": "性能优化插件 - 内存警告",
        "template": "内存使用率: {memory_percent:.1f}%\n警告阈值: {threshold:.1f}%\n\n时间: {timestamp}",
        "level": "warning",
    },
    "memory_critical": {
        "title": "性能优化插件 - 内存严重告警",
        "template": "内存使用率: {memory_percent:.1f}%\n严重阈值: {threshold:.1f}%\n已触发垃圾回收!\n\n时间: {timestamp}",
        "level": "critical",
    },
    "cache_hit_rate_low": {
        "title": "性能优化插件 - 缓存命中率过低",
        "template": "缓存模块: {cache_name}\n命中率: {hit_rate:.1%}\n阈值: {threshold:.1%}\n\n时间: {timestamp}",
        "level": "warning",
    },
    "error_log": {
        "title": "性能优化插件 - 错误日志",
        "template": "错误类型: {error_type}\n错误信息: {error_message}\n\n时间: {timestamp}",
        "level": "error",
    },
}


@dataclass
class NotificationConfig:
    """通知配置数据类"""

    # 全局开关
    enabled: bool = True
    # 模式: "qq" | "console" | "both"
    mode: str = "both"

    # QQ配置
    qq_target: int = 0
    qq_level: str = "warning"  # "all" | "warning" | "critical" | "error"
    qq_cooldown_seconds: float = 300.0
    qq_daily_limit: int = 50

    # 性能警告配置
    performance_warning_enabled: bool = True
    memory_warning_enabled: bool = True
    memory_critical_enabled: bool = True
    cache_hit_rate_enabled: bool = False
    cache_hit_rate_threshold: float = 0.5

    # 去重配置
    dedup_enabled: bool = True
    dedup_ttl_seconds: float = 3600.0  # 去重缓存过期时间

    # 频率限制
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "qq_target": self.qq_target,
            "qq_level": self.qq_level,
            "qq_cooldown_seconds": self.qq_cooldown_seconds,
            "qq_daily_limit": self.qq_daily_limit,
            "performance_warning_enabled": self.performance_warning_enabled,
            "memory_warning_enabled": self.memory_warning_enabled,
            "memory_critical_enabled": self.memory_critical_enabled,
            "cache_hit_rate_enabled": self.cache_hit_rate_enabled,
            "cache_hit_rate_threshold": self.cache_hit_rate_threshold,
            "dedup_enabled": self.dedup_enabled,
            "dedup_ttl_seconds": self.dedup_ttl_seconds,
            "rate_limit_enabled": self.rate_limit_enabled,
            "rate_limit_per_minute": self.rate_limit_per_minute,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotificationConfig":
        """从字典创建"""
        return cls(
            enabled=data.get("enabled", True),
            mode=data.get("mode", "both"),
            qq_target=data.get("qq_target", 0),
            qq_level=data.get("qq_level", "warning"),
            qq_cooldown_seconds=data.get("qq_cooldown_seconds", 300.0),
            qq_daily_limit=data.get("qq_daily_limit", 50),
            performance_warning_enabled=data.get("performance_warning_enabled", True),
            memory_warning_enabled=data.get("memory_warning_enabled", True),
            memory_critical_enabled=data.get("memory_critical_enabled", True),
            cache_hit_rate_enabled=data.get("cache_hit_rate_enabled", False),
            cache_hit_rate_threshold=data.get("cache_hit_rate_threshold", 0.5),
            dedup_enabled=data.get("dedup_enabled", True),
            dedup_ttl_seconds=data.get("dedup_ttl_seconds", 3600.0),
            rate_limit_enabled=data.get("rate_limit_enabled", True),
            rate_limit_per_minute=data.get("rate_limit_per_minute", 10),
        )


@dataclass
class NotificationRecord:
    """通知记录（用于去重和统计）"""

    dedup_key: str
    template_key: str
    level: str
    timestamp: float
    content_hash: str
    send_success: bool = True


class NotificationManager:
    """通知管理器

    管理通知的发送、冷却、去重和频率限制。
    """

    _instance: Optional["NotificationManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[NotificationConfig] = None):
        if self._initialized:
            return
        self._initialized = True

        self._config = config or NotificationConfig()
        self._channels: Dict[str, Any] = {}  # 渠道实例
        self._lock = threading.RLock()

        # 去重缓存
        self._dedup_cache: Dict[str, float] = {}

        # 频率限制
        self._send_history: List[float] = []

        # 通知记录
        self._notification_records: List[NotificationRecord] = []
        self._max_records: int = 1000

        # 统计信息
        self._stats = {
            "total_sent": 0,
            "total_failed": 0,
            "total_deduped": 0,
            "total_rate_limited": 0,
        }

        # 初始化渠道
        self._init_channels()

    def _init_channels(self):
        """初始化通知渠道"""
        # 延迟导入避免循环依赖
        try:
            from ..components.channels import (
                QQNotificationChannel,
                ConsoleNotificationChannel,
                NotificationLevel,
            )

            # 创建QQ渠道
            if self._config.qq_target > 0:
                qq_channel = QQNotificationChannel(
                    target_qq=self._config.qq_target,
                    cooldown_seconds=self._config.qq_cooldown_seconds,
                    daily_limit=self._config.qq_daily_limit,
                )
                qq_channel.disable()  # 默认禁用，等待Bot实例
                self._channels["qq"] = qq_channel

            # 创建控制台渠道
            console_channel = ConsoleNotificationChannel(
                min_level=NotificationLevel.from_string(self._config.qq_level)
            )
            self._channels["console"] = console_channel

            logger.info("[NotificationManager] 通知渠道初始化完成")

        except ImportError as e:
            logger.warning(f"[NotificationManager] 渠道导入失败: {e}")

    def set_config(self, config: NotificationConfig):
        """设置配置

        Args:
            config: 通知配置
        """
        with self._lock:
            self._config = config
            self._update_channels_config()

    def _update_channels_config(self):
        """更新渠道配置"""
        from ..components.channels import NotificationLevel

        # 更新QQ渠道配置
        if "qq" in self._channels:
            qq_channel = self._channels["qq"]
            qq_channel.target_qq = self._config.qq_target
            qq_channel.cooldown_seconds = self._config.qq_cooldown_seconds
            qq_channel.daily_limit = self._config.qq_daily_limit

        # 更新控制台渠道配置
        if "console" in self._channels:
            console_channel = self._channels["console"]
            console_channel.min_level = NotificationLevel.from_string(
                self._config.qq_level
            )

    def set_bot_instance(self, bot_instance: Any):
        """设置Bot实例

        Args:
            bot_instance: MaiBot实例
        """
        with self._lock:
            if "qq" in self._channels:
                self._channels["qq"].set_bot_instance(bot_instance)
                self._channels["qq"].enable()
                logger.info("[NotificationManager] QQ渠道已启用")

    def set_api_client(self, api_client: Any):
        """设置API客户端

        Args:
            api_client: API客户端实例
        """
        with self._lock:
            if "qq" in self._channels:
                self._channels["qq"].set_api_client(api_client)

    @property
    def config(self) -> NotificationConfig:
        """获取当前配置"""
        return self._config

    @property
    def enabled(self) -> bool:
        """是否启用通知"""
        return self._config.enabled

    def enable(self):
        """启用通知"""
        self._config.enabled = True

    def disable(self):
        """禁用通知"""
        self._config.enabled = False

    def _generate_dedup_key(
        self, template_key: str, level: str, content: str
    ) -> str:
        """生成去重键

        Args:
            template_key: 模板键
            level: 通知级别
            content: 消息内容

        Returns:
            去重键
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{template_key}:{level}:{content_hash}"

    def _is_duplicated(self, dedup_key: str) -> bool:
        """检查是否重复

        Args:
            dedup_key: 去重键

        Returns:
            是否重复
        """
        if not self._config.dedup_enabled:
            return False

        now = time.time()

        # 清理过期的去重缓存
        expired_keys = [
            k
            for k, v in self._dedup_cache.items()
            if now - v > self._config.dedup_ttl_seconds
        ]
        for k in expired_keys:
            del self._dedup_cache[k]

        # 检查是否存在
        if dedup_key in self._dedup_cache:
            return True

        # 记录新的去重键
        self._dedup_cache[dedup_key] = now
        return False

    def _check_rate_limit(self) -> bool:
        """检查频率限制

        Returns:
            是否允许发送（True=允许，False=被限制）
        """
        if not self._config.rate_limit_enabled:
            return True

        now = time.time()
        minute_ago = now - 60

        # 清理一分钟前的记录
        self._send_history = [t for t in self._send_history if t > minute_ago]

        # 检查是否超过限制
        if len(self._send_history) >= self._config.rate_limit_per_minute:
            return False

        # 记录本次发送
        self._send_history.append(now)
        return True

    def _get_active_channels(self) -> List[Any]:
        """获取活跃的渠道列表

        Returns:
            活跃渠道列表
        """
        channels = []

        if self._config.mode == "qq":
            if "qq" in self._channels and self._channels["qq"].is_available():
                channels.append(self._channels["qq"])
        elif self._config.mode == "console":
            if "console" in self._channels:
                channels.append(self._channels["console"])
        elif self._config.mode == "both":
            if "qq" in self._channels and self._channels["qq"].is_available():
                channels.append(self._channels["qq"])
            if "console" in self._channels:
                channels.append(self._channels["console"])

        return channels

    def create_message(
        self,
        template_key: str,
        level: str = "info",
        **kwargs,
    ) -> Optional[Any]:
        """创建通知消息

        Args:
            template_key: 模板键
            level: 通知级别
            **kwargs: 模板参数

        Returns:
            通知消息或None
        """
        from ..components.channels import NotificationMessage, NotificationLevel

        # 获取模板
        template_data = NOTIFICATION_TEMPLATES.get(template_key)
        if template_data is None:
            logger.warning(f"[NotificationManager] 未知模板: {template_key}")
            return None

        # 准备参数
        params = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs,
        }

        # 格式化内容
        title = template_data.get("title", template_key)
        template_str = template_data.get("template", "")
        try:
            content = template_str.format(**params)
        except KeyError as e:
            logger.warning(f"[NotificationManager] 模板参数缺失: {e}")
            content = template_str

        # 确定级别
        template_level = template_data.get("level", level)
        notification_level = NotificationLevel.from_string(template_level)

        # 创建消息
        message = NotificationMessage(
            title=title,
            content=content,
            level=notification_level,
            template_key=template_key,
            metadata=kwargs,
        )

        return message

    async def send(
        self,
        template_key: str,
        level: str = "info",
        **kwargs,
    ) -> bool:
        """发送通知

        Args:
            template_key: 模板键
            level: 通知级别
            **kwargs: 模板参数

        Returns:
            是否发送成功
        """
        # 检查是否启用
        if not self._config.enabled:
            return False

        # 创建消息
        message = self.create_message(template_key, level, **kwargs)
        if message is None:
            return False

        # 检查去重
        dedup_key = self._generate_dedup_key(
            template_key, message.level.value, message.content
        )
        message.dedup_key = dedup_key

        if self._is_duplicated(dedup_key):
            self._stats["total_deduped"] += 1
            logger.debug(f"[NotificationManager] 通知已去重: {dedup_key}")
            return True  # 去重视为成功

        # 检查频率限制
        if not self._check_rate_limit():
            self._stats["total_rate_limited"] += 1
            logger.warning("[NotificationManager] 通知被频率限制")
            return False

        # 获取活跃渠道
        channels = self._get_active_channels()
        if not channels:
            logger.warning("[NotificationManager] 无可用渠道")
            return False

        # 发送消息
        success = False
        for channel in channels:
            try:
                result = await channel.send(message)
                if result:
                    success = True
            except Exception as e:
                logger.error(f"[NotificationManager] 渠道发送失败: {e}")

        # 记录结果
        self._record_notification(
            dedup_key=dedup_key,
            template_key=template_key,
            level=message.level.value,
            content_hash=dedup_key.split(":")[-1],
            send_success=success,
        )

        # 更新统计
        if success:
            self._stats["total_sent"] += 1
        else:
            self._stats["total_failed"] += 1

        return success

    async def send_message(self, message: Any) -> bool:
        """发送已创建的消息

        Args:
            message: 通知消息

        Returns:
            是否发送成功
        """
        if not self._config.enabled:
            return False

        # 检查去重
        if message.dedup_key and self._is_duplicated(message.dedup_key):
            self._stats["total_deduped"] += 1
            return True

        # 检查频率限制
        if not self._check_rate_limit():
            self._stats["total_rate_limited"] += 1
            return False

        # 获取活跃渠道并发送
        channels = self._get_active_channels()
        if not channels:
            return False

        success = False
        for channel in channels:
            try:
                result = await channel.send(message)
                if result:
                    success = True
            except Exception as e:
                logger.error(f"[NotificationManager] 渠道发送失败: {e}")

        # 更新统计
        if success:
            self._stats["total_sent"] += 1
        else:
            self._stats["total_failed"] += 1

        return success

    def _record_notification(
        self,
        dedup_key: str,
        template_key: str,
        level: str,
        content_hash: str,
        send_success: bool,
    ):
        """记录通知

        Args:
            dedup_key: 去重键
            template_key: 模板键
            level: 通知级别
            content_hash: 内容哈希
            send_success: 是否发送成功
        """
        record = NotificationRecord(
            dedup_key=dedup_key,
            template_key=template_key,
            level=level,
            timestamp=time.time(),
            content_hash=content_hash,
            send_success=send_success,
        )

        self._notification_records.append(record)

        # 限制记录数量
        while len(self._notification_records) > self._max_records:
            self._notification_records.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        stats = self._stats.copy()
        stats["config"] = self._config.to_dict()
        stats["channels"] = {}

        for name, channel in self._channels.items():
            if hasattr(channel, "get_stats"):
                stats["channels"][name] = channel.get_stats()

        return stats

    def get_recent_notifications(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近的通知记录

        Args:
            limit: 返回数量限制

        Returns:
            通知记录列表
        """
        records = self._notification_records[-limit:]
        return [
            {
                "dedup_key": r.dedup_key,
                "template_key": r.template_key,
                "level": r.level,
                "timestamp": datetime.fromtimestamp(r.timestamp).isoformat(),
                "send_success": r.send_success,
            }
            for r in records
        ]

    def clear_dedup_cache(self):
        """清空去重缓存"""
        with self._lock:
            self._dedup_cache.clear()
            logger.info("[NotificationManager] 去重缓存已清空")

    def reset_stats(self):
        """重置统计信息"""
        with self._lock:
            self._stats = {
                "total_sent": 0,
                "total_failed": 0,
                "total_deduped": 0,
                "total_rate_limited": 0,
            }
            self._notification_records.clear()
            logger.info("[NotificationManager] 统计信息已重置")


# 全局实例获取函数
_notification_manager: Optional[NotificationManager] = None


def get_notification_manager(
    config: Optional[NotificationConfig] = None,
) -> NotificationManager:
    """获取通知管理器实例

    Args:
        config: 通知配置（仅首次调用时使用）

    Returns:
        通知管理器实例
    """
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager(config)
    return _notification_manager


def init_notification_manager(config: NotificationConfig) -> NotificationManager:
    """初始化通知管理器

    Args:
        config: 通知配置

    Returns:
        通知管理器实例
    """
    global _notification_manager
    _notification_manager = NotificationManager(config)
    return _notification_manager
