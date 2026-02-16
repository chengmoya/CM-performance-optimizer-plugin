"""
QQ通知渠道 - 通过MaiBot发送QQ私聊消息

支持：
- OneBot/NapCat 协议发送私聊消息
- 自动降级到日志输出
- 消息发送状态跟踪
"""

from __future__ import annotations

import asyncio
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


class QQNotificationChannel(NotificationChannel):
    """QQ通知渠道

    通过MaiBot的OneBot/NapCat协议发送私聊消息。
    支持自动降级到控���台输出。
    """

    def __init__(
        self,
        target_qq: int,
        name: str = "qq",
        cooldown_seconds: float = 300.0,
        daily_limit: int = 50,
    ):
        """初始化QQ通知渠道

        Args:
            target_qq: 目标QQ号
            name: 渠道名称
            cooldown_seconds: 冷却时间（秒）
            daily_limit: 每日发送限制
        """
        super().__init__(name)
        self._target_qq = target_qq
        self._cooldown_seconds = cooldown_seconds
        self._daily_limit = daily_limit

        # 发送状态跟踪
        self._last_send_time: float = 0
        self._daily_send_count: int = 0
        self._daily_reset_time: float = time.time()

        # API客户端（延迟初始化）
        self._api_client: Optional[Any] = None
        self._bot_instance: Optional[Any] = None

    def set_bot_instance(self, bot_instance: Any):
        """设置Bot实例

        Args:
            bot_instance: MaiBot实例或OneBot适配器
        """
        self._bot_instance = bot_instance

    def set_api_client(self, api_client: Any):
        """设置API客户端

        Args:
            api_client: API客户端实例
        """
        self._api_client = api_client

    @property
    def target_qq(self) -> int:
        """获取目标QQ号"""
        return self._target_qq

    @target_qq.setter
    def target_qq(self, value: int):
        """设置目标QQ号"""
        self._target_qq = value

    @property
    def cooldown_seconds(self) -> float:
        """获取冷却时间"""
        return self._cooldown_seconds

    @cooldown_seconds.setter
    def cooldown_seconds(self, value: float):
        """设置冷却时间"""
        self._cooldown_seconds = value

    @property
    def daily_limit(self) -> int:
        """获取每日限制"""
        return self._daily_limit

    @daily_limit.setter
    def daily_limit(self, value: int):
        """设置每日限制"""
        self._daily_limit = value

    def is_available(self) -> bool:
        """检查渠道是否可用

        Returns:
            渠道是否可用（QQ号有效且未超过每日限制）
        """
        if not self._enabled:
            return False

        if self._target_qq <= 0:
            logger.warning("[QQChannel] 目标QQ号无效")
            return False

        # 检查每日限制
        self._check_daily_reset()
        if self._daily_send_count >= self._daily_limit:
            logger.warning(
                f"[QQChannel] 已达每日发送限制: {self._daily_send_count}/{self._daily_limit}"
            )
            return False

        return True

    def _check_daily_reset(self):
        """检查并重置每日计数"""
        current_time = time.time()
        # 检查是否是新的一天（超过24小时）
        if current_time - self._daily_reset_time >= 86400:
            self._daily_send_count = 0
            self._daily_reset_time = current_time

    def _is_in_cooldown(self) -> bool:
        """检查是否在冷却中

        Returns:
            是否在冷却中
        """
        return (time.time() - self._last_send_time) < self._cooldown_seconds

    async def send(self, message: NotificationMessage) -> bool:
        """发送通知消息

        Args:
            message: 通知消息

        Returns:
            是否发送成功
        """
        if not self.is_available():
            return False

        if self._is_in_cooldown():
            logger.debug(
                f"[QQChannel] 消息在冷却中，跳过: {message.template_key}"
            )
            return False

        try:
            # 格式化消息
            full_message = message.format_full_message()

            # 尝试发送消息
            success = await self._send_private_message(self._target_qq, full_message)

            if success:
                self._last_send_time = time.time()
                self._daily_send_count += 1
                logger.info(
                    f"[QQChannel] ✓ 消息发送成功: {message.template_key} -> {self._target_qq}"
                )
            else:
                logger.warning(
                    f"[QQChannel] ✗ 消息发送失败: {message.template_key}"
                )

            return success

        except Exception as e:
            logger.error(f"[QQChannel] 发送消息异常: {e}")
            return False

    async def send_batch(self, messages: list[NotificationMessage]) -> Dict[str, bool]:
        """批量发送通知消息

        Args:
            messages: 通知消息列表

        Returns:
            消息去重键到发送结果的映射
        """
        results: Dict[str, bool] = {}

        for message in messages:
            # 批量发送时，每条消息之间添加小延迟
            if results:  # 不是第一条消息
                await asyncio.sleep(0.5)

            success = await self.send(message)
            results[message.dedup_key or message.template_key] = success

        return results

    async def _send_private_message(self, user_id: int, message: str) -> bool:
        """发送私聊消息

        尝试多种方式发送消息：
        1. 通过Bot实例发送
        2. 通过API客户端发送
        3. 降级到日志输出

        Args:
            user_id: 目标用户ID
            message: 消息内容

        Returns:
            是否发送成功
        """
        # 方式1: 通过Bot实例发送
        if self._bot_instance is not None:
            try:
                # 尝试调用 send_private_msg 方法（OneBot标准）
                if hasattr(self._bot_instance, "send_private_msg"):
                    await self._bot_instance.send_private_msg(
                        user_id=user_id, message=message
                    )
                    return True
                # 尝试调用 call_api 方法
                elif hasattr(self._bot_instance, "call_api"):
                    await self._bot_instance.call_api(
                        "send_private_msg", user_id=user_id, message=message
                    )
                    return True
                # 尝试异步发送方法
                elif hasattr(self._bot_instance, "send_private_message"):
                    await self._bot_instance.send_private_message(user_id, message)
                    return True
            except Exception as e:
                logger.warning(f"[QQChannel] Bot实例发送失败: {e}")

        # 方式2: 通过API客户端发送
        if self._api_client is not None:
            try:
                if hasattr(self._api_client, "send_private_msg"):
                    await self._api_client.send_private_msg(
                        user_id=user_id, message=message
                    )
                    return True
            except Exception as e:
                logger.warning(f"[QQChannel] API客户端发送失败: {e}")

        # 方式3: 尝试从全局获取Bot实例
        try:
            bot = await self._get_bot_instance()
            if bot is not None:
                if hasattr(bot, "send_private_msg"):
                    await bot.send_private_msg(user_id=user_id, message=message)
                    return True
                elif hasattr(bot, "call_api"):
                    await bot.call_api(
                        "send_private_msg", user_id=user_id, message=message
                    )
                    return True
        except Exception as e:
            logger.debug(f"[QQChannel] 全局Bot实例获取失败: {e}")

        # 降级：输出到日志
        logger.info(f"[QQChannel] [降级输出] 发送给 {user_id}:\n{message}")
        return True  # 降级输出视为成功

    async def _get_bot_instance(self) -> Optional[Any]:
        """尝试获取Bot实例

        Returns:
            Bot实例或None
        """
        # 尝试从MaiBot获取Bot实例
        try:
            # 方式1: 从全局变量获取
            from src.plugin_system.base import bot_manager

            if hasattr(bot_manager, "get_bot"):
                return bot_manager.get_bot()
            elif hasattr(bot_manager, "bot"):
                return bot_manager.bot
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[QQChannel] 获取Bot实例失败: {e}")

        # 方式2: 从nonebot获取
        try:
            import nonebot

            return nonebot.get_bot()
        except ImportError:
            pass
        except Exception:
            pass

        return None

    def reset_daily_count(self):
        """重置每日计数"""
        self._daily_send_count = 0
        self._daily_reset_time = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """获取渠道统计信息

        Returns:
            统计信息字典
        """
        return {
            "name": self._name,
            "enabled": self._enabled,
            "target_qq": self._target_qq,
            "cooldown_seconds": self._cooldown_seconds,
            "daily_limit": self._daily_limit,
            "daily_send_count": self._daily_send_count,
            "last_send_time": self._last_send_time,
            "in_cooldown": self._is_in_cooldown(),
        }
