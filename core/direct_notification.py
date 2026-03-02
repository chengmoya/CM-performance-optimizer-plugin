"""直连通知发送器。

用于绕过 channel manager，直接通过 chat_api + send_api 向管理员私聊发送告警。
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

try:
    from src.plugin_system import chat_api, send_api
except ImportError:
    from src.plugin_system.apis import chat_api, send_api


logger = logging.getLogger("CM_perf_opt.direct_notification")


class DirectNotificationSender:
    """直连通知发送器。"""

    def __init__(
        self,
        admin_qq: str = "",
        enabled: bool = True,
        cooldown_seconds: int = 300,
        platform: str = "qq",
    ):
        """初始化发送器。

        Args:
            admin_qq: 管理员 QQ。
            enabled: 是否启用发送。
            cooldown_seconds: 同标题通知冷却时间（秒）。
            platform: 平台标识，默认 qq。
        """
        self.admin_qq = (admin_qq or "").strip()
        self.enabled = bool(enabled)
        self.cooldown_seconds = max(0, int(cooldown_seconds))
        self.platform = platform

        # 按 title 记录最近发送时间戳
        self._last_sent_at: Dict[str, float] = {}

    def _should_skip_by_cooldown(self, title: str, now: float) -> bool:
        """检查是否应因冷却而跳过。"""
        if self.cooldown_seconds <= 0:
            return False

        last = self._last_sent_at.get(title)
        if last is None:
            return False

        return (now - last) < self.cooldown_seconds

    def _format_message(self, title: str, message: str, level: str) -> str:
        """格式化通知文本。"""
        normalized_level = (level or "info").upper()
        return f"[{normalized_level}] {title}\n\n{message}"

    async def send(self, title: str, message: str, level: str = "warning") -> bool:
        """发送通知。

        Returns:
            bool: 发送成功返回 True；其余场景返回 False。
        """
        if not self.enabled:
            logger.debug("[DirectNotificationSender] 已禁用，跳过发送: title=%s", title)
            return False

        if not self.admin_qq:
            logger.warning("[DirectNotificationSender] admin_qq 为空，无法发送通知")
            return False

        normalized_title = (title or "未命名通知").strip()
        normalized_message = (message or "").strip()

        now = time.time()
        if self._should_skip_by_cooldown(normalized_title, now):
            logger.debug(
                "[DirectNotificationSender] 冷却中，跳过发送: title=%s cooldown=%ss",
                normalized_title,
                self.cooldown_seconds,
            )
            return False

        try:
            stream = chat_api.get_stream_by_user_id(
                user_id=str(self.admin_qq),
                platform=self.platform,
            )
            if not stream:
                logger.warning(
                    "[DirectNotificationSender] 未找到管理员私聊流: admin_qq=%s platform=%s",
                    self.admin_qq,
                    self.platform,
                )
                return False

            text = self._format_message(
                title=normalized_title,
                message=normalized_message,
                level=level,
            )

            send_ok = await send_api.text_to_stream(
                text=text,
                stream_id=stream.stream_id,
                storage_message=False,
            )
            if not send_ok:
                logger.error(
                    "[DirectNotificationSender] 发送失败: admin_qq=%s title=%s",
                    self.admin_qq,
                    normalized_title,
                )
                return False

            self._last_sent_at[normalized_title] = now
            logger.info(
                "[DirectNotificationSender] 发送成功: admin_qq=%s title=%s level=%s",
                self.admin_qq,
                normalized_title,
                (level or "info").lower(),
            )
            return True

        except Exception as exc:
            logger.exception(
                "[DirectNotificationSender] 发送异常: admin_qq=%s title=%s error=%s",
                self.admin_qq,
                normalized_title,
                exc,
            )
            return False
