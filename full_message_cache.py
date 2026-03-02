"""
全量消息缓存模块 - 数据库镜像模式

功能：
1. 全量加载所有消息到内存
2. 拦截所有 find_messages 查询，优先从缓存读取
3. 拦截 store_message 写入，直接更新缓存
4. 支持增量加载（缓存未命中时从DB补充）
5. 双缓冲+缓慢加载+原子切换

设计原则：
- 与现有 MessageCacheModule 完全独立
- 两种模式互不干扰，通过配置切换
- 内存占用可控，支持LRU淘汰
"""

import sys
import asyncio
import time
import threading
import json
from typing import Optional, Dict, Any, List, Set, Tuple
from collections import OrderedDict

try:
    from src.common.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger("CM_full_msg_cache")


# ===== 统计类 =====
class ModuleStats:
    """单个模块的统计"""

    def __init__(self, name: str):
        self.name = name
        self.lock = threading.Lock()
        self.t_hit = self.t_miss = self.t_filtered = 0
        self.i_hit = self.i_miss = self.i_filtered = 0
        self.t_skipped = 0
        self.i_skipped = 0
        self.t_fast = self.t_slow = 0
        self.i_fast = self.i_slow = 0
        self.t_fast_time = self.t_slow_time = 0.0
        self.i_fast_time = self.i_slow_time = 0.0

    def hit(self):
        with self.lock:
            self.t_hit += 1
            self.i_hit += 1

    def miss(self, elapsed: float):
        with self.lock:
            self.t_miss += 1
            self.i_miss += 1
            if elapsed > 0.1:
                self.t_slow += 1
                self.i_slow += 1
                self.t_slow_time += elapsed
                self.i_slow_time += elapsed
            else:
                self.t_fast += 1
                self.i_fast += 1
                self.t_fast_time += elapsed
                self.i_fast_time += elapsed

    def skipped(self):
        with self.lock:
            self.t_skipped += 1
            self.i_skipped += 1

    def filtered(self):
        with self.lock:
            self.t_filtered += 1
            self.i_filtered += 1

    def reset_interval(self) -> Dict[str, Any]:
        with self.lock:
            r = {
                "i_hit": self.i_hit,
                "i_miss": self.i_miss,
                "i_filtered": self.i_filtered,
                "i_skipped": self.i_skipped,
                "i_fast": self.i_fast,
                "i_slow": self.i_slow,
                "i_fast_time": self.i_fast_time,
                "i_slow_time": self.i_slow_time,
            }
            self.i_hit = self.i_miss = self.i_filtered = 0
            self.i_skipped = 0
            self.i_fast = self.i_slow = 0
            self.i_fast_time = self.i_slow_time = 0.0
            return r

    def total(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "t_hit": self.t_hit,
                "t_miss": self.t_miss,
                "t_filtered": self.t_filtered,
                "t_skipped": self.t_skipped,
                "t_fast": self.t_fast,
                "t_slow": self.t_slow,
                "t_fast_time": self.t_fast_time,
                "t_slow_time": self.t_slow_time,
            }


# ===== 内存测量工具类 =====
class MemoryUtils:
    """内存测量工具类 - 递归计算对象的内存占用"""

    @staticmethod
    def get_size(obj, seen=None):
        """递归计算对象的内存占用（字节）"""
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0

        seen.add(obj_id)
        size = sys.getsizeof(obj)

        # 处理常见容器类型
        if isinstance(obj, dict):
            size += sum(MemoryUtils.get_size(k, seen) + MemoryUtils.get_size(v, seen) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sum(MemoryUtils.get_size(i, seen) for i in obj)
        elif isinstance(obj, OrderedDict):
            size += sum(MemoryUtils.get_size(k, seen) + MemoryUtils.get_size(v, seen) for k, v in obj.items())

        return size

    @staticmethod
    def format_size(bytes_size):
        """将字节转换为易读的格式"""
        if bytes_size < 1024:
            return f"{bytes_size:.2f} B"
        elif bytes_size < 1024 * 1024:
            return f"{bytes_size / 1024:.2f} KB"
        else:
            return f"{bytes_size / (1024 * 1024):.2f} MB"


# ===== 全量消息缓存核心类 =====
class FullMessageCache:
    """全量消息缓存 - 数据库镜像模式

    特性：
    - 双缓冲：buffer_a（当前使用）+ buffer_b（后台加载）
    - 缓慢加载：分批加载避免CPU峰值
    - 原子切换：加载完成后瞬间切换
    - 增量加载：缓存未命中时从DB补充
    - LRU淘汰：超过上限时自动淘汰最旧的chat
    """

    def __init__(
        self,
        batch_size: int = 500,
        batch_delay: float = 0.05,
        refresh_interval: int = 0,
        enable_incremental: bool = True,
        max_messages_per_chat: int = 10000,
        max_total_messages: int = 100000,
        enable_lru_eviction: bool = True,
        max_chats: int = 1000,
    ):
        # 双缓冲
        self.buffer_a = None      # 当前使用的缓存: {chat_id: {"messages": [...], "ts": timestamp}}
        self.buffer_b = None      # 后台加载的缓存
        self.buffer_lock = threading.Lock()

        # 加载配置
        self.batch_size = int(batch_size)
        self.batch_delay = float(batch_delay)
        self.refresh_interval = int(refresh_interval)
        self.enable_incremental = bool(enable_incremental)
        self.max_messages_per_chat = int(max_messages_per_chat)
        self.max_total_messages = int(max_total_messages)
        self.enable_lru_eviction = bool(enable_lru_eviction)
        self.max_chats = int(max_chats)

        # 状态
        self.loading = False
        self.load_lock = asyncio.Lock()
        self.last_refresh = 0
        self.stats = ModuleStats("full_message_cache")
        # 大循环让出事件循环的批次阈值
        self._yield_batch_size = max(100, self.batch_size * 5)

        # 增量加载状态
        self._incremental_loading: Set[str] = set()
        self._incremental_lock = threading.Lock()
        self._incremental_last_attempt: Dict[str, float] = {}

        # 统计信息
        self._total_messages = 0
        self._total_chats = 0
        self._stats_lock = threading.Lock()

        # 刷新循环控制
        self._running: bool = False
        self._refresh_task: Optional[asyncio.Task] = None

        # 启动时立即开始加载
        try:
            asyncio.get_running_loop().create_task(self._load_to_buffer_b())
        except RuntimeError:
            pass  # 没有运行中的事件循环，稍后加载

    def is_loaded(self) -> bool:
        """检查缓存是否已加载"""
        with self.buffer_lock:
            return self.buffer_a is not None

    def size(self) -> Dict[str, int]:
        """获取缓存大小统计"""
        with self.buffer_lock:
            if self.buffer_a is None:
                return {"chats": 0, "messages": 0}
            return {
                "chats": len(self.buffer_a),
                "messages": self._total_messages,
            }

    def get_memory_usage(self) -> int:
        """获取缓存内存使用量（字节）"""
        with self.buffer_lock:
            if self.buffer_a is None:
                return 0
            return int(MemoryUtils.get_size(self.buffer_a))

    def get_messages(
        self,
        chat_id: str,
        time_cond: Optional[Dict[str, Any]] = None,
        limit: int = 0,
        limit_mode: str = "latest",
        filter_bot: bool = False,
        filter_command: bool = False,
        filter_intercept_message_level: Optional[int] = None,
    ) -> Optional[List[Any]]:
        """从缓存获取消息

        Returns:
            Optional[List[Any]]: 缓存命中返回消息列表，未命中返回None
        """
        if not self.is_loaded():
            return None

        cid = str(chat_id)
        with self.buffer_lock:
            if self.buffer_a is None:
                return None

            chat_data = self.buffer_a.get(cid)
            if chat_data is None:
                return None

            messages = chat_data.get("messages", [])
            if not messages:
                return None

            # LRU touch
            if cid in self.buffer_a:
                self.buffer_a.move_to_end(cid)

        # 在锁外进行过滤（避免锁粒度过大）
        filtered = self._filter_messages(
            messages,
            time_cond,
            limit,
            limit_mode,
            filter_bot,
            filter_command,
            filter_intercept_message_level,
        )

        return filtered

    def _filter_messages(
        self,
        messages: List[Any],
        time_cond: Optional[Dict[str, Any]],
        limit: int,
        limit_mode: str,
        filter_bot: bool,
        filter_command: bool,
        filter_intercept_message_level: Optional[int],
    ) -> List[Any]:
        """过滤消息列表"""
        result = []

        # 解析时间条件
        start_ts = None
        end_ts = None
        if time_cond and isinstance(time_cond, dict):
            start_ts = time_cond.get("$gt", time_cond.get("$gte"))
            end_ts = time_cond.get("$lt", time_cond.get("$lte"))

        for msg in messages:
            try:
                # 过滤机器人消息
                if filter_bot:
                    is_bot = getattr(msg, "is_bot", False)
                    if is_bot:
                        continue

                # 过滤命令消息
                if filter_command:
                    is_command = getattr(msg, "is_command", False)
                    if is_command:
                        continue

                # 过滤拦截等级
                if filter_intercept_message_level is not None:
                    msg_level = getattr(msg, "intercept_message_level", 0)
                    if msg_level > filter_intercept_message_level:
                        continue

                # 时间范围过滤
                msg_time = getattr(msg, "time", 0)
                if start_ts is not None and msg_time < start_ts:
                    continue
                if end_ts is not None and msg_time > end_ts:
                    continue

                result.append(msg)
            except Exception:
                continue

        # 应用 limit
        if limit > 0:
            if limit_mode == "latest":
                if len(result) > limit:
                    result = result[-limit:]
            else:  # earliest
                if len(result) > limit:
                    result = result[:limit]

        return result

    async def add_message(self, chat_id: str, message: Any, chat_stream: Any = None) -> bool:
        """添加消息到缓存

        Args:
            chat_id: 聊天ID
            message: 消息对象 (MessageSending 或 MessageRecv)
            chat_stream: ChatStream 对象，用于获取 chat_info

        Returns:
            bool: 是否成功添加
        """
        if not self.is_loaded():
            return False

        # 转换消息对象为 DatabaseMessages
        db_msg = self._convert_to_database_messages(message, chat_stream)
        if db_msg is None:
            return False

        cid = str(chat_id)
        should_evict = False

        with self.buffer_lock:
            if self.buffer_a is None:
                return False

            # 获取或创建chat数据
            is_new_chat = cid not in self.buffer_a
            if is_new_chat:
                self.buffer_a[cid] = {"messages": [], "ts": time.time(), "msg_ids": set()}
                self.buffer_a.move_to_end(cid)

            chat_data = self.buffer_a[cid]
            messages = chat_data["messages"]

            # 使用集合加速 message_id 去重
            msg_ids = chat_data.get("msg_ids")
            if msg_ids is None:
                msg_ids = {
                    getattr(existing_msg, "message_id", None)
                    for existing_msg in messages
                    if getattr(existing_msg, "message_id", None) is not None
                }
                chat_data["msg_ids"] = msg_ids

            new_msg_id = getattr(db_msg, "message_id", None)
            if new_msg_id is not None and new_msg_id in msg_ids:
                # 消息已存在，跳过添加（不是错误，是正常的去重）
                return False

            removed_oldest = False
            # 检查单chat消息数上限
            if len(messages) >= self.max_messages_per_chat:
                # 移除最旧的消息
                popped_msg = messages.pop(0)
                popped_msg_id = getattr(popped_msg, "message_id", None)
                if popped_msg_id is not None:
                    msg_ids.discard(popped_msg_id)
                removed_oldest = True

            # 添加新消息
            messages.append(db_msg)
            if new_msg_id is not None:
                msg_ids.add(new_msg_id)
            chat_data["ts"] = time.time()
            self.buffer_a.move_to_end(cid)

            # 更新统计：若达到单chat上限发生替换，则总消息数净变化为 0
            with self._stats_lock:
                if is_new_chat:
                    self._total_chats += 1
                if not removed_oldest:
                    self._total_messages += 1

            should_evict = self.enable_lru_eviction

        # 锁外执行淘汰，避免锁内重操作与事件循环长时间阻塞
        if should_evict:
            await self._evict_if_needed()

        return True

    def _convert_to_database_messages(self, message: Any, chat_stream: Any = None) -> Optional[Any]:
        """将 MessageSending 或 MessageRecv 转换为 DatabaseMessages

        Args:
            message: MessageSending 或 MessageRecv 对象
            chat_stream: ChatStream 对象

        Returns:
            DatabaseMessages 对象，如果转换失败返回 None
        """
        try:
            from src.chat.message_receive.message import MessageSending, MessageRecv
            from src.common.data_models.database_data_model import (
                DatabaseMessages, DatabaseUserInfo, DatabaseGroupInfo, DatabaseChatInfo
            )

            # 通知消息不存储
            if isinstance(message, MessageRecv) and message.is_notify:
                return None

            # 获取消息基本信息
            msg_id = message.message_info.message_id
            msg_time = float(message.message_info.time)
            user_info_dict = message.message_info.user_info.to_dict()

            # 处理文本内容
            processed_plain_text = message.processed_plain_text or ""

            # 根据 MessageSending 或 MessageRecv 设置不同字段
            if isinstance(message, MessageSending):
                display_message = message.display_message or ""
                interest_value = 0
                is_mentioned = False
                is_at = False
                reply_probability_boost = 0.0
                reply_to = message.reply_to or ""
                priority_mode = ""
                priority_info = ""
                is_emoji = False
                is_picid = False
                is_notify = False
                is_command = False
                key_words = ""
                key_words_lite = ""
                selected_expressions = message.selected_expressions or ""
                intercept_message_level = 0
            else:  # MessageRecv
                display_message = ""
                interest_value = message.interest_value
                is_mentioned = message.is_mentioned
                is_at = message.is_at
                reply_probability_boost = message.reply_probability_boost
                reply_to = ""
                priority_mode = message.priority_mode or ""
                priority_info = json.dumps(message.priority_info) if message.priority_info else ""
                is_emoji = message.is_emoji
                is_picid = message.is_picid
                is_notify = message.is_notify
                is_command = message.is_command
                intercept_message_level = getattr(message, "intercept_message_level", 0)
                # 序列化关键词列表为JSON字符串
                key_words = json.dumps(message.key_words, ensure_ascii=False) if message.key_words else "[]"
                key_words_lite = json.dumps(message.key_words_lite, ensure_ascii=False) if message.key_words_lite else "[]"
                selected_expressions = ""

            # 获取 chat_info
            if chat_stream:
                chat_info_dict = chat_stream.to_dict()
                group_info_from_chat = chat_info_dict.get("group_info") or {}
                user_info_from_chat = chat_info_dict.get("user_info") or {}

                chat_info_stream_id = chat_info_dict.get("stream_id", "")
                chat_info_platform = chat_info_dict.get("platform", "")
                chat_info_create_time = float(chat_info_dict.get("create_time", 0.0))
                chat_info_last_active_time = float(chat_info_dict.get("last_active_time", 0.0))

                chat_info_user_id = user_info_from_chat.get("user_id", "")
                chat_info_user_nickname = user_info_from_chat.get("user_nickname", "")
                chat_info_user_cardname = user_info_from_chat.get("user_cardname")
                chat_info_user_platform = user_info_from_chat.get("platform", "")

                chat_info_group_id = group_info_from_chat.get("group_id")
                chat_info_group_name = group_info_from_chat.get("group_name")
                chat_info_group_platform = group_info_from_chat.get("platform")
            else:
                # 如果没有 chat_stream，使用默认值
                chat_info_stream_id = ""
                chat_info_platform = ""
                chat_info_create_time = 0.0
                chat_info_last_active_time = 0.0
                chat_info_user_id = ""
                chat_info_user_nickname = ""
                chat_info_user_cardname = None
                chat_info_user_platform = ""
                chat_info_group_id = None
                chat_info_group_name = None
                chat_info_group_platform = None

            # 构造 DatabaseMessages 对象
            db_msg = DatabaseMessages(
                message_id=msg_id,
                time=msg_time,
                chat_id=chat_stream.stream_id if chat_stream else "",
                reply_to=reply_to,
                interest_value=interest_value,
                key_words=key_words,
                key_words_lite=key_words_lite,
                is_mentioned=is_mentioned,
                is_at=is_at,
                reply_probability_boost=reply_probability_boost,
                processed_plain_text=processed_plain_text,
                display_message=display_message,
                priority_mode=priority_mode,
                priority_info=priority_info,
                is_emoji=is_emoji,
                is_picid=is_picid,
                is_command=is_command,
                intercept_message_level=intercept_message_level,
                is_notify=is_notify,
                selected_expressions=selected_expressions,
                # Flattened user_info (message sender)
                user_id=user_info_dict.get("user_id", ""),
                user_nickname=user_info_dict.get("user_nickname", ""),
                user_cardname=user_info_dict.get("user_cardname"),
                user_platform=user_info_dict.get("platform", ""),
                # Flattened chat_info
                chat_info_stream_id=chat_info_stream_id,
                chat_info_platform=chat_info_platform,
                chat_info_create_time=chat_info_create_time,
                chat_info_last_active_time=chat_info_last_active_time,
                chat_info_user_id=chat_info_user_id,
                chat_info_user_nickname=chat_info_user_nickname,
                chat_info_user_cardname=chat_info_user_cardname,
                chat_info_user_platform=chat_info_user_platform,
                chat_info_group_id=chat_info_group_id,
                chat_info_group_name=chat_info_group_name,
                chat_info_group_platform=chat_info_group_platform,
            )

            return db_msg

        except Exception as e:
            logger.error(f"[FullCache] 转换消息对象失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def remove_message(self, chat_id: str, message_id: str) -> bool:
        """从缓存删除消息

        Returns:
            bool: 是否成功删除
        """
        if not self.is_loaded():
            return False

        cid = str(chat_id)
        with self.buffer_lock:
            if self.buffer_a is None:
                return False

            chat_data = self.buffer_a.get(cid)
            if chat_data is None:
                return False

            messages = chat_data["messages"]

            # BUG FIX: 兼容 message_id 和 id 字段
            # 优先检查 message_id，然后回退到 id
            def _get_msg_id(m):
                msg_id = getattr(m, "message_id", None)
                if msg_id is not None:
                    return msg_id
                return getattr(m, "id", None)

            kept_messages = []
            removed_count = 0
            for m in messages:
                if _get_msg_id(m) == message_id:
                    removed_count += 1
                else:
                    kept_messages.append(m)

            if removed_count == 0:
                return False

            messages[:] = kept_messages

            msg_ids = chat_data.get("msg_ids")
            if isinstance(msg_ids, set):
                msg_ids.discard(message_id)

            with self._stats_lock:
                self._total_messages = max(0, self._total_messages - removed_count)

            if len(messages) == 0:
                # 如果chat没有消息了，删除chat
                del self.buffer_a[cid]
                with self._stats_lock:
                    self._total_chats = max(0, self._total_chats - 1)

            return True

    async def _evict_if_needed(self):
        """如果需要，执行LRU淘汰（异步，分批让出事件循环）"""
        if not self.enable_lru_eviction:
            return

        while True:
            with self.buffer_lock:
                if self.buffer_a is None:
                    return

                chat_count = len(self.buffer_a)
                message_count = self._total_messages
                if chat_count <= self.max_chats and message_count <= self.max_total_messages:
                    return

                # 锁内仅做快照，不做淘汰循环
                snapshot: List[Tuple[str, int]] = [
                    (chat_id, len(chat_data.get("messages", [])))
                    for chat_id, chat_data in self.buffer_a.items()
                ]

            # 锁外计算需要淘汰的 chat 列表（按 LRU 顺序，最旧优先）
            overflow_chats = max(0, chat_count - self.max_chats)
            overflow_messages = max(0, message_count - self.max_total_messages)

            evict_chat_ids: List[str] = []
            removed_messages_planned = 0
            for chat_id, msg_len in snapshot:
                if overflow_chats > 0 or removed_messages_planned < overflow_messages:
                    evict_chat_ids.append(chat_id)
                    removed_messages_planned += msg_len
                    if overflow_chats > 0:
                        overflow_chats -= 1
                else:
                    break

            if not evict_chat_ids:
                return

            # 分批淘汰，批次间让出事件循环
            for i in range(0, len(evict_chat_ids), self._yield_batch_size):
                batch_chat_ids = evict_chat_ids[i : i + self._yield_batch_size]
                removed_msgs = 0
                removed_chats = 0

                with self.buffer_lock:
                    if self.buffer_a is None:
                        return
                    for chat_id in batch_chat_ids:
                        old_chat_data = self.buffer_a.pop(chat_id, None)
                        if old_chat_data is None:
                            continue
                        old_messages = old_chat_data.get("messages", [])
                        removed_msgs += len(old_messages)
                        removed_chats += 1

                if removed_chats > 0:
                    with self._stats_lock:
                        self._total_messages = max(0, self._total_messages - removed_msgs)
                        self._total_chats = max(0, self._total_chats - removed_chats)

                await asyncio.sleep(0)

    def trigger_incremental_load(self, chat_id: str):
        """触发增量加载（不阻塞）"""
        if not self.enable_incremental:
            return

        cid = str(chat_id)
        now = time.time()

        with self._incremental_lock:
            # 防抖：同一chat 1秒内最多触发一次
            last_attempt = self._incremental_last_attempt.get(cid, 0.0)
            if now - last_attempt < 1.0:
                return
            self._incremental_last_attempt[cid] = now

            # 检查是否正在加载
            if cid in self._incremental_loading:
                return
            self._incremental_loading.add(cid)

        try:
            loop = asyncio.get_running_loop()
        except Exception:
            with self._incremental_lock:
                self._incremental_loading.discard(cid)
            return

        async def _load():
            try:
                await self._incremental_load_chat(cid)
            except Exception as e:
                logger.debug(f"[FullCache] 增量加载失败 chat_id={cid}: {e}")
            finally:
                with self._incremental_lock:
                    self._incremental_loading.discard(cid)

        try:
            loop.create_task(_load())
        except Exception:
            with self._incremental_lock:
                self._incremental_loading.discard(cid)

    async def _incremental_load_chat(self, chat_id: str):
        """增量加载单个chat的消息"""
        try:
            from src.common import message_repository

            # 查询该chat的所有消息（通过 to_thread 避免阻塞事件循环）
            def _sync_find() -> list:
                return message_repository.find_messages(
                    {"chat_id": chat_id},
                    sort=None,
                    limit=0,  # 不限制数量
                    limit_mode="latest",
                    filter_bot=False,
                    filter_command=False,
                    filter_intercept_message_level=None,
                )

            messages = await asyncio.to_thread(_sync_find)

            if not messages:
                return

            # 限制消息数量（锁外处理）
            if len(messages) > self.max_messages_per_chat:
                messages = messages[-self.max_messages_per_chat:]

            chat_already_exists = False
            # 更新缓存
            with self.buffer_lock:
                if self.buffer_a is None:
                    return

                # 统计更新必须在锁保护下完成，避免竞态条件
                with self._stats_lock:
                    # BUG FIX: 检查 chat 是否已存在，避免重复统计
                    chat_already_exists = chat_id in self.buffer_a
                    old_message_count = 0
                    if chat_already_exists:
                        old_message_count = len(self.buffer_a[chat_id].get("messages", []))

                    # 更新统计（只增加增量，避免重复统计）
                    # 减去旧消息数（如果chat已存在）
                    if chat_already_exists:
                        self._total_messages -= old_message_count
                    else:
                        # 新增的chat
                        self._total_chats += 1
                    # 加上新消息数
                    self._total_messages += len(messages)

                msg_ids = {
                    getattr(existing_msg, "message_id", None)
                    for existing_msg in messages
                    if getattr(existing_msg, "message_id", None) is not None
                }

                # 更新缓存数据（在统计更新之后，避免统计数据不一致）
                self.buffer_a[chat_id] = {
                    "messages": messages,
                    "msg_ids": msg_ids,
                    "ts": time.time(),
                }
                self.buffer_a.move_to_end(chat_id)

            # 锁外淘汰，避免锁内重操作
            if self.enable_lru_eviction:
                await self._evict_if_needed()

            logger.debug(f"[FullCache] 增量加载完成 chat_id={chat_id}, 消息数={len(messages)}, 已存在={chat_already_exists}")
        except Exception as e:
            logger.error(f"[FullCache] 增量加载异常 chat_id={chat_id}: {e}")

    async def _load_to_buffer_b(self):
        """缓慢加载数据到缓冲区B

        数据库查询通过 asyncio.to_thread() 在独立线程中执行；
        批内 Python 大循环按阈值让出事件循环，避免协程长时间饥饿。
        """
        async with self.load_lock:
            if self.loading:
                return
            self.loading = True

        try:
            logger.info("[FullCache] 开始全量加载消息缓存到缓冲区B...")

            from src.common.database.database_model import Messages
            from src.common.data_models.database_data_model import DatabaseMessages

            buffer_b_data: OrderedDict = OrderedDict()
            total_messages = 0
            total_chats = 0
            offset = 0
            processed_in_round = 0

            def _sync_fetch_batch(batch_offset: int):
                return list(
                    Messages.select()
                    .limit(self.batch_size)
                    .offset(batch_offset)
                )

            while True:
                # 每批同步查询放入线程，避免阻塞事件循环
                batch = await asyncio.to_thread(_sync_fetch_batch, offset)
                if not batch:
                    break

                # 批内大循环添加 yield 点
                for msg in batch:
                    chat_id = str(getattr(msg, "chat_id", ""))
                    if not chat_id:
                        continue

                    if chat_id not in buffer_b_data:
                        buffer_b_data[chat_id] = {
                            "messages": [],
                            "msg_ids": set(),
                            "ts": time.time(),
                        }
                        total_chats += 1

                    chat_data = buffer_b_data[chat_id]
                    messages = chat_data["messages"]
                    msg_ids = chat_data["msg_ids"]

                    # 限制单 chat 消息数
                    if len(messages) < self.max_messages_per_chat:
                        db_msg = DatabaseMessages(**msg.__data__)
                        messages.append(db_msg)
                        msg_id = getattr(db_msg, "message_id", None)
                        if msg_id is not None:
                            msg_ids.add(msg_id)
                        total_messages += 1

                    processed_in_round += 1
                    if processed_in_round >= self._yield_batch_size:
                        processed_in_round = 0
                        await asyncio.sleep(0)

                logger.debug(
                    f"[FullCache] 加载进度: {total_messages} 条消息, "
                    f"{total_chats} 个聊天"
                )

                offset += self.batch_size

                if total_messages >= self.max_total_messages:
                    logger.warning(
                        f"[FullCache] 达到总消息数上限 "
                        f"{self.max_total_messages}，停止加载"
                    )
                    break

                if self.batch_delay > 0:
                    await asyncio.sleep(self.batch_delay)
                else:
                    await asyncio.sleep(0)

            # 加载完成，原子切换
            with self.buffer_lock:
                self.buffer_b = buffer_b_data
                # 原子切换：buffer_b → buffer_a
                self.buffer_a, self.buffer_b = self.buffer_b, None

                # 更新统计
                with self._stats_lock:
                    self._total_messages = total_messages
                    self._total_chats = total_chats

            self.last_refresh = time.time()
            logger.info(
                f"[FullCache] 缓存加载完成并切换: "
                f"{total_messages} 条消息, {total_chats} 个聊天"
            )

        except Exception as e:
            logger.error(f"[FullCache] 缓存加载失败: {e}")
        finally:
            async with self.load_lock:
                self.loading = False

    async def _refresh_loop(self):
        """定期刷新循环（可通过 stop() 优雅退出）"""
        self._running = True
        try:
            while self._running:
                # 使用短间隔轮询以便及时响应 stop()
                elapsed = 0.0
                while elapsed < self.refresh_interval and self._running:
                    step = min(1.0, self.refresh_interval - elapsed)
                    await asyncio.sleep(step)
                    elapsed += step
                if not self._running:
                    break
                logger.info("[FullCache] 触发定期刷新...")
                await self._load_to_buffer_b()
        except asyncio.CancelledError:
            logger.info("[FullCache] 刷新循环已取消")
        finally:
            self._running = False

    def start(self):
        """启动定期刷新循环"""
        if self._running:
            return
        if self.refresh_interval <= 0:
            return
        try:
            loop = asyncio.get_running_loop()
            self._refresh_task = loop.create_task(self._refresh_loop())
            logger.info(
                f"[FullCache] 刷新循环已启动，间隔 {self.refresh_interval}s"
            )
        except RuntimeError:
            logger.warning("[FullCache] 无法启动刷新循环：没有运行中的事件循环")

    async def stop(self):
        """停止刷新循环并清理任务"""
        self._running = False
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None
        logger.info("[FullCache] 刷新循环已停止")

    def refresh(self):
        """手动刷新缓存"""
        try:
            asyncio.get_running_loop().create_task(self._load_to_buffer_b())
            logger.info("[FullCache] 已触发手动刷新")
        except RuntimeError:
            logger.warning("[FullCache] 无法触发刷新：没有运行中的事件循环")

    def clear(self):
        """清空缓存"""
        with self.buffer_lock:
            self.buffer_a = None
            self.buffer_b = None
        with self._stats_lock:
            self._total_messages = 0
            self._total_chats = 0
        logger.info("[FullCache] 缓存已清空")


# ===== 全量消息缓存模块 =====
class FullMessageCacheModule:
    """全量消息缓存模块 - 拦截所有数据库查询"""

    def __init__(
        self,
        batch_size: int = 500,
        batch_delay: float = 0.05,
        refresh_interval: int = 0,
        enable_incremental: bool = True,
        max_messages_per_chat: int = 10000,
        max_total_messages: int = 100000,
        enable_lru_eviction: bool = True,
        max_chats: int = 1000,
    ):
        self.cache = FullMessageCache(
            batch_size=batch_size,
            batch_delay=batch_delay,
            refresh_interval=refresh_interval,
            enable_incremental=enable_incremental,
            max_messages_per_chat=max_messages_per_chat,
            max_total_messages=max_total_messages,
            enable_lru_eviction=enable_lru_eviction,
            max_chats=max_chats,
        )
        self.stats = self.cache.stats

        self._orig_find_messages = None
        self._orig_store_message = None
        self._patched = False
        self._patched_store_message = False
        self._alias_patches: List[Tuple[str, str, Any]] = []
        self._patched_find_messages_func = None
        self._patched_store_message_func = None

    def get_memory_usage(self) -> int:
        """获取模块内存占用（字节）"""
        return self.cache.get_memory_usage()

    # ===== 代理属性，供统计报告使用 =====
    def size(self):
        """获取缓存大小统计"""
        return self.cache.size().get("messages", 0)

    @property
    def loading(self) -> bool:
        """是否正在加载"""
        return self.cache.loading

    @property
    def last_refresh(self) -> float:
        """上次刷新时间"""
        return self.cache.last_refresh

    @property
    def refresh_interval(self) -> int:
        """刷新间隔"""
        return self.cache.refresh_interval

    def apply_patch(self):
        """应用补丁"""
        if self._patched:
            return

        try:
            from src.common import message_repository

            # 保存原始函数
            self._orig_find_messages = message_repository.find_messages
            module = self

            def patched_find_messages(
                message_filter,
                sort=None,
                limit=0,
                limit_mode="latest",
                filter_bot=False,
                filter_command=False,
                filter_intercept_message_level=None,
            ):
                """全量缓存模式下的查询拦截"""
                mf = message_filter or {}
                chat_id = mf.get("chat_id")

                # 如果有 chat_id，优先从全量缓存读取
                if chat_id:
                    messages = module.cache.get_messages(
                        chat_id=chat_id,
                        time_cond=mf.get("time"),
                        limit=limit,
                        limit_mode=limit_mode,
                        filter_bot=filter_bot,
                        filter_command=filter_command,
                        filter_intercept_message_level=filter_intercept_message_level,
                    )

                    if messages is not None:
                        # 缓存命中
                        module.stats.hit()
                        return messages

                    # 缓存未命中，触发增量加载
                    if module.cache.enable_incremental:
                        module.cache.trigger_incremental_load(chat_id)

                # 回退到原始查询
                t0 = time.time()
                result = module._orig_find_messages(
                    message_filter, sort, limit, limit_mode,
                    filter_bot, filter_command, filter_intercept_message_level
                )
                module.stats.miss(time.time() - t0)
                return result

            # PatchChain 注册 find_messages（冲突检测 + 链式追踪）
            _pc = None
            try:
                _cm = sys.modules.get("CM_perf_opt_core")
                if _cm and hasattr(_cm, "get_patch_chain"):
                    _pc = _cm.get_patch_chain()
            except Exception:
                pass
            if _pc is not None:
                _pc.register_patch(
                    "find_messages", "full_message_cache",
                    self._orig_find_messages, patched_find_messages,
                )

            # 替换函数
            message_repository.find_messages = patched_find_messages
            self._patched_find_messages_func = patched_find_messages

            # 替换已导入的引用
            for n, m in list(sys.modules.items()):
                if m and getattr(m, "find_messages", None) is self._orig_find_messages:
                    try:
                        self._alias_patches.append((n, "find_messages", getattr(m, "find_messages", None)))
                    except Exception:
                        pass
                    setattr(m, "find_messages", patched_find_messages)
                    logger.debug(f"[FullCache] 替换 {n}.find_messages")

            # 写入侧补丁
            try:
                from src.chat.message_receive.storage import MessageStorage

                if not self._patched_store_message:
                    self._orig_store_message = MessageStorage.__dict__.get("store_message")
                    orig_store_callable = getattr(MessageStorage, "store_message", None)

                    async def patched_store_message(message, chat_stream):
                        # 先写入缓存（立即生效）
                        chat_id = getattr(chat_stream, "stream_id", None)
                        if chat_id:
                            await module.cache.add_message(chat_id, message, chat_stream)

                        # 异步写入数据库
                        if callable(orig_store_callable):
                            try:
                                await orig_store_callable(message, chat_stream)
                            except Exception as e:
                                # DB写入失败不影响缓存一致性
                                logger.warning(f"[FullCache] DB写入失败，但缓存已更新: {e}")

                    # PatchChain 注册 store_message
                    if _pc is not None:
                        _pc.register_patch(
                            "store_message", "full_message_cache",
                            orig_store_callable, patched_store_message,
                        )

                    MessageStorage.store_message = staticmethod(patched_store_message)
                    self._patched_store_message_func = patched_store_message

                    # 替换已导入的引用
                    for n, m in list(sys.modules.items()):
                        if m and getattr(m, "store_message", None) is orig_store_callable:
                            try:
                                self._alias_patches.append((n, "store_message", getattr(m, "store_message", None)))
                            except Exception:
                                pass
                            setattr(m, "store_message", patched_store_message)
                            logger.debug(f"[FullCache] 替换 {n}.store_message")

                    self._patched_store_message = True
                    logger.info("[FullCache] ✓ 写入侧补丁已应用")
            except Exception as e:
                logger.warning(f"[FullCache] 写入侧补丁失败: {e}")

            self._patched = True
            logger.info("[FullCache] ✓ 补丁应用成功")
        except Exception as e:
            logger.error(f"[FullCache] ✗ 补丁失败: {e}")

    def remove_patch(self):
        """移除补丁"""
        if not self._patched or not self._orig_find_messages:
            return

        try:
            from src.common import message_repository

            # 回滚查询补丁
            message_repository.find_messages = self._orig_find_messages

            # 回滚写入补丁
            try:
                if self._patched_store_message and self._orig_store_message is not None:
                    from src.chat.message_receive.storage import MessageStorage
                    MessageStorage.store_message = self._orig_store_message
                    self._patched_store_message = False
            except Exception as e:
                logger.warning(f"[FullCache] 回滚写入侧补丁失败: {e}")

            # 回滚已导入的别名引用
            try:
                patched_find = self._patched_find_messages_func
                patched_store = self._patched_store_message_func
                for mod_name, attr, original in list(self._alias_patches):
                    mod = sys.modules.get(mod_name)
                    if not mod:
                        continue
                    try:
                        cur = getattr(mod, attr, None)
                        if attr == "find_messages" and patched_find is not None and cur is patched_find:
                            setattr(mod, attr, original)
                        elif attr == "store_message" and patched_store is not None and cur is patched_store:
                            setattr(mod, attr, original)
                    except Exception:
                        continue
            finally:
                self._alias_patches.clear()
                self._patched_find_messages_func = None
                self._patched_store_message_func = None

            # PatchChain 取消注册
            try:
                _cm = sys.modules.get("CM_perf_opt_core")
                if _cm and hasattr(_cm, "get_patch_chain"):
                    _pc = _cm.get_patch_chain()
                    _pc.unregister_patch("find_messages", "full_message_cache")
                    _pc.unregister_patch("store_message", "full_message_cache")
            except Exception:
                pass

            self._patched = False
            logger.info("[FullCache] 补丁已移除")
        except Exception as e:
            logger.error(f"[FullCache] 移除补丁失败: {e}")
