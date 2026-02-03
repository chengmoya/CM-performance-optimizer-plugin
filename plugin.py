"""
CM 性能优化插件 v4.3.1

功能模块：
1. 消息缓存 (message_cache) - 缓存 find_messages 查询结果
2. 人物信息缓存 (person_cache) - 缓存人物信息查询
3. 表达式缓存 (expression_cache) - 双缓冲+缓慢加载+原子切换
4. 黑话缓存 (slang_cache) - 双缓冲+缓慢加载+原子切换+内容索引
5. 知识库图谱缓存 (kg_cache) - 双缓冲+缓慢加载+原子切换

安装：将目录放入 MaiBot/plugins/ 下，重启 MaiBot
依赖：无额外依赖
"""

import sys
import asyncio
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from collections import OrderedDict

try:
    from src.plugin_system.apis.plugin_register_api import register_plugin
    from src.plugin_system.base.base_plugin import BasePlugin
    from src.plugin_system.base.base_events_handler import BaseEventHandler
    from src.plugin_system.base.component_types import EventType
    from src.plugin_system.base.config_types import ConfigField, ConfigSection, ConfigLayout, ConfigTab
    from src.common.logger import get_logger
except ImportError:
    # 让本文件可被“独立 import”用于静态检查/离线测试
    class _FallbackEventType:
        ON_STOP = "on_stop"

    EventType = _FallbackEventType  # type: ignore

    class BasePlugin:
        def __init__(self, plugin_dir=None):
            pass

    class BaseEventHandler:
        def __init__(self, plugin_dir=None):
            pass

    class ConfigField:
        def __init__(self, **kw):
            pass

    class ConfigSection:
        def __init__(self, **kw):
            pass

    class ConfigLayout:
        def __init__(self, **kw):
            pass

    class ConfigTab:
        def __init__(self, **kw):
            pass

    def register_plugin(cls):
        return cls

    def get_logger(name):
        import logging

        return logging.getLogger(name)

logger = get_logger("CM_perf_opt")


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


# ===== 通用缓存类 =====
class TTLCache:
    """带TTL的LRU缓存"""
    def __init__(self, max_size=500, ttl=120.0):
        self.max_size, self.ttl = max_size, ttl
        self.data = OrderedDict()
        self.ts = {}
        self.lock = threading.Lock()
    
    def get(self, k):
        with self.lock:
            if k not in self.data: return None, False
            if time.time() - self.ts[k] > self.ttl:
                del self.data[k], self.ts[k]
                return None, False
            self.data.move_to_end(k)
            return self.data[k], True
    
    def _purge_expired_locked(self, now: float) -> int:
        """清理已过期的 key（需在 lock 内调用）。返回清理数量。"""
        removed = 0
        # OrderedDict 在迭代时不能修改，先复制 keys
        for kk in list(self.data.keys()):
            ts = self.ts.get(kk)
            if ts is None or (now - ts > self.ttl):
                self.data.pop(kk, None)
                self.ts.pop(kk, None)
                removed += 1
        return removed

    def set(self, k, v):
        with self.lock:
            now = time.time()
            # set 时机会性清理过期数据，避免“从未再次 get 的 key”长期占位
            self._purge_expired_locked(now)

            # 更新 LRU 顺序（如果是更新已有 key）
            if k in self.data:
                self.data.move_to_end(k)

            self.data[k] = v
            self.ts[k] = now

            # LRU 控制尺寸
            while len(self.data) > self.max_size:
                old = next(iter(self.data))
                del self.data[old], self.ts[old]
    
    def invalidate(self, k):
        with self.lock:
            if k in self.data:
                del self.data[k], self.ts[k]
    
    def clear(self):
        with self.lock:
            self.data.clear()
            self.ts.clear()
    
    def size(self): return len(self.data)
    
    def get_memory_usage(self):
        """获取缓存内存使用量（字节）"""
        with self.lock:
            data_size = MemoryUtils.get_size(self.data)
            ts_size = MemoryUtils.get_size(self.ts)
            return data_size + ts_size


# ===== 统计类 =====
class ModuleStats:
    """单个模块的统计。

    说明：
    - hit/miss：只统计“可缓存”的请求（也就是我们希望命中的那部分）。
    - skipped：不可缓存/主动跳过缓存的请求（例如高基数的范围查询、limit 过大等）。

    这样报告中的“命中率”才不会被大量跳过请求拉低。
    """

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
        """记录不可缓存/被跳过的情况（不进入命中率分母）。"""
        with self.lock:
            self.t_skipped += 1
            self.i_skipped += 1

    def filtered(self):
        """记录命中但被过滤的情况（如chat_id不匹配）。"""
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


def rate(hit, miss, filtered=0):
    t = hit + miss + filtered
    return (hit / t * 100) if t > 0 else 0


class ChatVersionTracker:
    """写入侧递增版本号。

    用于让“按 chat_id 拉最近上下文”的缓存键稳定命中，且在新消息写入后自动失效。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ver: Dict[str, int] = {}

    def get(self, chat_id: str) -> int:
        with self._lock:
            return int(self._ver.get(chat_id, 0))

    def bump(self, chat_id: str) -> int:
        with self._lock:
            v = int(self._ver.get(chat_id, 0)) + 1
            self._ver[chat_id] = v
            return v


_chat_versions = ChatVersionTracker()

# 对 `time: {"$lt": now}` / `{ "$lte": now }` 这类“拉取到当前上下文”查询的归一化窗口。
# 当 `$lt/$lte` 的时间戳距离当前时间超过该窗口时：默认不做归一化，且不缓存，避免误缓存历史查询。
MSG_CACHE_NORMALIZE_LT_WINDOW_SECONDS = 300.0


class MessageHotsetCache:
    """按 chat 缓存“最近 N 条消息”的热集（Hotset）。

    设计目标（对应用户选择的“全异步预热”）：
    - 首次访问不阻塞：仅在后台触发 warmup；当前请求仍走 DB。
    - warmup 完成后：后续滑动窗口范围查询可直接命中热集（无需 DB）。

    注意：这里的缓存是“消息集合缓存”，不是 query-cache。
    """

    def __init__(
        self,
        enabled: bool = True,
        per_chat_limit: int = 200,
        ttl: int = 300,
        max_chats: int = 500,
        ignore_time_limit_when_active: bool = True,
        active_time_window: int = 300,
    ):
        self.enabled = bool(enabled)
        self.per_chat_limit = int(per_chat_limit)
        self.ttl = float(ttl)
        self.max_chats = int(max_chats)
        self.ignore_time_limit_when_active = bool(ignore_time_limit_when_active)
        self.active_time_window = float(active_time_window)

        self._lock = threading.Lock()
        # chat_id -> {"ts": float, "messages": List[DatabaseMessages]}
        self._data: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

        # 运行时状态
        self._warming: set[str] = set()
        self._last_refresh_attempt: Dict[str, float] = {}

    def clear(self):
        with self._lock:
            self._data.clear()
            self._warming.clear()
            self._last_refresh_attempt.clear()

    def get_memory_usage(self) -> int:
        with self._lock:
            return int(MemoryUtils.get_size(self._data))

    def get_messages_if_fresh(self, chat_id: str) -> Optional[List[Any]]:
        """获取指定 chat 的热集快照（仅当 fresh）。

        说明：
        - 返回的是 *copy*（list(msgs)），避免调用方误修改内部缓存。
        - 不 fresh / 未命中返回 None。
        """
        if not self.enabled:
            return None

        cid = str(chat_id)
        now = time.time()
        with self._lock:
            if not self._is_fresh_locked(cid, now):
                return None
            ent = self._data.get(cid)
            if not ent:
                return None
            msgs = ent.get("messages") or []
            # LRU touch
            self._touch_locked(cid)
            # copy snapshot
            return list(msgs)

    def _touch_locked(self, chat_id: str):
        if chat_id in self._data:
            self._data.move_to_end(chat_id)

    def _is_fresh_locked(self, chat_id: str, now: float) -> bool:
        """检查缓存是否新鲜
        
        当聊天流激活时（ignore_time_limit_when_active=True），忽略时间限制
        """
        ent = self._data.get(chat_id)
        if not ent:
            return False
        ts = float(ent.get("ts", 0.0))
        
        # 如果启用了激活时忽略时间限制，检查聊天流是否激活
        if self.ignore_time_limit_when_active:
            try:
                from src.chat.message_receive.chat_stream import ChatManager
                chat_manager = ChatManager()
                stream = chat_manager.get_stream(chat_id)
                if stream:
                    # 聊天流存在且在活跃窗口内，忽略 TTL
                    if (now - stream.last_active_time) <= self.active_time_window:
                        return True
            except Exception:
                pass  # 检查失败，继续使用 TTL
        
        return (now - ts) <= self.ttl

    def try_get_range(
        self,
        chat_id: str,
        start_ts: float,
        end_ts: float,
        limit: int,
        limit_mode: str = "latest",
        filter_intercept_message_level: Optional[int] = None,
    ) -> Optional[List[Any]]:
        """尝试用热集回答范围查询。

        返回 None 表示无法回答（需要回退 DB）。
        
        Args:
            chat_id: 聊天ID
            start_ts: 开始时间戳
            end_ts: 结束时间戳
            limit: 返回消息数量限制
            limit_mode: 限制模式 ("latest" 或 "earliest")
            filter_intercept_message_level: 拦截消息等级过滤（可选）
        """
        if not self.enabled:
            return None

        try:
            chat_id = str(chat_id)
            start_ts_f = float(start_ts)
            end_ts_f = float(end_ts)
            lim = int(limit)
        except Exception:
            return None

        if lim <= 0:
            return None

        now = time.time()
        with self._lock:
            if not self._is_fresh_locked(chat_id, now):
                return None
            ent = self._data.get(chat_id)
            if not ent:
                return None
            msgs = ent.get("messages") or []
            if not msgs:
                return None

            # 检查聊天流是否激活，如果激活则忽略时间限制
            is_active = False
            if self.ignore_time_limit_when_active:
                try:
                    from src.chat.message_receive.chat_stream import ChatManager
                    chat_manager = ChatManager()
                    stream = chat_manager.get_stream(chat_id)
                    if stream and (now - stream.last_active_time) <= self.active_time_window:
                        is_active = True
                except Exception:
                    pass

            # 只有在热集覆盖 start_ts 时才回答（除非聊天流激活）
            if not is_active:
                try:
                    earliest = float(getattr(msgs[0], "time", 0.0))
                except Exception:
                    earliest = 0.0
                if earliest > start_ts_f:
                    return None

            self._touch_locked(chat_id)

        # 在锁外过滤（避免锁粒度过大）
        in_range = []
        for m in msgs:
            try:
                t = float(getattr(m, "time", 0.0))
            except Exception:
                continue
            
            # 如果聊天流激活，忽略时间限制，只按 limit 返回
            if is_active:
                # 只需要过滤拦截等级
                if filter_intercept_message_level is not None:
                    msg_level = getattr(m, "intercept_message_level", 0)
                    if msg_level > filter_intercept_message_level:
                        continue
                in_range.append(m)
            else:
                # 正常的时间范围过滤
                if t < start_ts_f:
                    continue
                if t > end_ts_f:
                    continue
                # 过滤拦截等级
                if filter_intercept_message_level is not None:
                    msg_level = getattr(m, "intercept_message_level", 0)
                    if msg_level > filter_intercept_message_level:
                        continue
                in_range.append(m)

        # 对齐 message_repository.find_messages(limit>0) 的行为：
        # - latest：取末尾 N 条，再按时间正序返回
        # - earliest：取开头 N 条（已经按时间正序）
        if limit_mode == "earliest":
            return in_range[:lim]

        # 默认为 latest
        if len(in_range) <= lim:
            return in_range
        return in_range[-lim:]

    def ensure_warmup(self, chat_id: str):
        """确保该 chat 已触发后台预热（不阻塞）。"""
        if not self.enabled:
            return
        cid = str(chat_id)

        now = time.time()
        with self._lock:
            # fresh 就不需要预热
            if self._is_fresh_locked(cid, now):
                self._touch_locked(cid)
                return

            # warmup 去重 + 简单防抖：同一 chat 1 秒内最多触发一次
            last = float(self._last_refresh_attempt.get(cid, 0.0))
            if now - last < 1.0:
                return
            self._last_refresh_attempt[cid] = now

            if cid in self._warming:
                return
            self._warming.add(cid)

        try:
            loop = asyncio.get_running_loop()
        except Exception:
            # 没有 running loop 时无法后台预热（保持“回退 DB”的正确性）
            with self._lock:
                self._warming.discard(cid)
            return

        async def _warm():
            try:
                from src.common import message_repository

                # 只拉该 chat 的最近 N 条（与 message_repository.limit_mode=latest 行为一致：返回仍为时间正序）
                res = message_repository.find_messages(
                    {"chat_id": cid},
                    sort=None,
                    limit=self.per_chat_limit,
                    limit_mode="latest",
                    filter_bot=False,
                    filter_command=False,
                    filter_intercept_message_level=None,
                )

                now2 = time.time()
                with self._lock:
                    self._data[cid] = {"ts": now2, "messages": res}
                    self._data.move_to_end(cid)

                    # LRU 淘汰
                    while len(self._data) > self.max_chats:
                        self._data.popitem(last=False)
            except Exception as e:
                logger.debug(f"[Hotset] warmup 失败 chat_id={cid}: {e}")
            finally:
                with self._lock:
                    self._warming.discard(cid)

        try:
            loop.create_task(_warm())
        except Exception:
            with self._lock:
                self._warming.discard(cid)


class PersonWarmupManager:
    """人物信息预热（全异步触发，不阻塞主流程）。

    设计目标（对应用户选择的“全异步预热”）：
    - 写入后触发：不 await，仅 create_task。
    - 防抖：同一 chat 的写入在 debounce 窗口内只触发一次。
    - TTL：同一 chat 在 ttl 窗口内只做一次预热（避免 DB 压力）。

    预热策略：
    - 从最近 N 条消息中提取参与者（platform + user_id -> person_id）
    - 最多预热 max_persons_per_chat 个
    - 对每个 person_id 构造 [`Person.__init__()`](../src/person_info/person_info.py:281)
      触发 [`Person.load_from_database()`](../src/person_info/person_info.py:442)
      以便被 [`PersonCacheModule.apply_patch()`](CM-performance-optimizer-plugin/plugin.py:847) 提前填充。
    """

    def __init__(
        self,
        enabled: bool = True,
        per_chat_message_sample: int = 30,
        max_persons_per_chat: int = 20,
        ttl: int = 120,
        debounce_seconds: float = 3.0,
        max_chats: int = 500,
        hotset: Optional[MessageHotsetCache] = None,
    ):
        self.enabled = bool(enabled)
        self.per_chat_message_sample = int(per_chat_message_sample)
        self.max_persons_per_chat = int(max_persons_per_chat)
        self.ttl = float(ttl)
        self.debounce_seconds = float(debounce_seconds)
        self.max_chats = int(max_chats)
        self.hotset = hotset

        self._lock = threading.Lock()
        # chat_id -> last_warm_ts
        self._last_warm: "OrderedDict[str, float]" = OrderedDict()
        # chat_id -> last_trigger_ts (for debounce)
        self._last_trigger: Dict[str, float] = {}
        self._warming: set[str] = set()

        # 统计（轻量）
        self.stats = {
            "triggered": 0,
            "debounce_skipped": 0,
            "fresh_skipped": 0,
            "loaded": 0,
            "errors": 0,
            "fallback_db_reads": 0,
        }

    def get_memory_usage(self) -> int:
        with self._lock:
            return int(MemoryUtils.get_size(self._last_warm) + MemoryUtils.get_size(self._last_trigger) + MemoryUtils.get_size(self._warming))

    def _touch_lru_locked(self, chat_id: str, ts: float):
        self._last_warm[chat_id] = ts
        self._last_warm.move_to_end(chat_id)
        while len(self._last_warm) > self.max_chats:
            self._last_warm.popitem(last=False)

    def ensure_warmup(self, chat_id: str):
        """确保该 chat 已触发后台预热（不阻塞）。"""
        if not self.enabled:
            return

        cid = str(chat_id)
        now = time.time()

        with self._lock:
            # TTL：近期已预热
            last_warm = float(self._last_warm.get(cid, 0.0))
            if last_warm and (now - last_warm) <= self.ttl:
                self.stats["fresh_skipped"] += 1
                self._last_warm.move_to_end(cid)
                return

            # debounce：短时间内只触发一次
            last_tr = float(self._last_trigger.get(cid, 0.0))
            if last_tr and (now - last_tr) < self.debounce_seconds:
                self.stats["debounce_skipped"] += 1
                return
            self._last_trigger[cid] = now

            if cid in self._warming:
                self.stats["debounce_skipped"] += 1
                return
            self._warming.add(cid)
            self.stats["triggered"] += 1

        try:
            loop = asyncio.get_running_loop()
        except Exception:
            with self._lock:
                self._warming.discard(cid)
            return

        async def _warm():
            try:
                msgs: List[Any] = []

                # 1) 优先复用热集（如果可用且 fresh）
                try:
                    if self.hotset:
                        snap = self.hotset.get_messages_if_fresh(cid)
                        if snap is not None:
                            # 取最近 N 条（hotset 本身是时间正序）
                            if self.per_chat_message_sample > 0:
                                msgs = snap[-self.per_chat_message_sample :]
                            else:
                                msgs = snap
                except Exception:
                    msgs = []

                # 2) fallback：查 DB 最近 N 条
                if not msgs:
                    try:
                        from src.common import message_repository

                        res = message_repository.find_messages(
                            {"chat_id": cid},
                            sort=None,
                            limit=max(1, self.per_chat_message_sample),
                            limit_mode="latest",
                            filter_bot=False,
                            filter_command=False,
                            filter_intercept_message_level=None,
                        )
                        msgs = res or []
                        with self._lock:
                            self.stats["fallback_db_reads"] += 1
                    except Exception:
                        msgs = []

                # 3) 提取 person_id 并预热
                try:
                    from src.person_info.person_info import Person, get_person_id

                    pids: List[str] = []
                    seen: set[str] = set()

                    for m in msgs:
                        platform = None
                        user_id = None
                        try:
                            ui = getattr(m, "user_info", None)
                            if ui is not None:
                                platform = getattr(ui, "platform", None) or platform
                                user_id = getattr(ui, "user_id", None) or user_id
                        except Exception:
                            pass

                        # 兼容直接字段
                        try:
                            platform = platform or getattr(m, "user_platform", None)
                        except Exception:
                            pass
                        try:
                            user_id = user_id or getattr(m, "user_id", None)
                        except Exception:
                            pass

                        if not platform or not user_id:
                            continue

                        try:
                            pid = get_person_id(str(platform), str(user_id))
                        except Exception:
                            continue

                        if not pid or pid in seen:
                            continue

                        seen.add(pid)
                        pids.append(pid)
                        if len(pids) >= self.max_persons_per_chat:
                            break

                    loaded_count = 0
                    for pid in pids:
                        try:
                            # 构造会触发 is_person_known + load_from_database
                            Person(person_id=pid)
                            loaded_count += 1
                        except Exception:
                            continue

                    with self._lock:
                        self.stats["loaded"] += int(loaded_count)
                except Exception as e:
                    with self._lock:
                        self.stats["errors"] += 1
                    logger.debug(f"[PersonWarmup] warmup 失败 chat_id={cid}: {e}")

                # 4) 记录 warm 成功时间
                with self._lock:
                    self._touch_lru_locked(cid, time.time())
            except Exception as e:
                with self._lock:
                    self.stats["errors"] += 1
                logger.debug(f"[PersonWarmup] warmup 失败 chat_id={cid}: {e}")
            finally:
                with self._lock:
                    self._warming.discard(cid)

        try:
            loop.create_task(_warm())
        except Exception:
            with self._lock:
                self._warming.discard(cid)


# ===== 消息缓存模块 =====
class MessageCacheModule:
    """消息查询缓存"""

    def __init__(
        self,
        max_size=2000,
        ttl=120.0,
        hotset_enabled: bool = True,
        hotset_per_chat_limit: int = 200,
        hotset_ttl: int = 300,
        hotset_max_chats: int = 500,
        hotset_bucket_enabled: bool = False,
        hotset_bucket_seconds: int = 5,
        hotset_ignore_time_limit_when_active: bool = True,
        hotset_active_time_window: int = 300,
    ):
        self.cache = TTLCache(max_size, ttl)
        self.stats = ModuleStats("message_cache")

        # 热集缓存（范围查询加速）
        self.hotset = MessageHotsetCache(
            enabled=hotset_enabled,
            per_chat_limit=hotset_per_chat_limit,
            ttl=hotset_ttl,
            max_chats=hotset_max_chats,
            ignore_time_limit_when_active=hotset_ignore_time_limit_when_active,
            active_time_window=hotset_active_time_window,
        )
        # 预留：滑动窗口分桶增强（默认关闭；后续实现 query-cache 提升时使用）
        self.hotset_bucket_enabled = bool(hotset_bucket_enabled)
        self.hotset_bucket_seconds = int(hotset_bucket_seconds)

        self._orig_func = None
        self._orig_store_message = None
        self._patched_store_message = False
        self._patched = False

    def get_memory_usage(self) -> int:
        """获取模块内存占用（字节）。

        这里复用底层 [`TTLCache.get_memory_usage()`](CM-performance-optimizer-plugin/plugin.py:126)。
        """
        try:
            return int(self.cache.get_memory_usage())
        except Exception:
            return 0
    
    @staticmethod
    def _make_cache_key(
        message_filter,
        sort,
        limit,
        limit_mode,
        filter_bot,
        filter_command,
        filter_intercept_message_level,
    ) -> Tuple[str, bool]:
        """生成稳定的缓存键。

        关键点：对常见的“拉取到当前上下文”查询进行归一化，避免时间戳不断变化导致 key 高基数。

        Returns:
            Tuple[str, bool]: (cache_key, cacheable)
        """
        import json

        mf = message_filter or {}
        mf_for_key = mf
        cacheable = True

        chat_id = mf.get("chat_id")
        time_cond = mf.get("time")

        # ---- 1) 处理 time 条件导致的高基数 key ----
        # 常见查询：{"chat_id": X, "time": {"$lt": now}} / {"$lte": now}
        # 归一化策略：当 `$lt/$lte` 时间戳接近当前时间（窗口内），将 time 替换为固定占位符，并附带 chat_version。
        # 范围查询（同时包含 $gt/$lt）通常为滑动窗口，默认不缓存。
        if chat_id and isinstance(time_cond, dict):
            has_lt = ("$lt" in time_cond) or ("$lte" in time_cond)
            has_gt = ("$gt" in time_cond) or ("$gte" in time_cond)

            if has_lt and has_gt:
                cacheable = False
            elif has_lt and not has_gt:
                ops = set(time_cond.keys())
                # 只对“单边上界”做归一化；如果还有其他操作符，先保守不归一化
                if ops.issubset({"$lt", "$lte"}):
                    ts_val = time_cond.get("$lt", time_cond.get("$lte"))
                    try:
                        now = time.time()
                        ts_float = float(ts_val)  # 兼容 int/float/str
                        # 太久以前的 `$lt` 大概率是历史查询：不归一化且不缓存，避免误缓存/爆 key
                        if ts_float < (now - MSG_CACHE_NORMALIZE_LT_WINDOW_SECONDS):
                            cacheable = False
                        else:
                            mf_for_key = dict(mf)
                            mf_for_key["time"] = {"$lt": "__NOW__"}
                    except Exception:
                        # 无法解析时间戳，保持原样
                        pass

        # ---- 2) 稳定序列化 ----
        # message_filter 可能包含复杂类型，这里用 default=str 做兜底。
        mf_str = json.dumps(mf_for_key, sort_keys=True, ensure_ascii=False, default=str)

        # 对 sort 参数进行稳定序列化（可能是字典、列表或 None）
        if sort is not None:
            if isinstance(sort, dict):
                sort_str = json.dumps(sort, sort_keys=True, ensure_ascii=False, default=str)
            elif isinstance(sort, list):
                sort_str = json.dumps(sort, ensure_ascii=False, default=str)
            else:
                sort_str = str(sort)
        else:
            sort_str = "None"

        # 处理 filter_intercept_message_level 可能为 None 的情况
        filter_level_str = str(filter_intercept_message_level) if filter_intercept_message_level is not None else "None"

        # ---- 3) 引入 chat_version 让“最新上下文”缓存可自动失效 ----
        # 只要 chat_id 存在，就将 version 纳入 key（新消息写入后 version+1）。
        cv = None
        if chat_id is not None:
            try:
                cv = _chat_versions.get(str(chat_id))
            except Exception:
                cv = None

        key = f"{mf_str}:{sort_str}:{limit}:{limit_mode}:{filter_bot}:{filter_command}:{filter_level_str}"
        if cv is not None:
            key = f"{key}:cv={cv}"

        return key, cacheable
    
    def apply_patch(self):
        if self._patched:
            return
        try:
            from src.common import message_repository

            self._orig_func = message_repository.find_messages
            module = self

            def patched(
                message_filter,
                sort=None,
                limit=0,
                limit_mode="latest",
                filter_bot=False,
                filter_command=False,
                filter_intercept_message_level=None,
            ):
                mf = message_filter or {}

                # ---- 0) hotset：优先处理常见滑动窗口 chat 范围查询 ----
                # 目标：加速 `get_recent_messages(chat_id, hours, limit)` 这类调用：
                #   filter = {"chat_id": X, "time": {"$gte": start, "$lte": end(now)}}
                # 策略：
                # - 若热集可回答：直接返回（不走 DB；不计入 query-cache hit/miss）
                # - 否则：异步触发 warmup（不阻塞）并回退到下方 query-cache/DB 逻辑
                try:
                    chat_id = mf.get("chat_id")
                    time_cond = mf.get("time")
                    if chat_id and isinstance(time_cond, dict):
                        start_ts = time_cond.get("$gt", time_cond.get("$gte"))
                        end_ts = time_cond.get("$lt", time_cond.get("$lte"))
                        if start_ts is not None and end_ts is not None and int(limit) > 0:
                            hot = module.hotset.try_get_range(
                                chat_id=str(chat_id),
                                start_ts=float(start_ts),
                                end_ts=float(end_ts),
                                limit=int(limit),
                                limit_mode=str(limit_mode or "latest"),
                                filter_intercept_message_level=filter_intercept_message_level,
                            )
                            if hot is not None:
                                return hot
                            module.hotset.ensure_warmup(str(chat_id))
                except Exception:
                    pass

                # ---- 1) query-cache：稳定 key 的查询结果缓存 ----
                # 使用稳定的缓存键生成方法（并返回该查询是否建议缓存）
                key, cacheable = module._make_cache_key(
                    message_filter,
                    sort,
                    limit,
                    limit_mode,
                    filter_bot,
                    filter_command,
                    filter_intercept_message_level,
                )

                # 不可缓存的请求：不计入 miss（否则命中率会被“主动跳过”的请求稀释）
                if not cacheable:
                    module.stats.skipped()
                    return module._orig_func(
                        message_filter,
                        sort,
                        limit,
                        limit_mode,
                        filter_bot,
                        filter_command,
                        filter_intercept_message_level,
                    )

                val, hit = module.cache.get(key)
                if hit:
                    module.stats.hit()
                    return val

                t0 = time.time()
                res = module._orig_func(
                    message_filter,
                    sort,
                    limit,
                    limit_mode,
                    filter_bot,
                    filter_command,
                    filter_intercept_message_level,
                )
                module.stats.miss(time.time() - t0)

                if 0 < limit <= 200:
                    module.cache.set(key, res)
                return res

            message_repository.find_messages = patched

            # 替换已导入的引用
            for n, m in list(sys.modules.items()):
                if m and getattr(m, "find_messages", None) is self._orig_func:
                    setattr(m, "find_messages", patched)
                    logger.debug(f"[MsgCache] 替换 {n}.find_messages")

            # 写入侧：补丁 MessageStorage.store_message，在写入后 bump chat_version
            # 注意：上游 store_message 内部会吞异常（try/except），因此无法严格判定“成功写入”。
            # 这里按“非通知消息调用过 store_message 即认为可能写入”，用于正确失效缓存。
            try:
                from src.chat.message_receive.storage import MessageStorage

                if not self._patched_store_message:
                    # ⚠️ 兼容性要点：原始实现是 @staticmethod。
                    # 如果直接赋值函数，会变成普通方法，实例调用时会多传一个 self，导致
                    # “takes 2 positional arguments but 3 were given”。
                    # 因此必须保存原 descriptor，并用 staticmethod 包装回写。
                    self._orig_store_message = MessageStorage.__dict__.get("store_message")
                    orig_store_callable = getattr(MessageStorage, "store_message", None)

                    async def patched_store_message(message, chat_stream):
                        if callable(orig_store_callable):
                            await orig_store_callable(message, chat_stream)
                        try:
                            # 通知消息不参与版本递增（与原逻辑一致）
                            if getattr(message, "is_notify", False):
                                return
                            chat_id = getattr(chat_stream, "stream_id", None)
                            if chat_id:
                                _chat_versions.bump(str(chat_id))
                                # 热集：写入后异步触发 warmup/top-up（不阻塞）
                                module.hotset.ensure_warmup(str(chat_id))

                                # 人物信息预热：写入后异步触发（不阻塞）
                                try:
                                    global _person_warmup
                                    if _person_warmup is not None:
                                        _person_warmup.ensure_warmup(str(chat_id))
                                except Exception:
                                    pass
                        except Exception:
                            # 版本递增/热集触发失败不应影响主流程
                            return

                    MessageStorage.store_message = staticmethod(patched_store_message)

                    # 替换已导入的引用（有些模块可能 `from ... import MessageStorage` 后取了 store_message 句柄）
                    for n, m in list(sys.modules.items()):
                        if m and getattr(m, "store_message", None) is orig_store_callable:
                            setattr(m, "store_message", patched_store_message)
                            logger.debug(f"[MsgCache] 替换 {n}.store_message")

                    self._patched_store_message = True
                    logger.info("[MsgCache] ✓ 写入侧版本补丁已应用")
            except Exception as e:
                logger.warning(f"[MsgCache] 写入侧版本补丁失败: {e}")

            self._patched = True
            logger.info("[MsgCache] ✓ 补丁应用成功")
        except Exception as e:
            logger.error(f"[MsgCache] ✗ 补丁失败: {e}")
    
    def remove_patch(self):
        if not self._patched or not self._orig_func:
            return
        try:
            from src.common import message_repository

            message_repository.find_messages = self._orig_func

            # 回滚写入侧补丁
            try:
                if self._patched_store_message and self._orig_store_message is not None:
                    from src.chat.message_receive.storage import MessageStorage

                    # 还原原始 descriptor（通常是 staticmethod）
                    MessageStorage.store_message = self._orig_store_message
                    self._patched_store_message = False
            except Exception as e:
                logger.warning(f"[MsgCache] 回滚写入侧版本补丁失败: {e}")

            self._patched = False
            logger.info("[MsgCache] 补丁已移除")
        except Exception as e:
            logger.error(f"[MsgCache] 移除补丁失败: {e}")


# ===== 人物信息缓存模块 (从person-cache-plugin整合) =====
class PersonCacheModule:
    """人物信息缓存"""
    # 缓存数据版本号，用于未来格式变更时的兼容性处理
    CACHE_VERSION = 1
    
    def __init__(self, max_size=3000, ttl=1800):
        self.cache = TTLCache(max_size, ttl)
        self.stats = ModuleStats("person_cache")
        self._orig_load = None
        self._orig_sync = None
        self._patched = False

    def get_memory_usage(self) -> int:
        """获取模块内存占用（字节）。

        这里复用底层 [`TTLCache.get_memory_usage()`](CM-performance-optimizer-plugin/plugin.py:126)。
        """
        try:
            return int(self.cache.get_memory_usage())
        except Exception:
            return 0
    
    @staticmethod
    def _serialize_person(person):
        """
        稳健序列化人物信息
        
        Args:
            person: Person 对象
            
        Returns:
            dict: 序列化后的数据，包含版本号
        """
        data = {
            "_version": PersonCacheModule.CACHE_VERSION,
            "user_id": getattr(person, "user_id", ""),
            "platform": getattr(person, "platform", ""),
            "is_known": getattr(person, "is_known", False),
            "nickname": getattr(person, "nickname", ""),
            "person_name": getattr(person, "person_name", None),
            "name_reason": getattr(person, "name_reason", None),
            "know_times": getattr(person, "know_times", 0),
            "know_since": getattr(person, "know_since", None),
            "last_know": getattr(person, "last_know", None),
            "memory_points": list(getattr(person, "memory_points", []) or []),
            "group_nick_name": list(getattr(person, "group_nick_name", []) or []),
        }
        return data
    
    @staticmethod
    def _deserialize_person(person, data):
        """
        反序列化人物信息到 Person 对象
        
        Args:
            person: Person 对象
            data: 序列化的数据
            
        Returns:
            bool: 是否成功反序列化
        """
        try:
            # 检查版本号
            version = data.get("_version", 0)
            if version != PersonCacheModule.CACHE_VERSION:
                logger.warning(f"[人物缓存] 缓存数据版本不匹配: 期望 {PersonCacheModule.CACHE_VERSION}, 实际 {version}")
                return False
            
            # 恢复属性
            for k, v in data.items():
                if k != "_version":  # 跳过版本号字段
                    setattr(person, k, v)
            return True
        except Exception as e:
            logger.error(f"[人物缓存] 反序列化失败: {e}")
            return False
    
    def apply_patch(self):
        if self._patched: return
        try:
            from src.person_info.person_info import Person
            self._orig_load = Person.load_from_database
            self._orig_sync = Person.sync_to_database
            module = self
            
            def cached_load(self_person):
                person_id = self_person.person_id
                cached = module.cache.get(person_id)
                if cached[1]:  # hit
                    module.stats.hit()
                    # 使用稳健的反序列化方法
                    if module._deserialize_person(self_person, cached[0]):
                        return
                    # 反序列化失败，降级到原始加载
                    logger.debug(f"[人物缓存] 缓存数据损坏，降级到原始加载: {person_id}")
                
                t0 = time.time()
                module._orig_load(self_person)
                module.stats.miss(time.time() - t0)
                
                if self_person.is_known:
                    # 使用稳健的序列化方法
                    data = module._serialize_person(self_person)
                    module.cache.set(person_id, data)
            
            def cached_sync(self_person):
                module.cache.invalidate(self_person.person_id)
                module._orig_sync(self_person)
            
            Person.load_from_database = cached_load
            Person.sync_to_database = cached_sync
            self._patched = True
            logger.info("[人物缓存] ✓ 补丁应用成功")
        except Exception as e:
            logger.error(f"[人物缓存] ✗ 补丁失败: {e}")
    
    def remove_patch(self):
        if not self._patched: return
        try:
            from src.person_info.person_info import Person
            if self._orig_load: Person.load_from_database = self._orig_load
            if self._orig_sync: Person.sync_to_database = self._orig_sync
            self._patched = False
            logger.info("[人物缓存] 补丁已移除")
        except Exception as e:
            logger.error(f"[人物缓存] 移除补丁失败: {e}")


# ===== 表达式缓存模块 (双缓冲 + 缓慢加载) =====
class ExpressionCacheModule:
    """表达式全量缓存 - 双缓冲 + 缓慢加载 + 原子切换"""
    def __init__(self, batch_size=100, batch_delay=0.05, refresh_interval=3600):
        # 双缓冲
        self.buffer_a = None      # 当前使用的缓存
        self.buffer_b = None      # 后台加载的缓存
        self.buffer_lock = threading.Lock()
        
        # 加载配置
        self.batch_size = batch_size        # 每批加载条数
        self.batch_delay = batch_delay      # 批次间延迟（秒）
        self.refresh_interval = refresh_interval  # 自动刷新间隔
        
        # 状态
        self.loading = False        # 是否正在加载
        self.load_lock = asyncio.Lock()
        self.last_refresh = 0       # 上次刷新时间
        self.stats = ModuleStats("expression_cache")
        
        # 启动时立即开始加载
        try:
            asyncio.get_running_loop().create_task(self._load_to_buffer_b())
        except RuntimeError:
            pass  # 没有运行中的事件循环，稍后加载
    
    def get_all(self):
        """获取当前缓存（从缓冲区A）"""
        with self.buffer_lock:
            # 如果缓冲区A为空，返回空列表（走数据库）
            if self.buffer_a is None:
                return []
            return self.buffer_a
    
    async def _load_to_buffer_b(self):
        """缓慢加载数据到缓冲区B"""
        async with self.load_lock:
            if self.loading:
                return
            self.loading = True
        
        try:
            logger.info("[ExprCache] 开始缓慢加载表达式缓存到缓冲区B...")
            
            # 清空缓冲区B
            buffer_b_data = []
            
            # 分批加载
            offset = 0
            from src.common.database.database_model import Expression
            while True:
                # 查询一批数据
                batch = list(Expression.select().limit(self.batch_size).offset(offset))
                if not batch:
                    break
                
                # 添加到缓冲区B
                buffer_b_data.extend(batch)
                
                # 记录进度
                logger.debug(f"[ExprCache] 加载进度: {len(buffer_b_data)} 条")
                
                # 休眠，避免CPU峰值
                await asyncio.sleep(self.batch_delay)
                
                offset += self.batch_size
            
            # 加载完成，原子切换
            with self.buffer_lock:
                self.buffer_b = buffer_b_data
                # 原子切换：buffer_b → buffer_a
                self.buffer_a, self.buffer_b = self.buffer_b, None
                
            self.last_refresh = time.time()
            logger.info(f"[ExprCache] 缓存加载完成并切换: {len(buffer_b_data)} 条")
            
        except Exception as e:
            logger.error(f"[ExprCache] 缓存加载失败: {e}")
        finally:
            async with self.load_lock:
                self.loading = False
    
    async def _refresh_loop(self):
        """定期刷新循环"""
        while True:
            await asyncio.sleep(self.refresh_interval)
            logger.info("[ExprCache] 触发定期刷新...")
            await self._load_to_buffer_b()
    
    def refresh(self):
        """手动刷新缓存"""
        try:
            asyncio.get_running_loop().create_task(self._load_to_buffer_b())
            logger.info("[ExprCache] 已触发手动刷新")
        except RuntimeError:
            logger.warning("[ExprCache] 无法触发刷新：没有运行中的事件循环")
    
    def size(self):
        """获取缓存大小"""
        with self.buffer_lock:
            return len(self.buffer_a) if self.buffer_a else 0
    
    def get_memory_usage(self):
        """获取缓存内存使用量（字节）"""
        with self.buffer_lock:
            if self.buffer_a is None:
                return 0
            return MemoryUtils.get_size(self.buffer_a)


# ===== 黑话缓存模块 (双缓冲 + 缓慢加载) =====
class JargonCacheModule:
    """黑话全量缓存 - 双缓冲 + 缓慢加载 + 原子切换"""
    def __init__(self, batch_size=100, batch_delay=0.05, refresh_interval=3600, enable_content_index=True):
        # 双缓冲
        self.buffer_a = None      # 当前使用的缓存
        self.buffer_b = None      # 后台加载的缓存
        self.buffer_lock = threading.Lock()
        
        # 内容索引
        self.content_index_a = None  # 当前使用的内容索引
        self.content_index_b = None  # 后台加载的内容索引
        self.enable_content_index = enable_content_index
        
        # 加载配置
        self.batch_size = batch_size        # 每批加载条数
        self.batch_delay = batch_delay      # 批次间延迟（秒）
        self.refresh_interval = refresh_interval  # 自动刷新间隔
        
        # 状态
        self.loading = False        # 是否正在加载
        self.load_lock = asyncio.Lock()
        self.last_refresh = 0       # 上次刷新时间
        self.stats = ModuleStats("jargon_cache")
        
        # 启动时立即开始加载
        try:
            asyncio.get_running_loop().create_task(self._load_to_buffer_b())
        except RuntimeError:
            pass  # 没有运行中的事件循环，稍后加载
    
    def get_all(self):
        """获取当前缓存（从缓冲区A）"""
        with self.buffer_lock:
            # 如果缓冲区A为空，返回空列表（走数据库）
            if self.buffer_a is None:
                return []
            return self.buffer_a
    
    def get_by_content(self, content: str):
        """通过内容精确匹配（使用索引）"""
        if not self.enable_content_index:
            return None
        with self.buffer_lock:
            if self.content_index_a is None:
                return None
            return self.content_index_a.get(content.lower())
    
    async def _load_to_buffer_b(self):
        """缓慢加载数据到缓冲区B"""
        async with self.load_lock:
            if self.loading:
                return
            self.loading = True
        
        try:
            logger.info("[JargonCache] 开始缓慢加载黑话缓存到缓冲区B...")
            
            # 清空缓冲区B
            buffer_b_data = []
            content_index_b = {} if self.enable_content_index else None
            
            # 分批加载
            offset = 0
            from src.common.database.database_model import Jargon
            while True:
                # 查询一批数据
                batch = list(Jargon.select().limit(self.batch_size).offset(offset))
                if not batch:
                    break
                
                # 添加到缓冲区B
                buffer_b_data.extend(batch)
                
                # 构建内容索引
                if self.enable_content_index:
                    for jargon in batch:
                        if jargon.content:
                            content_index_b[jargon.content.lower()] = jargon
                
                # 记录进度
                logger.debug(f"[JargonCache] 加载进度: {len(buffer_b_data)} 条")
                
                # 休眠，避免CPU峰值
                await asyncio.sleep(self.batch_delay)
                
                offset += self.batch_size
            
            # 加载完成，原子切换
            with self.buffer_lock:
                self.buffer_b = buffer_b_data
                self.content_index_b = content_index_b
                # 原子切换：buffer_b → buffer_a
                self.buffer_a, self.buffer_b = self.buffer_b, None
                self.content_index_a, self.content_index_b = self.content_index_b, None
                
            self.last_refresh = time.time()
            logger.info(f"[JargonCache] 缓存加载完成并切换: {len(buffer_b_data)} 条")
            
        except Exception as e:
            logger.error(f"[JargonCache] 缓存加载失败: {e}")
        finally:
            async with self.load_lock:
                self.loading = False
    
    async def _refresh_loop(self):
        """定期刷新循环"""
        while True:
            await asyncio.sleep(self.refresh_interval)
            logger.info("[JargonCache] 触发定期刷新...")
            await self._load_to_buffer_b()
    
    def refresh(self):
        """手动刷新缓存"""
        try:
            asyncio.get_running_loop().create_task(self._load_to_buffer_b())
            logger.info("[JargonCache] 已触发手动刷新")
        except RuntimeError:
            logger.warning("[JargonCache] 无法触发刷新：没有运行中的事件循环")
    
    def size(self):
        """获取缓存大小"""
        with self.buffer_lock:
            return len(self.buffer_a) if self.buffer_a else 0
    
    def get_memory_usage(self):
        """获取缓存内存使用量（字节）"""
        with self.buffer_lock:
            if self.buffer_a is None:
                return 0
            size = MemoryUtils.get_size(self.buffer_a)
            if self.content_index_a is not None:
                size += MemoryUtils.get_size(self.content_index_a)
            return size


# ===== 知识库图谱缓存模块 (双缓冲 + 缓慢加载) =====
class KGCacheModule:
    """知识库图谱全量缓存 - 双缓冲 + 缓慢加载 + 原子切换"""
    def __init__(self, batch_size=100, batch_delay=0.05, refresh_interval=3600):
        # 双缓冲
        self.buffer_a = None      # 当前使用的图数据
        self.buffer_b = None      # 后台加载的图数据
        self.buffer_lock = threading.Lock()
        
        # 缓存内容
        self.graph_a = None       # 图对象
        self.nodes_a = None       # 节点列表
        self.edges_a = None       # 边列表
        self.ent_appear_cnt_a = None  # 实体出现次数
        self.stored_paragraph_hashes_a = None  # 段落hash集合
        
        # 加载配置
        self.batch_size = batch_size        # 每批加载条数
        self.batch_delay = batch_delay      # 批次间延迟（秒）
        self.refresh_interval = refresh_interval  # 自动刷新间隔
        
        # 状态
        self.loading = False        # 是否正在加载
        self.load_lock = asyncio.Lock()
        self.last_refresh = 0       # 上次刷新时间
        self.stats = ModuleStats("kg_cache")
        
        # 启动时立即开始加载
        try:
            asyncio.get_running_loop().create_task(self._load_to_buffer_b())
        except RuntimeError:
            pass  # 没有运行中的事件循环，稍后加载
    
    def get_cached_data(self):
        """获取当前缓存的图数据"""
        with self.buffer_lock:
            if self.buffer_a is None:
                return None
            return {
                "graph": self.graph_a,
                "nodes": self.nodes_a,
                "edges": self.edges_a,
                "ent_appear_cnt": self.ent_appear_cnt_a,
                "stored_paragraph_hashes": self.stored_paragraph_hashes_a,
            }
    
    def is_loaded(self):
        """检查缓存是否已加载"""
        with self.buffer_lock:
            return self.buffer_a is not None
    
    async def _load_to_buffer_b(self):
        """缓慢加载数据到缓冲区B"""
        async with self.load_lock:
            if self.loading:
                return
            self.loading = True
        
        try:
            logger.info("[KGCache] 开始缓慢加载知识库图谱缓存到缓冲区B...")
            
            # 尝试加载知识库图谱
            from src.chat.knowledge.kg_manager import KGManager
            kg_manager = KGManager()
            
            # 检查文件是否存在
            import os
            if not os.path.exists(kg_manager.graph_data_path):
                logger.warning(f"[KGCache] 知识库图谱文件不存在: {kg_manager.graph_data_path}")
                self.loading = False
                return
            
            # 加载数据
            t0 = time.time()
            
            # 加载图谱
            from quick_algo import di_graph
            graph_b = di_graph.load_from_file(kg_manager.graph_data_path)
            nodes_b = graph_b.get_node_list()
            edges_b = graph_b.get_edge_list()
            
            logger.debug(f"[KGCache] 加载图谱: {len(nodes_b)} 个节点, {len(edges_b)} 条边")
            
            # 加载实体计数
            import pandas as pd
            ent_cnt_df = pd.read_parquet(kg_manager.ent_cnt_data_path, engine="pyarrow")
            ent_appear_cnt_b = dict({row["hash_key"]: row["appear_cnt"] for _, row in ent_cnt_df.iterrows()})
            
            logger.debug(f"[KGCache] 加载实体计数: {len(ent_appear_cnt_b)} 个实体")
            
            # 加载段落hash
            import json
            with open(kg_manager.pg_hash_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                stored_paragraph_hashes_b = set(data["stored_paragraph_hashes"])
            
            logger.debug(f"[KGCache] 加载段落hash: {len(stored_paragraph_hashes_b)} 个段落")
            
            # 模拟分批加载的延迟（避免CPU峰值）
            total_items = len(nodes_b) + len(edges_b) + len(ent_appear_cnt_b)
            batches = max(1, (total_items + self.batch_size - 1) // self.batch_size)
            for i in range(batches):
                await asyncio.sleep(self.batch_delay)
                if i % 10 == 0:
                    logger.debug(f"[KGCache] 加载进度: {i+1}/{batches} 批")
            
            # 加载完成，原子切换
            with self.buffer_lock:
                self.buffer_b = True
                self.graph_b = graph_b
                self.nodes_b = nodes_b
                self.edges_b = edges_b
                self.ent_appear_cnt_b = ent_appear_cnt_b
                self.stored_paragraph_hashes_b = stored_paragraph_hashes_b
                # 原子切换：buffer_b → buffer_a
                self.buffer_a, self.buffer_b = self.buffer_b, None
                self.graph_a, self.graph_b = self.graph_b, None
                self.nodes_a, self.nodes_b = self.nodes_b, None
                self.edges_a, self.edges_b = self.edges_b, None
                self.ent_appear_cnt_a, self.ent_appear_cnt_b = self.ent_appear_cnt_b, None
                self.stored_paragraph_hashes_a, self.stored_paragraph_hashes_b = self.stored_paragraph_hashes_b, None
                
            self.last_refresh = time.time()
            load_time = time.time() - t0
            logger.info(f"[KGCache] 缓存加载完成并切换: 节点{len(nodes_b)}个, 边{len(edges_b)}条, 耗时{load_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"[KGCache] 缓存加载失败: {e}")
        finally:
            async with self.load_lock:
                self.loading = False
    
    async def _refresh_loop(self):
        """定期刷新循环"""
        while True:
            await asyncio.sleep(self.refresh_interval)
            logger.info("[KGCache] 触发定期刷新...")
            await self._load_to_buffer_b()
    
    def refresh(self):
        """手动刷新缓存"""
        try:
            asyncio.get_running_loop().create_task(self._load_to_buffer_b())
            logger.info("[KGCache] 已触发手动刷新")
        except RuntimeError:
            logger.warning("[KGCache] 无法触发刷新：没有运行中的事件循环")
    
    def size(self):
        """获取缓存大小"""
        with self.buffer_lock:
            if self.nodes_a is None:
                return 0
            return {
                "nodes": len(self.nodes_a),
                "edges": len(self.edges_a),
                "entities": len(self.ent_appear_cnt_a),
                "paragraphs": len(self.stored_paragraph_hashes_a),
            }
    
    def get_memory_usage(self):
        """获取缓存内存使用量（字节）"""
        with self.buffer_lock:
            if self.buffer_a is None:
                return 0
            size = 0
            if self.graph_a is not None:
                size += MemoryUtils.get_size(self.graph_a)
            if self.nodes_a is not None:
                size += MemoryUtils.get_size(self.nodes_a)
            if self.edges_a is not None:
                size += MemoryUtils.get_size(self.edges_a)
            if self.ent_appear_cnt_a is not None:
                size += MemoryUtils.get_size(self.ent_appear_cnt_a)
            if self.stored_paragraph_hashes_a is not None:
                size += MemoryUtils.get_size(self.stored_paragraph_hashes_a)
            return size


# ===== 性能优化停止事件处理器 =====
class PerfOptStopHandler(BaseEventHandler):
    """性能优化停止事件处理器 - 确保插件停止时正确回滚补丁"""

    # 事件处理器配置 - 使用 ON_STOP 事件
    # 注意：EventsManager 注册的事件类型来自 EventType 枚举，必须与其 value 对齐。
    # 不能写成 "ON_STOP"，否则会被认为是未注册事件。
    event_type = EventType.ON_STOP
    handler_name = "perf_opt_stop_handler"
    handler_description = "插件停止时回滚所有补丁"
    weight = 10  # 高优先级，确保在其他处理器之前执行
    intercept_message = False
    
    def __init__(self):
        super().__init__()
    
    async def execute(self, message):
        """
        处理停止事件 - 调用 Optimizer.stop() 回滚补丁
        
        Args:
            message: MaiMessages 对象（ON_STOP 时为 None）
            
        Returns:
            Tuple[bool, bool, Optional[str], Optional[CustomEventHandlerResult], Optional[MaiMessages]]
        """
        global _opt
        if _opt:
            try:
                _opt.stop()
                logger.info("[PerfOpt] ✓ 插件已停止，补丁已回滚")
            except Exception as e:
                logger.error(f"[PerfOpt] 停止失败: {e}")
        return (True, True, None, None, None)


# ===== 主优化器 =====
class Optimizer:
    _inst = None
    
    def __new__(cls, *a, **kw):
        if not cls._inst:
            cls._inst = super().__new__(cls)
            cls._inst._ready = False
        return cls._inst
    
    def __init__(self, cfg=None):
        if self._ready: return
        cfg = cfg or {}
        self.start_time = time.time()
        self.interval = cfg.get("report_interval", 60)
        self.modules_cfg = cfg.get("modules", {})
        
        # 内存统计配置
        self.memory_stats_enabled = cfg.get("memory_stats_enabled", True)
        self.memory_stats_cache_ttl = cfg.get("memory_stats_cache_ttl", 60)
        self._memory_stats_cache = {}  # 模块内存统计缓存: {cache_key: (timestamp, size)}
        # 内存统计失败日志限流缓存: {cache_key: (timestamp, last_error)}
        self._memory_stats_error_cache = {}
        self._memory_stats_lock = threading.Lock()
        
        # 保存原始函数引用，用于补丁移除
        self._orig_expression_find_similar = None
        self._orig_jargon_match = None
        self._orig_kg_load = None
        
        # 初始化模块
        self.msg_cache = None
        self.full_msg_cache = None  # 全量消息缓存模块
        self.person_cache = None
        self.expr_cache = None
        self.jargon_cache = None
        self.kg_cache = None

        # 消息缓存模式选择
        message_cache_mode = cfg.get("message_cache_mode", "query")  # "query" 或 "full"

        if self.modules_cfg.get("message_cache", True):
            if message_cache_mode == "full":
                # 全量缓存模式
                from .full_message_cache import FullMessageCacheModule
                self.full_msg_cache = FullMessageCacheModule(
                    batch_size=cfg.get("message_cache_full_batch_size", 500),
                    batch_delay=cfg.get("message_cache_full_batch_delay", 0.05),
                    refresh_interval=cfg.get("message_cache_full_refresh_interval", 0),
                    enable_incremental=cfg.get("message_cache_full_enable_incremental", True),
                    max_messages_per_chat=cfg.get("message_cache_full_max_messages_per_chat", 10000),
                    max_total_messages=cfg.get("message_cache_full_max_total_messages", 100000),
                    enable_lru_eviction=cfg.get("message_cache_full_enable_lru_eviction", True),
                    max_chats=cfg.get("message_cache_full_max_chats", 1000),
                )
                logger.info("[PerfOpt] 使用全量消息缓存模式")
            else:
                # 查询缓存模式（默认）
                self.msg_cache = MessageCacheModule(
                    cfg.get("message_cache_size", 2000),
                    cfg.get("message_cache_ttl", 120.0),
                    hotset_enabled=cfg.get("message_hotset_enabled", True),
                    hotset_per_chat_limit=cfg.get("message_hotset_per_chat_limit", 200),
                    hotset_ttl=cfg.get("message_hotset_ttl", 300),
                    hotset_max_chats=cfg.get("message_hotset_max_chats", 500),
                    hotset_bucket_enabled=cfg.get("message_hotset_bucket_enabled", False),
                    hotset_bucket_seconds=cfg.get("message_hotset_bucket_seconds", 5),
                    hotset_ignore_time_limit_when_active=cfg.get("message_hotset_ignore_time_limit_when_active", True),
                    hotset_active_time_window=cfg.get("message_hotset_active_time_window", 300),
                )
                logger.info("[PerfOpt] 使用查询缓存模式")
        
        if self.modules_cfg.get("person_cache", True):
            self.person_cache = PersonCacheModule(
                cfg.get("person_cache_size", 3000),
                cfg.get("person_cache_ttl", 1800)
            )
        
        if self.modules_cfg.get("expression_cache", False):
            self.expr_cache = ExpressionCacheModule(
                batch_size=cfg.get("expression_cache_batch_size", 100),
                batch_delay=cfg.get("expression_cache_batch_delay", 0.05),
                refresh_interval=cfg.get("expression_cache_refresh_interval", 3600)
            )
        
        if self.modules_cfg.get("slang_cache", False):
            self.jargon_cache = JargonCacheModule(
                batch_size=cfg.get("slang_cache_batch_size", 100),
                batch_delay=cfg.get("slang_cache_batch_delay", 0.05),
                refresh_interval=cfg.get("slang_cache_refresh_interval", 3600),
                enable_content_index=cfg.get("slang_cache_enable_content_index", True)
            )
        
        if self.modules_cfg.get("kg_cache", False):
            self.kg_cache = KGCacheModule(
                batch_size=cfg.get("kg_cache_batch_size", 100),
                batch_delay=cfg.get("kg_cache_batch_delay", 0.05),
                refresh_interval=cfg.get("kg_cache_refresh_interval", 3600)
            )
        
        self._running = False
        self._ready = True
    
    @staticmethod
    def _check_dependencies(module_name):
        """
        检查模块所需的依赖是否可用
        
        Args:
            module_name: 模块名称 ("expression_cache", "jargon_cache", "kg_cache")
            
        Returns:
            bool: 依赖是否可用
        """
        try:
            if module_name == "expression_cache":
                from src.bw_learner.expression_learner import ExpressionLearner
                from src.common.database.database_model import Expression
                return True
            elif module_name == "jargon_cache":
                from src.bw_learner.jargon_explainer import JargonExplainer
                from src.common.database.database_model import Jargon
                from src.bw_learner.learner_utils import calculate_similarity
                from src.config.config import global_config
                return True
            elif module_name == "kg_cache":
                from src.chat.knowledge.kg_manager import KGManager
                from quick_algo import di_graph
                return True
        except ImportError as e:
            logger.warning(f"[PerfOpt] {module_name} 依赖检查失败: {e}")
            return False
        except Exception as e:
            logger.warning(f"[PerfOpt] {module_name} 依赖检查异常: {e}")
            return False
        return False
    
    def _get_module_memory_usage(self, module, cache_key: str, display_name: Optional[str] = None) -> int:
        """获取模块内存使用量（带缓存）。

        - `cache_key` 用于缓存/限流（稳定，不随展示名变化）
        - `display_name` 仅用于日志展示
        """
        if not self.memory_stats_enabled:
            return 0

        if not module:
            return 0

        current_time = time.time()
        display = display_name or cache_key

        with self._memory_stats_lock:
            # 1) 读取缓存
            if cache_key in self._memory_stats_cache:
                cache_time, cache_size = self._memory_stats_cache[cache_key]
                if current_time - cache_time < self.memory_stats_cache_ttl:
                    return int(cache_size)

            # 2) 防御式获取 get_memory_usage
            getter = getattr(module, "get_memory_usage", None)
            if not callable(getter):
                # 缺失方法也按“失败”处理，并进行日志限流
                self._memory_stats_cache[cache_key] = (current_time, 0)
                last = self._memory_stats_error_cache.get(cache_key)
                if not last or (current_time - last[0] >= self.memory_stats_cache_ttl):
                    self._memory_stats_error_cache[cache_key] = (current_time, "missing get_memory_usage")
                    logger.debug(f"[MemoryStats] 获取 {display} 内存失败: module has no callable get_memory_usage")
                return 0

            # 3) 重新测量
            try:
                size = int(getter())
                self._memory_stats_cache[cache_key] = (current_time, size)
                return size
            except Exception as e:
                # 失败也写入缓存，避免每次 report 都触发异常
                self._memory_stats_cache[cache_key] = (current_time, 0)

                # 日志限流：同一模块每 TTL 最多输出一次
                err_str = str(e)
                last = self._memory_stats_error_cache.get(cache_key)
                if not last or (current_time - last[0] >= self.memory_stats_cache_ttl):
                    self._memory_stats_error_cache[cache_key] = (current_time, err_str)
                    logger.debug(f"[MemoryStats] 获取 {display} 内存失败: {e}")

                return 0
    
    def apply_patches(self):
        # 消息缓存：根据模式选择应用哪个补丁
        if self.msg_cache:
            self.msg_cache.apply_patch()
        elif self.full_msg_cache:
            self.full_msg_cache.apply_patch()

        if self.person_cache:
            self.person_cache.apply_patch()
        
        # 表达式缓存拦截（带依赖检查）
        if self.expr_cache:
            if self._check_dependencies("expression_cache"):
                self._apply_expression_cache_patch()
            else:
                logger.warning("[PerfOpt] 表达式缓存依赖缺失，跳过启用")
                self.expr_cache = None
        
        # 黑话缓存拦截（带依赖检查）
        if self.jargon_cache:
            if self._check_dependencies("jargon_cache"):
                self._apply_jargon_cache_patch()
            else:
                logger.warning("[PerfOpt] 黑话缓存依赖缺失，跳过启用")
                self.jargon_cache = None
        
        # 知识库图谱缓存拦截（带依赖检查）
        if self.kg_cache:
            if self._check_dependencies("kg_cache"):
                self._apply_kg_cache_patch()
            else:
                logger.warning("[PerfOpt] 知识库图谱缓存依赖缺失，跳过启用")
                self.kg_cache = None
    
    def _apply_expression_cache_patch(self):
        """应用表达式缓存拦截"""
        try:
            from src.bw_learner.expression_learner import ExpressionLearner
            self._orig_expression_find_similar = ExpressionLearner._find_similar_situation_expression
            orig_find_similar = self._orig_expression_find_similar
            expr_cache = self.expr_cache
            stats = self.expr_cache.stats
            
            async def patched_find_similar(learner_self, situation: str, similarity_threshold: float = 0.75):
                # 从缓存获取所有表达式
                all_expressions = expr_cache.get_all()
                
                # 如果缓存未加载，走原逻辑
                if not all_expressions:
                    logger.debug("[ExprCache] 缓存未加载，使用数据库查询")
                    t0 = time.time()
                    result = await orig_find_similar(learner_self, situation, similarity_threshold)
                    stats.miss(time.time() - t0)
                    logger.debug(f"[ExprCache] 缓存未命中(未加载): 耗时={time.time()-t0:.3f}s")
                    return result
                
                # 在缓存中过滤当前 chat_id 的表达式
                chat_expressions = [expr for expr in all_expressions if expr.chat_id == learner_self.chat_id]
                
                # 先在所有表达式中查找匹配（用于统计被过滤的情况）
                best_match_all = None
                best_similarity_all = 0.0
                matched_chat_id_all = None
                
                for expr in all_expressions:
                    content_list = learner_self._parse_content_list(expr.content_list)
                    for existing_situation in content_list:
                        from src.bw_learner.learner_utils import calculate_similarity
                        similarity = calculate_similarity(situation, existing_situation)
                        if similarity >= similarity_threshold and similarity > best_similarity_all:
                            best_similarity_all = similarity
                            best_match_all = expr
                            matched_chat_id_all = expr.chat_id
                
                # 在当前 chat_id 的表达式中查找匹配
                best_match = None
                best_similarity = 0.0
                
                for expr in chat_expressions:
                    content_list = learner_self._parse_content_list(expr.content_list)
                    for existing_situation in content_list:
                        from src.bw_learner.learner_utils import calculate_similarity
                        similarity = calculate_similarity(situation, existing_situation)
                        if similarity >= similarity_threshold and similarity > best_similarity:
                            best_similarity = similarity
                            best_match = expr
                
                if best_match:
                    stats.hit()
                    logger.debug(f"[ExprCache] 缓存命中: 相似度={best_similarity:.3f}, 现有='{best_match.situation}', 新='{situation}'")
                elif best_match_all:
                    # 在缓存中找到匹配，但 chat_id 不匹配
                    stats.filtered()
                    logger.debug(f"[ExprCache] 缓存命中但被过滤: situation='{situation}', 匹配chat_id={matched_chat_id_all}, 查询chat_id={learner_self.chat_id}, 相似度={best_similarity_all:.3f}")
                else:
                    stats.miss(0.0)  # 缓存中未找到，但查询很快
                    logger.debug(f"[ExprCache] 缓存未命中(无匹配): situation='{situation}'")
                
                return best_match, best_similarity
            
            ExpressionLearner._find_similar_situation_expression = patched_find_similar
            logger.info("[ExprCache] ✓ 表达式缓存拦截已应用")
        except Exception as e:
            logger.error(f"[ExprCache] ✗ 表达式缓存拦截失败: {e}")
    
    def _apply_jargon_cache_patch(self):
        """应用黑话缓存拦截"""
        try:
            from src.bw_learner.jargon_explainer import JargonExplainer
            from src.bw_learner.learner_utils import is_bot_message, contains_bot_self_name, parse_chat_id_list, chat_id_list_contains
            from src.config.config import global_config
            import re
            
            self._orig_jargon_match = JargonExplainer.match_jargon_from_messages
            orig_match_jargon = self._orig_jargon_match
            jargon_cache = self.jargon_cache
            stats = self.jargon_cache.stats
            
            def patched_match_jargon(explainer_self, messages):
                # 从缓存获取所有黑话
                all_jargons = jargon_cache.get_all()
                
                # 如果缓存未加载，走原逻辑
                if not all_jargons:
                    logger.debug("[JargonCache] 缓存未加载，使用数据库查询")
                    t0 = time.time()
                    result = orig_match_jargon(explainer_self, messages)
                    stats.miss(time.time() - t0)
                    logger.debug(f"[JargonCache] 缓存未命中(未加载): 耗时={time.time()-t0:.3f}s, 消息数={len(messages)}")
                    return result
                
                # 收集所有消息的文本内容（跳过机器人消息）
                message_texts = []
                for msg in messages:
                    if is_bot_message(msg):
                        continue
                    
                    msg_text = (
                        getattr(msg, "display_message", None) or
                        getattr(msg, "processed_plain_text", None) or ""
                    ).strip()
                    if msg_text:
                        message_texts.append(msg_text)
                
                if not message_texts:
                    stats.miss(0.0)
                    logger.debug("[JargonCache] 缓存未命中(无有效消息)")
                    return []
                
                # 合并所有消息文本
                combined_text = " ".join(message_texts)
                
                # 根据 all_global_jargon 配置决定查询逻辑
                all_global_jargon = global_config.expression.all_global_jargon
                
                # 在缓存中过滤有meaning的黑话
                valid_jargons = [j for j in all_jargons if j.meaning and j.meaning.strip()]
                
                # 用于统计被过滤的匹配
                filtered_matches = []
                
                # 在合并文本中查找匹配
                matched_jargon = {}
                hit_count = 0
                
                for jargon in valid_jargons:
                    content = jargon.content or ""
                    if not content or not content.strip():
                        continue
                    
                    # 跳过包含机器人昵称的词条
                    if contains_bot_self_name(content):
                        continue
                    
                    # 检查chat_id（如果all_global=False）
                    if not all_global_jargon:
                        if jargon.is_global:
                            # 全局黑话，包含
                            pass
                        else:
                            # 检查chat_id列表是否包含当前chat_id
                            chat_id_list = parse_chat_id_list(jargon.chat_id)
                            if not chat_id_list_contains(chat_id_list, explainer_self.chat_id):
                                # 记录被过滤的匹配（用于统计）
                                # 检查是否在文本中匹配
                                pattern = re.escape(content)
                                if re.search(r"[\u4e00-\u9fff]", content):
                                    search_pattern = pattern
                                else:
                                    search_pattern = r"\b" + pattern + r"\b"
                                
                                if re.search(search_pattern, combined_text, re.IGNORECASE):
                                    filtered_matches.append((content, jargon.chat_id))
                                continue
                    
                    # 在文本中查找匹配（大小写不敏感）
                    pattern = re.escape(content)
                    # 使用单词边界或中文字符边界来匹配，避免部分匹配
                    if re.search(r"[\u4e00-\u9fff]", content):
                        # 包含中文，使用更宽松的匹配
                        search_pattern = pattern
                    else:
                        # 纯英文/数字，使用单词边界
                        search_pattern = r"\b" + pattern + r"\b"
                    
                    if re.search(search_pattern, combined_text, re.IGNORECASE):
                        # 找到匹配，记录（去重）
                        if content not in matched_jargon:
                            matched_jargon[content] = {"content": content}
                            hit_count += 1
                
                # 统计命中/未命中/被过滤
                if hit_count > 0:
                    stats.hit()
                    logger.debug(f"[JargonCache] 缓存命中: 匹配到 {hit_count} 个黑话: {list(matched_jargon.keys())}")
                elif filtered_matches:
                    stats.filtered()
                    filtered_sample = filtered_matches[:3]  # 只显示前3个
                    logger.debug(f"[JargonCache] 缓存命中但被过滤: 匹配到 {len(filtered_matches)} 个黑话但chat_id不匹配，示例: {filtered_sample}")
                else:
                    stats.miss(0.0)
                    logger.debug(f"[JargonCache] 缓存未命中(无匹配): 消息数={len(messages)}, 有效黑话数={len(valid_jargons)}, 文本长度={len(combined_text)}")
                
                return list(matched_jargon.values())
            
            JargonExplainer.match_jargon_from_messages = patched_match_jargon
            logger.info("[JargonCache] ✓ 黑话缓存拦截已应用")
        except Exception as e:
            logger.error(f"[JargonCache] ✗ 黑话缓存拦截失败: {e}")
    
    def _apply_kg_cache_patch(self):
        """应用知识库图谱缓存拦截"""
        try:
            from src.chat.knowledge.kg_manager import KGManager
            
            self._orig_kg_load = KGManager.load_from_file
            orig_load_from_file = self._orig_kg_load
            kg_cache = self.kg_cache
            stats = self.kg_cache.stats
            
            def patched_load_from_file(self_kg):
                # 从缓存获取图数据
                cached_data = kg_cache.get_cached_data()
                
                # 如果缓存未加载，走原逻辑
                if cached_data is None:
                    logger.debug("[KGCache] 缓存未加载，使用文件加载")
                    t0 = time.time()
                    result = orig_load_from_file(self_kg)
                    stats.miss(time.time() - t0)
                    logger.debug(f"[KGCache] 缓存未命中(未加载): 耗时={time.time()-t0:.3f}s")
                    return result
                
                # 使用缓存数据
                t0 = time.time()
                
                # 直接赋值缓存的数据
                self_kg.graph = cached_data["graph"]
                self_kg.ent_appear_cnt = cached_data["ent_appear_cnt"]
                self_kg.stored_paragraph_hashes = cached_data["stored_paragraph_hashes"]
                
                stats.hit()
                elapsed = time.time() - t0
                logger.debug(f"[KGCache] 缓存命中: 耗时={elapsed:.3f}s, 节点数={len(cached_data['nodes'])}, 边数={len(cached_data['edges'])}")
                
                return
            
            KGManager.load_from_file = patched_load_from_file
            logger.info("[KGCache] ✓ 知识库图谱缓存拦截已应用")
        except Exception as e:
            logger.error(f"[KGCache] ✗ 知识库图谱缓存拦截失败: {e}")
    
    async def _report_loop(self):
        # 如果 report_interval 为 0，则不启动报告循环
        if self.interval <= 0:
            logger.info("[PerfOpt] 统计报告已禁用 (report_interval=0)")
            return
        
        logger.info(f"[PerfOpt] 统计报告启动 (间隔{self.interval}s)")
        while self._running:
            await asyncio.sleep(self.interval)
            if not self._running: break
            self._print_report()
    
    def _print_report(self):
        uptime = int(time.time() - self.start_time)
        uptime_str = f"{uptime//3600}h{(uptime%3600)//60}m{uptime%60}s"
        
        # 构建完整的报告内容
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"📊 CM性能优化插件统计报告 | 运行时间: {uptime_str}")
        report_lines.append("=" * 80)
        
        # 消息缓存（根据模式显示）
        if self.msg_cache:
            report_lines.extend(self._build_module_stats_lines("📦 消息缓存(查询模式)", self.msg_cache))
            report_lines.append("")
        elif self.full_msg_cache:
            report_lines.extend(self._build_full_cache_stats_lines("📦 消息缓存(全量模式)", self.full_msg_cache))
            report_lines.append("")
        
        # 人物缓存
        if self.person_cache:
            report_lines.extend(self._build_module_stats_lines("👤 人物缓存", self.person_cache))
            report_lines.append("")
        
        # 表达式缓存
        if self.expr_cache:
            report_lines.extend(self._build_full_cache_stats_lines("🎭 表达式缓存", self.expr_cache))
            report_lines.append("")
        
        # 黑话缓存
        if self.jargon_cache:
            report_lines.extend(self._build_full_cache_stats_lines("🗣️ 黑话缓存", self.jargon_cache))
            report_lines.append("")
        
        # 知识库图谱缓存
        if self.kg_cache:
            report_lines.extend(self._build_kg_cache_stats_lines("🧠 知识库图谱缓存", self.kg_cache))
            report_lines.append("")
        
        # 计算总内存占用
        if self.memory_stats_enabled:
            total_memory = 0
            if self.msg_cache:
                total_memory += self._get_module_memory_usage(self.msg_cache, "message_cache", display_name="📦 消息缓存")
            elif self.full_msg_cache:
                total_memory += self._get_module_memory_usage(self.full_msg_cache, "full_message_cache", display_name="📦 消息缓存(全量)")
            if self.person_cache:
                total_memory += self._get_module_memory_usage(self.person_cache, "person_cache", display_name="👤 人物缓存")
            if self.expr_cache:
                total_memory += self._get_module_memory_usage(self.expr_cache, "expression_cache", display_name="🎭 表达式缓存")
            if self.jargon_cache:
                total_memory += self._get_module_memory_usage(self.jargon_cache, "jargon_cache", display_name="🗣️ 黑话缓存")
            if self.kg_cache:
                total_memory += self._get_module_memory_usage(self.kg_cache, "kg_cache", display_name="🧠 知识库图谱缓存")
            
            report_lines.append(f"📊 总内存占用: {MemoryUtils.format_size(total_memory)}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        # 一次性打印所有行，减少日志系统开销
        logger.info("\n".join(report_lines))
    
    def _build_full_cache_stats_lines(self, name: str, module):
        """构建全量缓存统计的行"""
        lines = []
        size = module.size()
        loading_status = "加载中" if module.loading else "已加载"
        last_refresh = time.time() - module.last_refresh if module.last_refresh > 0 else 0
        last_refresh_str = f"{int(last_refresh//60)}m{int(last_refresh%60)}s前" if last_refresh > 0 else "从未"
        
        # 显示命中统计
        t = module.stats.total()
        i = module.stats.reset_interval()

        # 兼容旧字段：使用 get() 避免 KeyError
        t_hit = t.get("t_hit", 0)
        t_miss = t.get("t_miss", 0)
        t_filtered = t.get("t_filtered", 0)
        t_skipped = t.get("t_skipped", 0)
        t_fast_time = t.get("t_fast_time", 0.0)
        t_slow_time = t.get("t_slow_time", 0.0)

        i_hit = i.get("i_hit", 0)
        i_miss = i.get("i_miss", 0)
        i_filtered = i.get("i_filtered", 0)
        i_skipped = i.get("i_skipped", 0)
        i_fast_time = i.get("i_fast_time", 0.0)
        i_slow_time = i.get("i_slow_time", 0.0)

        # 命中率口径：只统计“可缓存”的 hit/miss，跳过/被过滤不进入分母
        t_cacheable_total = t_hit + t_miss
        i_cacheable_total = i_hit + i_miss
        t_rate = (t_hit / t_cacheable_total * 100) if t_cacheable_total > 0 else 0
        i_rate = (i_hit / i_cacheable_total * 100) if i_cacheable_total > 0 else 0

        t_time = t_fast_time + t_slow_time
        i_time = i_fast_time + i_slow_time
        
        # 估算节省时间
        avg_time = t_time / t_miss if t_miss > 0 else 0.02
        saved = t_hit * avg_time
        
        # 获取内存占用
        memory_str = ""
        if self.memory_stats_enabled:
            cache_key = getattr(getattr(module, "stats", None), "name", None) or name
            memory_bytes = self._get_module_memory_usage(module, cache_key, display_name=name)
            memory_str = f" | 内存: {MemoryUtils.format_size(memory_bytes)}"
        
        lines.append(f"{name}")
        lines.append(f"  状态: {loading_status} | 大小: {size}条{memory_str} | 上次刷新: {last_refresh_str}")
        if module.refresh_interval > 0:
            lines.append(f"  自动刷新: 每{module.refresh_interval}秒")
        lines.append(
            f"  累计: 命中 {t_hit} | 未命中 {t_miss} | 跳过 {t_skipped} | 被过滤 {t_filtered} | 可缓存命中率 {t_rate:.1f}%"
        )
        lines.append(
            f"  本期: 命中 {i_hit} | 未命中 {i_miss} | 跳过 {i_skipped} | 被过滤 {i_filtered} | 可缓存命中率 {i_rate:.1f}%"
        )
        lines.append(f"  节省: {saved:.1f}秒 (平均{avg_time*1000:.1f}ms/次)")
        
        return lines
    
    def _build_module_stats_lines(self, name: str, module):
        """构建模块统计的行"""
        lines = []
        t = module.stats.total()
        i = module.stats.reset_interval()

        # 兼容旧字段：使用 get() 避免 KeyError
        t_hit = t.get("t_hit", 0)
        t_miss = t.get("t_miss", 0)
        t_filtered = t.get("t_filtered", 0)
        t_skipped = t.get("t_skipped", 0)
        t_fast = t.get("t_fast", 0)
        t_slow = t.get("t_slow", 0)
        t_fast_time = t.get("t_fast_time", 0.0)
        t_slow_time = t.get("t_slow_time", 0.0)

        i_hit = i.get("i_hit", 0)
        i_miss = i.get("i_miss", 0)
        i_filtered = i.get("i_filtered", 0)
        i_skipped = i.get("i_skipped", 0)
        i_fast = i.get("i_fast", 0)
        i_slow = i.get("i_slow", 0)
        i_fast_time = i.get("i_fast_time", 0.0)
        i_slow_time = i.get("i_slow_time", 0.0)

        # 命中率口径：只统计“可缓存”的 hit/miss
        t_cacheable_total = t_hit + t_miss
        i_cacheable_total = i_hit + i_miss
        t_rate = (t_hit / t_cacheable_total * 100) if t_cacheable_total > 0 else 0
        i_rate = (i_hit / i_cacheable_total * 100) if i_cacheable_total > 0 else 0

        t_time = t_fast_time + t_slow_time
        i_time = i_fast_time + i_slow_time
        
        # 估算节省时间
        avg_time = t_time / t_miss if t_miss > 0 else 0.03
        saved = t_hit * avg_time
        
        # 获取内存占用
        memory_str = ""
        if self.memory_stats_enabled:
            cache_key = getattr(getattr(module, "stats", None), "name", None) or name
            memory_bytes = self._get_module_memory_usage(module, cache_key, display_name=name)
            memory_str = f" | 内存: {MemoryUtils.format_size(memory_bytes)}"
        
        lines.append(f"{name}")
        lines.append(f"  缓存: {module.cache.size()}/{module.cache.max_size} | TTL: {module.cache.ttl}秒{memory_str}")
        lines.append(
            f"  累计: 命中 {t_hit} | 未命中 {t_miss} | 跳过 {t_skipped} | 被过滤 {t_filtered} | 可缓存命中率 {t_rate:.1f}%"
        )
        lines.append(f"  累计: 快 {t_fast}次/{t_fast_time:.2f}s | 慢 {t_slow}次/{t_slow_time:.2f}s")
        lines.append(
            f"  本期: 命中 {i_hit} | 未命中 {i_miss} | 跳过 {i_skipped} | 被过滤 {i_filtered} | 可缓存命中率 {i_rate:.1f}%"
        )
        lines.append(f"  节省: {saved:.1f}秒 (平均{avg_time*1000:.1f}ms/次)")
        
        return lines
    
    def _build_kg_cache_stats_lines(self, name: str, module):
        """构建知识库图谱缓存统计的行"""
        lines = []
        size = module.size()
        if size == 0:
            size_str = "未加载"
        else:
            size_str = f"节点{size['nodes']}个, 边{size['edges']}条, 实体{size['entities']}个, 段落{size['paragraphs']}个"
        
        loading_status = "加载中" if module.loading else "已加载"
        last_refresh = time.time() - module.last_refresh if module.last_refresh > 0 else 0
        last_refresh_str = f"{int(last_refresh//60)}m{int(last_refresh%60)}s前" if last_refresh > 0 else "从未"
        
        # 显示命中统计
        t = module.stats.total()
        i = module.stats.reset_interval()

        # 兼容旧字段：使用 get() 避免 KeyError
        t_hit = t.get("t_hit", 0)
        t_miss = t.get("t_miss", 0)
        t_filtered = t.get("t_filtered", 0)
        t_skipped = t.get("t_skipped", 0)
        t_fast_time = t.get("t_fast_time", 0.0)
        t_slow_time = t.get("t_slow_time", 0.0)

        i_hit = i.get("i_hit", 0)
        i_miss = i.get("i_miss", 0)
        i_filtered = i.get("i_filtered", 0)
        i_skipped = i.get("i_skipped", 0)
        i_fast_time = i.get("i_fast_time", 0.0)
        i_slow_time = i.get("i_slow_time", 0.0)

        # 命中率口径：只统计“可缓存”的 hit/miss
        t_cacheable_total = t_hit + t_miss
        i_cacheable_total = i_hit + i_miss
        t_rate = (t_hit / t_cacheable_total * 100) if t_cacheable_total > 0 else 0
        i_rate = (i_hit / i_cacheable_total * 100) if i_cacheable_total > 0 else 0

        t_time = t_fast_time + t_slow_time
        i_time = i_fast_time + i_slow_time
        
        # 估算节省时间
        avg_time = t_time / t_miss if t_miss > 0 else 0.5
        saved = t_hit * avg_time
        
        # 获取内存占用
        memory_str = ""
        if self.memory_stats_enabled:
            cache_key = getattr(getattr(module, "stats", None), "name", None) or name
            memory_bytes = self._get_module_memory_usage(module, cache_key, display_name=name)
            memory_str = f" | 内存: {MemoryUtils.format_size(memory_bytes)}"
        
        lines.append(f"{name}")
        lines.append(f"  状态: {loading_status} | 大小: {size_str}{memory_str} | 上次刷新: {last_refresh_str}")
        if module.refresh_interval > 0:
            lines.append(f"  自动刷新: 每{module.refresh_interval}秒")
        lines.append(
            f"  累计: 命中 {t_hit} | 未命中 {t_miss} | 跳过 {t_skipped} | 被过滤 {t_filtered} | 可缓存命中率 {t_rate:.1f}%"
        )
        lines.append(
            f"  本期: 命中 {i_hit} | 未命中 {i_miss} | 跳过 {i_skipped} | 被过滤 {i_filtered} | 可缓存命中率 {i_rate:.1f}%"
        )
        lines.append(f"  节省: {saved:.1f}秒 (平均{avg_time*1000:.1f}ms/次)")
        
        return lines
    
    def start(self):
        if self._running: return
        self._running = True
        try:
            asyncio.get_running_loop().create_task(self._report_loop())
            # 启动全量消息缓存的定期刷新
            if self.full_msg_cache and self.full_msg_cache.cache.refresh_interval > 0:
                asyncio.get_running_loop().create_task(self.full_msg_cache.cache._refresh_loop())
            # 启动表达式和黑话缓存的定期刷新
            if self.expr_cache and self.expr_cache.refresh_interval > 0:
                asyncio.get_running_loop().create_task(self.expr_cache._refresh_loop())
            if self.jargon_cache and self.jargon_cache.refresh_interval > 0:
                asyncio.get_running_loop().create_task(self.jargon_cache._refresh_loop())
            if self.kg_cache and self.kg_cache.refresh_interval > 0:
                asyncio.get_running_loop().create_task(self.kg_cache._refresh_loop())
        except Exception as e:
            logger.error(f"[PerfOpt] 启动后台任务失败: {e}")
    
    def stop(self):
        self._running = False
        # 消息缓存：根据模式选择移除哪个补丁
        if self.msg_cache: self.msg_cache.remove_patch()
        elif self.full_msg_cache: self.full_msg_cache.remove_patch()

        if self.person_cache: self.person_cache.remove_patch()
        
        # 移除表达式缓存补丁
        if self._orig_expression_find_similar:
            try:
                from src.bw_learner.expression_learner import ExpressionLearner
                ExpressionLearner._find_similar_situation_expression = self._orig_expression_find_similar
                logger.info("[ExprCache] 补丁已移除")
            except Exception as e:
                logger.error(f"[ExprCache] 移除补丁失败: {e}")
        
        # 移除黑话缓存补丁
        if self._orig_jargon_match:
            try:
                from src.bw_learner.jargon_explainer import JargonExplainer
                JargonExplainer.match_jargon_from_messages = self._orig_jargon_match
                logger.info("[JargonCache] 补丁已移除")
            except Exception as e:
                logger.error(f"[JargonCache] 移除补丁失败: {e}")
        
        # 移除知识库图谱缓存补丁
        if self._orig_kg_load:
            try:
                from src.chat.knowledge.kg_manager import KGManager
                KGManager.load_from_file = self._orig_kg_load
                logger.info("[KGCache] 补丁已移除")
            except Exception as e:
                logger.error(f"[KGCache] 移除补丁失败: {e}")


_opt: Optional[Optimizer] = None
_person_warmup: Optional[PersonWarmupManager] = None


def build_optimizer_cfg_from_plugin_config(plugin_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    从插件配置生成 Optimizer 所需的配置字典
    
    Args:
        plugin_config: 通过 self.get_config() 获取的插件配置
        
    Returns:
        Optimizer 所需的配置字典
    """
    cfg = {
        "report_interval": 60,
        "modules": {"message_cache": True, "person_cache": True, "expression_cache": False, "slang_cache": False, "kg_cache": False},
        "message_cache_size": 2000, "message_cache_ttl": 120.0,
        "person_cache_size": 3000, "person_cache_ttl": 1800,
        "expression_cache_batch_size": 100, "expression_cache_batch_delay": 0.05, "expression_cache_refresh_interval": 3600,
        "slang_cache_batch_size": 100, "slang_cache_batch_delay": 0.05, "slang_cache_refresh_interval": 3600, "slang_cache_enable_content_index": True,
        "kg_cache_batch_size": 100, "kg_cache_batch_delay": 0.05, "kg_cache_refresh_interval": 3600,
        "memory_stats_enabled": True,
        "memory_stats_cache_ttl": 60,
    }
    
    # plugin 标签页
    cfg["report_interval"] = plugin_config.get("plugin", {}).get("report_interval", 60)
    cfg["memory_stats_enabled"] = plugin_config.get("plugin", {}).get("memory_stats_enabled", True)
    cfg["memory_stats_cache_ttl"] = plugin_config.get("plugin", {}).get("memory_stats_cache_ttl", 60)
    
    # modules 标签页
    modules = plugin_config.get("modules", {})
    cfg["modules"]["message_cache"] = modules.get("message_cache_enabled", True)
    cfg["modules"]["person_cache"] = modules.get("person_cache_enabled", True)
    cfg["modules"]["expression_cache"] = modules.get("expression_cache_enabled", False)
    cfg["modules"]["slang_cache"] = modules.get("slang_cache_enabled", False)
    cfg["modules"]["kg_cache"] = modules.get("kg_cache_enabled", False)

    # message_cache 标签页
    cfg["message_cache_mode"] = plugin_config.get("message_cache", {}).get("cache_mode", "query")
    cfg["message_cache_size"] = plugin_config.get("message_cache", {}).get("max_size", 2000)
    cfg["message_cache_ttl"] = plugin_config.get("message_cache", {}).get("ttl", 120.0)

    # message_cache_full 标签页
    full_cache_cfg = plugin_config.get("message_cache_full", {})
    if not isinstance(full_cache_cfg, dict):
        full_cache_cfg = {}

    cfg["message_cache_full_batch_size"] = full_cache_cfg.get("batch_size", 500)
    full_batch_delay_str = full_cache_cfg.get("batch_delay", "0.05")
    try:
        cfg["message_cache_full_batch_delay"] = float(full_batch_delay_str)
    except (ValueError, TypeError):
        cfg["message_cache_full_batch_delay"] = 0.05
    cfg["message_cache_full_refresh_interval"] = full_cache_cfg.get("refresh_interval", 0)
    cfg["message_cache_full_enable_incremental"] = full_cache_cfg.get("enable_incremental", True)
    cfg["message_cache_full_max_messages_per_chat"] = full_cache_cfg.get("max_messages_per_chat", 10000)
    cfg["message_cache_full_max_total_messages"] = full_cache_cfg.get("max_total_messages", 100000)
    cfg["message_cache_full_enable_lru_eviction"] = full_cache_cfg.get("enable_lru_eviction", True)
    cfg["message_cache_full_max_chats"] = full_cache_cfg.get("max_chats", 1000)

    # message_cache_hotset 标签页（运行时兼容读取 `[message_cache.hotset]` 与 `[message_cache_hotset]`）
    hotset_cfg = plugin_config.get("message_cache_hotset", {})
    if not isinstance(hotset_cfg, dict):
        hotset_cfg = {}

    # 兼容旧/手写配置：允许 [message_cache] 下嵌套 hotset
    nested_hotset_cfg = plugin_config.get("message_cache", {}).get("hotset", {})
    if isinstance(nested_hotset_cfg, dict) and nested_hotset_cfg:
        # nested 优先级更高（用户手写更直观）
        hotset_cfg = {**hotset_cfg, **nested_hotset_cfg}

    cfg["message_hotset_enabled"] = hotset_cfg.get("enabled", True)
    cfg["message_hotset_per_chat_limit"] = hotset_cfg.get("per_chat_limit", 200)
    cfg["message_hotset_ttl"] = hotset_cfg.get("ttl", 300)
    cfg["message_hotset_max_chats"] = hotset_cfg.get("max_chats", 500)
    cfg["message_hotset_bucket_enabled"] = hotset_cfg.get("bucket_enabled", False)
    cfg["message_hotset_bucket_seconds"] = hotset_cfg.get("bucket_seconds", 5)
    cfg["message_hotset_ignore_time_limit_when_active"] = hotset_cfg.get("ignore_time_limit_when_active", True)
    cfg["message_hotset_active_time_window"] = hotset_cfg.get("active_time_window", 300)

    # person_cache 标签页
    cfg["person_cache_size"] = plugin_config.get("person_cache", {}).get("max_size", 3000)
    cfg["person_cache_ttl"] = plugin_config.get("person_cache", {}).get("ttl", 1800)

    # person_cache_warmup 标签页（运行时兼容读取 `[person_cache.warmup]` 与 `[person_cache_warmup]`）
    warmup_cfg = plugin_config.get("person_cache_warmup", {})
    if not isinstance(warmup_cfg, dict):
        warmup_cfg = {}

    nested_warmup_cfg = plugin_config.get("person_cache", {}).get("warmup", {})
    if isinstance(nested_warmup_cfg, dict) and nested_warmup_cfg:
        warmup_cfg = {**warmup_cfg, **nested_warmup_cfg}

    cfg["person_warmup_enabled"] = warmup_cfg.get("enabled", True)
    cfg["person_warmup_per_chat_message_sample"] = warmup_cfg.get("per_chat_message_sample", 30)
    cfg["person_warmup_max_persons_per_chat"] = warmup_cfg.get("max_persons_per_chat", 20)
    cfg["person_warmup_ttl"] = warmup_cfg.get("ttl", 120)
    cfg["person_warmup_debounce_seconds"] = warmup_cfg.get("debounce_seconds", 3.0)
    cfg["person_warmup_max_chats"] = warmup_cfg.get("max_chats", 500)
    
    # expression_cache 标签页
    cfg["expression_cache_batch_size"] = plugin_config.get("expression_cache", {}).get("batch_size", 100)
    expr_batch_delay_str = plugin_config.get("expression_cache", {}).get("batch_delay", "0.05")
    try:
        cfg["expression_cache_batch_delay"] = float(expr_batch_delay_str)
    except (ValueError, TypeError):
        cfg["expression_cache_batch_delay"] = 0.05
    cfg["expression_cache_refresh_interval"] = plugin_config.get("expression_cache", {}).get("refresh_interval", 3600)
    
    # slang_cache 标签页
    cfg["slang_cache_batch_size"] = plugin_config.get("slang_cache", {}).get("batch_size", 100)
    slang_batch_delay_str = plugin_config.get("slang_cache", {}).get("batch_delay", "0.05")
    try:
        cfg["slang_cache_batch_delay"] = float(slang_batch_delay_str)
    except (ValueError, TypeError):
        cfg["slang_cache_batch_delay"] = 0.05
    cfg["slang_cache_refresh_interval"] = plugin_config.get("slang_cache", {}).get("refresh_interval", 3600)
    cfg["slang_cache_enable_content_index"] = plugin_config.get("slang_cache", {}).get("enable_content_index", True)
    
    # kg_cache 标签页
    cfg["kg_cache_batch_size"] = plugin_config.get("kg_cache", {}).get("batch_size", 100)
    kg_batch_delay_str = plugin_config.get("kg_cache", {}).get("batch_delay", "0.05")
    try:
        cfg["kg_cache_batch_delay"] = float(kg_batch_delay_str)
    except (ValueError, TypeError):
        cfg["kg_cache_batch_delay"] = 0.05
    cfg["kg_cache_refresh_interval"] = plugin_config.get("kg_cache", {}).get("refresh_interval", 3600)
    
    return cfg


config_fields = {
    # ===== 插件基本配置 (第1个标签页) =====
    "plugin": {
        "enabled": ConfigField(type=bool, default=True, description="是否启用插件"),
        # 注意：宿主使用 plugin.config_version 做配置迁移（不是 plugin.version）
        "config_version": ConfigField(type=str, default="4.5.0", description="配置结构版本号（用于自动迁移/生成默认配置）"),
        "version": ConfigField(type=str, default="4.5.0", description="插件版本号，用于追踪更新"),
        "report_interval": ConfigField(type=int, default=60, description="统计报告输出间隔(秒)，设置0可关闭定时报告", min=0, max=600),
        "log_level": ConfigField(type=str, default="INFO", description="日志输出等级", choices=["DEBUG", "INFO", "WARNING", "ERROR"]),
        "memory_stats_enabled": ConfigField(type=bool, default=True, description="内存统计: 在统计报告中显示各模块的内存占用情况。关闭后不显示内存信息，可减少CPU开销"),
        "memory_stats_cache_ttl": ConfigField(type=int, default=60, description="内存统计缓存时间(秒)。内存测量有一定开销，缓存结果可避免频繁测量。建议60-300秒", min=10, max=600),
    },
    # ===== 模块开关 (第2个标签页) =====
    "modules": {
        "message_cache_enabled": ConfigField(type=bool, default=True, description="消息缓存: 拦截find_messages数据库查询，缓存结果避免重复查询。命中率通常>95%，可节省大量数据库IO"),
        "person_cache_enabled": ConfigField(type=bool, default=True, description="人物信息缓存: 拦截人物信息加载，按QQ号缓存昵称等信息。人物信息变化慢，缓存效果好"),
        "expression_cache_enabled": ConfigField(type=bool, default=False, description="表达式缓存: 双缓冲+缓慢加载+原子切换，全量缓存表达式数据。启动后约10秒完成加载"),
        "slang_cache_enabled": ConfigField(type=bool, default=False, description="黑话缓存: 双缓冲+缓慢加载+原子切换+内容索引，O(1)查找速度。启动后约10秒完成加载"),
        "kg_cache_enabled": ConfigField(type=bool, default=False, description="知识库图谱缓存: 双缓冲+缓慢加载+原子切换，全量缓存知识库图谱数据。启动后约5-10秒完成加载，查询速度提升80-90%"),
    },
    # ===== 消息缓存配置 (第3个标签页) =====
    "message_cache": {
        "cache_mode": ConfigField(type=str, default="query", description="缓存模式: query=查询缓存(默认，内存占用小), full=全量镜像(内存占用大但性能极致)", choices=["query", "full"]),
        "max_size": ConfigField(type=int, default=2000, description="最大缓存条目数(仅query模式)。每条约占用1-5KB内存，2000条约占用2-10MB。超过后自动清理最旧的条目", min=100, max=10000),
        "ttl": ConfigField(type=float, default=120.0, description="缓存过期时间(秒，仅query模式)。消息变化快，建议60-180秒。过长可能导致消息不同步", min=10.0, max=600.0),
    },
    # ===== 消息全量缓存配置 (第4个标签页) =====
    "message_cache_full": {
        "batch_size": ConfigField(type=int, default=500, description="每批加载的条数。默认500条，10万条约需20秒加载完成。增大此值可加快加载但会增加CPU峰值", min=100, max=2000),
        "batch_delay": ConfigField(type=str, default="0.05", description="批次间延迟(秒)。用于平滑加载避免CPU峰值，增大此值可降低CPU占用但延长加载时间", choices=["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0"]),
        "refresh_interval": ConfigField(type=int, default=0, description="自动刷新间隔(秒)。设置为0表示不自动刷新，仅启动时加载一次。建议0(不自动刷新)", min=0, max=86400),
        "enable_incremental": ConfigField(type=bool, default=True, description="启用增量加载。缓存未命中时自动从DB补充数据。建议开启"),
        "max_messages_per_chat": ConfigField(type=int, default=10000, description="单个chat最大缓存消息数。超过后自动删除最旧的消息", min=1000, max=50000),
        "max_total_messages": ConfigField(type=int, default=100000, description="总消息数上限(内存保护)。超过后自动淘汰最旧的chat", min=10000, max=1000000),
        "enable_lru_eviction": ConfigField(type=bool, default=True, description="启用LRU淘汰。超过上限时自动淘汰最旧的chat。建议开启"),
        "max_chats": ConfigField(type=int, default=1000, description="最多同时维护多少个chat的缓存(LRU淘汰)", min=100, max=10000),
    },
    # ===== 消息热集缓存（Hotset）(第4个标签页) =====
    # 说明：WebUI 的 schema 目前按“顶层 section”生成，这里使用独立 section。
    # 运行时兼容读取 `[message_cache.hotset]` 与 `[message_cache_hotset]` 两种写法。
    "message_cache_hotset": {
        "enabled": ConfigField(type=bool, default=True, description="启用消息热集缓存（按 chat 保存最近 N 条消息）。首次访问不阻塞：后台预热；预热完成后范围查询可直接命中热集"),
        "per_chat_limit": ConfigField(type=int, default=200, description="每个 chat 缓存最近多少条消息（建议 100-400）", min=10, max=2000),
        "ttl": ConfigField(type=int, default=300, description="单个 chat 热集的有效期(秒)。过期后将重新预热。当聊天流激活时此限制会被忽略", min=10, max=3600),
        "max_chats": ConfigField(type=int, default=500, description="最多同时维护多少个 chat 的热集（LRU淘汰）", min=10, max=5000),
        "bucket_enabled": ConfigField(type=bool, default=False, description="（默认关闭）对滑动窗口查询的 end_time 做分桶归一化，提高 query-cache 命中率；开启后会进行二次过滤保证正确"),
        "bucket_seconds": ConfigField(type=int, default=5, description="分桶粒度(秒)。仅 bucket_enabled=true 时生效", min=1, max=60),
        "ignore_time_limit_when_active": ConfigField(type=bool, default=True, description="（默认开启）当聊天流激活时，忽略时间限制，直接返回最近 N 条消息。这样可以提高聊天流激活时的缓存命中率"),
        "active_time_window": ConfigField(type=int, default=300, description="聊天流激活时间窗口(秒)。在此时间内认为聊天流是激活的，忽略时间限制", min=60, max=3600),
    },
    # ===== 人物信息缓存配置 (第5个标签页) =====
    "person_cache": {
        "max_size": ConfigField(type=int, default=3000, description="最大缓存条目数。每条约占用0.5-2KB内存，3000条约占用1.5-6MB。建议大于活跃用户数", min=100, max=10000),
        "ttl": ConfigField(type=int, default=1800, description="缓存过期时间(秒)。人物信息变化慢，建议1800秒(30分钟)。过期后自动刷新", min=60, max=7200),
    },
    # ===== 表达式缓存配置 (第6个标签页) =====
    "expression_cache": {
        "batch_size": ConfigField(type=int, default=100, description="每批加载的条数。默认100条，2万条约需10秒加载完成。增大此值可加快加载但会增加CPU峰值", min=10, max=1000),
        "batch_delay": ConfigField(type=str, default="0.05", description="批次间延迟(秒)。用于平滑加载避免CPU峰值，增大此值可降低CPU占用但延长加载时间", choices=["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0"]),
        "refresh_interval": ConfigField(type=int, default=3600, description="自动刷新间隔(秒)。设置为0表示不自动刷新，仅启动时加载一次。建议3600秒(1小时)", min=0, max=86400),
    },
    # ===== 黑话缓存配置 (第7个标签页) =====
    "slang_cache": {
        "batch_size": ConfigField(type=int, default=100, description="每批加载的条数。默认100条，2万条约需10秒加载完成。增大此值可加快加载但会增加CPU峰值", min=10, max=1000),
        "batch_delay": ConfigField(type=str, default="0.05", description="批次间延迟(秒)。用于平滑加载避免CPU峰值，增大此值可降低CPU占用但延长加载时间", choices=["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0"]),
        "refresh_interval": ConfigField(type=int, default=3600, description="自动刷新间隔(秒)。设置为0表示不自动刷新，仅启动时加载一次。建议3600秒(1小时)", min=0, max=86400),
        "enable_content_index": ConfigField(type=bool, default=True, description="启用内容索引。启用后可通过内容快速查找黑话，O(1)查找速度。会额外占用内存，每条约0.1KB"),
    },
    # ===== 知识库图谱缓存配置 (第8个标签页) =====
    "kg_cache": {
        "batch_size": ConfigField(type=int, default=100, description="每批加载的条数。默认100条，2万条约需10秒加载完成。增大此值可加快加载但会增加CPU峰值", min=10, max=1000),
        "batch_delay": ConfigField(type=str, default="0.05", description="批次间延迟(秒)。用于平滑加载避免CPU峰值，增大此值可降低CPU占用但延长加载时间", choices=["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0"]),
        "refresh_interval": ConfigField(type=int, default=3600, description="自动刷新间隔(秒)。设置为0表示不自动刷新，仅启动时加载一次。建议3600秒(1小时)", min=0, max=86400),
    },
}

# 配置节描述
config_section_descriptions = {
    "plugin": ConfigSection(
        title="插件设置",
        description="基础配置：启用/禁用、统计报告间隔、日志等级。内存占用约10-20MB，CPU开销极低",
        icon="🔧",
        collapsed=False,
        order=0
    ),
    "modules": ConfigSection(
        title="功能模块",
        description="选择启用的缓存模块。消息缓存命中率通常>95%，人物信息缓存命中率>90%。可根据需要单独开关",
        icon="📦",
        collapsed=False,
        order=1
    ),
    "message_cache": ConfigSection(
        title="消息缓存",
        description="缓存消息查询结果。支持两种模式：query=查询缓存(内存占用小，命中率95%)，full=全量镜像(内存占用大，命中率99%+)。原理：拦截数据库查询，优先从缓存读取。效果：减少约95-99%的数据库查询",
        icon="💬",
        collapsed=True,
        order=2
    ),
    "message_cache_full": ConfigSection(
        title="消息全量缓存",
        description="全量镜像模式：将所有消息加载到内存，拦截所有数据库查询。特性：双缓冲+缓慢加载+原子切换+增量加载+LRU淘汰。效果：查询延迟降低80%，命中率接近100%。注意：内存占用较大(200MB-1GB)",
        icon="💾",
        collapsed=True,
        order=3
    ),
    "message_cache_hotset": ConfigSection(
        title="消息热集",
        description="按 chat 缓存最近 N 条消息，用于加速 get_recent_messages 这类滑动窗口查询。策略：首次访问不阻塞（后台预热）；预热完成后优先命中热集，否则回退数据库。注意：仅query模式生效",
        icon="🔥",
        collapsed=True,
        order=4
    ),
    "person_cache": ConfigSection(
        title="人物信息缓存",
        description="缓存人物信息(昵称、备注等)。原理：按QQ号缓存，避免重复查询数据库。效果：减少约90%的人物信息查询",
        icon="👤",
        collapsed=True,
        order=4
    ),
    "expression_cache": ConfigSection(
        title="表达式缓存",
        description="全量缓存表达式数据。原理：双缓冲+缓慢加载+原子切换，避免CPU峰值。效果：启动后约10秒完成加载，后续查询直接从内存读取",
        icon="🎭",
        collapsed=True,
        order=4
    ),
    "slang_cache": ConfigSection(
        title="黑话缓存",
        description="全量缓存黑话/网络用语数据。原理：双缓冲+缓慢加载+原子切换，支持内容索引O(1)查找。效果：启动后约10秒完成加载，黑话匹配速度提升100倍以上",
        icon="🗣️",
        collapsed=True,
        order=5
    ),
    "kg_cache": ConfigSection(
        title="知识库图谱缓存",
        description="全量缓存知识库图谱数据。原理：双缓冲+缓慢加载+原子切换，避免CPU峰值。效果：启动后约5-10秒完成加载，知识库查询速度提升80-90%，消除文件IO开销",
        icon="🧠",
        collapsed=True,
        order=6
    ),
}

# 布局配置 - 使用标签页布局
config_layout = ConfigLayout(
    type="tabs",
    tabs=[
        ConfigTab(id="plugin", title="插件", icon="🔧", sections=["plugin"], order=0),
        ConfigTab(id="modules", title="模块开关", icon="📦", sections=["modules"], order=1),
        ConfigTab(id="message_cache", title="消息缓存", icon="💬", sections=["message_cache"], order=2),
        ConfigTab(id="message_cache_full", title="消息全量缓存", icon="💾", sections=["message_cache_full"], order=3),
        ConfigTab(id="message_cache_hotset", title="消息热集", icon="🔥", sections=["message_cache_hotset"], order=4),
        ConfigTab(id="person_cache", title="人物信息缓存", icon="👤", sections=["person_cache"], order=5),
        ConfigTab(id="expression_cache", title="表达式缓存", icon="🎭", sections=["expression_cache"], order=6),
        ConfigTab(id="slang_cache", title="黑话缓存", icon="🗣️", sections=["slang_cache"], order=7),
        ConfigTab(id="kg_cache", title="知识库图谱缓存", icon="🧠", sections=["kg_cache"], order=8),
    ]
)


@register_plugin
class PerformanceOptimizerPlugin(BasePlugin):
    plugin_name = "CM-performance-optimizer"
    plugin_version = "4.5.0"
    plugin_description = "性能优化 - 消息缓存（查询模式/全量模式） + 消息热集 + 人物信息缓存 + 表达式缓存 + 黑话缓存 + 知识库图谱缓存"
    plugin_author = "城陌"
    enable_plugin = True
    config_file_name = "config.toml"
    dependencies = []
    python_dependencies = []
    config_schema = config_fields
    config_section_descriptions = config_section_descriptions
    config_layout = config_layout
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global _opt
        
        try:
            # 使用 PluginBase 的配置系统读取配置
            plugin_config = self.config  # self.config 由 PluginBase._load_plugin_config() 加载
            
            # 检查插件是否启用
            enabled = plugin_config.get("plugin", {}).get("enabled", True)
            
            # 获取日志等级
            log_level = plugin_config.get("plugin", {}).get("log_level", "INFO")
            
            # 应用日志等级（在输出任何日志之前设置）
            import logging
            level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
            if log_level.upper() in level_map:
                logger.setLevel(level_map[log_level.upper()])
            
            # 现在输出启动日志
            logger.info("[PerfOpt] CM-performance-optimizer v4.5.0 启动")
            logger.info(f"[PerfOpt] 日志等级: {log_level.upper()}")
            
            if not enabled:
                logger.info("[PerfOpt] 插件已禁用")
                return
            
            # 使用辅助函数生成 Optimizer 配置
            cfg = build_optimizer_cfg_from_plugin_config(plugin_config)
            
            _opt = Optimizer(cfg)
            _opt.apply_patches()
            _opt.start()
            logger.info("[PerfOpt] ✓ 插件启动完成")
        except Exception as e:
            logger.error(f"[PerfOpt] 启动失败: {e}")
    
    def get_plugin_components(self):
        """返回插件组件列表"""
        components = []
        
        # 添加停止事件处理器（始终添加，确保插件停止时正确回滚）
        components.append((PerfOptStopHandler.get_handler_info(), PerfOptStopHandler))
        
        return components