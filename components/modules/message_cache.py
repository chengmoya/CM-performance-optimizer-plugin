"""
消息缓存模块 - MessageCacheModule 和 MessageHotsetCache
"""

import sys
import asyncio
import time
import threading
import json
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple, List, Callable, Awaitable, cast
from pathlib import Path

# 从公共模块导入动态加载函数
try:
    from core.compat import load_core_module, CoreModuleLoadError
except ImportError:
    # 回退定义
    def load_core_module(caller_path=None, module_name="CM_perf_opt_core", submodules=None):
        """Fallback load_core_module 实现"""
        module_name = "CM_perf_opt_core"
        if module_name in sys.modules:
            return sys.modules[module_name]
        
        current_dir = Path(__file__).parent
        plugin_dir = current_dir.parent.parent
        core_init = plugin_dir / "core" / "__init__.py"
        
        if not core_init.exists():
            raise ImportError(f"Core module not found at {core_init}")
        
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, core_init)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load core module spec from {core_init}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    
    class CoreModuleLoadError(ImportError):
        """Core 模块加载失败异常"""
        pass


# 尝试加载核心模块，失败时使用内置实现
try:
    core = load_core_module(Path(__file__).parent)
    TTLCache = core.TTLCache
    ModuleStats = core.ModuleStats
    MemoryUtils = core.MemoryUtils
    ChatVersionTracker = core.ChatVersionTracker
    rate = core.rate
except (ImportError, CoreModuleLoadError) as e:
    # 内置实现
    class ModuleStats:
        """Fallback ModuleStats 实现"""

        def __init__(self, name: str):
            self.name = name
            self._lock = threading.Lock()
            self._hits = 0
            self._misses = 0
            self._skipped = 0
            self._filtered = 0
            self._unselected = 0
            self._total_time = 0.0

        def hit(self):
            with self._lock:
                self._hits += 1

        def miss(self, elapsed: float = 0.0):
            with self._lock:
                self._misses += 1
                self._total_time += elapsed

        def skipped(self):
            with self._lock:
                self._skipped += 1

        def filtered(self):
            with self._lock:
                self._filtered += 1

        def unselected(self):
            with self._lock:
                self._unselected += 1

        def total(self) -> Dict[str, Any]:
            with self._lock:
                return {
                    "hits": self._hits,
                    "misses": self._misses,
                    "skipped": self._skipped,
                    "filtered": self._filtered,
                    "unselected": self._unselected,
                    "hit_rate": (self._hits / max(1, self._hits + self._misses)),
                    "avg_response_time": (self._total_time / max(1, self._misses)),
                }

        def reset_interval(self) -> Dict[str, Any]:
            with self._lock:
                stats = self.total()
                self._hits = 0
                self._misses = 0
                self._skipped = 0
                self._filtered = 0
                self._unselected = 0
                self._total_time = 0.0
                return stats
    
    class MemoryUtils:
        """Fallback MemoryUtils 实现"""
        @staticmethod
        def get_size(obj, seen=None) -> int:
            """估算对象内存占用"""
            if seen is None:
                seen = set()
            
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)
            
            try:
                size = sys.getsizeof(obj)
            except Exception:
                size = 0
            
            if isinstance(obj, dict):
                size += sum(MemoryUtils.get_size(v, seen) for v in obj.values())
                size += sum(MemoryUtils.get_size(k, seen) for k in obj.keys())
            elif isinstance(obj, (list, tuple, set)):
                size += sum(MemoryUtils.get_size(item, seen) for item in obj)
            
            return size
    
    class ChatVersionTracker:
        """Fallback ChatVersionTracker 实现"""
        def __init__(self):
            self._lock = threading.Lock()
            self._versions: Dict[str, int] = {}
        
        def bump(self, chat_id: str) -> int:
            with self._lock:
                self._versions[chat_id] = self._versions.get(chat_id, 0) + 1
                return self._versions[chat_id]
        
        def get(self, chat_id: str) -> Optional[int]:
            with self._lock:
                return self._versions.get(chat_id)
    
    class TTLCache:
        """Fallback TTLCache 实现（支持同步和异步访问）"""
        def __init__(self, max_size=500, ttl=120.0):
            self.max_size = max_size
            self.ttl = ttl
            self._data: Dict[str, tuple] = {}
            self._sync_lock = threading.RLock()
            self._lock = asyncio.Lock()  # 保留用于向后兼容
        
        # ========== 同步方法 ==========
        
        def get_sync(self, k):
            """同步获取缓存值"""
            with self._sync_lock:
                if k in self._data:
                    value, expiry = self._data[k]
                    if time.time() <= expiry:
                        return value, True
                    else:
                        del self._data[k]
                return None, False
        
        def set_sync(self, k, v):
            """同步设置缓存值"""
            with self._sync_lock:
                expiry = time.time() + self.ttl
                self._data[k] = (v, expiry)
                
                # LRU eviction
                if len(self._data) > self.max_size:
                    oldest_key = min(self._data.keys(), key=lambda x: self._data[x][1])
                    del self._data[oldest_key]
        
        def invalidate_sync(self, k):
            """同步使缓存失效"""
            with self._sync_lock:
                self._data.pop(k, None)
        
        def clear_sync(self):
            """同步清空缓存"""
            with self._sync_lock:
                self._data.clear()
        
        def get_memory_usage_sync(self) -> int:
            """同步获取缓存内存使用量"""
            with self._sync_lock:
                return MemoryUtils.get_size(self._data)
        
        # ========== 异步方法（通过 to_thread 调用同步方法）==========
        
        async def get(self, k):
            return await asyncio.to_thread(self.get_sync, k)
        
        async def set(self, k, v):
            await asyncio.to_thread(self.set_sync, k, v)
        
        async def invalidate(self, k):
            await asyncio.to_thread(self.invalidate_sync, k)
        
        async def clear(self):
            await asyncio.to_thread(self.clear_sync)
        
        async def get_memory_usage(self):
            return await asyncio.to_thread(self.get_memory_usage_sync)
    
    def rate(name: str):
        """Fallback rate 装饰器"""
        def decorator(func):
            return func
        return decorator

try:
    import orjson
except ImportError:
    orjson = None

try:
    from src.common.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger("CM_perf_opt")

# 全局版本跟踪器
_chat_versions = ChatVersionTracker()

# 对 `time: {"$lt": now}` / `{ "$lte": now }` 这类"拉取到当前上下文"查询的归一化窗口。
# 当 `$lt/$lte` 的时间戳距离当前时间超过该窗口时：默认不做归一化，且不缓存，避免误缓存历史查询。
MSG_CACHE_NORMALIZE_LT_WINDOW_SECONDS = 300.0


def _quantize_time_fields_for_key(message_filter: Any, quantum: float = 3.0) -> Any:
    """对 filter 中的 time 字段做量化，仅用于生成缓存 key。

    说明：
    - 只量化顶层 key == "time" 且其值为 dict 的情况。
    - 量化窗口 quantum 单位为秒，默认 3 秒。
    - 仅用于 key 生成，实际查询仍使用原始 filter。

    示例：
        1707123456.789 -> 1707123456.0（向下取整到 quantum 的倍数）

    Args:
        message_filter: 原始过滤器（通常是 dict）。
        quantum: 量化窗口（秒）。

    Returns:
        量化后的过滤器副本（若无需量化则返回原对象）。
    """

    if not isinstance(message_filter, dict):
        return message_filter

    try:
        q = float(quantum)
    except Exception:
        return message_filter

    if q <= 0:
        return message_filter

    if "time" not in message_filter or not isinstance(message_filter.get("time"), dict):
        return message_filter

    original_time = message_filter.get("time")
    if not isinstance(original_time, dict):
        return message_filter

    changed = False
    quantized_time: Dict[str, Any] = {}

    for op, op_value in original_time.items():
        # 仅量化数值；其他类型保持原样
        if isinstance(op_value, (int, float)):
            quantized = (float(op_value) // q) * q
            if quantized != float(op_value):
                changed = True
            quantized_time[op] = quantized
            continue

        # 支持数值字符串（例如 "1707123456.789"）
        if isinstance(op_value, str):
            try:
                fval = float(op_value)
                quantized = (fval // q) * q
                if quantized != fval:
                    changed = True
                quantized_time[op] = quantized
                continue
            except Exception:
                pass

        quantized_time[op] = op_value

    if not changed:
        return message_filter

    out = dict(message_filter)
    out["time"] = quantized_time

    try:
        logger.debug(
            f"[MsgCache] key time-quantize quantum={q}s time={original_time}->{quantized_time}"
        )
    except Exception:
        pass

    return out


class _MessageCacheStatsProxy:
    """MessageCache 统计代理。

    兼容 core.ModuleStats 的接口，同时追加 MessageHotsetCache / 写入侧失效 等扩展统计。

    StatsReporter 约定：对象提供 total()/reset_interval()，返回 dict。
    该代理会先合并 base 的 dict，再追加扩展字段。
    """

    def __init__(self, base: ModuleStats):
        self._base = base
        self._lock = threading.Lock()

        # 累计（t_*)
        self.t_hotset_hit = 0
        self.t_hotset_miss = 0
        self.t_hotset_warmup_scheduled = 0
        self.t_hotset_warmup_success = 0
        self.t_hotset_warmup_error = 0
        self.t_write_bump = 0
        self.t_write_invalidate_hotset = 0

        # 间隔（i_*)
        self.i_hotset_hit = 0
        self.i_hotset_miss = 0
        self.i_hotset_warmup_scheduled = 0
        self.i_hotset_warmup_success = 0
        self.i_hotset_warmup_error = 0
        self.i_write_bump = 0
        self.i_write_invalidate_hotset = 0

    # ---- base stats proxy ----

    def hit(self) -> None:
        self._base.hit()

    def miss(self, elapsed: float) -> None:
        self._base.miss(elapsed)

    def skipped(self) -> None:
        if hasattr(self._base, "skipped"):
            self._base.skipped()

    def filtered(self) -> None:
        if hasattr(self._base, "filtered"):
            self._base.filtered()

    def unselected(self) -> None:
        if hasattr(self._base, "unselected"):
            self._base.unselected()

    # ---- extra stats ----

    def hotset_hit(self) -> None:
        with self._lock:
            self.t_hotset_hit += 1
            self.i_hotset_hit += 1

    def hotset_miss(self) -> None:
        with self._lock:
            self.t_hotset_miss += 1
            self.i_hotset_miss += 1

    def hotset_warmup_scheduled(self) -> None:
        with self._lock:
            self.t_hotset_warmup_scheduled += 1
            self.i_hotset_warmup_scheduled += 1

    def hotset_warmup_success(self) -> None:
        with self._lock:
            self.t_hotset_warmup_success += 1
            self.i_hotset_warmup_success += 1

    def hotset_warmup_error(self) -> None:
        with self._lock:
            self.t_hotset_warmup_error += 1
            self.i_hotset_warmup_error += 1

    def write_bump(self) -> None:
        with self._lock:
            self.t_write_bump += 1
            self.i_write_bump += 1

    def write_invalidate_hotset(self) -> None:
        with self._lock:
            self.t_write_invalidate_hotset += 1
            self.i_write_invalidate_hotset += 1

    def total(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            if hasattr(self._base, "total"):
                base_total = self._base.total()
                if isinstance(base_total, dict):
                    out.update(base_total)
        except Exception:
            pass

        with self._lock:
            out.update(
                {
                    "t_hotset_hit": self.t_hotset_hit,
                    "t_hotset_miss": self.t_hotset_miss,
                    "t_hotset_warmup_scheduled": self.t_hotset_warmup_scheduled,
                    "t_hotset_warmup_success": self.t_hotset_warmup_success,
                    "t_hotset_warmup_error": self.t_hotset_warmup_error,
                    "t_write_bump": self.t_write_bump,
                    "t_write_invalidate_hotset": self.t_write_invalidate_hotset,
                }
            )
        return out

    def reset_interval(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            if hasattr(self._base, "reset_interval"):
                base_interval = self._base.reset_interval()
                if isinstance(base_interval, dict):
                    out.update(base_interval)
        except Exception:
            pass

        with self._lock:
            out.update(
                {
                    "i_hotset_hit": self.i_hotset_hit,
                    "i_hotset_miss": self.i_hotset_miss,
                    "i_hotset_warmup_scheduled": self.i_hotset_warmup_scheduled,
                    "i_hotset_warmup_success": self.i_hotset_warmup_success,
                    "i_hotset_warmup_error": self.i_hotset_warmup_error,
                    "i_write_bump": self.i_write_bump,
                    "i_write_invalidate_hotset": self.i_write_invalidate_hotset,
                }
            )

            self.i_hotset_hit = 0
            self.i_hotset_miss = 0
            self.i_hotset_warmup_scheduled = 0
            self.i_hotset_warmup_success = 0
            self.i_hotset_warmup_error = 0
            self.i_write_bump = 0
            self.i_write_invalidate_hotset = 0

        return out


class MessageHotsetCache:
    """按 chat 缓存"最近 N 条消息"的热集（Hotset）。

    设计目标（对应用户选择的"全异步预热"）：
    - 首次访问不阻塞：仅在后台触发 warmup；当前请求仍走 DB。
    - warmup 完成后：后续滑动窗口范围查询可直接命中热集（无需 DB）。

    注意：这里的缓存是"消息集合缓存"，不是 query-cache。
    """

    def __init__(
        self,
        enabled: bool = True,
        per_chat_limit: int = 200,
        ttl: int = 300,
        max_chats: int = 500,
        ignore_time_limit_when_active: bool = True,
        active_time_window: int = 300,
        stats: Optional[Any] = None,
    ):
        self.enabled = bool(enabled)
        self.per_chat_limit = int(per_chat_limit)
        self.ttl = float(ttl)
        self.max_chats = int(max_chats)
        self.ignore_time_limit_when_active = bool(ignore_time_limit_when_active)
        self.active_time_window = float(active_time_window)
        self._stats = stats

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

    def try_get_latest(
        self,
        chat_id: str,
        limit: int,
        limit_mode: str = "latest",
        filter_intercept_message_level: Optional[int] = None,
    ) -> Optional[List[Any]]:
        """尝试用热集回答“仅按 chat_id + limit”的查询。

        典型调用：
        - find_messages({"chat_id": X}, limit=N)
        - find_messages({"chat_id": X, "time": {"$lte": now}}, limit=N)

        返回 None 表示无法回答（需要回退 DB）。
        """
        if not self.enabled:
            return None

        try:
            cid = str(chat_id)
            lim = int(limit)
        except Exception:
            return None

        if lim <= 0:
            return None

        snap = self.get_messages_if_fresh(cid)
        if not snap:
            return None

        if filter_intercept_message_level is not None:
            out: List[Any] = []
            for m in snap:
                try:
                    msg_level = getattr(m, "intercept_message_level", 0)
                    if int(msg_level) > int(filter_intercept_message_level):
                        continue
                except Exception:
                    continue
                out.append(m)
        else:
            out = snap

        if not out:
            return None

        if limit_mode == "earliest":
            return out[:lim]

        if len(out) <= lim:
            return out
        return out[-lim:]

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

    def ensure_warmup(self, chat_id: str, force: bool = False):
        """确保该 chat 已触发后台预热（不阻塞）。

        Args:
            chat_id: 聊天 ID
            force: 是否强制刷新。用于写入侧触发：避免热集在 TTL 内返回旧数据。

        修复问题3: 使用"写时复制"策略，避免刷新期间查询返回空数据。
        之前是先删除旧数据再加载新数据，期间查询可能返回空。
        现在是先加载新数据到临时位置，成功后再原子替换旧数据。
        """
        if not self.enabled:
            return
        cid = str(chat_id)

        now = time.time()
        scheduled = False
        with self._lock:
            # fresh 就不需要预热（仅在非强制刷新时检查）
            if (not force) and self._is_fresh_locked(cid, now):
                self._touch_locked(cid)
                return

            # warmup 去重 + 简单防抖：同一 chat 1 秒内最多触发一次
            last = float(self._last_refresh_attempt.get(cid, 0.0))
            if (not force) and (now - last < 1.0):
                return
            self._last_refresh_attempt[cid] = now

            if cid in self._warming:
                return
            self._warming.add(cid)
            scheduled = True

        if scheduled and self._stats is not None and hasattr(self._stats, "hotset_warmup_scheduled"):
            try:
                self._stats.hotset_warmup_scheduled()
            except Exception:
                pass

        try:
            loop = asyncio.get_running_loop()
        except Exception:
            # 没有 running loop 时无法后台预热（保持"回退 DB"的正确性）
            with self._lock:
                self._warming.discard(cid)
            return

        async def _warm():
            ok = False
            new_data = None
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
                # 修复问题3: 先构建新数据，成功后再原子替换
                # 这样在刷新期间，查询仍然可以返回旧数据（即使过期），而不是空
                new_data = {"ts": now2, "messages": res}

                with self._lock:
                    # 只有在 force=True 时才删除旧数据（在写入之后）
                    # 否则保留旧数据，直到新数据加载完成
                    if force:
                        self._data.pop(cid, None)

                    # 原子替换：直接写入新数据
                    self._data[cid] = new_data
                    self._data.move_to_end(cid)

                    # LRU 淘汰
                    while len(self._data) > self.max_chats:
                        self._data.popitem(last=False)

                ok = True
            except Exception as e:
                logger.debug(f"[Hotset] warmup 失败 chat_id={cid}: {e}")
            finally:
                try:
                    if self._stats is not None:
                        if ok and hasattr(self._stats, "hotset_warmup_success"):
                            self._stats.hotset_warmup_success()
                        elif (not ok) and hasattr(self._stats, "hotset_warmup_error"):
                            self._stats.hotset_warmup_error()
                except Exception:
                    pass

                with self._lock:
                    self._warming.discard(cid)

        try:
            loop.create_task(_warm())
        except Exception:
            with self._lock:
                self._warming.discard(cid)


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

        # 统计：兼容 StatsReporter 的 total/reset_interval，同时追加 hotset / 写入侧等扩展字段
        self._base_stats = ModuleStats("message_cache")
        self.stats = _MessageCacheStatsProxy(self._base_stats)

        # 热集缓存（范围查询加速）
        self.hotset = MessageHotsetCache(
            enabled=hotset_enabled,
            per_chat_limit=hotset_per_chat_limit,
            ttl=hotset_ttl,
            max_chats=hotset_max_chats,
            ignore_time_limit_when_active=hotset_ignore_time_limit_when_active,
            active_time_window=hotset_active_time_window,
            stats=self.stats,
        )
        # 预留：滑动窗口分桶增强（默认关闭；后续实现 query-cache 提升时使用）
        self.hotset_bucket_enabled = bool(hotset_bucket_enabled)
        self.hotset_bucket_seconds = int(hotset_bucket_seconds)

        self._orig_func = None
        self._orig_store_message = None
        self._patched_store_message = False
        self._patched = False
        self._alias_patches: List[Tuple[str, str, Any]] = []
        self._patched_find_messages_func = None
        self._patched_store_message_func = None

    def get_memory_usage(self) -> int:
        """获取模块内存占用（字节）。

        这里复用底层 TTLCache.get_memory_usage_sync()。
        """
        try:
            return int(self.cache.get_memory_usage_sync())
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

        关键点：对常见的"拉取到当前上下文"查询进行归一化，避免时间戳不断变化导致 key 高基数。

        Returns:
            Tuple[str, bool]: (cache_key, cacheable)
        """
        mf = message_filter or {}
        mf_for_key = mf
        cacheable = True

        chat_id = mf.get("chat_id")
        time_cond = mf.get("time")

        # 修复：query-cache 必须包含 chat_id（避免跨群缓存混用）
        if chat_id is None or str(chat_id) == "":
            cacheable = False
            try:
                logger.warning(
                    "[MsgCache] 禁止缓存无 chat_id 的查询，避免跨群命中 message_filter=%r",
                    mf,
                )
            except Exception:
                pass
            return "", cacheable

        # ---- 1) 处理 time 条件导致的高基数 key ----
        # 常见查询：{"chat_id": X, "time": {"$lt": now}} / {"$lte": now}
        # 归一化策略：当 `$lt/$lte` 时间戳接近当前时间（窗口内），将 time 替换为固定占位符，并附带 chat_version。
        # 范围查询（同时包含 $gt/$lt）通常为滑动窗口：对 $gt/$lt 进行时间窗口量化，降低 key 基数。
        if chat_id and isinstance(time_cond, dict):
            has_lt = ("$lt" in time_cond) or ("$lte" in time_cond)
            has_gt = ("$gt" in time_cond) or ("$gte" in time_cond)

            if has_lt and has_gt:
                # 范围查询：将边界量化到固定桶（默认 3 秒），提升相似查询的复用率
                bucket_seconds = 3.0
                gt_val = time_cond.get("$gt", time_cond.get("$gte"))
                lt_val = time_cond.get("$lt", time_cond.get("$lte"))
                try:
                    gt_float = float(gt_val)
                    lt_float = float(lt_val)
                    bucket_int = int(bucket_seconds)
                    if bucket_int <= 0:
                        raise ValueError("bucket_seconds must be > 0")
                    gt_bucket = int(gt_float // bucket_seconds) * bucket_int
                    lt_bucket = int(lt_float // bucket_seconds) * bucket_int

                    mf_for_key = dict(mf)
                    mf_for_key["time"] = {"$gt": gt_bucket, "$lt": lt_bucket}

                    try:
                        logger.debug(
                            f"[MsgCache] range-key bucket chat_id={chat_id} gt={gt_val}->{gt_bucket} lt={lt_val}->{lt_bucket}"
                        )
                    except Exception:
                        pass
                except Exception as e:
                    # 无法解析/量化时间戳时：保守退回不缓存，避免误命中
                    cacheable = False
                    try:
                        logger.debug(
                            f"[MsgCache] range-key bucket failed chat_id={chat_id} time={time_cond} err={e}"
                        )
                    except Exception:
                        pass
            elif has_lt and not has_gt:
                ops = set(time_cond.keys())
                # 只对"单边上界"做归一化；如果还有其他操作符，先保守不归一化
                if ops.issubset({"$lt", "$lte"}):
                    ts_val = time_cond.get("$lt", time_cond.get("$lte"))
                    # BUG FIX: 添加边界检查，避免 ts_val 为 None 时触发 UnboundLocalError
                    if ts_val is None:
                        cacheable = False
                    else:
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

        # ---- 2) time 量化（仅用于 key 生成；不影响实际查询 filter） ----
        mf_for_key = _quantize_time_fields_for_key(mf_for_key, quantum=3.0)

        # ---- 3) 稳定序列化 ----
        # message_filter 可能包含复杂类型，这里用 default=str 做兜底。
        # 优先使用 orjson 提升性能，降级到 json
        if orjson is not None:
            try:
                # orjson 不支持 sort_keys，需要先排序
                if isinstance(mf_for_key, dict):
                    sorted_mf = dict(sorted(mf_for_key.items()))
                    mf_str = orjson.dumps(sorted_mf, default=str).decode('utf-8')
                else:
                    mf_str = orjson.dumps(mf_for_key, default=str).decode('utf-8')
            except Exception:
                mf_str = json.dumps(mf_for_key, sort_keys=True, ensure_ascii=False, default=str)
        else:
            mf_str = json.dumps(mf_for_key, sort_keys=True, ensure_ascii=False, default=str)

        # 对 sort 参数进行稳定序列化（可能是字典、列表或 None）
        if sort is not None:
            if isinstance(sort, dict):
                if orjson is not None:
                    try:
                        sorted_sort = dict(sorted(sort.items()))
                        sort_str = orjson.dumps(sorted_sort, default=str).decode('utf-8')
                    except Exception:
                        sort_str = json.dumps(sort, sort_keys=True, ensure_ascii=False, default=str)
                else:
                    sort_str = json.dumps(sort, sort_keys=True, ensure_ascii=False, default=str)
            elif isinstance(sort, list):
                if orjson is not None:
                    try:
                        sort_str = orjson.dumps(sort, default=str).decode('utf-8')
                    except Exception:
                        sort_str = json.dumps(sort, ensure_ascii=False, default=str)
                else:
                    sort_str = json.dumps(sort, ensure_ascii=False, default=str)
            else:
                sort_str = str(sort)
        else:
            sort_str = "None"

        # 处理 filter_intercept_message_level 可能为 None 的情况
        filter_level_str = str(filter_intercept_message_level) if filter_intercept_message_level is not None else "None"

        # ---- 3) 引入 chat_version 让"最新上下文"缓存可自动失效 ----
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
                """同步版本的 patched 函数，与原始 find_messages 签名一致"""
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
                    lim = int(limit)

                    if chat_id and lim > 0:
                        hot = None

                        if time_cond is None:
                            # 仅 chat_id + limit：直接走热集切片，避免 query-cache 与 hotset 双份存储
                            hot = module.hotset.try_get_latest(
                                chat_id=str(chat_id),
                                limit=lim,
                                limit_mode=str(limit_mode or "latest"),
                                filter_intercept_message_level=filter_intercept_message_level,
                            )
                        elif isinstance(time_cond, dict):
                            has_lt = ("$lt" in time_cond) or ("$lte" in time_cond)
                            has_gt = ("$gt" in time_cond) or ("$gte" in time_cond)

                            if has_lt and has_gt:
                                start_ts = time_cond.get("$gt", time_cond.get("$gte"))
                                end_ts = time_cond.get("$lt", time_cond.get("$lte"))
                                if start_ts is not None and end_ts is not None:
                                    hot = module.hotset.try_get_range(
                                        chat_id=str(chat_id),
                                        start_ts=float(start_ts),
                                        end_ts=float(end_ts),
                                        limit=lim,
                                        limit_mode=str(limit_mode or "latest"),
                                        filter_intercept_message_level=filter_intercept_message_level,
                                    )
                            elif has_lt and (not has_gt):
                                # 仅上界：语义上是“拉到当前上下文”，也可用热集切片回答
                                hot = module.hotset.try_get_latest(
                                    chat_id=str(chat_id),
                                    limit=lim,
                                    limit_mode=str(limit_mode or "latest"),
                                    filter_intercept_message_level=filter_intercept_message_level,
                                )

                        if hot is not None:
                            if hasattr(module.stats, "hotset_hit"):
                                module.stats.hotset_hit()
                            # hotset 命中时，本次请求未走 query-cache（视为"未选择"）
                            if hasattr(module.stats, "unselected"):
                                module.stats.unselected()
                            return hot

                        # miss：后台触发 warmup（不阻塞）并回退 query-cache/DB
                        if hasattr(module.stats, "hotset_miss"):
                            module.stats.hotset_miss()
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

                # 不可缓存的请求：不计入 miss（否则命中率会被"主动跳过"的请求稀释）
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

                chat_id_for_cache = mf.get("chat_id")
                has_chat_id = chat_id_for_cache is not None and str(chat_id_for_cache) != ""
                if cacheable and not has_chat_id:
                    try:
                        logger.warning(
                            "[MsgCache] 发现无 chat_id 的缓存请求，已防御拦截 message_filter=%s",
                            mf,
                        )
                    except Exception:
                        pass

                def _extract_chat_id(item: Any) -> Optional[str]:
                    if isinstance(item, dict):
                        val = item.get("chat_id")
                    else:
                        val = getattr(item, "chat_id", None)
                    if val is None:
                        return None
                    return str(val)

                def _all_in_chat(items: Any, expected: Any) -> bool:
                    try:
                        exp = str(expected)
                        if not isinstance(items, list):
                            return False
                        for it in items:
                            cid = _extract_chat_id(it)
                            if cid is None or cid != exp:
                                return False
                        return True
                    except Exception:
                        return False

                # 使用同步缓存方法
                val, hit = module.cache.get_sync(key)
                if hit:
                    if has_chat_id and not _all_in_chat(val, chat_id_for_cache):
                        try:
                            logger.warning(
                                "[MsgCache] query-cache 命中但 chat_id 不一致，已失效 key_len=%s chat_id=%s",
                                len(key),
                                chat_id_for_cache,
                            )
                        except Exception:
                            pass
                        try:
                            module.cache.invalidate_sync(key)
                        except Exception:
                            pass
                        if hasattr(module.stats, "filtered"):
                            module.stats.filtered()
                        hit = False

                    if hit:
                        module.stats.hit()
                        try:
                            logger.debug(
                                "[MsgCache] query-cache hit has_chat_id=%s chat_id=%s key_len=%s",
                                has_chat_id,
                                chat_id_for_cache,
                                len(key),
                            )
                        except Exception:
                            pass
                        return val

                try:
                    logger.debug(
                        "[MsgCache] query-cache miss has_chat_id=%s chat_id=%s key_len=%s",
                        has_chat_id,
                        chat_id_for_cache,
                        len(key),
                    )
                except Exception:
                    pass

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
                    # 写入 query-cache：除非请求已被 hotset 命中（命中时已在上方 return）
                    if not has_chat_id:
                        try:
                            logger.warning(
                                "[MsgCache] 拒绝写入无 chat_id 的缓存 message_filter=%s",
                                mf,
                            )
                        except Exception:
                            pass
                    else:
                        try:
                            module.cache.set_sync(key, res)
                            try:
                                logger.debug(
                                    "[MsgCache] query-cache set limit=%s has_chat_id=%s chat_id=%s key_len=%s",
                                    limit,
                                    has_chat_id,
                                    chat_id_for_cache,
                                    len(key),
                                )
                            except Exception:
                                pass
                        except Exception as e:
                            try:
                                logger.debug(f"[MsgCache] query-cache set failed err={e}")
                            except Exception:
                                pass
                return res

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
                    "find_messages", "message_cache",
                    self._orig_func, patched,
                )

            message_repository.find_messages = patched
            self._patched_find_messages_func = patched

            # 替换已导入的引用
            for n, m in list(sys.modules.items()):
                if m and getattr(m, "find_messages", None) is self._orig_func:
                    # record & patch alias import
                    try:
                        self._alias_patches.append((n, "find_messages", getattr(m, "find_messages", None)))
                    except Exception:
                        pass
                    setattr(m, "find_messages", patched)
                    logger.debug(f"[MsgCache] 替换 {n}.find_messages")

            # 写入侧：补丁 MessageStorage.store_message，在写入后 bump chat_version
            # 注意：上游 store_message 内部会吞异常（try/except），因此无法严格判定"成功写入"。
            # 这里按"非通知消息调用过 store_message 即认为可能写入"，用于正确失效缓存。
            try:
                from src.chat.message_receive.storage import MessageStorage

                if not self._patched_store_message:
                    # ⚠️ 兼容性要点：原始实现是 @staticmethod。
                    # 如果直接赋值函数，会变成普通方法，实例调用时会多传一个 self，导致
                    # "takes 2 positional arguments but 3 were given"。
                    # 因此必须保存原 descriptor，并用 staticmethod 包装回写。
                    self._orig_store_message = MessageStorage.__dict__.get("store_message")
                    orig_store_callable = getattr(MessageStorage, "store_message", None)

                    async def patched_store_message(message, chat_stream):
                        # 修复问题2: 将版本递增移到数据库写入之前，避免竞态条件
                        # 先递增版本，确保在写入之前就使缓存失效
                        try:
                            # 通知消息不参与版本递增（与原逻辑一致）
                            if not getattr(message, "is_notify", False):
                                chat_id = getattr(chat_stream, "stream_id", None)
                                if chat_id:
                                    # 版本递增必须在数据库写入之前执行，确保查询侧看到的版本是最新的
                                    _chat_versions.bump(str(chat_id))
                                    if hasattr(module.stats, "write_bump"):
                                        module.stats.write_bump()
                        except Exception:
                            # 版本递增失败不应影响主流程，继续执行写入
                            pass

                        # 执行原始的数据库写入操作
                        if callable(orig_store_callable):
                            await cast(
                                Callable[[Any, Any], Awaitable[None]],
                                orig_store_callable,
                            )(message, chat_stream)

                        # 写入后：避免 hotset 在 TTL 内返回旧数据 -> 强制刷新
                        try:
                            if not getattr(message, "is_notify", False):
                                chat_id = getattr(chat_stream, "stream_id", None)
                                if chat_id:
                                    module.hotset.ensure_warmup(str(chat_id), force=True)
                                    if hasattr(module.stats, "write_invalidate_hotset"):
                                        module.stats.write_invalidate_hotset()
                        except Exception:
                            # 热集刷新失败不应影响主流程
                            pass

                    # PatchChain 注册 store_message
                    if _pc is not None:
                        _pc.register_patch(
                            "store_message", "message_cache",
                            orig_store_callable, patched_store_message,
                        )

                    MessageStorage.store_message = staticmethod(patched_store_message)
                    self._patched_store_message_func = patched_store_message

                    # 替换已导入的引用（有些模块可能 `from ... import MessageStorage` 后取了 store_message 句柄）
                    for n, m in list(sys.modules.items()):
                        if m and getattr(m, "store_message", None) is orig_store_callable:
                            try:
                                self._alias_patches.append((n, "store_message", getattr(m, "store_message", None)))
                            except Exception:
                                pass
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

            # 回滚已导入的别名引用（例如: from src.common.message_repository import find_messages）
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
                    _pc.unregister_patch("find_messages", "message_cache")
                    _pc.unregister_patch("store_message", "message_cache")
            except Exception:
                pass

            self._patched = False
            logger.info("[MsgCache] 补丁已移除")
        except Exception as e:
            logger.error(f"[MsgCache] 移除补丁失败: {e}")


def apply_message_cache(cache_manager) -> Optional[MessageCacheModule]:
    """应用消息缓存补丁
    
    Args:
        cache_manager: 缓存管理器实例
        
    Returns:
        MessageCacheModule 实例，失败时返回 None
    """
    try:
        # 创建消息缓存模块（内部会自动创建热集缓存）
        cache = MessageCacheModule()
        
        # 注册到缓存管理器
        cache_manager.register_cache("message_cache", cache)
        
        # 应用补丁
        cache.apply_patch()
        
        logger.info("[MsgCache] ✓ 消息缓存模块已初始化")
        return cache
        
    except Exception as e:
        logger.error(f"[MsgCache] 初始化失败: {e}")
        return None
