"""
人物信息缓存模块 - PersonCacheModule 和 PersonWarmupManager
"""

import sys
import asyncio
import time
import threading
from collections import OrderedDict
from typing import Optional, Dict, Any, List
from pathlib import Path

# 动态加载核心模块，避免相对导入问题
def _load_core_module():
    """动态加载 core 模块，避免相对导入问题"""
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

# 尝试加载核心模块，失败时使用内置实现
try:
    core = _load_core_module()
    TTLCache = core.TTLCache
    ModuleStats = core.ModuleStats
    MemoryUtils = core.MemoryUtils
except Exception as e:
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
        
        def total(self) -> Dict[str, Any]:
            with self._lock:
                return {
                    "hits": self._hits,
                    "misses": self._misses,
                    "skipped": self._skipped,
                    "filtered": self._filtered,
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

try:
    from src.common.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger("CM_perf_opt")


class PersonWarmupManager:
    """人物信息预热（全异步触发，不阻塞主流程）。

    设计目标（对应用户选择的"全异步预热"）：
    - 写入后触发：不 await，仅 create_task。
    - 防抖：同一 chat 的写入在 debounce 窗口内只触发一次。
    - TTL：同一 chat 在 ttl 窗口内只做一次预热（避免 DB 压力）。

    预热策略：
    - 从最近 N 条消息中提取参与者（platform + user_id -> person_id）
    - 最多预热 max_persons_per_chat 个
    - 对每个 person_id 构造 Person.__init__()
      触发 Person.load_from_database()
      以便被 PersonCacheModule.apply_patch() 提前填充。
    """

    def __init__(
        self,
        enabled: bool = True,
        per_chat_message_sample: int = 30,
        max_persons_per_chat: int = 20,
        ttl: int = 120,
        debounce_seconds: float = 3.0,
        max_chats: int = 500,
        hotset=None,
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

        这里复用底层 TTLCache.get_memory_usage_sync()。
        """
        try:
            return int(self.cache.get_memory_usage_sync())
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
                """同步版本的 cached_load 函数，与原始 load_from_database 签名一致"""
                person_id = self_person.person_id
                cached = module.cache.get_sync(person_id)
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
                    module.cache.set_sync(person_id, data)
            
            def cached_sync(self_person):
                """同步版本的 cached_sync 函数，与原始 sync_to_database 签名一致"""
                module.cache.invalidate_sync(self_person.person_id)
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


def apply_person_cache(cache_manager) -> Optional[PersonCacheModule]:
    """应用人物信息缓存补丁
    
    Args:
        cache_manager: 缓存管理器实例
        
    Returns:
        PersonCacheModule 实例，失败时返回 None
    """
    try:
        # 创建人物缓存模块
        cache = PersonCacheModule()
        
        # 注册到缓存管理器
        cache_manager.register_cache("person_cache", cache)
        
        # 应用补丁
        cache.apply_patch()
        
        logger.info("[PersonCache] ✓ 人物信息缓存模块已初始化")
        return cache
        
    except Exception as e:
        logger.error(f"[PersonCache] 初始化失败: {e}")
        return None
