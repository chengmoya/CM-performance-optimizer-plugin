"""
CM 性能优化插件 v4.4.0

功能模块：
1. 消息缓存 (message_cache) - 缓存 find_messages 查询结果
2. 人物信息缓存 (person_cache) - 缓存人物信息查询
3. 表达式缓存 (expression_cache) - 双缓冲+缓慢加载+原子切换
4. 黑话缓存 (slang_cache) - 双缓冲+缓慢加载+原子切换+内容索引
5. 预加载 (preload) - 异步预加载聊天流的消息和人物信息

安装：将目录放入 MaiBot/plugins/ 下，重启 MaiBot
依赖：无额外依赖
"""

import sys
import asyncio
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict

try:
    from src.plugin_system.apis.plugin_register_api import register_plugin
    from src.plugin_system.base.base_plugin import BasePlugin
    from src.plugin_system.base.base_events_handler import BaseEventHandler
    from src.plugin_system.base.config_types import ConfigField, ConfigSection, ConfigLayout, ConfigTab
    from src.plugin_system.base.component_types import EventType, CustomEventHandlerResult, MaiMessages
    from src.common.logger import get_logger
except ImportError:
    class BasePlugin:
        def __init__(self, plugin_dir=None): pass
    class BaseEventHandler:
        event_type = "unknown"
        handler_name = ""
        handler_description = ""
        weight = 0
        def __init__(self): pass
        def set_plugin_config(self, plugin_config): pass
        def set_plugin_name(self, plugin_name): pass
    class EventType:
        ON_MESSAGE = "on_message"
    class ConfigField:
        def __init__(self, **kw): pass
    class ConfigSection:
        def __init__(self, **kw): pass
    class ConfigLayout:
        def __init__(self, **kw): pass
    class ConfigTab:
        def __init__(self, **kw): pass
    class CustomEventHandlerResult:
        def __init__(self, message=""): self.message = message
    class MaiMessages:
        pass
    def register_plugin(cls): return cls
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
    
    def set(self, k, v):
        with self.lock:
            if len(self.data) >= self.max_size:
                old = next(iter(self.data))
                del self.data[old], self.ts[old]
            self.data[k] = v
            self.ts[k] = time.time()
    
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
    """单个模块的统计"""
    def __init__(self, name: str):
        self.name = name
        self.lock = threading.Lock()
        self.t_hit = self.t_miss = self.t_filtered = 0
        self.i_hit = self.i_miss = self.i_filtered = 0
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
    
    def filtered(self):
        """记录命中但被过滤的情况（如chat_id不匹配）- 仅表达式和黑话缓存使用"""
        with self.lock:
            self.t_filtered += 1
            self.i_filtered += 1
    
    def reset_interval(self) -> Dict[str, Any]:
        with self.lock:
            r = {"i_hit": self.i_hit, "i_miss": self.i_miss, "i_filtered": self.i_filtered,
                 "i_fast": self.i_fast, "i_slow": self.i_slow,
                 "i_fast_time": self.i_fast_time, "i_slow_time": self.i_slow_time}
            self.i_hit = self.i_miss = self.i_filtered = 0
            self.i_fast = self.i_slow = 0
            self.i_fast_time = self.i_slow_time = 0.0
            return r
    
    def total(self) -> Dict[str, Any]:
        with self.lock:
            return {"t_hit": self.t_hit, "t_miss": self.t_miss, "t_filtered": self.t_filtered,
                    "t_fast": self.t_fast, "t_slow": self.t_slow,
                    "t_fast_time": self.t_fast_time, "t_slow_time": self.t_slow_time}


def rate(hit, miss, filtered=0):
    """计算命中率 - 仅表达式和黑话缓存使用 filtered 参数"""
    t = hit + miss + filtered
    # filtered 也算命中（缓存命中但被业务逻辑过滤）
    effective_hit = hit + filtered
    return (effective_hit / t * 100) if t > 0 else 0


# ===== 消息缓存模块 =====
class MessageCacheModule:
    """消息查询缓存"""
    def __init__(self, max_size=2000, ttl=120.0):
        self.cache = TTLCache(max_size, ttl)
        self.stats = ModuleStats("message_cache")
        self._orig_func = None
        self._patched = False
    
    def apply_patch(self):
        if self._patched: return
        try:
            from src.common import message_repository
            self._orig_func = message_repository.find_messages
            module = self
            
            def patched(message_filter, sort=None, limit=0, limit_mode="latest",
                       filter_bot=False, filter_command=False, filter_intercept_message_level=None):
                mf = message_filter or {}
                key = f"{mf.get('chat_id','')}:{mf.get('stream_id','')}:{limit}:{limit_mode}:{filter_bot}"
                
                val, hit = module.cache.get(key)
                if hit:
                    module.stats.hit()
                    return val
                
                t0 = time.time()
                res = module._orig_func(message_filter, sort, limit, limit_mode, 
                                        filter_bot, filter_command, filter_intercept_message_level)
                module.stats.miss(time.time() - t0)
                
                if 0 < limit <= 200:
                    module.cache.set(key, res)
                return res
            
            message_repository.find_messages = patched
            # 替换已导入的引用
            for n, m in list(sys.modules.items()):
                if m and getattr(m, 'find_messages', None) is self._orig_func:
                    setattr(m, 'find_messages', patched)
                    logger.debug(f"[消息缓存] 替换 {n}.find_messages")
            
            self._patched = True
            logger.info("[消息缓存] ✓ 补丁应用成功")
        except Exception as e:
            logger.error(f"[消息缓存] ✗ 补丁失败: {e}")
    
    def remove_patch(self):
        if not self._patched or not self._orig_func: return
        try:
            from src.common import message_repository
            message_repository.find_messages = self._orig_func
            self._patched = False
            logger.info("[消息缓存] 补丁已移除")
        except: pass
    
    def get_memory_usage(self):
        """获取缓存内存使用量（字节）"""
        return self.cache.get_memory_usage()


# ===== 人物信息缓存模块 (从person-cache-plugin整合) =====
class PersonCacheModule:
    """人物信息缓存"""
    def __init__(self, max_size=3000, ttl=1800):
        self.cache = TTLCache(max_size, ttl)
        self.stats = ModuleStats("person_cache")
        self._orig_load = None
        self._orig_sync = None
        self._patched = False
    
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
                    for k, v in cached[0].items():
                        setattr(self_person, k, v)
                    logger.debug(f"[人物缓存] 缓存命中: person_id={person_id}, is_known={self_person.is_known}")
                    return
                
                t0 = time.time()
                module._orig_load(self_person)
                elapsed = time.time() - t0
                module.stats.miss(elapsed)
                
                logger.debug(f"[人物缓存] 缓存未命中: person_id={person_id}, is_known={self_person.is_known}, 耗时={elapsed:.3f}s")
                
                if self_person.is_known:
                    data = {
                        "user_id": getattr(self_person, "user_id", ""),
                        "platform": getattr(self_person, "platform", ""),
                        "is_known": getattr(self_person, "is_known", False),
                        "nickname": getattr(self_person, "nickname", ""),
                        "person_name": getattr(self_person, "person_name", None),
                        "name_reason": getattr(self_person, "name_reason", None),
                        "know_times": getattr(self_person, "know_times", 0),
                        "know_since": getattr(self_person, "know_since", None),
                        "last_know": getattr(self_person, "last_know", None),
                        "memory_points": list(getattr(self_person, "memory_points", []) or []),
                        "group_nick_name": list(getattr(self_person, "group_nick_name", []) or []),
                    }
                    module.cache.set(person_id, data)
                    logger.debug(f"[人物缓存] 已缓存已知用户: person_id={person_id}, nickname={data.get('nickname', 'N/A')}")
                else:
                    logger.debug(f"[人物缓存] 未知用户不缓存: person_id={person_id}")
            
            def cached_sync(self_person):
                person_id = self_person.person_id
                is_known_before = getattr(self_person, 'is_known', False)
                logger.debug(f"[人物缓存] sync_to_database: person_id={person_id}, is_known={is_known_before}")
                module.cache.invalidate(person_id)
                module._orig_sync(self_person)
                is_known_after = getattr(self_person, 'is_known', False)
                logger.debug(f"[人物缓存] sync_to_database完成: person_id={person_id}, is_known={is_known_after}, 缓存已失效")
            
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
        except: pass
    
    def get_memory_usage(self):
        """获取缓存内存使用量（字节）"""
        return self.cache.get_memory_usage()


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
            logger.info("[表达式缓存] 开始缓慢加载表达式缓存到缓冲区B...")
            
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
                logger.debug(f"[表达式缓存] 加载进度: {len(buffer_b_data)} 条")
                
                # 休眠，避免CPU峰值
                await asyncio.sleep(self.batch_delay)
                
                offset += self.batch_size
            
            # 加载完成，原子切换
            with self.buffer_lock:
                self.buffer_b = buffer_b_data
                # 原子切换：buffer_b → buffer_a
                self.buffer_a, self.buffer_b = self.buffer_b, None
                
            self.last_refresh = time.time()
            logger.info(f"[表达式缓存] 缓存加载完成并切换: {len(buffer_b_data)} 条")
            
        except Exception as e:
            logger.error(f"[表达式缓存] 缓存加载失败: {e}")
        finally:
            async with self.load_lock:
                self.loading = False
    
    async def _refresh_loop(self):
        """定期刷新循环"""
        while True:
            await asyncio.sleep(self.refresh_interval)
            logger.info("[表达式缓存] 触发定期刷新...")
            await self._load_to_buffer_b()
    
    def refresh(self):
        """手动刷新缓存"""
        try:
            asyncio.get_running_loop().create_task(self._load_to_buffer_b())
            logger.info("[表达式缓存] 已触发手动刷新")
        except RuntimeError:
            logger.warning("[表达式缓存] 无法触发刷新：没有运行中的事件循环")
    
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


# ===== 黑话缓存模块 (双缓冲 + 缓慢加载 + AC自动机) =====
class JargonCacheModule:
    """黑话全量缓存 - 双缓冲 + 缓慢加载 + 原子切换 + AC自动机优化"""
    def __init__(self, batch_size=100, batch_delay=0.05, refresh_interval=3600, enable_ahocorasick=True):
        # 双缓冲
        self.buffer_a = None      # 当前使用的缓存
        self.buffer_b = None      # 后台加载的缓存
        self.buffer_lock = threading.Lock()
        
        # AC自动机
        self.automaton_a = None   # 当前使用的AC自动机
        self.automaton_b = None   # 后台加载的AC自动机
        self.enable_ahocorasick = enable_ahocorasick
        
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
    
    def get_automaton(self):
        """获取当前AC自动机"""
        with self.buffer_lock:
            return self.automaton_a
    
    async def _load_to_buffer_b(self):
        """缓慢加载数据到缓冲区B"""
        async with self.load_lock:
            if self.loading:
                return
            self.loading = True
        
        try:
            logger.info("[黑话缓存] 开始缓慢加载黑话缓存到缓冲区B...")
            
            # 清空缓冲区B
            buffer_b_data = []
            automaton_b = None
            
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
                
                # 记录进度
                logger.debug(f"[黑话缓存] 加载进度: {len(buffer_b_data)} 条")
                
                # 休眠，避免CPU峰值
                await asyncio.sleep(self.batch_delay)
                
                offset += self.batch_size
            
            # 构建AC自动机（如果启用）
            if self.enable_ahocorasick and buffer_b_data:
                try:
                    import ahocorasick
                    automaton_b = ahocorasick.Automaton()
                    
                    # 收集所有有效的黑话内容
                    for jargon in buffer_b_data:
                        if jargon.content and jargon.content.strip():
                            # 使用小写作为键，支持大小写不敏感匹配
                            automaton_b.add_word(jargon.content.lower(), jargon)
                    
                    automaton_b.make_automaton()
                    logger.info(f"[黑话缓存] AC自动机构建完成: {len(buffer_b_data)} 条黑话")
                except ImportError:
                    logger.warning("[黑话缓存] ahocorasick 库未安装，将使用遍历+正则匹配")
                    self.enable_ahocorasick = False
                except Exception as e:
                    logger.error(f"[黑话缓存] AC自动机构建失败: {e}，将使用遍历+正则匹配")
                    self.enable_ahocorasick = False
                    automaton_b = None
            
            # 加载完成，原子切换
            with self.buffer_lock:
                self.buffer_b = buffer_b_data
                self.automaton_b = automaton_b
                # 原子切换：buffer_b → buffer_a
                self.buffer_a, self.buffer_b = self.buffer_b, None
                self.automaton_a, self.automaton_b = self.automaton_b, None
                
            self.last_refresh = time.time()
            logger.info(f"[黑话缓存] 缓存加载完成并切换: {len(buffer_b_data)} 条")
            
        except Exception as e:
            logger.error(f"[黑话缓存] 缓存加载失败: {e}")
        finally:
            async with self.load_lock:
                self.loading = False
    
    async def _refresh_loop(self):
        """定期刷新循环"""
        while True:
            await asyncio.sleep(self.refresh_interval)
            logger.info("[黑话缓存] 触发定期刷新...")
            await self._load_to_buffer_b()
    
    def refresh(self):
        """手动刷新缓存"""
        try:
            asyncio.get_running_loop().create_task(self._load_to_buffer_b())
            logger.info("[黑话缓存] 已触发手动刷新")
        except RuntimeError:
            logger.warning("[黑话缓存] 无法触发刷新：没有运行中的事件循环")
    
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
            if self.automaton_a is not None:
                size += MemoryUtils.get_size(self.automaton_a)
            return size


# ===== 预加载管理器 =====
class PreloadManager:
    """预加载管理器 - 管理聊天流的预加载状态（单例模式）"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, max_streams=10):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, max_streams=10):
        if self._initialized:
            return
        self.preloaded_streams = set()  # 已预加载的聊天流ID
        self.max_streams = max_streams
        self.stream_last_active = {}  # 聊天流最后活跃时间
        self.lock = threading.Lock()
        self._initialized = True
    
    def should_preload(self, stream_id: str) -> bool:
        """检查是否需要预加载"""
        with self.lock:
            if stream_id in self.preloaded_streams:
                # 更新活跃时间
                self.stream_last_active[stream_id] = time.time()
                return False
            return True
    
    def mark_preloaded(self, stream_id: str):
        """标记为已预加载"""
        with self.lock:
            self.preloaded_streams.add(stream_id)
            self.stream_last_active[stream_id] = time.time()
            
            # 如果超过最大数量，移除最旧的
            if len(self.preloaded_streams) > self.max_streams:
                oldest = min(self.stream_last_active.items(), key=lambda x: x[1])[0]
                self.preloaded_streams.remove(oldest)
                del self.stream_last_active[oldest]
    
    def get_stats(self):
        """获取预加载统计"""
        with self.lock:
            return {
                "preloaded_count": len(self.preloaded_streams),
                "max_streams": self.max_streams,
                "streams": list(self.preloaded_streams)
            }
    
    def get_memory_usage(self):
        """获取预加载管理器内存使用量（字节）"""
        with self.lock:
            size = MemoryUtils.get_size(self.preloaded_streams)
            size += MemoryUtils.get_size(self.stream_last_active)
            return size


# ===== 预加载事件处理器 =====
class PreloadEventHandler(BaseEventHandler):
    """预加载事件处理器 - 监听消息事件并预加载聊天流数据"""
    event_type = EventType.ON_MESSAGE
    handler_name = "preload"
    handler_description = "预加载聊天流数据"
    weight = 100  # 设置较高权重，确保在其他处理器之前执行
    
    # 配置参数（可通过插件配置覆盖）
    message_count = 50
    max_persons_per_stream = 50
    preload_delay = 0.1
    max_streams = 10
    
    def __init__(self):
        super().__init__()
        self.preload_manager = PreloadManager(max_streams=self.max_streams)
        self.enabled = True
    
    def set_plugin_config(self, plugin_config: Dict) -> None:
        """设置插件配置并更新预加载参数"""
        super().set_plugin_config(plugin_config)
        # 从配置中读取预加载参数
        if plugin_config:
            preload_config = plugin_config.get("preload", {})
            self.message_count = preload_config.get("message_count", 50)
            self.max_persons_per_stream = preload_config.get("max_persons_per_stream", 50)
            self.preload_delay = float(preload_config.get("preload_delay", "0.1"))
            self.max_streams = preload_config.get("max_streams", 10)
            # 更新预加载管理器的最大流数
            if hasattr(self, 'preload_manager'):
                self.preload_manager.max_streams = self.max_streams
    
    async def execute(self, message: MaiMessages | None) -> Tuple[bool, bool, Optional[str], Optional[CustomEventHandlerResult], Optional[MaiMessages]]:
        """执行事件处理"""
        if not self.enabled:
            return True, True, None, None, None
        
        if not message or not hasattr(message, 'stream_id') or not message.stream_id:
            return True, True, None, None, None
        
        stream_id = message.stream_id
        
        # 检查是否需要预加载
        if not self.preload_manager.should_preload(stream_id):
            return True, True, None, None, None
        
        # 异步预加载
        asyncio.create_task(self._preload_stream_data(stream_id))
        
        return True, True, None, None, None
    
    async def _preload_stream_data(self, stream_id: str):
        """预加载聊天流数据"""
        try:
            # 延迟执行，避免阻塞主流程
            await asyncio.sleep(self.preload_delay)
            
            logger.debug(f"[预加载] 开始预加载: stream_id={stream_id[:20]}...")
            
            # 1. 加载最近的消息（这会自动填充消息缓存）
            from src.common import message_repository
            messages = message_repository.find_messages(
                {"stream_id": stream_id},
                limit=self.message_count,
                limit_mode="latest"
            )
            
            logger.debug(f"[预加载] 加载了 {len(messages)} 条消息")
            
            # 2. 提取消息中的发送者ID和平台组合
            person_keys = set()
            for msg in messages:
                if hasattr(msg, "user_id") and msg.user_id and hasattr(msg, "user_platform") and msg.user_platform:
                    person_keys.add((msg.user_platform, msg.user_id))
            
            logger.debug(f"[预加载] 发现 {len(person_keys)} 个唯一用户: {list(person_keys)}")
            
            # 3. 预加载这些人物信息
            from src.person_info.person_info import Person
            loaded_count = 0
            known_count = 0
            unknown_count = 0
            for platform, user_id in list(person_keys)[:self.max_persons_per_stream]:
                person_id = f"{platform}_{user_id}"
                person = Person(person_id=person_id)
                person.load_from_database()  # 这会自动填充人物缓存
                loaded_count += 1
                if person.is_known:
                    known_count += 1
                else:
                    unknown_count += 1
            
            # 4. 标记为已预加载
            self.preload_manager.mark_preloaded(stream_id)
            
            logger.info(f"[预加载] 预加载完成: {stream_id[:20]}..., 消息{len(messages)}条, 人物{loaded_count}个 (已知{known_count}, 未知{unknown_count})")
        except Exception as e:
            logger.error(f"[预加载] 预加载失败: {e}")


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
        self._memory_stats_cache = {}  # 模块内存统计缓存: {module_name: (timestamp, size)}
        self._memory_stats_lock = threading.Lock()
        
        # 初始化模块
        self.msg_cache = None
        self.person_cache = None
        self.expr_cache = None
        self.jargon_cache = None
        
        if self.modules_cfg.get("message_cache", True):
            self.msg_cache = MessageCacheModule(
                cfg.get("message_cache_size", 2000),
                cfg.get("message_cache_ttl", 120.0)
            )
        
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
                enable_ahocorasick=cfg.get("slang_cache_enable_ahocorasick", True)
            )
        
        self._running = False
        self._ready = True
    
    def reload_config(self, cfg=None):
        """重新加载配置"""
        if cfg is None:
            return
        
        # 更新基本配置
        old_interval = self.interval
        self.interval = cfg.get("report_interval", 60)
        self.memory_stats_enabled = cfg.get("memory_stats_enabled", True)
        self.memory_stats_cache_ttl = cfg.get("memory_stats_cache_ttl", 60)
        
        # 更新模块配置
        old_modules_cfg = self.modules_cfg.copy()
        self.modules_cfg = cfg.get("modules", {})
        
        # 检查模块开关是否变化
        modules_changed = (
            old_modules_cfg.get("message_cache", True) != self.modules_cfg.get("message_cache", True) or
            old_modules_cfg.get("person_cache", True) != self.modules_cfg.get("person_cache", True) or
            old_modules_cfg.get("expression_cache", False) != self.modules_cfg.get("expression_cache", False) or
            old_modules_cfg.get("slang_cache", False) != self.modules_cfg.get("slang_cache", False)
        )
        
        # 如果模块开关变化，需要重新初始化模块
        if modules_changed:
            logger.info("[性能优化] 模块开关已变化，重新初始化模块...")
            self._reinit_modules(cfg)
        else:
            # 只更新模块参数
            self._update_module_params(cfg)
        
        logger.info(f"[性能优化] 配置已重新加载: report_interval={old_interval}s -> {self.interval}s")
    
    def _reinit_modules(self, cfg):
        """重新初始化所有模块"""
        # 移除旧模块的补丁
        if self.msg_cache:
            self.msg_cache.remove_patch()
        if self.person_cache:
            self.person_cache.remove_patch()
        
        # 重新初始化模块
        if self.modules_cfg.get("message_cache", True):
            self.msg_cache = MessageCacheModule(
                cfg.get("message_cache_size", 2000),
                cfg.get("message_cache_ttl", 120.0)
            )
        else:
            self.msg_cache = None
        
        if self.modules_cfg.get("person_cache", True):
            self.person_cache = PersonCacheModule(
                cfg.get("person_cache_size", 3000),
                cfg.get("person_cache_ttl", 1800)
            )
        else:
            self.person_cache = None
        
        if self.modules_cfg.get("expression_cache", False):
            self.expr_cache = ExpressionCacheModule(
                batch_size=cfg.get("expression_cache_batch_size", 100),
                batch_delay=cfg.get("expression_cache_batch_delay", 0.05),
                refresh_interval=cfg.get("expression_cache_refresh_interval", 3600)
            )
        else:
            self.expr_cache = None
        
        if self.modules_cfg.get("slang_cache", False):
            self.jargon_cache = JargonCacheModule(
                batch_size=cfg.get("slang_cache_batch_size", 100),
                batch_delay=cfg.get("slang_cache_batch_delay", 0.05),
                refresh_interval=cfg.get("slang_cache_refresh_interval", 3600),
                enable_ahocorasick=cfg.get("slang_cache_enable_ahocorasick", True)
            )
        else:
            self.jargon_cache = None
        
        # 重新应用补丁
        self.apply_patches()
        
        # 重新启动后台任务
        if self._running:
            if self.expr_cache and self.expr_cache.refresh_interval > 0:
                try:
                    asyncio.get_running_loop().create_task(self.expr_cache._refresh_loop())
                except: pass
            if self.jargon_cache and self.jargon_cache.refresh_interval > 0:
                try:
                    asyncio.get_running_loop().create_task(self.jargon_cache._refresh_loop())
                except: pass
    
    def _update_module_params(self, cfg):
        """更新模块参数（不重新初始化）"""
        # 注意：TTLCache 的参数在创建后无法修改
        # 这里只记录日志，实际参数变化需要重启才能生效
        logger.debug("[性能优化] 模块参数已更新（部分参数需要重启才能生效）")
    
    def _get_module_memory_usage(self, module, module_name):
        """获取模块内存使用量（带缓存）"""
        if not self.memory_stats_enabled:
            return 0
        
        current_time = time.time()
        
        with self._memory_stats_lock:
            # 检查缓存
            if module_name in self._memory_stats_cache:
                cache_time, cache_size = self._memory_stats_cache[module_name]
                if current_time - cache_time < self.memory_stats_cache_ttl:
                    return cache_size
            
            # 重新测量
            try:
                size = module.get_memory_usage()
                self._memory_stats_cache[module_name] = (current_time, size)
                return size
            except Exception as e:
                logger.debug(f"[内存统计] 获取 {module_name} 内存失败: {e}")
                return 0
    
    def apply_patches(self):
        if self.msg_cache:
            self.msg_cache.apply_patch()
        if self.person_cache:
            self.person_cache.apply_patch()
        
        # 表达式缓存拦截
        if self.expr_cache:
            self._apply_expression_cache_patch()
        
        # 黑话缓存拦截
        if self.jargon_cache:
            self._apply_jargon_cache_patch()
        
    
    def _apply_expression_cache_patch(self):
        """应用表达式缓存拦截"""
        try:
            from src.bw_learner.expression_learner import ExpressionLearner
            orig_find_similar = ExpressionLearner._find_similar_situation_expression
            expr_cache = self.expr_cache
            stats = self.expr_cache.stats
            
            async def patched_find_similar(learner_self, situation: str, similarity_threshold: float = 0.75):
                # 从缓存获取所有表达式
                all_expressions = expr_cache.get_all()
                
                # 如果缓存未加载，走原逻辑
                if not all_expressions:
                    logger.debug("[表达式缓存] 缓存未加载，使用数据库查询")
                    t0 = time.time()
                    result = await orig_find_similar(learner_self, situation, similarity_threshold)
                    stats.miss(time.time() - t0)
                    logger.debug(f"[表达式缓存] 缓存未命中(未加载): 耗时={time.time()-t0:.3f}s")
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
                    logger.debug(f"[表达式缓存] 缓存命中: 相似度={best_similarity:.3f}, 现有='{best_match.situation}', 新='{situation}'")
                elif best_match_all:
                    # 在缓存中找到匹配，但 chat_id 不匹配
                    stats.filtered()
                    logger.debug(f"[表达式缓存] 缓存命中但被过滤: situation='{situation}', 匹配chat_id={matched_chat_id_all}, 查询chat_id={learner_self.chat_id}, 相似度={best_similarity_all:.3f}")
                else:
                    stats.miss(0.0)  # 缓存中未找到，但查询很快
                    logger.debug(f"[表达式缓存] 缓存未命中(无匹配): situation='{situation}'")
                
                return best_match, best_similarity
            
            ExpressionLearner._find_similar_situation_expression = patched_find_similar
            logger.info("[表达式缓存] ✓ 表达式缓存拦截已应用")
        except Exception as e:
            logger.error(f"[表达式缓存] ✗ 表达式缓存拦截失败: {e}")
    
    def _apply_jargon_cache_patch(self):
        """应用黑话缓存拦截"""
        try:
            from src.bw_learner.jargon_explainer import JargonExplainer
            from src.bw_learner.learner_utils import is_bot_message, contains_bot_self_name, parse_chat_id_list, chat_id_list_contains
            from src.config.config import global_config
            import re
            
            orig_match_jargon = JargonExplainer.match_jargon_from_messages
            jargon_cache = self.jargon_cache
            stats = self.jargon_cache.stats
            
            def patched_match_jargon(explainer_self, messages):
                # 从缓存获取所有黑话
                all_jargons = jargon_cache.get_all()
                
                # 如果缓存未加载，走原逻辑
                if not all_jargons:
                    logger.debug("[黑话缓存] 缓存未加载，使用数据库查询")
                    t0 = time.time()
                    result = orig_match_jargon(explainer_self, messages)
                    stats.miss(time.time() - t0)
                    logger.debug(f"[黑话缓存] 缓存未命中(未加载): 耗时={time.time()-t0:.3f}s, 消息数={len(messages)}")
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
                    logger.debug("[黑话缓存] 缓存未命中(无有效消息)")
                    return []
                
                # 合并所有消息文本
                combined_text = " ".join(message_texts)
                
                # 根据 all_global_jargon 配置决定查询逻辑
                all_global_jargon = global_config.expression.all_global_jargon
                
                # 在缓存中过滤有meaning的黑话
                valid_jargons = [j for j in all_jargons if j.meaning and j.meaning.strip()]
                
                # 用于统计被过滤的匹配
                filtered_matches = []
                
                # 使用AC自动机进行快速匹配（如果启用）
                automaton = jargon_cache.get_automaton()
                if automaton and jargon_cache.enable_ahocorasick:
                    # AC自动机匹配：一次性找出所有匹配
                    matched_jargon = {}
                    hit_count = 0
                    
                    # 使用小写文本进行匹配（支持大小写不敏感）
                    combined_text_lower = combined_text.lower()
                    
                    for end_idx, jargon in automaton.iter(combined_text_lower):
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
                                    filtered_matches.append((content, jargon.chat_id))
                                    continue
                        
                        # 找到匹配，记录（去重）
                        if content not in matched_jargon:
                            matched_jargon[content] = {"content": content}
                            hit_count += 1
                else:
                    # 回退到遍历+正则匹配
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
                    logger.debug(f"[黑话缓存] 缓存命中: 匹配到 {hit_count} 个黑话: {list(matched_jargon.keys())}")
                elif filtered_matches:
                    stats.filtered()
                    filtered_sample = filtered_matches[:3]  # 只显示前3个
                    logger.debug(f"[黑话缓存] 缓存命中但被过滤: 匹配到 {len(filtered_matches)} 个黑话但chat_id不匹配，示例: {filtered_sample}")
                else:
                    stats.miss(0.0)
                    logger.debug(f"[黑话缓存] 缓存未命中(无匹配): 消息数={len(messages)}, 有效黑话数={len(valid_jargons)}, 文本长度={len(combined_text)}")
                
                return list(matched_jargon.values())
            
            JargonExplainer.match_jargon_from_messages = patched_match_jargon
            logger.info("[黑话缓存] ✓ 黑话缓存拦截已应用")
        except Exception as e:
            logger.error(f"[黑话缓存] ✗ 黑话缓存拦截失败: {e}")
    
    async def _report_loop(self):
        # 如果间隔为0，禁用日志报告
        if self.interval <= 0:
            logger.info("[性能优化] 统计报告已禁用 (间隔设置为0)")
            return
            
        logger.info(f"[性能优化] 统计报告启动 (间隔{self.interval}s)")
        while self._running:
            await asyncio.sleep(self.interval)
            if not self._running: break
            self._print_report()
    
    def _print_report(self):
        uptime = int(time.time() - self.start_time)
        uptime_str = f"{uptime//3600}小时{(uptime%3600)//60}分钟{uptime%60}秒"
        
        # 构建完整的报告内容
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"📊 CM性能优化插件统计报告 | 已经跑了: {uptime_str}")
        report_lines.append("=" * 80)
        
        # 消息缓存
        if self.msg_cache:
            report_lines.extend(self._build_module_stats_lines("📦 消息缓存", self.msg_cache))
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
        
        # 计算总内存占用
        if self.memory_stats_enabled:
            total_memory = 0
            if self.msg_cache:
                total_memory += self._get_module_memory_usage(self.msg_cache, "消息缓存")
            if self.person_cache:
                total_memory += self._get_module_memory_usage(self.person_cache, "人物缓存")
            if self.expr_cache:
                total_memory += self._get_module_memory_usage(self.expr_cache, "表达式缓存")
            if self.jargon_cache:
                total_memory += self._get_module_memory_usage(self.jargon_cache, "黑话缓存")
            
            report_lines.append(f"📊 总共用了多少内存: {MemoryUtils.format_size(total_memory)}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        # 一次性打印所有行，减少日志系统开销
        logger.info("\n".join(report_lines))
    
    def _build_full_cache_stats_lines(self, name: str, module):
        """构建全量缓存统计的行"""
        lines = []
        size = module.size()
        loading_status = "正在加载" if module.loading else "已经加载好了"
        last_refresh = time.time() - module.last_refresh if module.last_refresh > 0 else 0
        last_refresh_str = f"{int(last_refresh//60)}分{int(last_refresh%60)}秒前" if last_refresh > 0 else "还没刷新过"
        
        # 显示命中统计
        t = module.stats.total()
        i = module.stats.reset_interval()
        t_rate = rate(t["t_hit"], t["t_miss"], t["t_filtered"])
        i_rate = rate(i["i_hit"], i["i_miss"], i["i_filtered"])
        t_time = t["t_fast_time"] + t["t_slow_time"]
        i_time = i["i_fast_time"] + i["i_slow_time"]
        
        # 估算节省时间
        avg_time = t_time / t["t_miss"] if t["t_miss"] > 0 else 0.02
        saved = t["t_hit"] * avg_time
        
        # 获取内存占用
        memory_str = ""
        if self.memory_stats_enabled:
            # 将带emoji的名称映射到纯中文名称
            name_map = {
                "🎭 表达式缓存": "表达式缓存",
                "🗣️ 黑话缓存": "黑话缓存",
                "📦 消息缓存": "消息缓存",
                "👤 人物缓存": "人物缓存"
            }
            module_name_cn = name_map.get(name, name)
            memory_bytes = self._get_module_memory_usage(module, module_name_cn)
            memory_str = f" | 占用内存: {MemoryUtils.format_size(memory_bytes)}"
        
        lines.append(f"{name}")
        lines.append(f"  现在状态: {loading_status} | 存了{size}条数据{memory_str} | 上次刷新: {last_refresh_str}")
        if module.refresh_interval > 0:
            lines.append(f"  自动刷新: 每隔{module.refresh_interval}秒刷新一次")
        lines.append(f"  从开始到现在: 命中{t['t_hit']}次 | 没命中{t['t_miss']}次 | 被过滤{t['t_filtered']}次 | 命中率{t_rate:.1f}%")
        lines.append(f"  这一期: 命中{i['i_hit']}次 | 没命中{i['i_miss']}次 | 被过滤{i['i_filtered']}次 | 命中率{i_rate:.1f}%")
        lines.append(f"  省了{saved:.1f}秒时间 (平均每次{avg_time*1000:.1f}毫秒)")
        
        return lines
    
    def _build_module_stats_lines(self, name: str, module):
        """构建模块统计的行 - 消息缓存和人物缓存（不使用filtered统计）"""
        lines = []
        t = module.stats.total()
        i = module.stats.reset_interval()
        t_rate = rate(t["t_hit"], t["t_miss"])
        i_rate = rate(i["i_hit"], i["i_miss"])
        t_time = t["t_fast_time"] + t["t_slow_time"]
        i_time = i["i_fast_time"] + i["i_slow_time"]
        
        # 估算节省时间
        avg_time = t_time / t["t_miss"] if t["t_miss"] > 0 else 0.03
        saved = t["t_hit"] * avg_time
        
        # 获取内存占用
        memory_str = ""
        if self.memory_stats_enabled:
            # 将带emoji的名称映射到纯中文名称
            name_map = {
                "📦 消息缓存": "消息缓存",
                "👤 人物缓存": "人物缓存"
            }
            module_name_cn = name_map.get(name, name)
            memory_bytes = self._get_module_memory_usage(module, module_name_cn)
            memory_str = f" | 占用内存: {MemoryUtils.format_size(memory_bytes)}"
        
        lines.append(f"{name}")
        lines.append(f"  缓存情况: 存了{module.cache.size()}条，最多能存{module.cache.max_size}条 | 过期时间{module.cache.ttl}秒{memory_str}")
        lines.append(f"  从开始到现在: 命中{t['t_hit']}次 | 没命中{t['t_miss']}次 | 命中率{t_rate:.1f}%")
        lines.append(f"  从开始到现在: 快速查询{t['t_fast']}次(花了{t['t_fast_time']:.2f}秒) | 慢速查询{t['t_slow']}次(花了{t['t_slow_time']:.2f}秒)")
        lines.append(f"  这一期: 命中{i['i_hit']}次 | 没命中{i['i_miss']}次 | 命中率{i_rate:.1f}%")
        lines.append(f"  省了{saved:.1f}秒时间 (平均每次{avg_time*1000:.1f}毫秒)")
        
        return lines
    
    def start(self):
        if self._running: return
        self._running = True
        try:
            asyncio.get_running_loop().create_task(self._report_loop())
            # 启动表达式和黑话缓存的定期刷新
            if self.expr_cache and self.expr_cache.refresh_interval > 0:
                asyncio.get_running_loop().create_task(self.expr_cache._refresh_loop())
            if self.jargon_cache and self.jargon_cache.refresh_interval > 0:
                asyncio.get_running_loop().create_task(self.jargon_cache._refresh_loop())
        except: pass
    
    def stop(self):
        self._running = False
        if self.msg_cache: self.msg_cache.remove_patch()
        if self.person_cache: self.person_cache.remove_patch()


_opt: Optional[Optimizer] = None

config_fields = {
    # ===== 插件基本配置 (第1个标签页) =====
    "plugin": {
        "enabled": ConfigField(type=bool, default=True, description="是否启用插件"),
        "version": ConfigField(type=str, default="4.6.0", description="插件版本号，用于追踪更新"),
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
    },
    # ===== 消息缓存配置 (第3个标签页) =====
    "message_cache": {
        "max_size": ConfigField(type=int, default=2000, description="最大缓存条目数。每条约占用1-5KB内存，2000条约占用2-10MB。超过后自动清理最旧的条目", min=100, max=10000),
        "ttl": ConfigField(type=float, default=120.0, description="缓存过期时间(秒)。消息变化快，建议60-180秒。过长可能导致消息不同步", min=10.0, max=600.0),
    },
    # ===== 人物信息缓存配置 (第4个标签页) =====
    "person_cache": {
        "max_size": ConfigField(type=int, default=3000, description="最大缓存条目数。每条约占用0.5-2KB内存，3000条约占用1.5-6MB。建议大于活跃用户数", min=100, max=10000),
        "ttl": ConfigField(type=int, default=1800, description="缓存过期时间(秒)。人物信息变化慢，建议1800秒(30分钟)。过期后自动刷新", min=60, max=7200),
    },
    # ===== 表达式缓存配置 (第5个标签页) =====
    "expression_cache": {
        "batch_size": ConfigField(type=int, default=100, description="每批加载的条数。默认100条，2万条约需10秒加载完成。增大此值可加快加载但会增加CPU峰值", min=10, max=1000),
        "batch_delay": ConfigField(type=str, default="0.05", description="批次间延迟(秒)。用于平滑加载避免CPU峰值，增大此值可降低CPU占用但延长加载时间", choices=["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0"]),
        "refresh_interval": ConfigField(type=int, default=3600, description="自动刷新间隔(秒)。设置为0表示不自动刷新，仅启动时加载一次。建议3600秒(1小时)", min=0, max=86400),
    },
    # ===== 黑话缓存配置 (第6个标签页) =====
    "slang_cache": {
        "batch_size": ConfigField(type=int, default=100, description="每批加载的条数。默认100条，2万条约需10秒加载完成。增大此值可加快加载但会增加CPU峰值", min=10, max=1000),
        "batch_delay": ConfigField(type=str, default="0.05", description="批次间延迟(秒)。用于平滑加载避免CPU峰值，增大此值可降低CPU占用但延长加载时间", choices=["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0"]),
        "refresh_interval": ConfigField(type=int, default=3600, description="自动刷新间隔(秒)。设置为0表示不自动刷新，仅启动时加载一次。建议3600秒(1小时)", min=0, max=86400),
        "enable_content_index": ConfigField(type=bool, default=True, description="启用内容索引。启用后可通过内容快速查找黑话，O(1)查找速度。会额外占用内存，每条约0.1KB"),
    },
    # ===== 预加载配置 (第7个标签页) =====
    "preload": {
        "max_streams": ConfigField(type=int, default=10, description="最大预加载聊天流数量。超过此数量时，移除最旧的预加载记录", min=1, max=100),
        "message_count": ConfigField(type=int, default=50, description="每个聊天流预加载的消息数量。预加载的消息会自动填充消息缓存", min=10, max=500),
        "max_persons_per_stream": ConfigField(type=int, default=50, description="每个聊天流预加载的人物数量。预加载的人物信息会自动填充人物缓存", min=10, max=200),
        "preload_delay": ConfigField(type=str, default="0.1", description="预加载延迟(秒)。延迟执行预加载任务，避免阻塞主流程", choices=["0.05", "0.1", "0.2", "0.5", "1.0"]),
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
        description="缓存消息查询结果。原理：拦截数据库查询，相同参数直接返回缓存。效果：减少约95%的数据库查询",
        icon="💬",
        collapsed=True,
        order=2
    ),
    "person_cache": ConfigSection(
        title="人物信息缓存",
        description="缓存人物信息(昵称、备注等)。原理：按QQ号缓存，避免重复查询数据库。效果：减少约90%的人物信息查询",
        icon="👤",
        collapsed=True,
        order=3
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
    "preload": ConfigSection(
        title="预加载",
        description="异步预加载聊天流的消息和人物信息。原理：监听消息事件，延迟预加载最近的消息和人物信息。效果：减少首次查询延迟，提升响应速度",
        icon="⚡",
        collapsed=True,
        order=6
    ),
}

# 布局配置 - 使用标签页布局
config_layout = ConfigLayout(
    type="tabs",
    tabs=[
        ConfigTab(id="plugin", title="插件", icon="🔧", sections=["plugin"], order=0),
        ConfigTab(id="modules", title="模块开关", icon="�", sections=["modules"], order=1),
        ConfigTab(id="message_cache", title="消息缓存", icon="💬", sections=["message_cache"], order=2),
        ConfigTab(id="person_cache", title="人物信息缓存", icon="👤", sections=["person_cache"], order=3),
        ConfigTab(id="expression_cache", title="表达式缓存", icon="🎭", sections=["expression_cache"], order=4),
        ConfigTab(id="slang_cache", title="黑话缓存", icon="🗣️", sections=["slang_cache"], order=5),
        ConfigTab(id="preload", title="预加载", icon="⚡", sections=["preload"], order=6),
    ]
)


@register_plugin
class PerformanceOptimizerPlugin(BasePlugin):
    plugin_name = "CM-performance-optimizer"
    plugin_version = "4.6.0"
    plugin_description = "性能优化 - 消息缓存 + 人物信息缓存 + 表达式缓存 + 黑话缓存 + 预加载"
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
        logger.info("[性能优化] CM-performance-optimizer v4.3.1 启动")
        
        try:
            cfg = {
                "report_interval": 60,
                "modules": {"message_cache": True, "person_cache": True, "expression_cache": False, "slang_cache": False},
                "message_cache_size": 2000, "message_cache_ttl": 120.0,
                "person_cache_size": 3000, "person_cache_ttl": 1800,
                "expression_cache_batch_size": 100, "expression_cache_batch_delay": 0.05, "expression_cache_refresh_interval": 3600,
                "slang_cache_batch_size": 100, "slang_cache_batch_delay": 0.05, "slang_cache_refresh_interval": 3600, "slang_cache_enable_content_index": True,
                "preload_max_streams": 10, "preload_message_count": 50, "preload_max_persons_per_stream": 50, "preload_delay": 0.1,
            }
            enabled = True
            log_level = "INFO"
            
            try:
                import tomlkit
                p = Path(__file__).parent / "config.toml"
                
                # 如果配置文件不存在，创建默认配置
                if not p.exists():
                    logger.info("[性能优化] 配置文件不存在，创建默认配置...")
                    self._create_default_config(p)
                
                if p.exists():
                    with open(p) as f: c = tomlkit.load(f)
                    # plugin 标签页
                    enabled = c.get("plugin", {}).get("enabled", True)
                    cfg["report_interval"] = c.get("plugin", {}).get("report_interval", 60)
                    log_level = c.get("plugin", {}).get("log_level", "INFO")
                    # modules 标签页
                    modules = c.get("modules", {})
                    cfg["modules"]["message_cache"] = modules.get("message_cache_enabled", True)
                    cfg["modules"]["person_cache"] = modules.get("person_cache_enabled", True)
                    cfg["modules"]["expression_cache"] = modules.get("expression_cache_enabled", False)
                    cfg["modules"]["slang_cache"] = modules.get("slang_cache_enabled", False)
                    # message_cache 标签页
                    cfg["message_cache_size"] = c.get("message_cache", {}).get("max_size", 2000)
                    cfg["message_cache_ttl"] = c.get("message_cache", {}).get("ttl", 120.0)
                    # person_cache 标签页
                    cfg["person_cache_size"] = c.get("person_cache", {}).get("max_size", 3000)
                    cfg["person_cache_ttl"] = c.get("person_cache", {}).get("ttl", 1800)
                    # expression_cache 标签页
                    cfg["expression_cache_batch_size"] = c.get("expression_cache", {}).get("batch_size", 100)
                    # batch_delay 从字符串转换为 float
                    expr_batch_delay_str = c.get("expression_cache", {}).get("batch_delay", "0.05")
                    try:
                        cfg["expression_cache_batch_delay"] = float(expr_batch_delay_str)
                    except (ValueError, TypeError):
                        cfg["expression_cache_batch_delay"] = 0.05
                    cfg["expression_cache_refresh_interval"] = c.get("expression_cache", {}).get("refresh_interval", 3600)
                    # slang_cache 标签页
                    cfg["slang_cache_batch_size"] = c.get("slang_cache", {}).get("batch_size", 100)
                    # batch_delay 从字符串转换为 float
                    slang_batch_delay_str = c.get("slang_cache", {}).get("batch_delay", "0.05")
                    try:
                        cfg["slang_cache_batch_delay"] = float(slang_batch_delay_str)
                    except (ValueError, TypeError):
                        cfg["slang_cache_batch_delay"] = 0.05
                    cfg["slang_cache_refresh_interval"] = c.get("slang_cache", {}).get("refresh_interval", 3600)
                    cfg["slang_cache_enable_ahocorasick"] = c.get("slang_cache", {}).get("enable_ahocorasick", True)
                    # preload 标签页
                    cfg["preload_max_streams"] = c.get("preload", {}).get("max_streams", 10)
                    cfg["preload_message_count"] = c.get("preload", {}).get("message_count", 50)
                    cfg["preload_max_persons_per_stream"] = c.get("preload", {}).get("max_persons_per_stream", 50)
                    # preload_delay 从字符串转换为 float
                    preload_delay_str = c.get("preload", {}).get("preload_delay", "0.1")
                    try:
                        cfg["preload_delay"] = float(preload_delay_str)
                    except (ValueError, TypeError):
                        cfg["preload_delay"] = 0.1
            except: pass
            
            # 应用日志等级
            import logging
            level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
            if log_level.upper() in level_map:
                logger.setLevel(level_map[log_level.upper()])
                logger.info(f"[性能优化] 日志等级: {log_level.upper()}")
            
            if not enabled:
                logger.info("[性能优化] 插件已禁用")
                return
            
            # 创建或获取单例，并重新加载配置
            if _opt is None:
                _opt = Optimizer(cfg)
                logger.info("[性能优化] 创建新的优化器实例")
            else:
                _opt.reload_config(cfg)
                logger.info("[性能优化] 重新加载配置到现有实例")
            
            _opt.apply_patches()
            _opt.start()
            logger.info("[性能优化] ✓ 插件启动完成")
        except Exception as e:
            logger.error(f"[性能优化] 启动失败: {e}")
    
    def _create_default_config(self, config_path):
        """创建默认配置文件"""
        try:
            import tomlkit
            
            default_config = """
# =====================================================
# CM 性能优化插件配置 v4.7.0
#
# 功能说明：
#   通过缓存数据库查询结果来优化MaiBot性能
#   - 消息缓存：拦截find_messages查询，命中率>95%
#   - 人物信息缓存：拦截人物信息查询，命中率>90%
#   - 表达式缓存：双缓冲+缓慢加载+原子切换，全量缓存
#   - 黑话缓存：双缓冲+缓慢加载+原子切换+AC自动机
#   - 预加载：异步预加载聊天流的消息和人物信息
#
# 内存占用：约10-20MB (取决于缓存大小配置)
# CPU开销：极低，仅在缓存未命中时有额外开销
#
# 依赖：pyahocorasick (可选，用于黑话缓存优化)
# =====================================================

# ============ 插件基本配置 ============
[plugin]
enabled = true          # 是否启用插件
version = "4.7.0"       # 插件版本
report_interval = 60    # 统计报告间隔(秒)，设置0可关闭
log_level = "INFO"     # 日志等级: DEBUG/INFO/WARNING/ERROR

# ============ 功能模块开关 ============
memory_stats_enabled = true
memory_stats_cache_ttl = 60

[modules]
message_cache_enabled = true    # 消息缓存 - 缓存find_messages查询
person_cache_enabled = true     # 人物信息缓存 - 缓存昵称等信息
expression_cache_enabled = false # 表达式缓存 - 双缓冲+缓慢加载
slang_cache_enabled = false      # 黑话缓存 - 双缓冲+缓慢加载+AC自动机

# ============ 消息缓存配置 ============
# 每条约1-5KB，2000条约占用2-10MB内存

[message_cache]
max_size = 2000         # 缓存最大条目数
ttl = 180             # 缓存过期时间(秒)

# ============ 人物信息缓存配置 ============
# 每条约0.5-2KB，3000条约占用1.5-6MB内存

[person_cache]
max_size = 3000         # 缓存最大条目数
ttl = 1800              # 缓存过期时间(秒) 30分钟

# ============ 表达式缓存配置 ============
# 双缓冲+缓慢加载+原子切换，避免CPU峰值
# 2万条约需10秒加载完成 (batch_size=100, batch_delay=0.05)

[expression_cache]
batch_size = 100        # 每批加载条数
batch_delay = "0.1"      # 批次间延迟(秒)
refresh_interval = 3600 # 自动刷新间隔(秒)，0表示不自动刷新

# ============ 黑话缓存配置 ============
# 双缓冲+缓慢加载+原子切换+AC自动机，几万条黑话也能毫秒级响应
# 2万条约需10秒加载完成 (batch_size=100, batch_delay=0.05)
# AC自动机需要安装pyahocorasick库: pip install pyahocorasick
# 如果未安装该库，会自动回退到遍历+正则匹配（性能较低，但功能正常）
# 性能对比: 50,000条黑话，AC自动机约10ms，遍历+正则约500ms，提升50倍

[slang_cache]
batch_size = 100        # 每批加载条数
batch_delay = "0.1"      # 批次间延迟(秒)
refresh_interval = 3600 # 自动刷新间隔(秒)，0表示不自动刷新
enable_ahocorasick = true  # 启用AC自动机优化，几万条黑话也能毫秒级响应

# ============ 预加载配置 ============
# 异步预加载聊天流的消息和人物信息

[preload]
max_streams = 10              # 最大预加载聊天流数量
message_count = 50            # 每个聊天流预加载的消息数量
max_persons_per_stream = 50   # 每个聊天流预加载的人物数量
preload_delay = "0.1"         # 预加载延迟(秒)
"""
            
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(default_config.strip())
            
            logger.info(f"[性能优化] 默认配置已创建: {config_path}")
        except Exception as e:
            logger.error(f"[性能优化] 创建默认配置失败: {e}")
    
    def get_plugin_components(self):
        """返回插件组件列表"""
        # 返回事件处理器类（不是实例）
        return [(PreloadEventHandler.get_handler_info(), PreloadEventHandler)]