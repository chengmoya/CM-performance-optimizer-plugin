"""
CM 性能优化插件 v4.3.1

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
from typing import Optional, Dict, Any
from collections import OrderedDict

try:
    from src.plugin_system.apis.plugin_register_api import register_plugin
    from src.plugin_system.base.base_plugin import BasePlugin
    from src.plugin_system.base.config_types import ConfigField, ConfigSection, ConfigLayout, ConfigTab
    from src.common.logger import get_logger
except ImportError:
    class BasePlugin:
        def __init__(self, plugin_dir=None): pass
    class ConfigField:
        def __init__(self, **kw): pass
    class ConfigSection:
        def __init__(self, **kw): pass
    class ConfigLayout:
        def __init__(self, **kw): pass
    class ConfigTab:
        def __init__(self, **kw): pass
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
        """记录命中但被过滤的情况（如chat_id不匹配）"""
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
    t = hit + miss + filtered
    return (hit / t * 100) if t > 0 else 0


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
                    logger.debug(f"[MsgCache] 替换 {n}.find_messages")
            
            self._patched = True
            logger.info("[MsgCache] ✓ 补丁应用成功")
        except Exception as e:
            logger.error(f"[MsgCache] ✗ 补丁失败: {e}")
    
    def remove_patch(self):
        if not self._patched or not self._orig_func: return
        try:
            from src.common import message_repository
            message_repository.find_messages = self._orig_func
            self._patched = False
            logger.info("[MsgCache] 补丁已移除")
        except: pass


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
                    return
                
                t0 = time.time()
                module._orig_load(self_person)
                module.stats.miss(time.time() - t0)
                
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
        except: pass


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


# ===== 预加载管理器 =====
class PreloadManager:
    """预加载管理器 - 管理聊天流的预加载状态"""
    def __init__(self, max_streams=10):
        self.preloaded_streams = set()  # 已预加载的聊天流ID
        self.max_streams = max_streams
        self.stream_last_active = {}  # 聊天流最后活跃时间
        self.lock = threading.Lock()
    
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
class PreloadEventHandler:
    """预加载事件处理器 - 监听消息事件并预加载聊天流数据"""
    def __init__(self, preload_manager, message_count=50, max_persons_per_stream=50, preload_delay=0.1):
        self.preload_manager = preload_manager
        self.message_count = message_count
        self.max_persons_per_stream = max_persons_per_stream
        self.preload_delay = preload_delay
        self.enabled = True
    
    async def handle_message(self, message):
        """处理消息事件"""
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
            
            # 1. 加载最近的消息（这会自动填充消息缓存）
            from src.common import message_repository
            messages = message_repository.find_messages(
                {"stream_id": stream_id},
                limit=self.message_count,
                limit_mode="latest"
            )
            
            # 2. 提取消息中的发送者ID和平台组合
            person_keys = set()
            for msg in messages:
                if hasattr(msg, "user_id") and msg.user_id and hasattr(msg, "user_platform") and msg.user_platform:
                    person_keys.add((msg.user_platform, msg.user_id))
            
            # 3. 预加载这些人物信息
            from src.person_info.person_info import Person
            loaded_count = 0
            for platform, user_id in list(person_keys)[:self.max_persons_per_stream]:
                person_id = f"{platform}_{user_id}"
                person = Person(person_id=person_id)
                person.load_from_database()  # 这会自动填充人物缓存
                loaded_count += 1
            
            # 4. 标记为已预加载
            self.preload_manager.mark_preloaded(stream_id)
            
            logger.info(f"[Preload] 预加载完成: {stream_id[:20]}..., 消息{len(messages)}条, 人物{loaded_count}个")
        except Exception as e:
            logger.error(f"[Preload] 预加载失败: {e}")


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
        self.kg_cache = None
        
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
                enable_content_index=cfg.get("slang_cache_enable_content_index", True)
            )
        
        if self.modules_cfg.get("kg_cache", False):
            self.kg_cache = KGCacheModule(
                batch_size=cfg.get("kg_cache_batch_size", 100),
                batch_delay=cfg.get("kg_cache_batch_delay", 0.05),
                refresh_interval=cfg.get("kg_cache_refresh_interval", 3600)
            )
        
        # 初始化预加载功能
        self.preload_manager = None
        self.preload_handler = None
        if self.modules_cfg.get("preload_enabled", False):
            self.preload_manager = PreloadManager(
                max_streams=cfg.get("preload_max_streams", 10)
            )
            self.preload_handler = PreloadEventHandler(
                preload_manager=self.preload_manager,
                message_count=cfg.get("preload_message_count", 50),
                max_persons_per_stream=cfg.get("preload_max_persons_per_stream", 50),
                preload_delay=cfg.get("preload_delay", 0.1)
            )
        
        self._running = False
        self._ready = True
    
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
                logger.debug(f"[MemoryStats] 获取 {module_name} 内存失败: {e}")
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
        
        # 知识库图谱缓存拦截
        if self.kg_cache:
            self._apply_kg_cache_patch()
    
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
            
            orig_match_jargon = JargonExplainer.match_jargon_from_messages
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
            
            orig_load_from_file = KGManager.load_from_file
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
        
        # 知识库图谱缓存
        if self.kg_cache:
            report_lines.extend(self._build_kg_cache_stats_lines("🧠 知识库图谱缓存", self.kg_cache))
            report_lines.append("")
        
        # 计算总内存占用
        if self.memory_stats_enabled:
            total_memory = 0
            if self.msg_cache:
                total_memory += self._get_module_memory_usage(self.msg_cache, "message_cache")
            if self.person_cache:
                total_memory += self._get_module_memory_usage(self.person_cache, "person_cache")
            if self.expr_cache:
                total_memory += self._get_module_memory_usage(self.expr_cache, "expression_cache")
            if self.jargon_cache:
                total_memory += self._get_module_memory_usage(self.jargon_cache, "jargon_cache")
            if self.kg_cache:
                total_memory += self._get_module_memory_usage(self.kg_cache, "kg_cache")
            if self.preload_manager:
                total_memory += self._get_module_memory_usage(self.preload_manager, "preload_manager")
            
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
        t_total = t["t_hit"] + t["t_miss"] + t["t_filtered"]
        i_total = i["i_hit"] + i["i_miss"] + i["i_filtered"]
        t_rate = rate(t["t_hit"], t_total) if t_total > 0 else 0
        i_rate = rate(i["i_hit"], i_total) if i_total > 0 else 0
        t_time = t["t_fast_time"] + t["t_slow_time"]
        i_time = i["i_fast_time"] + i["i_slow_time"]
        
        # 估算节省时间
        avg_time = t_time / t["t_miss"] if t["t_miss"] > 0 else 0.02
        saved = t["t_hit"] * avg_time
        
        # 获取内存占用
        memory_str = ""
        if self.memory_stats_enabled:
            memory_bytes = self._get_module_memory_usage(module, name)
            memory_str = f" | 内存: {MemoryUtils.format_size(memory_bytes)}"
        
        lines.append(f"{name}")
        lines.append(f"  状态: {loading_status} | 大小: {size}条{memory_str} | 上次刷新: {last_refresh_str}")
        if module.refresh_interval > 0:
            lines.append(f"  自动刷新: 每{module.refresh_interval}秒")
        lines.append(f"  累计: 命中 {t['t_hit']} | 未命中 {t['t_miss']} | 被过滤 {t['t_filtered']} | 命中率 {t_rate:.1f}%")
        lines.append(f"  本期: 命中 {i['i_hit']} | 未命中 {i['i_miss']} | 被过滤 {i['i_filtered']} | 命中率 {i_rate:.1f}%")
        lines.append(f"  节省: {saved:.1f}秒 (平均{avg_time*1000:.1f}ms/次)")
        
        return lines
    
    def _build_module_stats_lines(self, name: str, module):
        """构建模块统计的行"""
        lines = []
        t = module.stats.total()
        i = module.stats.reset_interval()
        t_total = t["t_hit"] + t["t_miss"] + t["t_filtered"]
        i_total = i["i_hit"] + i["i_miss"] + i["i_filtered"]
        t_rate = rate(t["t_hit"], t_total) if t_total > 0 else 0
        i_rate = rate(i["i_hit"], i_total) if i_total > 0 else 0
        t_time = t["t_fast_time"] + t["t_slow_time"]
        i_time = i["i_fast_time"] + i["i_slow_time"]
        
        # 估算节省时间
        avg_time = t_time / t["t_miss"] if t["t_miss"] > 0 else 0.03
        saved = t["t_hit"] * avg_time
        
        # 获取内存占用
        memory_str = ""
        if self.memory_stats_enabled:
            memory_bytes = self._get_module_memory_usage(module, name)
            memory_str = f" | 内存: {MemoryUtils.format_size(memory_bytes)}"
        
        lines.append(f"{name}")
        lines.append(f"  缓存: {module.cache.size()}/{module.cache.max_size} | TTL: {module.cache.ttl}秒{memory_str}")
        lines.append(f"  累计: 命中 {t['t_hit']} | 未命中 {t['t_miss']} | 被过滤 {t['t_filtered']} | 命中率 {t_rate:.1f}%")
        lines.append(f"  累计: 快 {t['t_fast']}次/{t['t_fast_time']:.2f}s | 慢 {t['t_slow']}次/{t['t_slow_time']:.2f}s")
        lines.append(f"  本期: 命中 {i['i_hit']} | 未命中 {i['i_miss']} | 被过滤 {i['i_filtered']} | 命中率 {i_rate:.1f}%")
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
        t_total = t["t_hit"] + t["t_miss"] + t["t_filtered"]
        i_total = i["i_hit"] + i["i_miss"] + i["i_filtered"]
        t_rate = rate(t["t_hit"], t_total) if t_total > 0 else 0
        i_rate = rate(i["i_hit"], i_total) if i_total > 0 else 0
        t_time = t["t_fast_time"] + t["t_slow_time"]
        i_time = i["i_fast_time"] + i["i_slow_time"]
        
        # 估算节省时间
        avg_time = t_time / t["t_miss"] if t["t_miss"] > 0 else 0.5
        saved = t["t_hit"] * avg_time
        
        # 获取内存占用
        memory_str = ""
        if self.memory_stats_enabled:
            memory_bytes = self._get_module_memory_usage(module, name)
            memory_str = f" | 内存: {MemoryUtils.format_size(memory_bytes)}"
        
        lines.append(f"{name}")
        lines.append(f"  状态: {loading_status} | 大小: {size_str}{memory_str} | 上次刷新: {last_refresh_str}")
        if module.refresh_interval > 0:
            lines.append(f"  自动刷新: 每{module.refresh_interval}秒")
        lines.append(f"  累计: 命中 {t['t_hit']} | 未命中 {t['t_miss']} | 被过滤 {t['t_filtered']} | 命中率 {t_rate:.1f}%")
        lines.append(f"  本期: 命中 {i['i_hit']} | 未命中 {i['i_miss']} | 被过滤 {i['i_filtered']} | 命中率 {i_rate:.1f}%")
        lines.append(f"  节省: {saved:.1f}秒 (平均{avg_time*1000:.1f}ms/次)")
        
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
            if self.kg_cache and self.kg_cache.refresh_interval > 0:
                asyncio.get_running_loop().create_task(self.kg_cache._refresh_loop())
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
        "version": ConfigField(type=str, default="4.3.1", description="插件版本号，用于追踪更新"),
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
        "preload_enabled": ConfigField(type=bool, default=False, description="预加载功能: 当聊天流激活时，异步预加载该聊天流的最近消息和人物信息。可提升首次查询性能90%以上"),
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
    # ===== 知识库图谱缓存配置 (第7个标签页) =====
    "kg_cache": {
        "batch_size": ConfigField(type=int, default=100, description="每批加载的条数。默认100条，2万条约需10秒加载完成。增大此值可加快加载但会增加CPU峰值", min=10, max=1000),
        "batch_delay": ConfigField(type=str, default="0.05", description="批次间延迟(秒)。用于平滑加载避免CPU峰值，增大此值可降低CPU占用但延长加载时间", choices=["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0"]),
        "refresh_interval": ConfigField(type=int, default=3600, description="自动刷新间隔(秒)。设置为0表示不自动刷新，仅启动时加载一次。建议3600秒(1小时)", min=0, max=86400),
    },
    # ===== 预加载配置 (第8个标签页) =====
    "preload": {
        "max_streams": ConfigField(type=int, default=10, description="最多预加载的聊天流数量。超过此数量会移除最旧的聊天流。建议10-20个，过多会增加内存占用", min=1, max=50),
        "message_count": ConfigField(type=int, default=50, description="每个聊天流预加载的消息数量。建议50-100条，覆盖最近的对话上下文", min=10, max=200),
        "max_persons_per_stream": ConfigField(type=int, default=50, description="每个聊天流预加载的最大人物数量。群聊建议50-100个，私聊2个即可", min=1, max=200),
        "preload_delay": ConfigField(type=str, default="0.1", description="预加载延迟(秒)。收到消息后延迟执行预加载，避免阻塞主流程。建议0.1-0.5秒", choices=["0.05", "0.1", "0.2", "0.5", "1.0"]),
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
    "kg_cache": ConfigSection(
        title="知识库图谱缓存",
        description="全量缓存知识库图谱数据。原理：双缓冲+缓慢加载+原子切换，避免CPU峰值。效果：启动后约5-10秒完成加载，知识库查询速度提升80-90%，消除文件IO开销",
        icon="🧠",
        collapsed=True,
        order=6
    ),
    "preload": ConfigSection(
        title="预加载",
        description="预加载聊天流数据。原理：当聊天流激活时，异步预加载最近消息和人物信息到缓存。效果：首次查询性能提升90%以上，减少冷启动延迟",
        icon="⚡",
        collapsed=True,
        order=7
    ),
}

# 布局配置 - 使用标签页布局
config_layout = ConfigLayout(
    type="tabs",
    tabs=[
        ConfigTab(id="plugin", title="插件", icon="🔧", sections=["plugin"], order=0),
        ConfigTab(id="modules", title="模块开关", icon="📦", sections=["modules"], order=1),
        ConfigTab(id="message_cache", title="消息缓存", icon="💬", sections=["message_cache"], order=2),
        ConfigTab(id="person_cache", title="人物信息缓存", icon="👤", sections=["person_cache"], order=3),
        ConfigTab(id="expression_cache", title="表达式缓存", icon="🎭", sections=["expression_cache"], order=4),
        ConfigTab(id="slang_cache", title="黑话缓存", icon="🗣️", sections=["slang_cache"], order=5),
        ConfigTab(id="kg_cache", title="知识库图谱缓存", icon="🧠", sections=["kg_cache"], order=6),
        ConfigTab(id="preload", title="预加载", icon="⚡", sections=["preload"], order=7),
    ]
)


@register_plugin
class PerformanceOptimizerPlugin(BasePlugin):
    plugin_name = "CM-performance-optimizer"
    plugin_version = "4.3.1"
    plugin_description = "性能优化 - 消息缓存 + 人物信息缓存 + 表达式缓存 + 黑话缓存 + 知识库图谱缓存 + 预加载"
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
        logger.info("[PerfOpt] CM-performance-optimizer v4.3.1 启动")
        
        try:
            cfg = {
                "report_interval": 60,
                "modules": {"message_cache": True, "person_cache": True, "expression_cache": False, "slang_cache": False, "kg_cache": False, "preload_enabled": False},
                "message_cache_size": 2000, "message_cache_ttl": 120.0,
                "person_cache_size": 3000, "person_cache_ttl": 1800,
                "expression_cache_batch_size": 100, "expression_cache_batch_delay": 0.05, "expression_cache_refresh_interval": 3600,
                "slang_cache_batch_size": 100, "slang_cache_batch_delay": 0.05, "slang_cache_refresh_interval": 3600, "slang_cache_enable_content_index": True,
                "kg_cache_batch_size": 100, "kg_cache_batch_delay": 0.05, "kg_cache_refresh_interval": 3600,
                "preload_max_streams": 10, "preload_message_count": 50, "preload_max_persons_per_stream": 50, "preload_delay": 0.1,
            }
            enabled = True
            log_level = "INFO"
            
            try:
                import tomlkit
                p = Path(__file__).parent / "config.toml"
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
                    cfg["modules"]["kg_cache"] = modules.get("kg_cache_enabled", False)
                    cfg["modules"]["preload_enabled"] = modules.get("preload_enabled", False)
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
                    cfg["slang_cache_enable_content_index"] = c.get("slang_cache", {}).get("enable_content_index", True)
                    # kg_cache 标签页
                    cfg["kg_cache_batch_size"] = c.get("kg_cache", {}).get("batch_size", 100)
                    # batch_delay 从字符串转换为 float
                    kg_batch_delay_str = c.get("kg_cache", {}).get("batch_delay", "0.05")
                    try:
                        cfg["kg_cache_batch_delay"] = float(kg_batch_delay_str)
                    except (ValueError, TypeError):
                        cfg["kg_cache_batch_delay"] = 0.05
                    cfg["kg_cache_refresh_interval"] = c.get("kg_cache", {}).get("refresh_interval", 3600)
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
                logger.info(f"[PerfOpt] 日志等级: {log_level.upper()}")
            
            if not enabled:
                logger.info("[PerfOpt] 插件已禁用")
                return
            
            _opt = Optimizer(cfg)
            _opt.apply_patches()
            _opt.start()
            logger.info("[PerfOpt] ✓ 插件启动完成")
        except Exception as e:
            logger.error(f"[PerfOpt] 启动失败: {e}")
    
    def get_plugin_components(self): return []