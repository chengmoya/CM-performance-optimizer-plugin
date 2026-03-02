"""
核心缓存模块 - TTLCache 和 MemoryUtils
"""

import sys
import asyncio
import time
import threading
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple

# ========== 常量定义 ==========
# 内存单位转换常量
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024


class MemoryUtils:
    """内存测量工具类 - 递归计算对象的内存占用"""
    
    @staticmethod
    def get_size(obj: Any, seen: Optional[set] = None) -> int:
        """递归计算对象的内存占用（字节）。
        
        Args:
            obj: 要计算内存占用的对象
            seen: 已访问对象的 ID 集合，用于防止循环引用
            
        Returns:
            对象的内存占用字节数
        """
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
    def format_size(bytes_size: int) -> str:
        """将字节转换为易读的格式。
        
        Args:
            bytes_size: 字节数
            
        Returns:
            易读的内存大小字符串（如 "1.50 KB"）
        """
        if bytes_size < BYTES_PER_KB:
            return f"{bytes_size:.2f} B"
        elif bytes_size < BYTES_PER_MB:
            return f"{bytes_size / BYTES_PER_KB:.2f} KB"
        else:
            return f"{bytes_size / BYTES_PER_MB:.2f} MB"


class TTLCache:
    """带TTL的LRU缓存（支持同步和异步访问）
    
    使用 threading.RLock 作为底层锁，确保线程安全。
    同步方法直接使用锁，异步方法通过 asyncio.to_thread 调用同步方法。
    """
    
    def __init__(self, max_size: int = 500, ttl: float = 120.0):
        self.max_size = max_size
        self.ttl = ttl
        self.data: OrderedDict = OrderedDict()
        self.ts: Dict[str, float] = {}
        self._lock = threading.RLock()
        # 保留异步锁用于向后兼容（某些代码可能直接访问 self.lock）
        self.lock = asyncio.Lock()
    
    def _purge_expired_locked(self, now: float) -> int:
        """清理已过期的 key（需在 _lock 内调用）。返回清理数量。"""
        removed = 0
        for kk in list(self.data.keys()):
            ts = self.ts.get(kk)
            if ts is None or (now - ts > self.ttl):
                self.data.pop(kk, None)
                self.ts.pop(kk, None)
                removed += 1
        return removed
    
    # ========== 同步方法 ==========
    
    def get_sync(self, k: str) -> Tuple[Any, bool]:
        """同步获取缓存值
        
        Args:
            k: 缓存键
            
        Returns:
            (value, hit): 值和是否命中的元组
        """
        with self._lock:
            if k not in self.data:
                return None, False
            if time.time() - self.ts[k] > self.ttl:
                del self.data[k], self.ts[k]
                return None, False
            self.data.move_to_end(k)
            return self.data[k], True
    
    def set_sync(self, k: str, v: Any) -> None:
        """同步设置缓存值
        
        Args:
            k: 缓存键
            v: 缓存值
        """
        with self._lock:
            now = time.time()
            self._purge_expired_locked(now)
            
            if k in self.data:
                self.data.move_to_end(k)
            
            self.data[k] = v
            self.ts[k] = now
            
            while len(self.data) > self.max_size:
                old = next(iter(self.data))
                del self.data[old], self.ts[old]
    
    def invalidate_sync(self, k: str) -> None:
        """同步使缓存失效
        
        Args:
            k: 缓存键
        """
        with self._lock:
            if k in self.data:
                del self.data[k], self.ts[k]
    
    def clear_sync(self) -> None:
        """同步清空缓存"""
        with self._lock:
            self.data.clear()
            self.ts.clear()
    
    def get_memory_usage_sync(self) -> int:
        """同步获取缓存内存使用量（字节）"""
        with self._lock:
            return MemoryUtils.get_size(self.data) + MemoryUtils.get_size(self.ts)
    
    # ========== 异步方法（通过 to_thread 调用同步方法）==========
    
    async def get(self, k: str) -> Tuple[Any, bool]:
        """异步获取缓存值
        
        Args:
            k: 缓存键
            
        Returns:
            (value, hit): 值和是否命中的元组
        """
        return await asyncio.to_thread(self.get_sync, k)
    
    async def set(self, k: str, v: Any) -> None:
        """异步设置缓存值
        
        Args:
            k: 缓存键
            v: 缓存值
        """
        await asyncio.to_thread(self.set_sync, k, v)
    
    async def invalidate(self, k: str) -> None:
        """异步使缓存失效
        
        Args:
            k: 缓存键
        """
        await asyncio.to_thread(self.invalidate_sync, k)
    
    async def clear(self) -> None:
        """异步清空缓存"""
        await asyncio.to_thread(self.clear_sync)
    
    def size(self) -> int:
        """获取缓存大小（线程安全）"""
        with self._lock:
            return len(self.data)
    
    async def get_memory_usage(self) -> int:
        """异步获取缓存内存使用量（字节）"""
        return await asyncio.to_thread(self.get_memory_usage_sync)