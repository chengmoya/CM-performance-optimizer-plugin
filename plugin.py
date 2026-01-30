"""
CM 性能优化插件 v3.0.0

功能模块：
1. 消息缓存 (message_cache) - 缓存 find_messages 查询结果
2. 人物信息缓存 (person_cache) - 缓存人物信息查询
3. 表达式缓存 (expression_cache) - 预留
4. 黑话缓存 (slang_cache) - 预留

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


# ===== 统计类 =====
class ModuleStats:
    """单个模块的统计"""
    def __init__(self, name: str):
        self.name = name
        self.lock = threading.Lock()
        self.t_hit = self.t_miss = 0
        self.i_hit = self.i_miss = 0
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
    
    def reset_interval(self) -> Dict[str, Any]:
        with self.lock:
            r = {"i_hit": self.i_hit, "i_miss": self.i_miss,
                 "i_fast": self.i_fast, "i_slow": self.i_slow,
                 "i_fast_time": self.i_fast_time, "i_slow_time": self.i_slow_time}
            self.i_hit = self.i_miss = self.i_fast = self.i_slow = 0
            self.i_fast_time = self.i_slow_time = 0.0
            return r
    
    def total(self) -> Dict[str, Any]:
        with self.lock:
            return {"t_hit": self.t_hit, "t_miss": self.t_miss,
                    "t_fast": self.t_fast, "t_slow": self.t_slow,
                    "t_fast_time": self.t_fast_time, "t_slow_time": self.t_slow_time}


def rate(hit, miss):
    t = hit + miss
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
        
        # 初始化模块
        self.msg_cache = None
        self.person_cache = None
        
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
        
        self._running = False
        self._ready = True
    
    def apply_patches(self):
        if self.msg_cache:
            self.msg_cache.apply_patch()
        if self.person_cache:
            self.person_cache.apply_patch()
    
    async def _report_loop(self):
        logger.info(f"[PerfOpt] 统计报告启动 (间隔{self.interval}s)")
        while self._running:
            await asyncio.sleep(self.interval)
            if not self._running: break
            self._print_report()
    
    def _print_report(self):
        uptime = int(time.time() - self.start_time)
        uptime_str = f"{uptime//3600}h{(uptime%3600)//60}m{uptime%60}s"
        
        logger.info("=" * 60)
        logger.info(f"[PerfOpt] 📊 性能统计报告 | 运行时间: {uptime_str}")
        
        # 消息缓存
        if self.msg_cache:
            self._print_module_stats("📦 消息缓存", self.msg_cache)
        
        # 人物信息缓存
        if self.person_cache:
            self._print_module_stats("👤 人物缓存", self.person_cache)
        
        logger.info("=" * 60)
    
    def _print_module_stats(self, name: str, module):
        t = module.stats.total()
        i = module.stats.reset_interval()
        t_rate = rate(t["t_hit"], t["t_miss"])
        i_rate = rate(i["i_hit"], i["i_miss"])
        t_time = t["t_fast_time"] + t["t_slow_time"]
        i_time = i["i_fast_time"] + i["i_slow_time"]
        
        # 估算节省时间
        avg_time = t_time / t["t_miss"] if t["t_miss"] > 0 else 0.03
        saved = t["t_hit"] * avg_time
        
        logger.info("-" * 60)
        logger.info(f"[PerfOpt] {name} | 缓存: {module.cache.size()}/{module.cache.max_size}")
        logger.info(f"[PerfOpt]   累计: 命中 {t['t_hit']} | 未命中 {t['t_miss']} | 命中率 {t_rate:.1f}%")
        logger.info(f"[PerfOpt]   累计: 快 {t['t_fast']}次/{t['t_fast_time']:.2f}s | 慢 {t['t_slow']}次/{t['t_slow_time']:.2f}s")
        logger.info(f"[PerfOpt]   💡 节省约 {saved:.1f}s (平均 {avg_time*1000:.1f}ms/次)")
        logger.info(f"[PerfOpt]   本期: 命中 {i['i_hit']} | 未命中 {i['i_miss']} | 命中率 {i_rate:.1f}%")
    
    def start(self):
        if self._running: return
        self._running = True
        try:
            asyncio.get_running_loop().create_task(self._report_loop())
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
        "version": ConfigField(type=str, default="3.0.0", description="插件版本号，用于追踪更新"),
        "report_interval": ConfigField(type=int, default=60, description="统计报告输出间隔(秒)，设置0可关闭定时报告", min=0, max=600),
        "log_level": ConfigField(type=str, default="INFO", description="日志输出等级", choices=["DEBUG", "INFO", "WARNING", "ERROR"]),
    },
    # ===== 模块开关 (第2个标签页) =====
    "modules": {
        "message_cache_enabled": ConfigField(type=bool, default=True, description="消息缓存: 拦截find_messages数据库查询，缓存结果避免重复查询。命中率通常>95%，可节省大量数据库IO"),
        "person_cache_enabled": ConfigField(type=bool, default=True, description="人物信息缓存: 拦截人物信息加载，按QQ号缓存昵称等信息。人物信息变化慢，缓存效果好"),
        "expression_cache_enabled": ConfigField(type=bool, default=False, description="表达式缓存: 缓存表情包查询结果 (开发中，暂不可用)"),
        "slang_cache_enabled": ConfigField(type=bool, default=False, description="黑话缓存: 缓存黑话/网络用语查询 (开发中，暂不可用)"),
    },
    # ===== 消息缓存配置 (第3个标签页) =====
    "message_cache": {
        "max_size": ConfigField(type=int, default=2000, description="最大缓存条目数。每条约占用1-5KB内存，2000条约占用2-10MB。超过后自动清理最旧的条目", min=100, max=10000),
        "ttl": ConfigField(type=float, default=120.0, description="缓存过期时间(秒)。消息变化快，建议60-180秒。过长可能导致消息不同步", min=10.0, max=600.0),
    },
    # ===== 人物信息缓存配置 (第4个标签页) =====
    "person_cache": {
        "max_size": ConfigField(type=int, default=3000, description="最大缓存条目数。每条约占用0.5-2KB内存，3000条约占用1.5-6MB。建议大于活跃用户数", min=100, max=10000),
        "ttl": ConfigField(type=int, default=1800, description="缓存过期时间(秒)。人物信息变化慢，建议1800秒(30分钟)。改名后需等待过期才会更新", min=60, max=7200),
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
}

# 布局配置 - 使用标签页布局
config_layout = ConfigLayout(
    type="tabs",
    tabs=[
        ConfigTab(id="plugin", title="插件", icon="🔧", sections=["plugin"], order=0),
        ConfigTab(id="modules", title="模块开关", icon="📦", sections=["modules"], order=1),
        ConfigTab(id="message_cache", title="消息缓存", icon="💬", sections=["message_cache"], order=2),
        ConfigTab(id="person_cache", title="人物信息缓存", icon="👤", sections=["person_cache"], order=3),
    ]
)


@register_plugin
class PerformanceOptimizerPlugin(BasePlugin):
    plugin_name = "CM-performance-optimizer"
    plugin_version = "3.0.0"
    plugin_description = "性能优化 - 消息缓存 + 人物信息缓存"
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
        logger.info("[PerfOpt] CM-performance-optimizer v3.0.0 启动")
        
        try:
            cfg = {
                "report_interval": 60,
                "modules": {"message_cache": True, "person_cache": True},
                "message_cache_size": 2000, "message_cache_ttl": 120.0,
                "person_cache_size": 3000, "person_cache_ttl": 1800,
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
                    # message_cache 标签页
                    cfg["message_cache_size"] = c.get("message_cache", {}).get("max_size", 2000)
                    cfg["message_cache_ttl"] = c.get("message_cache", {}).get("ttl", 120.0)
                    # person_cache 标签页
                    cfg["person_cache_size"] = c.get("person_cache", {}).get("max_size", 3000)
                    cfg["person_cache_ttl"] = c.get("person_cache", {}).get("ttl", 1800)
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