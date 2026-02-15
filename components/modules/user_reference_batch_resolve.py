"""User reference batch resolve optimization.

该模块通过 monkey-patch 注入带 TTL 缓存的 `name_resolver`，
用于优化 [`src.chat.utils.chat_message_builder.replace_user_references()`](../src/chat/utils/chat_message_builder.py:22)
在同一条消息中重复解析同一用户引用时的性能开销。

关键分层关系：
- 本模块在上层缓存 (platform, user_id) -> display_name，命中时可跳过 Person 创建与 DB 查询。
- [`person_cache`](CM-performance-optimizer-plugin/components/modules/person_cache.py:1) 在下层缓存 person_id -> 属性数据。
两者互补，不冲突。

模块风格参考：[`levenshtein_fast.py`](CM-performance-optimizer-plugin/components/modules/levenshtein_fast.py:1)
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from src.common.logger import get_logger
except ImportError:  # pragma: no cover
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")


def _load_core_module():
    """动态加载 core 模块，避免相对导入在插件加载方式下失效。"""

    module_name = "CM_perf_opt_core"
    if module_name in sys.modules:
        return sys.modules[module_name]

    current_dir = Path(__file__).parent
    plugin_dir = current_dir.parent.parent  # components/modules -> components -> plugin root
    core_init = plugin_dir / "core" / "__init__.py"

    if not core_init.exists():
        raise ImportError(f"Core module not found at {core_init}")

    spec = importlib.util.spec_from_file_location(module_name, core_init)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load core module from {core_init}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    # 预加载常用子模块（提高兼容性；失败不致命）
    for submodule in ["cache", "utils", "config", "monitor", "module_config"]:
        sub_path = plugin_dir / "core" / f"{submodule}.py"
        if not sub_path.exists():
            continue
        sub_name = f"CM_perf_opt_core_{submodule}"
        if sub_name in sys.modules:
            continue
        sub_spec = importlib.util.spec_from_file_location(sub_name, sub_path)
        if sub_spec is None or sub_spec.loader is None:
            continue
        sub_mod = importlib.util.module_from_spec(sub_spec)
        sys.modules[sub_name] = sub_mod
        try:
            sub_spec.loader.exec_module(sub_mod)
        except Exception:
            pass

    spec.loader.exec_module(module)
    return module


# ========== 核心模块加载（TTLCache / ModuleStats / MemoryUtils） ==========
try:
    # 在正确包结构中优先相对导入
    from ..core import MemoryUtils, ModuleStats, TTLCache  # type: ignore
except Exception:
    try:
        _core = _load_core_module()
        TTLCache = _core.TTLCache
        ModuleStats = _core.ModuleStats
        MemoryUtils = _core.MemoryUtils
    except Exception as e:  # pragma: no cover
        logger.warning(
            f"[UserRefBatchResolve] 无法加载核心模块，使用内置实现: {e}"
        )

        class ModuleStats:  # type: ignore[no-redef]
            """Fallback ModuleStats（提供 total/reset_interval 的 dict 输出）。"""

            def __init__(self, name: str):
                self.name = str(name)

            def total(self) -> Dict[str, Any]:
                return {}

            def reset_interval(self) -> Dict[str, Any]:
                return {}

        class MemoryUtils:  # type: ignore[no-redef]
            """Fallback MemoryUtils。"""

            @staticmethod
            def get_size(obj: Any, seen: Optional[set[int]] = None) -> int:
                if seen is None:
                    seen = set()
                oid = id(obj)
                if oid in seen:
                    return 0
                seen.add(oid)

                try:
                    size = sys.getsizeof(obj)
                except Exception:
                    size = 0

                if isinstance(obj, dict):
                    size += sum(MemoryUtils.get_size(k, seen) for k in obj.keys())
                    size += sum(MemoryUtils.get_size(v, seen) for v in obj.values())
                elif isinstance(obj, (list, tuple, set, frozenset)):
                    size += sum(MemoryUtils.get_size(i, seen) for i in obj)
                return int(size)

        class TTLCache:  # type: ignore[no-redef]
            """Fallback TTLCache（仅提供 get_sync/set_sync/get_memory_usage_sync）。"""

            def __init__(self, max_size: int = 500, ttl: float = 120.0):
                self.max_size = int(max_size)
                self.ttl = float(ttl)
                self._data: Dict[str, Tuple[Any, float]] = {}
                self._lock = threading.RLock()

            def get_sync(self, k: str):
                with self._lock:
                    if k in self._data:
                        v, expiry = self._data[k]
                        if time.time() <= expiry:
                            return v, True
                        self._data.pop(k, None)
                    return None, False

            def set_sync(self, k: str, v: Any) -> None:
                with self._lock:
                    self._data[str(k)] = (v, time.time() + self.ttl)
                    # 简化版清理：超量时移除最早过期的 key
                    if len(self._data) > self.max_size:
                        oldest_key = min(self._data.keys(), key=lambda x: self._data[x][1])
                        self._data.pop(oldest_key, None)

            def clear_sync(self) -> None:
                with self._lock:
                    self._data.clear()

            def get_memory_usage_sync(self) -> int:
                with self._lock:
                    return int(MemoryUtils.get_size(self._data))


class _UserReferenceBatchResolveStats:
    """统计代理：提供 total()/reset_interval() 以兼容 StatsReporter。"""

    def __init__(self, base: Any):
        self._base = base
        self._lock = threading.Lock()

        # 累计（t_）
        self.t_cache_hits = 0
        self.t_cache_misses = 0
        self.t_bypassed = 0

        # 间隔（i_）
        self.i_cache_hits = 0
        self.i_cache_misses = 0
        self.i_bypassed = 0

    def cache_hit(self) -> None:
        with self._lock:
            self.t_cache_hits += 1
            self.i_cache_hits += 1
        try:
            if hasattr(self._base, "hit"):
                self._base.hit()
        except Exception:
            pass

    def cache_miss(self, elapsed: float = 0.0) -> None:
        with self._lock:
            self.t_cache_misses += 1
            self.i_cache_misses += 1
        try:
            if hasattr(self._base, "miss"):
                try:
                    self._base.miss(float(elapsed))
                except TypeError:
                    self._base.miss()
        except Exception:
            pass

    def bypassed(self) -> None:
        with self._lock:
            self.t_bypassed += 1
            self.i_bypassed += 1

    def total(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # 合并 base.total（若存在）
        try:
            if hasattr(self._base, "total"):
                base_total = self._base.total()
                if isinstance(base_total, dict):
                    out.update(base_total)
        except Exception:
            pass

        with self._lock:
            hits = int(self.t_cache_hits)
            misses = int(self.t_cache_misses)
            bypassed = int(self.t_bypassed)

        out.update(
            {
                "cache_hits": hits,
                "cache_misses": misses,
                "bypassed": bypassed,
                "hit_rate": (hits / max(1, hits + misses)),
            }
        )
        return out

    def reset_interval(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # 合并 base.reset_interval（若存在）
        try:
            if hasattr(self._base, "reset_interval"):
                base_interval = self._base.reset_interval()
                if isinstance(base_interval, dict):
                    out.update(base_interval)
        except Exception:
            pass

        with self._lock:
            hits = int(self.i_cache_hits)
            misses = int(self.i_cache_misses)
            bypassed = int(self.i_bypassed)
            self.i_cache_hits = 0
            self.i_cache_misses = 0
            self.i_bypassed = 0

        out.update(
            {
                "i_cache_hits": hits,
                "i_cache_misses": misses,
                "i_bypassed": bypassed,
                "i_hit_rate": (hits / max(1, hits + misses)),
            }
        )
        return out


class UserReferenceBatchResolveModule:
    """用户引用批量解析（带 TTL 缓存）补丁模块。"""

    def __init__(self, max_size: int = 2000, ttl: int = 600):
        """初始化缓存模块。

        Args:
            max_size: 缓存最大条目数。
            ttl: 缓存 TTL（秒）。
        """

        self.cache = TTLCache(int(max_size), float(ttl))
        self._base_stats = ModuleStats("user_reference_batch_resolve")
        self.stats = _UserReferenceBatchResolveStats(self._base_stats)

        self._lock = threading.RLock()
        self._patched = False

        self._orig_func: Optional[Callable[..., str]] = None
        self._patched_func: Optional[Callable[..., str]] = None
        self._alias_patches: List[Tuple[str, str, Any]] = []

    def apply_patch(self) -> None:
        """应用 monkey-patch：注入带缓存的 name_resolver。"""

        with self._lock:
            if self._patched:
                return

            try:
                from src.chat.utils import chat_message_builder

                self._orig_func = getattr(chat_message_builder, "replace_user_references", None)
                if not callable(self._orig_func):
                    raise AttributeError(
                        "chat_message_builder.replace_user_references 不存在或不可调用"
                    )

                module = self
                original_func = self._orig_func

                def patched_replace_user_references(
                    content: Optional[str],
                    platform: str,
                    name_resolver: Optional[Callable[[str, str], str]] = None,
                    replace_bot_name: bool = True,
                ) -> str:
                    if name_resolver is not None:
                        module.stats.bypassed()
                        return str(
                            original_func(content, platform, name_resolver, replace_bot_name)
                        )

                    def cached_resolver(pf: str, user_id: str) -> str:
                        # 延迟导入，避免模块加载顺序问题
                        try:
                            from src.chat.utils.utils import is_bot_self
                            from src.config.config import global_config
                            from src.person_info.person_info import Person
                        except Exception:
                            return str(user_id)

                        # 机器人自身：不缓存（通常也不会走到此处，但保持一致）
                        try:
                            if replace_bot_name and is_bot_self(pf, user_id):
                                return f"{global_config.bot.nickname}(你)"
                        except Exception:
                            pass

                        cache_key = f"{pf}:{user_id}"
                        try:
                            cached_val, hit = module.cache.get_sync(cache_key)
                        except Exception:
                            cached_val, hit = None, False

                        if hit:
                            module.stats.cache_hit()
                            try:
                                return str(cached_val)
                            except Exception:
                                return str(user_id)

                        # miss：创建 Person（person_cache 的 cached_load 会在下层生效）
                        t0 = time.time()
                        try:
                            person = Person(platform=pf, user_id=user_id)
                            display_name = getattr(person, "person_name", None) or user_id
                        except Exception:
                            display_name = user_id

                        module.stats.cache_miss(time.time() - t0)
                        try:
                            module.cache.set_sync(cache_key, str(display_name))
                        except Exception:
                            pass
                        return str(display_name)

                    return str(
                        original_func(content, platform, cached_resolver, replace_bot_name)
                    )

                # PatchChain 注册（冲突检测 + 链式追踪）
                _pc = None
                try:
                    _cm = sys.modules.get("CM_perf_opt_core")
                    if _cm and hasattr(_cm, "get_patch_chain"):
                        _pc = _cm.get_patch_chain()
                except Exception:
                    pass
                if _pc is not None:
                    _pc.register_patch(
                        "replace_user_references", "user_reference_batch_resolve",
                        original_func, patched_replace_user_references,
                    )

                # 1) patch 主目标
                chat_message_builder.replace_user_references = patched_replace_user_references  # type: ignore[assignment]
                self._patched_func = patched_replace_user_references

                # 2) patch alias import（参考 message_cache 的做法）
                for mod_name, mod in list(sys.modules.items()):
                    try:
                        if mod and getattr(mod, "replace_user_references", None) is original_func:
                            try:
                                self._alias_patches.append(
                                    (
                                        str(mod_name),
                                        "replace_user_references",
                                        getattr(mod, "replace_user_references", None),
                                    )
                                )
                            except Exception:
                                pass
                            setattr(mod, "replace_user_references", patched_replace_user_references)
                    except Exception:
                        continue

                self._patched = True
                logger.info("[UserRefBatchResolve] ✓ 补丁应用成功")
            except Exception as e:
                logger.error(f"[UserRefBatchResolve] ✗ 补丁失败: {e}")

    def remove_patch(self) -> None:
        """移除 monkey-patch，恢复原始函数。"""

        with self._lock:
            if not self._patched:
                return

            try:
                from src.chat.utils import chat_message_builder

                if self._orig_func is not None:
                    chat_message_builder.replace_user_references = self._orig_func  # type: ignore[assignment]

                # 恢复 alias import
                for mod_name, attr, original in list(self._alias_patches):
                    mod = sys.modules.get(mod_name)
                    if not mod:
                        continue
                    try:
                        current = getattr(mod, attr, None)
                        # 仅当仍为本模块 patched 时才回滚，避免覆盖其他 patch
                        if self._patched_func is not None and current is self._patched_func:
                            setattr(mod, attr, original)
                    except Exception:
                        continue

                self._alias_patches.clear()
                self._patched_func = None

                # PatchChain 取消注册
                try:
                    _cm = sys.modules.get("CM_perf_opt_core")
                    if _cm and hasattr(_cm, "get_patch_chain"):
                        _pc = _cm.get_patch_chain()
                        _pc.unregister_patch("replace_user_references", "user_reference_batch_resolve")
                except Exception:
                    pass

                self._patched = False
                logger.info("[UserRefBatchResolve] 补丁已移除")
            except Exception as e:
                logger.error(f"[UserRefBatchResolve] 移除补丁失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """返回统计信息。"""

        try:
            return dict(self.stats.total())
        except Exception:
            return {
                "cache_hits": 0,
                "cache_misses": 0,
                "bypassed": 0,
                "hit_rate": 0.0,
            }

    def get_memory_usage(self) -> int:
        """返回缓存内存占用（字节）。"""

        try:
            if hasattr(self.cache, "get_memory_usage_sync"):
                return int(self.cache.get_memory_usage_sync())
        except Exception:
            pass

        # fallback：粗略估算
        try:
            return int(MemoryUtils.get_size({"cache": getattr(self.cache, "_data", {})}))
        except Exception:
            return 0


def apply_user_reference_batch_resolve(
    cache_manager: Any,
) -> Optional[UserReferenceBatchResolveModule]:
    """工厂函数：创建模块、注册到 cache_manager 并应用补丁。"""

    try:
        mod = UserReferenceBatchResolveModule()
        cache_manager.register_cache("user_reference_batch_resolve", mod)
        mod.apply_patch()
        logger.info("[UserRefBatchResolve] ✓ 模块已初始化")
        return mod
    except Exception as e:
        logger.error(f"[UserRefBatchResolve] 初始化失败: {e}")
        return None
