"""Levenshtein 距离快速模块（rapidfuzz 加速）。

该模块用于通过 monkey-patch 将 [`src.person_info.person_info.levenshtein_distance()`](../src/person_info/person_info.py:132)
替换为 rapidfuzz 的 C 扩展实现，从而降低长文本或高频调用下的 CPU 消耗。

设计目标：
- rapidfuzz 可用：走快速路径，统计 hit
- rapidfuzz 不可用：优雅降级，仍然加载成功，统计 miss，并标记 degraded

模块风格参考：[`person_cache.py`](CM-performance-optimizer-plugin/components/modules/person_cache.py:1)
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    from src.common.logger import get_logger
except ImportError:  # pragma: no cover
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")


# ========== rapidfuzz 探测（模块级） ==========
try:
    from rapidfuzz.distance import Levenshtein as rf_levenshtein

    _HAS_RAPIDFUZZ = True
except Exception:  # pragma: no cover
    rf_levenshtein = None  # type: ignore[assignment]
    _HAS_RAPIDFUZZ = False


# 从公共模块导入动态加载函数
try:
    from core.compat import load_core_module, CoreModuleLoadError
except ImportError:
    # 回退定义
    import importlib.util
    
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
        
        spec = importlib.util.spec_from_file_location(module_name, core_init)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load core module from {core_init}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    
    class CoreModuleLoadError(ImportError):
        """Core 模块加载失败异常"""
        pass


# ========== 核心模块加载（ModuleStats / MemoryUtils） ==========
try:
    # 在正确包结构中优先相对导入
    from ..core import MemoryUtils, ModuleStats
except Exception:
    try:
        _core = load_core_module(Path(__file__).parent)
        ModuleStats = _core.ModuleStats
        MemoryUtils = _core.MemoryUtils
    except (ImportError, CoreModuleLoadError) as e:  # pragma: no cover
        logger.warning(f"[LevenshteinFast] 无法加载核心模块，使用内置实现: {e}")

        class ModuleStats:  # type: ignore[no-redef]
            """Fallback ModuleStats（仅提供 total/reset_interval/hit/miss）。"""

            def __init__(self, name: str):
                self.name = str(name)
                self._lock = threading.Lock()
                self.t_hit = 0
                self.t_miss = 0
                self.i_hit = 0
                self.i_miss = 0

            def hit(self) -> None:
                with self._lock:
                    self.t_hit += 1
                    self.i_hit += 1

            def miss(self, elapsed: float = 0.0) -> None:
                _ = elapsed
                with self._lock:
                    self.t_miss += 1
                    self.i_miss += 1

            def total(self) -> Dict[str, Any]:
                with self._lock:
                    return {"t_hit": self.t_hit, "t_miss": self.t_miss}

            def reset_interval(self) -> Dict[str, Any]:
                with self._lock:
                    r = {"i_hit": self.i_hit, "i_miss": self.i_miss}
                    self.i_hit = 0
                    self.i_miss = 0
                    return r

        class MemoryUtils:  # type: ignore[no-redef]
            """Fallback MemoryUtils（满足接口即可）。"""

            @staticmethod
            def get_size(obj: Any, seen: Optional[set[int]] = None) -> int:
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
                    size += sum(MemoryUtils.get_size(k, seen) for k in obj.keys())
                    size += sum(MemoryUtils.get_size(v, seen) for v in obj.values())
                elif isinstance(obj, (list, tuple, set, frozenset)):
                    size += sum(MemoryUtils.get_size(i, seen) for i in obj)
                return int(size)


def _stats_hit(stats: Any) -> None:
    try:
        if hasattr(stats, "hit"):
            stats.hit()
    except Exception:
        pass


def _stats_miss(stats: Any) -> None:
    """对不同 ModuleStats 实现保持兼容：miss(elapsed) / miss()。"""

    try:
        if not hasattr(stats, "miss"):
            return
        try:
            stats.miss(0.0)
        except TypeError:
            stats.miss()
    except Exception:
        pass


class LevenshteinFastModule:
    """Levenshtein 加速补丁模块。"""

    def __init__(self):
        """初始化并探测 rapidfuzz 可用性。"""

        self.stats = ModuleStats("levenshtein_fast")

        self._patched = False
        self._lock = threading.RLock()

        self._has_rapidfuzz = bool(_HAS_RAPIDFUZZ)
        self._degraded = not self._has_rapidfuzz

        # 原始函数引用
        self._orig_func: Optional[Callable[[str, str], int]] = None

        # 由于 calculate_string_similarity 通过模块级全局名调用，
        # 这里也同步替换其 __globals__ 中的引用，以确保 patch 行为一致。
        self._orig_globals_ref: Optional[Callable[[str, str], int]] = None

        # 轻量统计（便于直接展示）
        self._calls = 0
        self._fast_calls = 0

    @property
    def degraded(self) -> bool:
        """是否处于降级模式（rapidfuzz 不可用）。"""

        return bool(self._degraded)

    def apply_patch(self) -> None:
        """应用 monkey-patch。"""

        with self._lock:
            if self._patched:
                return

            try:
                from src.person_info import person_info

                self._orig_func = getattr(person_info, "levenshtein_distance", None)
                if not callable(self._orig_func):
                    raise AttributeError("person_info.levenshtein_distance 不存在或不可调用")

                module = self
                original_levenshtein = self._orig_func

                def fast_levenshtein(s1: str, s2: str) -> int:
                    """快速 Levenshtein：优先 rapidfuzz，不可用则回退原始实现。"""

                    # CPython GIL 保证整数 += 原子性，无需加锁
                    module._calls += 1

                    if module._has_rapidfuzz and rf_levenshtein is not None:
                        try:
                            d = int(rf_levenshtein.distance(s1, s2))
                            module._fast_calls += 1
                            _stats_hit(module.stats)
                            return d
                        except Exception:
                            # rapidfuzz 调用异常时也回退（保持稳定性）
                            pass

                    _stats_miss(module.stats)
                    return int(original_levenshtein(s1, s2))

                # 1) patch 主目标：模块函数本体
                person_info.levenshtein_distance = fast_levenshtein  # type: ignore[assignment]

                # 2) patch calculate_string_similarity 的 globals 引用（按需求）
                try:
                    css = getattr(person_info, "calculate_string_similarity", None)
                    if callable(css) and hasattr(css, "__globals__"):
                        g = css.__globals__  # type: ignore[attr-defined]
                        if isinstance(g, dict):
                            self._orig_globals_ref = g.get("levenshtein_distance")
                            g["levenshtein_distance"] = fast_levenshtein
                except Exception:
                    # globals patch 失败不影响主 patch
                    pass

                self._patched = True
                logger.info(
                    "[LevenshteinFast] ✓ 补丁应用成功"
                    + ("（rapidfuzz 可用）" if self._has_rapidfuzz else "（降级：rapidfuzz 不可用）")
                )
            except Exception as e:
                logger.error(f"[LevenshteinFast] ✗ 补丁失败: {e}")

    def remove_patch(self) -> None:
        """移除 monkey-patch 并恢复原始实现。"""

        with self._lock:
            if not self._patched:
                return

            try:
                from src.person_info import person_info

                if self._orig_func is not None:
                    person_info.levenshtein_distance = self._orig_func  # type: ignore[assignment]

                # 恢复 calculate_string_similarity.__globals__ 中的引用
                try:
                    css = getattr(person_info, "calculate_string_similarity", None)
                    if callable(css) and hasattr(css, "__globals__"):
                        g = css.__globals__  # type: ignore[attr-defined]
                        if isinstance(g, dict):
                            if self._orig_globals_ref is not None:
                                g["levenshtein_distance"] = self._orig_globals_ref
                except Exception:
                    pass

                self._patched = False
                logger.info("[LevenshteinFast] 补丁已移除")
            except Exception as e:
                logger.error(f"[LevenshteinFast] 移除补丁失败: {e}")

    def get_memory_usage(self) -> int:
        """返回模块内存占用（字节）。"""

        # 该模块只持有少量引用，按需估算
        try:
            return int(
                MemoryUtils.get_size(
                    {
                        "orig": self._orig_func,
                        "globals_ref": self._orig_globals_ref,
                        "calls": self._calls,
                        "fast_calls": self._fast_calls,
                    }
                )
            )
        except Exception:
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """返回统计信息（包含累计和间隔统计字段）。"""

        # CPython GIL 保证整数读取原子性，无需加锁
        calls = int(self._calls)
        fast_calls = int(self._fast_calls)

        out: Dict[str, Any] = {
            "degraded": bool(self._degraded),
            "has_rapidfuzz": bool(self._has_rapidfuzz),
            "calls": calls,
            "fast_calls": fast_calls,
            "fallback_calls": calls - fast_calls,
        }

        # 尽量复用 core.ModuleStats 的输出结构，方便 StatsReporter 汇总
        try:
            if hasattr(self.stats, "total"):
                total = self.stats.total()
                if isinstance(total, dict):
                    out.update(total)
        except Exception:
            pass

        return out


def apply_levenshtein_fast(cache_manager) -> Optional[LevenshteinFastModule]:
    """工厂函数：创建模块、注册到 cache_manager 并应用补丁。"""

    try:
        mod = LevenshteinFastModule()
        cache_manager.register_cache("levenshtein_fast", mod)
        mod.apply_patch()
        logger.info("[LevenshteinFast] ✓ 模块已初始化")
        return mod
    except Exception as e:
        logger.error(f"[LevenshteinFast] 初始化失败: {e}")
        return None
