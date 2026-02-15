"""图片描述批量替换优化模块（image_desc_bulk_lookup）。

该模块通过 monkey-patch 将
[`src.chat.message_receive.storage.MessageStorage.replace_image_descriptions()`](../src/chat/message_receive/storage.py:174)
从“每个匹配项一次 DB 查询”优化为“单次 WHERE IN 批量查询 + 本地替换”，
以减少包含多张图片描述文本的 DB round-trip 次数。

设计目标：
- 稳定性优先：批量查询失败时优雅回退到原始实现
- 无缓存：不引入持久缓存与额外内存占用
- 提供统计：calls / bulk_queries / saved_queries / fallbacks

模块风格参考：[`levenshtein_fast.py`](CM-performance-optimizer-plugin/components/modules/levenshtein_fast.py:1)
"""

from __future__ import annotations

import importlib.util
import re
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

_PATTERN = re.compile(r"\[图片[：:]\s*([^\]]+)\]")


def _load_core_module():
    """动态加载 core 模块，避免相对导入问题。"""

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

    # 尝试预加载 core 子模块，避免 __init__ 中的相对导入失败
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
        sub_spec.loader.exec_module(sub_mod)

    spec.loader.exec_module(module)
    return module


# ========== 核心模块加载（ModuleStats / MemoryUtils） ==========
try:
    # 在正确包结构中优先相对导入
    from ..core import MemoryUtils, ModuleStats
except Exception:
    try:
        _core = _load_core_module()
        ModuleStats = _core.ModuleStats
        MemoryUtils = _core.MemoryUtils
    except Exception as e:  # pragma: no cover
        logger.warning(f"[ImageDescBulkLookup] 无法加载核心模块，使用内置实现: {e}")

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
                _ = seen
                _ = obj
                return 0


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


class ImageDescBulkLookupModule:
    """image_desc_bulk_lookup 批量查询替换补丁模块。"""

    def __init__(self):
        """初始化统计与状态。"""

        self.stats = ModuleStats("image_desc_bulk_lookup")

        self._lock = threading.RLock()
        self._patched = False

        # 保存原始 staticmethod 描述符（用于 remove_patch 完整恢复）
        self._orig_descriptor: Any = None
        # 保存原始可调用对象（用于 fallback 调用）
        self._orig_callable: Optional[Callable[[str], str]] = None

        # 轻量统计
        self._calls = 0
        self._bulk_queries = 0
        self._saved_queries = 0
        self._fallbacks = 0

    def apply_patch(self) -> None:
        """应用 monkey-patch。"""

        with self._lock:
            if self._patched:
                return

            try:
                from src.chat.message_receive.storage import MessageStorage

                # 取原始 staticmethod 描述符
                self._orig_descriptor = MessageStorage.__dict__.get(
                    "replace_image_descriptions"
                )

                if isinstance(self._orig_descriptor, staticmethod):
                    self._orig_callable = self._orig_descriptor.__func__
                else:
                    # 兼容意外情况：不是 staticmethod，但仍可尝试当作 callable
                    self._orig_callable = (
                        self._orig_descriptor
                        if callable(self._orig_descriptor)
                        else None
                    )

                if not callable(self._orig_callable):
                    raise AttributeError(
                        "MessageStorage.replace_image_descriptions 不存在或不可调用"
                    )

                module = self
                original_func = self._orig_callable

                def bulk_replace_image_descriptions(text: str) -> str:
                    """批量替换实现：单次 WHERE IN 查询 + 本地替换。

                    批量查询失败时回退原始实现。
                    """

                    if not text:
                        return text

                    with module._lock:
                        module._calls += 1

                    matches = list(_PATTERN.finditer(text))
                    if not matches:
                        return text

                    # 统计：原实现会对每个 match 都查一次
                    match_count = len(matches)

                    # 提取唯一描述
                    descriptions = list(
                        {
                            m.group(1).strip()
                            for m in matches
                            if m.group(1) and m.group(1).strip()
                        }
                    )
                    if not descriptions:
                        return text

                    desc_to_id: Dict[str, str] = {}

                    # 第二步：单次批量查询（WHERE IN）
                    try:
                        from src.common.database.database_model import Images

                        images = (
                            Images.select()
                            .where(Images.description.in_(descriptions))
                            .order_by(Images.timestamp.desc())
                        )

                        for img in images:
                            # 由于已按 timestamp desc 排序，首次写入即最新
                            try:
                                desc = str(getattr(img, "description", ""))
                                if not desc:
                                    continue
                                if desc not in desc_to_id:
                                    desc_to_id[desc] = str(getattr(img, "image_id"))
                            except Exception:
                                continue

                        with module._lock:
                            module._bulk_queries += 1
                            module._saved_queries += max(0, match_count - 1)

                        _stats_hit(module.stats)

                    except Exception as e:
                        with module._lock:
                            module._fallbacks += 1
                        _stats_miss(module.stats)
                        logger.debug(
                            f"[ImageDescBulkLookup] 批量查询失败，回退原始实现: {e}"
                        )
                        return str(original_func(text))

                    # 第三步：替换
                    def replace_match(match: re.Match[str]) -> str:
                        description = match.group(1).strip()
                        image_id = desc_to_id.get(description)
                        if image_id:
                            return f"[picid:{image_id}]"
                        return match.group(0)

                    try:
                        return _PATTERN.sub(replace_match, text)
                    except Exception as e:
                        # 替换阶段也采用回退（最大化稳健性）
                        with module._lock:
                            module._fallbacks += 1
                        _stats_miss(module.stats)
                        logger.debug(
                            f"[ImageDescBulkLookup] 替换阶段异常，回退原始实现: {e}"
                        )
                        return str(original_func(text))

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
                        "replace_image_descriptions", "image_desc_bulk_lookup",
                        original_func, bulk_replace_image_descriptions,
                    )

                MessageStorage.replace_image_descriptions = staticmethod(
                    bulk_replace_image_descriptions
                )

                self._patched = True
                logger.info("[ImageDescBulkLookup] ✓ 补丁应用成功")

            except Exception as e:
                logger.error(f"[ImageDescBulkLookup] ✗ 补丁失败: {e}")

    def remove_patch(self) -> None:
        """移除 monkey-patch 并恢复原始实现。"""

        with self._lock:
            if not self._patched:
                return

            try:
                from src.chat.message_receive.storage import MessageStorage

                if self._orig_descriptor is not None:
                    # 完整恢复原始 staticmethod 描述符
                    MessageStorage.replace_image_descriptions = self._orig_descriptor

                # PatchChain 取消注册
                try:
                    _cm = sys.modules.get("CM_perf_opt_core")
                    if _cm and hasattr(_cm, "get_patch_chain"):
                        _pc = _cm.get_patch_chain()
                        _pc.unregister_patch("replace_image_descriptions", "image_desc_bulk_lookup")
                except Exception:
                    pass

                self._patched = False
                logger.info("[ImageDescBulkLookup] 补丁已移除")

            except Exception as e:
                logger.error(f"[ImageDescBulkLookup] 移除补丁失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """返回统计信息。"""

        with self._lock:
            return {
                "calls": int(self._calls),
                "bulk_queries": int(self._bulk_queries),
                "saved_queries": int(self._saved_queries),
                "fallbacks": int(self._fallbacks),
            }

    def get_memory_usage(self) -> int:
        """该模块无缓存，内存占用按 0 返回。"""

        return 0


def apply_image_desc_bulk_lookup(cache_manager) -> Optional[ImageDescBulkLookupModule]:
    """工厂函数：创建模块、注册到 cache_manager 并应用补丁。"""

    try:
        mod = ImageDescBulkLookupModule()
        cache_manager.register_cache("image_desc_bulk_lookup", mod)
        mod.apply_patch()
        logger.info("[ImageDescBulkLookup] ✓ 模块已初始化")
        return mod
    except Exception as e:
        logger.error(f"[ImageDescBulkLookup] 初始化失败: {e}")
        return None
