"""正则预编译优化模块（regex_precompile）。

该模块通过 monkey-patch 将 MaiBot 消息处理热路径中的多处字符串形式正则调用替换为：

- **策略 A（静态模式预编译）**：模块加载时 `re.compile()`
- **策略 B（动态模式 LRU 缓存）**：对包含动态部分（如 account）的模式，用 `functools.lru_cache` 缓存编译结果
- **策略 C（函数级替换）**：整体替换目标函数/方法实现，仅将 `re.search/re.sub` 替换为预编译 pattern 调用，不改变业务逻辑

覆盖目标：

- [`src.chat.utils.utils.is_mentioned_bot_in_message()`](../src/chat/utils/utils.py:117)

实现风格参考：
- [`user_reference_batch_resolve.py`](CM-performance-optimizer-plugin/components/modules/user_reference_batch_resolve.py:1)
- [`image_desc_bulk_lookup.py`](CM-performance-optimizer-plugin/components/modules/image_desc_bulk_lookup.py:1)

注意：
- monkey-patch **保持完全相同的函数签名/返回值**。
- 若导入/patch 任一环节失败，将**优雅降级**，不影响主程序运行。
"""

from __future__ import annotations

import re
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, cast

try:
    from src.common.logger import get_logger
except ImportError:  # pragma: no cover
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")


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


# ============================
# 静态模式：模块级预编译
# ============================

# utils.is_mentioned_bot_in_message（静态）
_RE_REPLY_YOU_PAREN: Pattern[str] = re.compile(r"\[回复 .*?\(你\)：")
_RE_REPLY_YOU_FULLWIDTH: Pattern[str] = re.compile(r"\[回复 .*?（你）：")

_RE_CLEAN_AT_PAREN: Pattern[str] = re.compile(r"@(.+?)（(\d+)）")
_RE_CLEAN_AT_ANGLE: Pattern[str] = re.compile(r"@<(.+?)(?=:(\d+))\:(\d+)>")
_RE_CLEAN_REPLY_PAREN: Pattern[str] = re.compile(
    r"\[回复 (.+?)\(((\d+)|未知id|你)\)：(.+?)\]，说："
)
_RE_CLEAN_REPLY_ANGLE: Pattern[str] = re.compile(
    r"\[回复<(.+?)(?=:(\d+))\:(\d+)>：(.+?)\]，说："
)


# ============================
# 动态模式：LRU 缓存编译结果
# ============================


@lru_cache(maxsize=256)
def _re_qq_at(current_account: str) -> Pattern[str]:
    """QQ 平台 @<name:qq_id> 检测（account 动态）。"""

    return re.compile(rf"@<(.+?):{re.escape(str(current_account))}>")


@lru_cache(maxsize=256)
def _re_other_at(current_account: str) -> Pattern[str]:
    """其他平台 @account 检测（account 动态，IGNORECASE）。"""

    return re.compile(
        rf"@{re.escape(str(current_account))}(\b|$)", flags=re.IGNORECASE
    )


@lru_cache(maxsize=256)
def _re_reply_id_paren(current_account: str) -> Pattern[str]:
    """回复格式（括号 ID）检测（account 动态）。"""

    return re.compile(
        rf"\[回复 (.+?)\({re.escape(str(current_account))}\)：(.+?)\]，说："
    )


@lru_cache(maxsize=256)
def _re_reply_id_angle(current_account: str) -> Pattern[str]:
    """回复格式（尖括号 ID）检测（account 动态）。"""

    esc = re.escape(str(current_account))
    return re.compile(
        rf"\[回复<(.+?)(?=:{esc}>)\:{esc}>：(.+?)\]，说："
    )


# ========== 核心模块加载（ModuleStats / MemoryUtils）==========
try:
    from ..core import MemoryUtils, ModuleStats
except Exception:
    try:
        _core = load_core_module(Path(__file__).parent)
        ModuleStats = _core.ModuleStats
        MemoryUtils = _core.MemoryUtils
    except (ImportError, CoreModuleLoadError) as e:  # pragma: no cover
        logger.warning(f"[RegexPrecompile] 无法加载核心模块，使用内置实现: {e}")

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
                _ = obj
                _ = seen
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


class RegexPrecompileModule:
    """regex_precompile 预编译正则补丁模块。"""

    def __init__(self):
        self.stats = ModuleStats("regex_precompile")

        self._lock = threading.RLock()
        self._patched = False
        self._degraded = False

        # 保存原始实现引用，便于 revert
        self._originals: Dict[str, Any] = {}

        # 保存 patched 引用（仅在 revert 时用于"只回滚自己打的补丁"）
        self._patched_refs: Dict[str, Any] = {}

        # alias import 补丁（sys.modules 扫描替换）
        self._alias_patches: List[Tuple[str, str, Any, Any]] = []

        # 轻量统计
        self._patch_attempts = 0
        self._patch_success = 0

    def apply_patch(self) -> None:
        """应用预编译正则替换（monkey-patch）。"""

        with self._lock:
            self._patch_attempts += 1
            if self._patched:
                return

            # 记录是否已写入 patch，用于异常回滚
            applied_utils = False

            try:
                # is_mentioned_bot_in_message
                from src.chat.utils import utils as utils_mod

                orig_is_mentioned = getattr(utils_mod, "is_mentioned_bot_in_message", None)
                if not callable(orig_is_mentioned):
                    raise AttributeError(
                        "utils.is_mentioned_bot_in_message 不存在或不可调用"
                    )

                def patched_is_mentioned_bot_in_message(message: Any) -> Tuple[bool, bool, float]:
                    """检查消息是否提到了机器人（预编译正则版）。"""

                    text = getattr(message, "processed_plain_text", "") or ""
                    platform = (
                        getattr(getattr(message, "message_info", None), "platform", "")
                        or ""
                    )

                    platforms_list = getattr(utils_mod.global_config.bot, "platforms", []) or []
                    platform_accounts = utils_mod.parse_platform_accounts(platforms_list)
                    qq_account = str(getattr(utils_mod.global_config.bot, "qq_account", "") or "")

                    current_account = utils_mod.get_current_platform_account(
                        platform, platform_accounts, qq_account
                    )

                    nickname = str(utils_mod.global_config.bot.nickname or "")
                    alias_names = list(
                        getattr(utils_mod.global_config.bot, "alias_names", []) or []
                    )
                    keywords = [nickname] + alias_names

                    reply_probability = 0.0
                    is_at = False
                    is_mentioned = False

                    add_cfg = (
                        getattr(getattr(message, "message_info", None), "additional_config", None)
                        or {}
                    )
                    if isinstance(add_cfg, dict):
                        if add_cfg.get("at_bot") or add_cfg.get("is_mentioned"):
                            is_mentioned = True
                            try:
                                if add_cfg.get("is_mentioned") not in (None, ""):
                                    reply_probability = float(add_cfg.get("is_mentioned"))  # type: ignore[arg-type]
                            except Exception:
                                pass

                    if getattr(message, "is_mentioned", False):
                        is_mentioned = True

                    def _has_mention_bot(seg: Any) -> bool:
                        try:
                            if seg is None:
                                return False
                            if getattr(seg, "type", None) == "mention_bot":
                                return True
                            if getattr(seg, "type", None) == "seglist":
                                for s in getattr(seg, "data", []) or []:
                                    if _has_mention_bot(s):
                                        return True
                            return False
                        except Exception:
                            return False

                    if _has_mention_bot(getattr(message, "message_segment", None)):
                        is_at = True
                        is_mentioned = True

                    if current_account and not is_at and not is_mentioned:
                        if platform == "qq":
                            if _re_qq_at(str(current_account)).search(text):
                                is_at = True
                                is_mentioned = True
                        else:
                            if _re_other_at(str(current_account)).search(text):
                                is_at = True
                                is_mentioned = True

                    if not is_mentioned:
                        if _RE_REPLY_YOU_PAREN.search(text) or _RE_REPLY_YOU_FULLWIDTH.search(text):
                            is_mentioned = True
                        elif current_account:
                            if _re_reply_id_paren(str(current_account)).search(text):
                                is_mentioned = True
                            elif _re_reply_id_angle(str(current_account)).search(text):
                                is_mentioned = True

                    if not is_mentioned and keywords:
                        msg_content = text
                        msg_content = _RE_CLEAN_AT_PAREN.sub("", msg_content)
                        msg_content = _RE_CLEAN_AT_ANGLE.sub("", msg_content)
                        msg_content = _RE_CLEAN_REPLY_PAREN.sub("", msg_content)
                        msg_content = _RE_CLEAN_REPLY_ANGLE.sub("", msg_content)
                        for kw in keywords:
                            if kw and kw in msg_content:
                                is_mentioned = True
                                break

                    if is_at and getattr(utils_mod.global_config.chat, "at_bot_inevitable_reply", 1):
                        reply_probability = 1.0
                        utils_mod.logger.debug("被@，回复概率设置为100%")
                    elif is_mentioned and getattr(
                        utils_mod.global_config.chat, "mentioned_bot_reply", 1
                    ):
                        reply_probability = max(reply_probability, 1.0)
                        utils_mod.logger.debug("被提及，回复概率设置为100%")

                    return bool(is_mentioned), bool(is_at), float(reply_probability)

                # ============================
                # 写入 monkey-patch（先记录 originals + patched refs，保证可回滚且不干扰其它补丁）
                # ============================

                self._originals["utils.is_mentioned_bot_in_message"] = orig_is_mentioned

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
                        "is_mentioned_bot_in_message", "regex_precompile",
                        orig_is_mentioned, patched_is_mentioned_bot_in_message,
                    )

                # is_mentioned_bot_in_message
                utils_mod.is_mentioned_bot_in_message = patched_is_mentioned_bot_in_message  # type: ignore[assignment]
                self._patched_refs["utils.is_mentioned_bot_in_message"] = (
                    patched_is_mentioned_bot_in_message
                )
                applied_utils = True

                # ============================
                # patch alias import（覆盖 from ... import xxx 场景）
                # ============================

                for mod_name, mod in list(sys.modules.items()):
                    if mod is None:
                        continue

                    # is_mentioned_bot_in_message
                    try:
                        if (
                            getattr(mod, "is_mentioned_bot_in_message", None)
                            is orig_is_mentioned
                        ):
                            self._alias_patches.append(
                                (
                                    str(mod_name),
                                    "is_mentioned_bot_in_message",
                                    orig_is_mentioned,
                                    patched_is_mentioned_bot_in_message,
                                )
                            )
                            setattr(
                                mod,
                                "is_mentioned_bot_in_message",
                                patched_is_mentioned_bot_in_message,
                            )
                    except Exception:
                        pass

                self._patched = True
                self._patch_success += 1
                _stats_hit(self.stats)
                logger.info("[RegexPrecompile] ✓ 补丁应用成功")

            except Exception as e:
                # 尝试回滚可能已写入的 patch（保持稳定性优先）
                try:
                    if applied_utils:
                        self.revert_patch()
                except Exception:
                    pass

                self._degraded = True
                _stats_miss(self.stats)
                logger.error(f"[RegexPrecompile] ✗ 补丁失败（降级）: {e}")

    def revert_patch(self) -> None:
        """恢复原始函数/方法。"""

        with self._lock:
            # 1) revert alias import
            for mod_name, attr, original, patched in list(self._alias_patches):
                mod = sys.modules.get(mod_name)
                if not mod:
                    continue
                try:
                    if getattr(mod, attr, None) is patched:
                        setattr(mod, attr, original)
                except Exception:
                    continue
            self._alias_patches.clear()

            # 2) revert 主目标
            try:
                from src.chat.utils import utils as utils_mod

                # is_mentioned_bot_in_message：仅当仍为我们 patched 才恢复
                orig_is_mentioned = self._originals.get(
                    "utils.is_mentioned_bot_in_message"
                )
                patched_is_mentioned = self._patched_refs.get(
                    "utils.is_mentioned_bot_in_message"
                )
                if (
                    callable(orig_is_mentioned)
                    and patched_is_mentioned is not None
                    and getattr(utils_mod, "is_mentioned_bot_in_message", None)
                    is patched_is_mentioned
                ):
                    utils_mod.is_mentioned_bot_in_message = orig_is_mentioned  # type: ignore[assignment]

                # PatchChain 取消注册
                try:
                    _cm = sys.modules.get("CM_perf_opt_core")
                    if _cm and hasattr(_cm, "get_patch_chain"):
                        _pc = _cm.get_patch_chain()
                        _pc.unregister_patch("is_mentioned_bot_in_message", "regex_precompile")
                except Exception:
                    pass

                self._patched = False
                self._patched_refs.clear()
                logger.info("[RegexPrecompile] 补丁已移除")

            except Exception as e:
                logger.error(f"[RegexPrecompile] 移除补丁失败: {e}")

    def is_degraded(self) -> bool:
        """是否处于降级模式。"""

        return bool(self._degraded)

    def get_memory_usage(self) -> int:
        """预编译正则占用极小，按 0 返回。"""

        return 0

    def get_stats(self) -> Dict[str, Any]:
        """返回统计信息（供 StatsReporter 汇总）。"""

        out: Dict[str, Any] = {
            "patched": bool(self._patched),
            "degraded": bool(self._degraded),
            "patch_attempts": int(self._patch_attempts),
            "patch_success": int(self._patch_success),
            "dynamic_cache": {
                "qq_at": _re_qq_at.cache_info()._asdict(),
                "other_at": _re_other_at.cache_info()._asdict(),
                "reply_id_paren": _re_reply_id_paren.cache_info()._asdict(),
                "reply_id_angle": _re_reply_id_angle.cache_info()._asdict(),
            },
        }

        try:
            if hasattr(self.stats, "total"):
                total = self.stats.total()
                if isinstance(total, dict):
                    out.update(total)
        except Exception:
            pass

        return out


def apply_regex_precompile(cache_manager: Any) -> Optional[RegexPrecompileModule]:
    """工厂函数：创建模块、注册到 cache_manager 并应用补丁。"""

    try:
        mod = RegexPrecompileModule()
        cache_manager.register_cache("regex_precompile", mod)
        mod.apply_patch()
        logger.info("[RegexPrecompile] ✓ 模块已初始化")
        return mod
    except Exception as e:
        logger.error(f"[RegexPrecompile] 初始化失败: {e}")
        return None
