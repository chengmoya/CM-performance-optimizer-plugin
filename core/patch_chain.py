"""PatchChain 管理器 - 解决多模块函数 monkey-patch 冲突。

当多个优化模块需要 patch 同一个函数时，后加载的模块会覆盖前者的 patch，
导致前者的优化逻辑被跳过。PatchChain 通过维护一个有序链表来跟踪每个函数
的所有 patch，确保：

1. 每个模块 patch 时能感知当前函数已被谁 patch（冲突检测）
2. 后 patch 的模块包装的是"最新版本"的函数（链式调用）
3. 取消 patch 时能正确移除自己的注册信息

PatchChain 注册目标矩阵（与各模块实际 register_patch 保持一致）：
| 目标函数                      | 注册模块                                      |
|-------------------------------|-----------------------------------------------|
| is_mentioned_bot_in_message   | regex_precompile                              |
| replace_user_references       | user_reference_batch_resolve                  |
| replace_image_descriptions    | image_desc_bulk_lookup                        |
| store_message                 | message_cache ↔ full_message_cache            |
| find_messages                 | message_cache ↔ full_message_cache            |
"""

from __future__ import annotations

import sys
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from src.common.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger("CM_perf_opt")


class PatchChain:
    """管理函数 patch 链，确保多个模块可以链式 patch 同一函数。

    单例模式：全局只有一个 PatchChain 实例，所有模块共享。
    """

    _instance: Optional["PatchChain"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PatchChain":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        # {函数标识: [(模块名, 原始函数, 补丁函数), ...]}
        self._chains: Dict[str, List[Tuple[str, Any, Any]]] = {}
        # 最小扫描索引: {func_id: {候选模块名集合}}
        self._reference_index: Dict[str, Set[str]] = {}
        # 兼容历史 alias 导入路径（仅用于最小扫描扩展）
        self._known_prefixes: Set[str] = {
            "CM_perf_opt",
            "src.chat",
            "src.person_info",
        }
        # 最近一次 unregister 扫描统计，便于观测与最小自测
        self._last_scan_stats: Dict[str, Any] = {}
        self._rlock = threading.RLock()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def register_patch(
        self,
        func_id: str,
        module_name: str,
        original_func: Any,
        patched_func: Any,
    ) -> Any:
        """注册一个 patch，返回应该包装的函数。

        返回值是调用者的 wrapper 应当委托的"上游"函数：
        - 如果 func_id 尚无任何 patch：返回 original_func
        - 如果已有其他模块 patch：返回链中最新的 patched_func（即上一个 patch）

        Args:
            func_id: 函数唯一标识，如 ``"replace_user_references"``。
            module_name: 当前模块名，如 ``"regex_precompile"``。
            original_func: 模块看到的"原始"函数引用。
            patched_func: 当前模块准备写入的 patched 版本。

        Returns:
            调用者应当包装（wrap/delegate）的目标函数。
        """
        with self._rlock:
            if func_id not in self._chains:
                self._chains[func_id] = []

            chain = self._chains[func_id]

            # 检测冲突
            for existing_module, _, _ in chain:
                if existing_module != module_name:
                    logger.warning(
                        f"[PatchChain] ⚠️ 检测到冲突: {func_id} 已被 "
                        f"{existing_module} patch，现在 {module_name} 也将 patch 此函数"
                    )

            # 获取当前应该包装的函数（链中最后一个 patch 的结果，或原始函数）
            wrapper_target = original_func
            if chain:
                _, _, last_patched = chain[-1]
                wrapper_target = last_patched

            chain.append((module_name, original_func, patched_func))
            self._record_reference_candidates(
                func_id=func_id,
                module_name=module_name,
                original_func=original_func,
                patched_func=patched_func,
            )
            logger.debug(
                f"[PatchChain] 注册: {func_id} <- {module_name} "
                f"(链长度={len(chain)})"
            )
            return wrapper_target

    def unregister_patch(self, func_id: str, module_name: str) -> Optional[Any]:
        """取消注册一个 patch，并尝试恢复正确的函数引用。

        Returns:
            当前链应生效的函数引用（链为空时为原始函数）。
        """
        with self._rlock:
            chain = self._chains.get(func_id)
            if not chain:
                return None

            removed_items = [item for item in chain if item[0] == module_name]
            if not removed_items:
                return None

            before = len(chain)
            new_chain = [item for item in chain if item[0] != module_name]
            after = len(new_chain)

            removed_module, removed_original, removed_patched = removed_items[-1]

            if new_chain:
                _, _, new_active = new_chain[-1]
            else:
                new_active = removed_original

            if new_chain:
                self._chains[func_id] = new_chain
            else:
                self._chains.pop(func_id, None)

            if before != after:
                logger.debug(
                    f"[PatchChain] 取消注册: {func_id} x {removed_module} "
                    f"(链长度={before}->{after})"
                )

            index_candidates = self._reference_index.get(func_id)
            scanned, updated, used_fallback = self._restore_func_reference(
                func_id=func_id,
                removed_patched=removed_patched,
                new_active=new_active,
                index_candidates=index_candidates,
            )
            self._last_scan_stats = {
                "func_id": func_id,
                "scanned": scanned,
                "updated": updated,
                "used_fallback": used_fallback,
            }

            if not new_chain:
                self._reference_index.pop(func_id, None)

            return new_active

    def get_chain(self, func_id: str) -> List[Tuple[str, Any, Any]]:
        """获取某个函数的完整 patch 链（只读副本）。"""
        with self._rlock:
            return list(self._chains.get(func_id, []))

    def summary(self) -> Dict[str, List[str]]:
        """返回所有已注册的 patch 摘要：{func_id: [module_name, ...]}。"""
        with self._rlock:
            return {
                fid: [m for m, _, _ in chain]
                for fid, chain in self._chains.items()
                if chain
            }

    def get_last_scan_stats(self) -> Dict[str, Any]:
        """返回最近一次 unregister 的扫描统计。"""
        with self._rlock:
            return dict(self._last_scan_stats)

    def _record_reference_candidates(
        self,
        func_id: str,
        module_name: str,
        original_func: Any,
        patched_func: Any,
    ) -> None:
        """记录可能持有函数引用的模块集合（最小扫描索引）。"""
        candidates = self._reference_index.setdefault(func_id, set())

        # 1) patch 的逻辑模块名（例如 message_cache）
        if module_name:
            candidates.add(str(module_name))

        # 2) 原函数与补丁函数声明模块
        for fn in (original_func, patched_func):
            mod_name = getattr(fn, "__module__", None)
            if isinstance(mod_name, str) and mod_name:
                candidates.add(mod_name)
                # 同步收集更稳定的前缀，用于 alias 模块扩展
                parts = mod_name.split(".")
                if len(parts) >= 2:
                    self._known_prefixes.add(".".join(parts[:2]))
                else:
                    self._known_prefixes.add(parts[0])

        # 3) 注册时记录当前实际持有目标引用的模块（一次扫描，换取注销时最小扫描）
        try:
            for mod_name, mod in list(sys.modules.items()):
                if mod is None:
                    continue
                try:
                    if getattr(mod, func_id, None) is original_func:
                        candidates.add(str(mod_name))
                except Exception:
                    continue
        except Exception:
            pass

    def _collect_minimal_scan_modules(
        self,
        func_id: str,
        index_candidates: Optional[Set[str]],
    ) -> List[Any]:
        """基于索引 + 前缀扩展构建最小扫描模块列表。"""
        modules: List[Any] = []
        seen: Set[int] = set()

        if not index_candidates:
            return modules

        # 直接命中的模块名
        for mod_name in index_candidates:
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            obj_id = id(mod)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            modules.append(mod)

        # 前缀扩展（避免全量扫描，仅覆盖已知命名空间）
        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            if not any(
                mod_name == prefix or mod_name.startswith(prefix + ".")
                for prefix in self._known_prefixes
            ):
                continue
            # 进一步限制：仅保留可能含目标 attr 的模块
            if not hasattr(mod, func_id):
                continue
            obj_id = id(mod)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            modules.append(mod)

        return modules

    def _scan_and_replace(
        self,
        modules: List[Any],
        func_id: str,
        removed_patched: Any,
        new_active: Any,
    ) -> Tuple[int, int]:
        """在给定模块列表中替换目标引用，返回 (扫描数, 替换数)。"""
        scanned = 0
        updated = 0

        for mod in modules:
            scanned += 1
            try:
                current = getattr(mod, func_id, None)
                if current is removed_patched:
                    setattr(mod, func_id, new_active)
                    updated += 1
            except Exception:
                continue

        return scanned, updated

    def _restore_func_reference(
        self,
        func_id: str,
        removed_patched: Any,
        new_active: Any,
        index_candidates: Optional[Set[str]],
    ) -> Tuple[int, int, bool]:
        """恢复函数引用：默认最小扫描，必要时回退全量扫描。"""
        # 默认路径：最小扫描
        minimal_modules = self._collect_minimal_scan_modules(func_id, index_candidates)
        scanned, updated = self._scan_and_replace(
            minimal_modules,
            func_id,
            removed_patched,
            new_active,
        )

        # fallback 条件：索引缺失/失效（候选为空）
        if minimal_modules:
            return scanned, updated, False

        logger.warning(
            f"[PatchChain] {func_id} 最小扫描索引缺失/失效，回退全量 sys.modules 扫描"
        )
        try:
            full_modules = [mod for mod in list(sys.modules.values()) if mod is not None]
            full_scanned, full_updated = self._scan_and_replace(
                full_modules,
                func_id,
                removed_patched,
                new_active,
            )
            return scanned + full_scanned, updated + full_updated, True
        except Exception as e:
            logger.debug(f"[PatchChain] 恢复引用时出现异常: {e}")
            return scanned, updated, True

    @classmethod
    def reset(cls) -> None:
        """重置单例状态（用于热重载场景）
        
        此方法会：
        1. 清空所有 patch 链记录
        2. 重置单例实例为 None
        
        线程安全：使用类锁确保原子操作
        """
        with cls._lock:
            if cls._instance is not None:
                try:
                    with cls._instance._rlock:
                        cls._instance._chains.clear()
                        cls._instance._reference_index.clear()
                        cls._instance._last_scan_stats.clear()
                    logger.debug("[PatchChain] 单例已重置")
                except Exception as e:
                    logger.error(f"[PatchChain] 重置失败: {e}")
                finally:
                    cls._instance = None


def get_patch_chain() -> PatchChain:
    """获取全局 PatchChain 单例的便捷函数。"""
    return PatchChain()
