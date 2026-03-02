"""PatchChain 最小扫描策略自测脚本。

验证点：
1) unregister 后引用回退正确
2) 不会误改其他模块属性
3) 默认路径扫描数量显著小于全量 sys.modules
4) 索引缺失时触发 fallback 并记录标记

运行：
    python CM-performance-optimizer-plugin/scripts/patch_chain_min_scan_selftest.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_patch_chain_class():
    root = Path(__file__).resolve().parents[1]
    file_path = root / "core" / "patch_chain.py"
    spec = importlib.util.spec_from_file_location("cm_perf_opt_patch_chain", file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 PatchChain: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PatchChain


def _build_function(name: str, module_name: str):
    def _fn(*args, **kwargs):
        return (name, args, kwargs)

    _fn.__name__ = name
    _fn.__module__ = module_name
    return _fn


def main() -> int:
    PatchChain = _load_patch_chain_class()
    pc = PatchChain()
    PatchChain.reset()
    pc = PatchChain()

    func_id = "find_messages"

    owner_mod_name = "src.chat.message_receive.storage"
    alias_mod_name = "CM_perf_opt_message_cache_alias"
    unrelated_mod_name = "random_unrelated_module"

    owner_mod = types.ModuleType(owner_mod_name)
    alias_mod = types.ModuleType(alias_mod_name)
    unrelated_mod = types.ModuleType(unrelated_mod_name)

    orig = _build_function("orig_find_messages", owner_mod_name)
    patched_a = _build_function("patched_a", "CM_perf_opt_message_cache")
    patched_b = _build_function("patched_b", "CM_perf_opt_full_message_cache")
    untouched = _build_function("untouched", unrelated_mod_name)

    owner_mod.find_messages = orig
    alias_mod.find_messages = orig
    unrelated_mod.find_messages = untouched

    sys.modules[owner_mod_name] = owner_mod
    sys.modules[alias_mod_name] = alias_mod
    sys.modules[unrelated_mod_name] = unrelated_mod

    try:
        # 第1层 patch
        up = pc.register_patch(func_id, "message_cache", orig, patched_a)
        assert up is orig, "第1层 wrapper_target 应为 original"
        owner_mod.find_messages = patched_a
        alias_mod.find_messages = patched_a

        # 第2层 patch
        up = pc.register_patch(func_id, "full_message_cache", patched_a, patched_b)
        assert up is patched_a, "第2层 wrapper_target 应为上一层 patched"
        owner_mod.find_messages = patched_b
        alias_mod.find_messages = patched_b

        total_modules = len(sys.modules)

        # 卸载第2层：应回退到 patched_a
        active_after_second_unreg = pc.unregister_patch(func_id, "full_message_cache")
        stats_min = pc.get_last_scan_stats()

        assert active_after_second_unreg is patched_a, "卸载第2层后链顶应为 patched_a"
        assert owner_mod.find_messages is patched_a, "owner 模块引用未回退到 patched_a"
        assert alias_mod.find_messages is patched_a, "alias 模块引用未回退到 patched_a"
        assert unrelated_mod.find_messages is untouched, "误修改了无关模块属性"
        assert stats_min.get("used_fallback") is False, "默认路径不应触发 fallback"

        scanned_min = int(stats_min.get("scanned", 0))
        assert scanned_min > 0, "最小扫描计数异常"
        assert scanned_min < total_modules, (
            f"扫描未缩小: scanned={scanned_min}, total={total_modules}"
        )

        # 人工清理索引，触发 fallback
        try:
            pc._reference_index.pop(func_id, None)  # noqa: SLF001
        except Exception:
            pass

        # 卸载第1层：应回退到 orig，并触发 fallback
        active_after_first_unreg = pc.unregister_patch(func_id, "message_cache")
        stats_fallback = pc.get_last_scan_stats()

        assert active_after_first_unreg is orig, "卸载第1层后链顶应回退到 original"
        assert owner_mod.find_messages is orig, "owner 模块引用未回退到 original"
        assert alias_mod.find_messages is orig, "alias 模块引用未回退到 original"
        assert unrelated_mod.find_messages is untouched, "fallback 误修改了无关模块"
        assert stats_fallback.get("used_fallback") is True, "索引缺失时应触发 fallback"

        print("[PASS] PatchChain 最小扫描自测通过")
        print(f"[INFO] 默认路径扫描: {stats_min}")
        print(f"[INFO] fallback 路径扫描: {stats_fallback}")
        return 0

    finally:
        for key in (owner_mod_name, alias_mod_name, unrelated_mod_name):
            sys.modules.pop(key, None)
        PatchChain.reset()


if __name__ == "__main__":
    raise SystemExit(main())
