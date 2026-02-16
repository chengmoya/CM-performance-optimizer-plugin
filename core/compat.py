"""
兼容性模块 - 提供动态模块加载和安全验证功能

此模块提供：
1. load_core_module() - 动态加载 core 模块的公共函数
2. validate_file_path() - 文件路径遍历安全验证
3. validate_json_schema() - JSON Schema 验证
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# 模块名称常量
CORE_MODULE_NAME = "CM_perf_opt_core"
CORE_SUBMODULES = ["cache", "utils", "config", "monitor", "module_config", "expiration"]


def load_core_module(
    caller_path: Optional[Path] = None,
    module_name: str = CORE_MODULE_NAME,
    submodules: Optional[List[str]] = None,
) -> Any:
    """动态加载 core 模块，避免相对导入问题。

    此函数被各模块共享使用，避免代码重复。

    Args:
        caller_path: 调用者文件路径，默认使用 __file__ 推断
        module_name: 模块注册名称
        submodules: 需要预加载的子模块列表

    Returns:
        加载的 core 模块对象

    Raises:
        ImportError: 当 core 模块不存在或加载失败时
        FileNotFoundError: 当 core/__init__.py 不存在时
    """
    # 如果模块已加载，直接返回
    if module_name in sys.modules:
        return sys.modules[module_name]

    # 推断 core 目录位置
    if caller_path is None:
        # 获取调用者文件路径
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_file = frame.f_back.f_code.co_filename
            caller_path = Path(caller_file).parent
        else:
            caller_path = Path.cwd()

    # 定位 core 目录
    # 假设调用者在 components/modules/ 目录下
    plugin_dir = caller_path
    while plugin_dir.name != "components" and plugin_dir.parent != plugin_dir:
        plugin_dir = plugin_dir.parent

    if plugin_dir.name == "components":
        plugin_dir = plugin_dir.parent

    core_init = plugin_dir / "core" / "__init__.py"

    if not core_init.exists():
        raise ImportError(f"Core module not found at {core_init}")

    # 创建模块规格
    spec = importlib.util.spec_from_file_location(module_name, core_init)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load core module spec from {core_init}")

    # 创建并注册模块
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    # 预加载子模块
    submodules_to_load = submodules or CORE_SUBMODULES
    for submodule in submodules_to_load:
        sub_path = plugin_dir / "core" / f"{submodule}.py"
        if sub_path.exists():
            sub_name = f"{module_name}_{submodule}"
            if sub_name not in sys.modules:
                sub_spec = importlib.util.spec_from_file_location(sub_name, sub_path)
                if sub_spec and sub_spec.loader:
                    sub_module = importlib.util.module_from_spec(sub_spec)
                    sys.modules[sub_name] = sub_module
                    sub_spec.loader.exec_module(sub_module)

    # 执行主模块
    spec.loader.exec_module(module)
    return module


def validate_file_path(
    file_path: Union[str, Path],
    base_dir: Union[str, Path],
    allow_create: bool = False,
) -> Path:
    """验证文件路径安全性，防止路径遍历攻击。

    Args:
        file_path: 待验证的文件路径
        base_dir: 基准目录，文件必须在此目录下
        allow_create: 是否允许创建不存在的路径

    Returns:
        解析后的安全路径

    Raises:
        ValueError: 当路径存在遍历风险时
        FileNotFoundError: 当路径不存在且不允许创建时
    """
    # 转换为 Path 对象
    file_path = Path(file_path)
    base_dir = Path(base_dir).resolve()

    # 解析绝对路径（解析 .. 和符号链接）
    resolved_path = file_path.resolve()

    # 验证路径是否在基准目录下
    try:
        resolved_path.relative_to(base_dir)
    except ValueError:
        raise ValueError(
            f"路径遍历风险: 文件路径 '{file_path}' 不在基准目录 '{base_dir}' 内"
        )

    # 检查路径是否存在
    if not resolved_path.exists() and not allow_create:
        raise FileNotFoundError(f"文件不存在: {resolved_path}")

    return resolved_path


def validate_json_schema(
    data: Any,
    schema: Dict[str, Any],
    path_hint: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """验证 JSON 数据是否符合指定的 Schema。

    Args:
        data: 待验证的数据
        schema: JSON Schema 定义
        path_hint: 用于错误提示的路径信息

    Returns:
        (是否有效, 错误信息) 元组
    """
    try:
        # 尝试导入 jsonschema 库
        import jsonschema

        jsonschema.validate(instance=data, schema=schema)
        return True, None
    except ImportError:
        # jsonschema 不可用，使用基础验证
        return _basic_schema_validate(data, schema, path_hint)
    except jsonschema.ValidationError as e:
        path_str = path_hint or "JSON数据"
        return False, f"{path_str} 验证失败: {e.message}"
    except jsonschema.SchemaError as e:
        return False, f"Schema 定义错误: {e.message}"


def _basic_schema_validate(
    data: Any,
    schema: Dict[str, Any],
    path_hint: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """基础 Schema 验证（当 jsonschema 库不可用时使用）。

    支持的 Schema 类型：
    - type: 数据类型验证
    - required: 必需字段验证
    - properties: 对象属性验证
    - items: 数组元素验证
    - minItems/maxItems: 数组长度验证
    """
    path_str = path_hint or "数据"

    # 类型映射
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    # 验证类型
    if "type" in schema:
        expected_type = schema["type"]
        if expected_type in type_map:
            if not isinstance(data, type_map[expected_type]):
                return False, f"{path_str}: 期望类型 {expected_type}，实际类型 {type(data).__name__}"
        elif expected_type == "array":
            if not isinstance(data, list):
                return False, f"{path_str}: 期望类型 array，实际类型 {type(data).__name__}"
        elif expected_type == "object":
            if not isinstance(data, dict):
                return False, f"{path_str}: 期望类型 object，实际类型 {type(data).__name__}"

    # 验证对象属性
    if isinstance(data, dict):
        # 必需字段
        if "required" in schema:
            for field in schema["required"]:
                if field not in data:
                    return False, f"{path_str}: 缺少必需字段 '{field}'"

        # 属性验证
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                if prop_name in data:
                    valid, error = _basic_schema_validate(
                        data[prop_name],
                        prop_schema,
                        f"{path_str}.{prop_name}",
                    )
                    if not valid:
                        return False, error

    # 验证数组
    if isinstance(data, list):
        # 长度验证
        if "minItems" in schema and len(data) < schema["minItems"]:
            return False, f"{path_str}: 数组长度 {len(data)} 小于最小值 {schema['minItems']}"
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            return False, f"{path_str}: 数组长度 {len(data)} 大于最大值 {schema['maxItems']}"

        # 元素验证
        if "items" in schema:
            for i, item in enumerate(data):
                valid, error = _basic_schema_validate(
                    item,
                    schema["items"],
                    f"{path_str}[{i}]",
                )
                if not valid:
                    return False, error

    return True, None


# 预定义的 JSON Schema
PARAGRAPH_HASH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["stored_paragraph_hashes"],
    "properties": {
        "stored_paragraph_hashes": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
}

ENTITY_COUNT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": {"type": "number"},
}


def safe_load_json_file(
    file_path: Union[str, Path],
    base_dir: Union[str, Path],
    schema: Optional[Dict[str, Any]] = None,
    encoding: str = "utf-8",
) -> Tuple[Optional[Any], Optional[str]]:
    """安全加载 JSON 文件，包含路径验证和 Schema 验证。

    Args:
        file_path: JSON 文件路径
        base_dir: 基准目录
        schema: 可选的 JSON Schema
        encoding: 文件编码

    Returns:
        (解析的数据, 错误信息) 元组。成功时错误信息为 None。
    """
    try:
        # 验证路径安全性
        validated_path = validate_file_path(file_path, base_dir)
    except ValueError as e:
        return None, f"路径验证失败: {e}"
    except FileNotFoundError:
        return None, f"文件不存在: {file_path}"

    try:
        with open(validated_path, "r", encoding=encoding) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败: {e}"
    except UnicodeDecodeError as e:
        return None, f"编码错误: {e}"
    except OSError as e:
        return None, f"文件读取错误: {e}"

    # Schema 验证
    if schema is not None:
        valid, error = validate_json_schema(data, schema, str(validated_path))
        if not valid:
            return None, error

    return data, None


# 异常类型定义
class CoreModuleLoadError(ImportError):
    """Core 模块加载失败异常"""

    def __init__(self, message: str, path: Optional[Path] = None):
        self.path = path
        super().__init__(message)


class PathTraversalError(ValueError):
    """路径遍历安全异常"""

    def __init__(self, message: str, path: Optional[Path] = None, base_dir: Optional[Path] = None):
        self.path = path
        self.base_dir = base_dir
        super().__init__(message)


class JsonValidationError(ValueError):
    """JSON Schema 验证失败异常"""

    def __init__(self, message: str, path: Optional[str] = None):
        self.path = path
        super().__init__(message)


__all__ = [
    # 模块加载
    "load_core_module",
    "CORE_MODULE_NAME",
    "CORE_SUBMODULES",
    # 路径验证
    "validate_file_path",
    # JSON 验证
    "validate_json_schema",
    "safe_load_json_file",
    "PARAGRAPH_HASH_SCHEMA",
    "ENTITY_COUNT_SCHEMA",
    # 异常
    "CoreModuleLoadError",
    "PathTraversalError",
    "JsonValidationError",
]
