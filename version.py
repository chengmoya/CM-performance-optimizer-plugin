"""
统一版本管理模块

本模块集中管理插件的所有版本信息，确保版本一致性:
- PLUGIN_VERSION: 插件版本号
- CONFIG_VERSION: 配置文件版本号
- MIN_COMPATIBLE_VERSION: 最小兼容版本号

所有其他模块应从此模块导入版本信息，避免版本不一致问题。
"""

# 插件版本号
PLUGIN_VERSION = "6.0.0"

# 配置文件版本号 (用于配置迁移)
CONFIG_VERSION = "6.0.0"

# 最小兼容的配置版本
MIN_COMPATIBLE_VERSION = "1.0.0"

# MaiBot 最低版本要求
MAIBOT_MIN_VERSION = "0.12.0"

__all__ = [
    "PLUGIN_VERSION",
    "CONFIG_VERSION", 
    "MIN_COMPATIBLE_VERSION",
    "MAIBOT_MIN_VERSION",
]
