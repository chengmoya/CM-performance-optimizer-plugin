---
name: version-management
description: 统一版本号管理技能 - 当需要修改插件版本号时使用此技能，确保所有文件版本一致。包含 version.py 作为唯一版本源，其他文件导入或同步版本号。
---

# 何时使用

当需要执行以下操作时使用此技能：
- 修改插件版本号（PLUGIN_VERSION）
- 修改配置文件版本号（CONFIG_VERSION）
- 更新最低兼容版本（MIN_COMPATIBLE_VERSION）
- 更新 MaiBot 最低版本要求（MAIBOT_MIN_VERSION）
- 发布新版本时需要同步所有文件的版本号

# 何时不使用

- 仅查看当前版本号而不修改时
- 与版本管理无关的配置修改
- 代码重构不涉及版本变更

# 版本号管理规范

## 核心原则

**唯一版本源**：`version.py` 是所有版本信息的唯一真实来源。修改版本号时，只需修改 `version.py` 一个文件，其他文件通过导入或同步机制保持一致。

## 受版本号影响的所有文件清单

| 文件路径 | 版本号字段 | 同步方式 |
|---------|-----------|---------|
| `version.py` | PLUGIN_VERSION, CONFIG_VERSION, MIN_COMPATIBLE_VERSION, MAIBOT_MIN_VERSION | **源文件** - 手动修改 |
| `plugin.py` | 从 version.py 导入 | 自动同步 - 无需手动修改 |
| `core/config.py` | 从 version.py 导入 | 自动同步 - 无需手动修改 |
| `_manifest.json` | version 字段 | **需要手动同步** |
| `config.toml` | config_version 字段 | **需要手动同步** |
| `README.md` | 文档中的版本号 | **需要手动同步（如有）** |

## 版本号变量说明

在 `version.py` 中定义以下版本变量：

```python
PLUGIN_VERSION = "x.y.z"          # 插件版本号，遵循语义化版本
CONFIG_VERSION = "x.y.z"           # 配置文件版本，用于配置迁移
MIN_COMPATIBLE_VERSION = "x.y.z"   # 最小兼容的配置版本
MAIBOT_MIN_VERSION = "x.y.z"       # 最低兼容的 MaiBot 版本
```

# 工作流程

## 步骤 1：修改 version.py（核心步骤）

编辑 `version.py`，修改需要更新的版本号变量：

```python
# 修改前
PLUGIN_VERSION = "6.0.0"

# 修改后
PLUGIN_VERSION = "6.1.0"
```

## 步骤 2：同步 _manifest.json

找到并修改 `version` 字段，使其与 `PLUGIN_VERSION` 一致：

```json
{
  "version": "6.1.0"
}
```

## 步骤 3：同步 config.toml

找到并修改 `config_version` 字段，使其与 `CONFIG_VERSION` 一致：

```toml
config_version = "6.1.0"
```

## 步骤 4：检查 README.md

检查 `README.md` 中是否有硬编码的版本号，如有则同步更新。

## 步骤 5：验证导入正常

确认以下导入语句能正常工作：
- `plugin.py`: `from version import PLUGIN_VERSION, CONFIG_VERSION, MAIBOT_MIN_VERSION`
- `core/config.py`: `from version import CONFIG_VERSION, MIN_COMPATIBLE_VERSION`

# 版本号格式规范

遵循语义化版本（Semantic Versioning）：
- **主版本号 (MAJOR)**：不兼容的 API 变更
- **次版本号 (MINOR)**：向后兼容的新功能
- **修订号 (PATCH)**：向后兼容的问题修复

示例：`6.1.0` 表示第 6 主版本、第 1 次版本、无修订号

# 常见问题

## Q: 为什么 _manifest.json 和 config.toml 不能自动导入 version.py？

A: 这些是配置文件格式（JSON、TOML），无法直接导入 Python 模块。因此需要手动保持同步。

## Q: 修改版本号后需要做什么？

A: 
1. 确保所有受影响的文件都已同步
2. 如有 CHANGELOG.md，建议添加版本更新记录
3. 运行测试确保没有引入问题
