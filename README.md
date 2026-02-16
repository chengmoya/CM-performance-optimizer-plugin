# CM-performance-optimizer（性能优化插件）

[![Version](https://img.shields.io/badge/version-6.0.0-blue.svg)](https://github.com/chengmoya/CM-performance-optimizer-plugin)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MaiBot](https://img.shields.io/badge/MaiBot-%3E%3D0.12.0-purple.svg)](https://github.com/Mai-with-u/MaiBot)

CM-performance-optimizer 是一个面向 MaiBot 的全栈性能优化插件，通过多级缓存、算法优化和运行时调优，显著提升 MaiBot 的响应速度和资源利用率。

> **核心特性**：���存模块（5个）+ 系统优化模块（8个）+ 通知系统 + 错误日志监控

---

## 📖 目录

- [功能模块列表](#功能模块列表)
- [安装说明](#安装说明)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [性能优化建议](#性能优化建议)
- [可选依赖说明](#可选依赖说明)
- [注意事项](#注意事项)
- [更新日志](#更新日志)
- [许可证](#许可证)

---

## 功能模块列表

### 缓存模块（5个）

| 模块名称 | 功能描述 | 关键技术 |
|---------|---------|---------|
| **message_cache** | 消息查询缓存，缓存 `find_messages` 查询结果 | TTL 缓存 + 可选 `orjson` 加速序列化 |
| **person_cache** | 人物信息缓存，缓存 Person 的 `load_from_database` 结果 | TTL 缓存 + 预热机制 |
| **expression_cache** | 表达式全量缓存 | 双缓冲 + 原子切换 + 后台加载 |
| **jargon_cache** | 黑话全量缓存 | 双缓冲 + 内容索引 + Aho-Corasick 自动机 |
| **kg_cache** | 知识图谱全量缓存 | 双缓冲 + 文件哈希校验 + Parquet 格式 |

### 系统性能优化模块（8个）

| 模块名称 | 功能描述 | 关键技术 |
|---------|---------|---------|
| **levenshtein_fast** | 编辑距离计算加速 | `rapidfuzz` C 扩展替代纯 Python |
| **image_desc_bulk_lookup** | 图片描述批量查询 | `WHERE IN` 批量查询降低数据库往返 |
| **lightweight_profiler** | 轻量 SQL 性能剖析器 | 可开关的纯观测层 |
| **user_reference_batch_resolve** | @用户引用解析优化 | TTL 缓存层减少重复解析 |
| **db_tuning** | SQLite 运行时参数调优 | PRAGMA 配置 + 索引自检 |
| **message_repository_fastpath** | 消息计数快速路径 | COUNT + 短 TTL 缓存 |
| **jargon_matcher_automaton** | 黑话匹配加速 | Aho-Corasick 自动机算法 |
| **asyncio_loop_pool** | 线程本地事件循环池 | 避免频繁创建事件循环 |

### 通知系统

- **QQ 通知渠道**：通过 MaiBot 发送私聊消息通知
- **控制台通知渠道**：终端输出通知信息
- **错误日志监控**：自动捕获 ERROR 及以上级别日志并发送通知

---

## 安装说明

### 前置要求

- MaiBot 版本 >= 0.12.0
- Python >= 3.8

### 安装步骤

1. **克隆或下载插件**

   将插件目录放入 MaiBot 的 `plugins` 目录下：

   ```bash
   cd MaiBot/plugins
   git clone https://github.com/chengmoya/CM-performance-optimizer-plugin.git
   ```

   或直接下载压缩包解压到 `MaiBot/plugins/CM-performance-optimizer-plugin/`。

2. **安装依赖**

   ```bash
   pip install -r CM-performance-optimizer-plugin/requirements.txt
   ```

   > **提示**：大部分依赖为可选依赖，未安装时插件仍可运行，但对应增强模块会自动降级或禁用。

3. **创建配置文件**

   首次运行时，插件会自动在 `MaiBot/data/plugins/CM-performance-optimizer/` 目录下生成 `config.toml` 配置文件。

   也可以手动复制配置模板：

   ```bash
   cp CM-performance-optimizer-plugin/config.example.toml MaiBot/data/plugins/CM-performance-optimizer/config.toml
   ```

4. **重启 MaiBot**

   ```bash
   # 重启 MaiBot 以加载插件
   python main.py
   ```

---

## 配置说明

> ⚠️ **重要提示**：配置修改后需要**重启 MaiBot** 才能生效！

### 配置文件位置

配置文件位于 `MaiBot/data/plugins/CM-performance-optimizer/config.toml`。

### 配置结构概览

```toml
[plugin]           # 插件基本配置
[modules]          # 功能模块开关
[message_cache]    # 消息缓存配置
[person_cache]     # 人物信息缓存配置
[expression_cache] # 表达式缓存配置
[jargon_cache]     # 黑话缓存配置
[kg_cache]         # 知识图谱缓存配置
[db_tuning]        # 数据库调优配置
[lightweight_profiler] # 性能剖析配置
[advanced]         # 高级配置
[monitoring]       # 监控配置
[notification]     # 通知配置
```

### 插件基本配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| `enabled` | boolean | `true` | 是否启用插件 |
| `config_version` | string | `"2.0.0"` | 配置文件版本（用于配置迁移，请勿手动修改） |
| `log_level` | string | `"INFO"` | 日志级别，可选值：`DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL` |

### 功能模块开关

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| `message_cache_enabled` | boolean | `true` | 消息缓存开关 |
| `message_repository_fastpath_enabled` | boolean | `true` | 消息仓库快速路径开关 |
| `person_cache_enabled` | boolean | `true` | 人物信息缓存开关 |
| `expression_cache_enabled` | boolean | `true` | 表达式缓存开关 |
| `jargon_cache_enabled` | boolean | `true` | 黑话缓存开关 |
| `jargon_matcher_automaton_enabled` | boolean | `true` | 黑话匹配自动机开关 |
| `kg_cache_enabled` | boolean | `true` | 知识图谱缓存开关 |
| `levenshtein_fast_enabled` | boolean | `true` | Levenshtein 距离加速开关 |
| `image_desc_bulk_lookup_enabled` | boolean | `true` | 图片描述批量查询开关 |
| `user_reference_batch_resolve_enabled` | boolean | `true` | 用户引用批量解析开关 |
| `regex_precompile_enabled` | boolean | `true` | 正则表达式预编译开关 |
| `typo_generator_cache_enabled` | boolean | `true` | 错别字生成器缓存开关 |
| `db_tuning_enabled` | boolean | `true` | SQLite 数据库调优开关 |
| `lightweight_profiler_enabled` | boolean | `false` | 轻量性能剖析器开关 |
| `asyncio_loop_pool_enabled` | boolean | `false` | 异步事件循环池开关（高风险，默认关闭） |

### 消息缓存配置

| 配置项 | 类型 | 默认值 | 约束范围 | 说明 |
|-------|------|-------|---------|------|
| `per_chat_limit` | integer | `200` | 50-1000 | 每个聊天的缓存消息数量 |
| `ttl` | integer | `300` | 60-3600 | 缓存过期时间（秒） |
| `max_chats` | integer | `500` | 100-2000 | 最大缓存聊天数 |
| `mode` | string | `"query"` | `query` / `full` | 缓存模式：`query` 仅缓存查询结果，`full` 缓存完整消息数据 |
| `ignore_time_limit_when_active` | boolean | `true` | - | 活跃聊天的缓存是否忽略 TTL 限制 |
| `active_time_window` | integer | `300` | 60-1800 | 判断聊天是否活跃的时间窗口（秒） |

### 人物信息缓存配置

| 配置项 | 类型 | 默认值 | 约束范围 | 说明 |
|-------|------|-------|---------|------|
| `max_size` | integer | `3000` | 500-10000 | 最大缓存条目数 |
| `ttl` | integer | `1800` | 300-7200 | 缓存过期时间（秒），默认 30 分钟 |
| `warmup_enabled` | boolean | `true` | - | 是否启用预热功能 |
| `warmup_per_chat_sample` | integer | `30` | 10-100 | 预热时每聊天采样消息数 |
| `warmup_max_persons` | integer | `20` | 5-50 | 每聊天最多预热人数 |
| `warmup_ttl` | integer | `120` | 30-600 | 预热记录过期时间（秒） |
| `warmup_debounce_seconds` | float | `3.0` | 0.5-10.0 | 预热防抖时间（秒） |

### 表达式缓存配置

| 配置项 | 类型 | 默认值 | 约束范围 | 说明 |
|-------|------|-------|---------|------|
| `batch_size` | integer | `100` | 50-500 | 批量处理大小 |
| `batch_delay` | float | `0.05` | 0.01-0.5 | 批量处理延迟（秒） |
| `refresh_interval` | integer | `3600` | 600-86400 | 自动刷新间隔（秒），默认 1 小时 |
| `incremental_refresh_interval` | integer | `600` | 60-3600 | 增量刷新间隔（秒），默认 10 分钟 |
| `incremental_threshold_ratio` | float | `0.1` | 0.01-1.0 | 增量刷新阈值比例 |
| `full_rebuild_interval` | integer | `86400` | 3600-604800 | 全量重建间隔（秒），默认 24 小时 |
| `deletion_check_interval` | integer | `10` | 1-100 | 删除检测间隔 |

### 黑话缓存配置

| 配置项 | 类型 | 默认值 | ���束范围 | 说明 |
|-------|------|-------|---------|------|
| `batch_size` | integer | `100` | 50-500 | 批量处理大小 |
| `batch_delay` | float | `0.05` | 0.01-0.5 | 批量处理延迟（秒） |
| `refresh_interval` | integer | `3600` | 600-86400 | 自动刷新间隔（秒） |
| `enable_content_index` | boolean | `true` | - | 是否启用内容索引 |
| `incremental_refresh_interval` | integer | `600` | 60-3600 | 增量刷新间隔（秒） |
| `incremental_threshold_ratio` | float | `0.1` | 0.01-1.0 | 增量刷新阈值比例 |
| `full_rebuild_interval` | integer | `86400` | 3600-604800 | 全量重建间隔（秒） |
| `deletion_check_interval` | integer | `10` | 1-100 | 删除检测间隔 |

### 知识图谱缓存配置

| 配置项 | 类型 | 默认值 | 约束范围 | 说明 |
|-------|------|-------|---------|------|
| `batch_size` | integer | `100` | 50-500 | 批量处理大小 |
| `batch_delay` | float | `0.05` | 0.01-0.5 | 批量处理延迟（秒） |
| `refresh_interval` | integer | `3600` | 600-86400 | 自动刷新间隔（秒） |
| `incremental_refresh_interval` | integer | `600` | 60-3600 | 增量刷新间隔（秒） |
| `incremental_threshold_ratio` | float | `0.1` | 0.01-1.0 | 增量刷新阈值比例 |
| `full_rebuild_interval` | integer | `86400` | 3600-604800 | 全量重建间隔（秒） |
| `deletion_check_interval` | integer | `10` | 1-100 | 删除检测间隔 |
| `use_parquet` | boolean | `true` | - | 是否启用 Parquet 格式存储 |

### 数据库调优配置

| 配置项 | 类型 | 默认值 | 约束范围 | 说明 |
|-------|------|-------|---------|------|
| `mmap_size` | integer | `268435456` | 0+ | SQLite mmap 大小（字节），0 表示禁用，默认 256MB |
| `wal_checkpoint_interval` | integer | `300` | 0-86400 | WAL checkpoint 周期（秒），0 表示禁用自动 checkpoint |

### 轻量性能剖析配置

| 配置项 | 类型 | 默认值 | 约束范围 | 说明 |
|-------|------|-------|---------|------|
| `sample_rate` | float | `0.1` | 0.0-1.0 | 采样率，0.1 表示 10% 的操作会被记录 |

### 高级配置

| 配置项 | 类型 | 默认值 | 约束范围 | 说明 |
|-------|------|-------|---------|------|
| `enable_async_io` | boolean | `true` | - | 是否启用异步 IO 优化（需要 `aiofiles`） |
| `enable_orjson` | boolean | `true` | - | 是否启用 orjson 加速（需要 `orjson`） |
| `thread_pool_size` | integer | `4` | 1-32 | 线程池大小（预留配置，暂未实际使用） |
| `gc_interval` | integer | `300` | 60-3600 | 垃圾回收间隔（秒） |
| `strict_validation` | boolean | `false` | - | 是否启用严格验证 |
| `enable_change_notifications` | boolean | `true` | - | 是否启用配置变更通知 |

### 监控配置

| 配置项 | 类型 | 默认值 | 约束范围 | 说明 |
|-------|------|-------|---------|------|
| `enable_stats` | boolean | `true` | - | 是否启用统计功能 |
| `stats_interval` | integer | `60` | 10-3600 | 统计报告间隔（秒） |
| `enable_memory_monitor` | boolean | `true` | - | 是否启用内存监控 |
| `memory_warning_threshold` | float | `0.8` | 0.1-1.0 | 内存警告阈值（0.8 表示 80%） |
| `memory_critical_threshold` | float | `0.9` | 0.1-1.0 | 内存严重阈值（0.9 表示 90%） |
| `enable_health_check` | boolean | `true` | - | 是否启用健康检查 |
| `health_check_interval` | integer | `30` | 10-300 | 健康检查间隔（秒） |

### 通知配置

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| `enabled` | boolean | `true` | 是否启用通知功能 |
| `admin_qq` | string | `""` | 接收通知的 QQ 号，留空则不发送 QQ 通知 |

---

## 使用示例

### 基础配置示例

```toml
# config.toml - 基础配置示例

[plugin]
enabled = true
log_level = "INFO"

[modules]
message_cache_enabled = true
person_cache_enabled = true
expression_cache_enabled = true
jargon_cache_enabled = true
kg_cache_enabled = false  # 知识图谱缓存需要额外依赖

[notification]
enabled = true
admin_qq = "123456789"  # 替换为你的 QQ 号
```

### 高性能配置示例（内存充足）

```toml
# config.toml - 高性能配置（适用于内存 >= 4GB 的服务器）

[plugin]
enabled = true
log_level = "WARNING"

[modules]
# 开启所有缓存模块
message_cache_enabled = true
person_cache_enabled = true
expression_cache_enabled = true
jargon_cache_enabled = true
kg_cache_enabled = true
levenshtein_fast_enabled = true
jargon_matcher_automaton_enabled = true

[message_cache]
per_chat_limit = 500
ttl = 600
max_chats = 1000
mode = "full"

[person_cache]
max_size = 5000
ttl = 3600
warmup_enabled = true

[db_tuning]
mmap_size = 536870912  # 512MB
wal_checkpoint_interval = 300
```

### 低内存配置示例

```toml
# config.toml - 低内存配置（适用于内存 <= 1GB 的服务器）

[plugin]
enabled = true
log_level = "WARNING"

[modules]
message_cache_enabled = true
person_cache_enabled = true
expression_cache_enabled = false
jargon_cache_enabled = true
kg_cache_enabled = false
lightweight_profiler_enabled = false

[message_cache]
per_chat_limit = 50
ttl = 120
max_chats = 100

[person_cache]
max_size = 500
ttl = 600
warmup_enabled = false

[monitoring]
enable_memory_monitor = true
memory_warning_threshold = 0.7
memory_critical_threshold = 0.85
```

### 调试配置示例

```toml
# config.toml - 调试配置

[plugin]
enabled = true
log_level = "DEBUG"

[modules]
lightweight_profiler_enabled = true

[lightweight_profiler]
sample_rate = 0.5  # 50% 采样率

[monitoring]
enable_stats = true
stats_interval = 30
```

---

## 性能优化建议

### 内存充足场景

1. **增大缓存容量**：提高 `per_chat_limit`、`max_size` 等参数
2. **延长 TTL**：增大 `ttl` 值使缓存更持久
3. **启用预热**：开启 `warmup_enabled` 预加载常用数据
4. **启用全量缓存模式**：设置 `message_cache.mode = "full"`
5. **启用知识图谱缓存**：设置 `kg_cache_enabled = true`

### 内存紧张场景

1. **减小缓存容量**：降低 `per_chat_limit`、`max_size` 等参数
2. **缩短 TTL**：减小 `ttl` 值加快缓存释放
3. **禁用预热**：关闭 `warmup_enabled`
4. **使用查询模式**：设置 `message_cache.mode = "query"`
5. **禁用知识图谱缓存**：设置 `kg_cache_enabled = false`

### 高频使用场景

1. **启用预热功能**：加速首次访问
2. **增大活跃时间窗口**：提高 `active_time_window`
3. **启用快速路径**：确保 `message_repository_fastpath_enabled = true`
4. **启用黑话自动机**：设置 `jargon_matcher_automaton_enabled = true`

### 调试场景

1. **启用性能剖析器**：设置 `lightweight_profiler_enabled = true`
2. **提高日志级别**：设置 `log_level = "DEBUG"`
3. **启用统计报告**：设置 `enable_stats = true`

---

## 可选依赖说明

| 依赖包 | 版本要求 | 影响模块 | 说明 |
|-------|---------|---------|------|
| `aiofiles` | >=0.8.0 | `kg_cache`、异步 IO 优化 | 异步文件读写 |
| `orjson` | >=3.8.0 | `message_cache` 等缓存模块 | 高性能 JSON 序列化 |
| `psutil` | >=5.9.0 | `lightweight_profiler`、`monitor` | 系统指标采集 |
| `rapidfuzz` | >=3.0.0 | `levenshtein_fast` | C 扩展编辑距离计算 |
| `pyahocorasick` | >=2.0.0 | `jargon_matcher_automaton` | Aho-Corasick 自动机 |
| `pandas` | >=1.0.0 | `kg_cache` | 数据处理 |
| `pyarrow` | >=10.0.0 | `kg_cache` | 高效序列化/Parquet 支持 |
| `quick-algo` | >=0.1.0 | `kg_cache` | 图算法库 |
| `tomli` | >=2.0.0 | `core.config` | Python < 3.11 的 TOML 后备 |
| `json-repair` | >=0.7.0 | `expression_cache` | 修复极端 JSON 脏数据 |

> **注意**：未安装可选依赖时，插件仍可正常加载运行，但对应的增强模块会自动降级或禁用，并在日志中提示。

---

## 注意事项

### 兼容性

- **MaiBot 版本**：要求 >= 0.12.0，建议使用最新 main 分支
- **Python 版本**：要求 >= 3.8

### 高风险模块

- **asyncio_loop_pool**：属于"运行时行为改变"类优化，可能影响与其他插件/线程模型的兼容性，**默认关闭**。启用前请充分测试。

### 通知系统

- 配置 `admin_qq` 后，插件会在内存占用过高或发生错误时发送 QQ 私聊通知
- 留空 `admin_qq` 则不发送 QQ 通知，仅输出到控制台

### 性能监控

- 启用 `lightweight_profiler` 会产生轻微性能开销
- 生产环境建议保持 `log_level = "INFO"` 或 `"WARNING"`

---

## 更新日志

### [6.0.0] - 2026-02-16

#### 新增
- **通知系统**（QQ 消息 + 控制台）
  - QQ 通知渠道（通过 MaiBot 发送消息）
  - 控制台通知渠道
  - 通知冷却控制（防止频繁发送）
  - 每日发送限制
  - 通知去重机制
- **错误日志通知**
  - 捕获 ERROR 及以上级别日志
  - 自动发送到配置的 QQ 号
  - 支持堆栈跟踪包含
  - 错误去重窗口配置
- **缓存过期机制增强**
  - 增量刷新模式（仅更新变化的数据）
  - 全量重建模式（完全重新加载）
  - 可配置的刷新间隔
- **人物缓存过期模式**
  - 30 分钟默认 TTL
  - 最大缓存大小限制
  - 自动过期清理

#### 变更
- 版本号升级到 6.0.0
- 插件描述更新，包含通知系统功能
- 配置文件结构优化，新增通知和过期配置节

### [5.2.0] - 2026-02-16

#### 变更
- 统一版本号到 5.2.0（包括配置系统版本号）

### [5.1.0] - 2026-02-15

#### 变更
- 统一版本号到 5.1.0
- 清理遗留注释和调试代码
- 优化代码结构和可读性
- 完善类型注解和文档

### [5.0.0] - 2026-02-15

#### 新增
- 表达式缓存模块（expression_cache）
- 拼写错误生成器缓存（typo_generator_cache）
- Levenshtein 距离加速模块（levenshtein_fast）
- 轻量级性能分析器（lightweight_profiler）
- 异步循环池（asyncio_loop_pool）
- 数据库调优模块（db_tuning）
- 正则预编译模块（regex_precompile）
- 用户引用批量解析（user_reference_batch_resolve）
- 消息仓库快速路径（message_repository_fastpath）
- 图片描述批量查询（image_desc_bulk_lookup）

#### 变更
- 重构核心架构，模块化设计
- 优化配置系统，支持热更新
- 完善类型注解和错误处理

### [4.7.0] - 2026-02-01

#### 变更
- 黑话缓存支持 Aho-Corasick 算法优化
- 几万条黑话也能毫秒级响应
- 性能提升 100 倍以上

### [4.5.0] - 2026-02-03

#### 新增
- 全量消息缓存模式

### [4.3.1] - 2026-01-31

#### 变更
- 优化内存统计功能
- 更新文档

### [4.2.0] - 2026-01-31

#### 新增
- 知识库图谱缓存功能（kg_cache）

### [3.0.0] - 2026-01-31

#### 新增
- 消息缓存模块（message_cache）
- 人物信息缓存（person_cache）
- 黑话缓存（jargon_cache）

---

## 许可证

本项目采用 MIT 许可证，详见 [`LICENSE`](LICENSE) 文件。

---

## 链接

- **插件仓库**：[https://github.com/chengmoya/CM-performance-optimizer-plugin](https://github.com/chengmoya/CM-performance-optimizer-plugin)
- **MaiBot 主项目**：[https://github.com/Mai-with-u/MaiBot](https://github.com/Mai-with-u/MaiBot)
- **MaiBot 开发文档**：[https://docs.mai-mai.org/develop/](https://docs.mai-mai.org/develop/)
