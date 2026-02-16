# CM-performance-optimizer（性能优化插件）

[![Version](https://img.shields.io/badge/version-5.2.0-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

CM-performance-optimizer 是一个面向 MaiBot 的性能优化插件。

- **缓存模块（5 个）**：通过 TTL 缓存 / 全量双缓冲缓存，减少数据库与文件 IO。
- **系统性能优化模块（8 个）**：通过算法/查询/配置/运行时策略优化，降低关键路径开销。

> 说明：本插件尽量做到“无硬依赖即可运行”，未安装可选依赖时会自动降级/禁用对应增强模块，并在日志中提示。

---

## 功能模块列表（13 个）

### 缓存模块（5 个）

1. **message_cache**：消息查询缓存（TTL 缓存 `find_messages` 结果；可选使用 `orjson` 加速序列化）
2. **person_cache**：人物信息缓存（缓存 Person 的 `load_from_database` 结果，减少重复查询）
3. **expression_cache**：表达式全量缓存（双缓冲 + 原子切换；启动后后台加载）
4. **jargon_cache**：黑话全量缓存（双缓冲 + 内容索引，加速匹配/查找）
5. **kg_cache**：知识图谱全量缓存（双缓冲 + 文件哈希校验；可选依赖 `aiofiles/pandas/pyarrow/quick-algo`）

### 系统性能优化模块（8 个）

6. **levenshtein_fast**：用 `rapidfuzz` 的 C 扩展替代纯 Python 编辑距离计算（可选）
7. **image_desc_bulk_lookup**：图片描述替换改为批量 `WHERE IN` 查询，降低数据库往返
8. **lightweight_profiler**：可开关的轻量 SQL 性能剖析器（纯观测层；可选 `psutil` 提供更丰富指标）
9. **user_reference_batch_resolve**：@用户引用解析引入 TTL 缓存层，减少重复解析/查询
10. **db_tuning**：SQLite 运行时参数调优（PRAGMA）+ 索引自检（可配置 mmap、checkpoint 周期）
11. **message_repository_fastpath**：`count_messages` 快速路径（COUNT + 短 TTL 缓存）
12. **jargon_matcher_automaton**：黑话匹配升级为 Aho-Corasick 自动机（可选 `pyahocorasick`）
13. **asyncio_loop_pool**：线程本地事件循环池（避免频繁 `new_event_loop`；**高风险默认关闭**）

---

## 安装

1. 将插件目录放入 `MaiBot/plugins/` 下：

   - `MaiBot/plugins/CM-performance-optimizer-plugin/`

2. 安装依赖（建议，尤其是离线测试/容器环境）：

```bash
pip install -r CM-performance-optimizer-plugin/requirements.txt
```

3. 重启 MaiBot。

---

## 配置

### 配置文件位置

默认配置文件为 [`config.toml`](config.toml:1)。插件会在首次运行时生成/更新配置。

### 常用配置示例

> 该示例覆盖“模块开关 + 关键参数”。更细粒度配置会持续以 `config.toml` 的注释/默认值为准。

```toml
[plugin]
enabled = true
log_level = "INFO"
config_version = "2.0.0"  # 配置结构版本（用于自动迁移；不是插件版本）

[performance]
# 13 个模块开关（按需开关）
enable_message_cache = true
enable_person_cache = true
enable_expression_cache = true
enable_jargon_cache = true
enable_kg_cache = false

enable_levenshtein_fast = true
enable_image_desc_bulk_lookup = true
enable_lightweight_profiler = false
enable_user_reference_batch_resolve = true
enable_message_repository_fastpath = true
enable_db_tuning = true
enable_jargon_matcher_automaton = true
enable_asyncio_loop_pool = false

# db_tuning 参数
db_mmap_size = 268435456
# db_wal_checkpoint_interval = 300

# profiler 参数（仅 enable_lightweight_profiler=true 时生效）
# profiler_sample_rate = 0.1

[advanced]
# 可选：依赖 aiofiles/orjson 时更明显
enable_async_io = true
enable_orjson = true
thread_pool_size = 4
gc_interval = 300

[monitoring]
enable_stats = true
stats_interval = 60
enable_memory_monitor = true
memory_warning_threshold = 0.8
memory_critical_threshold = 0.9

# 模块细节参数（使用 TOML 子表表达嵌套配置）
[modules.message_cache]
enabled = true
per_chat_limit = 200
ttl = 300
max_chats = 500
ignore_time_limit_when_active = true
active_time_window = 300

[modules.person_cache]
max_size = 3000
ttl = 1800
warmup_enabled = true
warmup_per_chat_sample = 30
warmup_max_persons = 20
warmup_ttl = 120

[modules.expression_cache]
refresh_interval = 3600

[modules.jargon_cache]
refresh_interval = 3600

[modules.kg_cache]
refresh_interval = 3600
```

### 配置项说明（摘要）

- `plugin.enabled`：是否启用插件
- `plugin.log_level`：日志级别
- `performance.*`：13 个模块的开关与关键参数（例如 `db_mmap_size`）
- `advanced.*`：异步 IO / JSON 加速与线程池等
- `monitoring.*`：统计报告与内存监控
- `modules.*`：各缓存模块的细粒度参数（见示例中的 `modules.message_cache` 等子表）

---

## 可选依赖说明

| 依赖 | 用途 | 影响模块 |
|---|---|---|
| `aiofiles>=0.8.0` | 异步文件读写 | `kg_cache` / 异步 IO 优化 |
| `orjson>=3.8.0` | JSON 序列化加速 | `message_cache` 等缓存序列化路径 |
| `psutil>=5.9.0` | 系统指标采集 | `lightweight_profiler` / `core.monitor` |
| `rapidfuzz>=3.0.0` | C 扩展编辑距离 | `levenshtein_fast` |
| `pyahocorasick>=2.0.0` | Aho-Corasick 自动机 | `jargon_matcher_automaton` |
| `pandas>=1.0.0` | 数据处理 | `kg_cache` |
| `pyarrow>=10.0.0` | 高效序列化/Parquet | `kg_cache` |
| `quick-algo>=0.1.0` | 图算法 | `kg_cache` |
| `tomli>=2.0.0` | Python<3.11 TOML 后备 | `core.config` |
| `json-repair` | 修复极端 JSON 脏数据 | `expression_cache` |

---

## 注意事项与兼容性

- **MaiBot 版本**：以 [`_manifest.json`](_manifest.json:1) 的 `host_application.min_version` 为准（当前为 `0.12.0`），建议使用 MaiBot 最新 main 分支。
- **asyncio_loop_pool**：属于“运行时行为改变”类优化，可能影响与其他插件/线程模型的兼容性，默认关闭。
- **可选依赖缺失**：不会阻止插件加载，但对应增强模块会降级/禁用。

---

## 版本

- 插件版本：`5.0.0`
- 版本来源：以 [`_manifest.json`](_manifest.json:1) 的 `version` 字段为准，并在代码与文档中保持一致。

---

## 许可证

MIT License，见 [`LICENSE`](LICENSE:1)。
