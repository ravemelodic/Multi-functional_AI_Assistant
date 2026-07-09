# FastAPI 课程数据管理接口文档

> 本文档描述了 Telegram 智能助手中台的数据管理 API，用于向 Milvus 向量数据库导入课程/作业数据。

**基础地址**: `http://localhost:8000`

---

## 目录

- [1. 健康检查](#1-健康检查)
- [2. 上传文件](#2-上传文件)
- [3. 直接注入 JSON](#3-直接注入-json)
- [4. 查看统计](#4-查看统计)
- [5. 管理后台](#5-管理后台)
- [6. 错误处理](#6-错误处理)
- [7. curl 速查表](#7-curl-速查表)

---

## 1. 健康检查

检查 API 服务及 Milvus 连接状态。

```
GET /api/health
```

**请求示例**:

```bash
curl http://localhost:8000/api/health
```

**响应示例**:

```json
{
  "status": "ok",
  "milvus_host": "milvus",
  "milvus_port": "19530",
  "collection": "course_documents"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `"ok"` 表示服务正常 |
| `milvus_host` | string | Milvus 主机地址 |
| `milvus_port` | string | Milvus 端口 |
| `collection` | string | Milvus 集合名称 |

---

## 2. 上传文件

上传 CSV 或 JSON 文件，解析后存入 Milvus。

```
POST /api/upload
Content-Type: multipart/form-data
```

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | 是 | CSV 或 JSON 文件 |
| `source_type` | string | 否 | 数据类型：`"courses"`（课程，默认）或 `"assignments"`（作业） |

**CSV 文件格式**:

课程数据 (`source_type=courses`):
```csv
course_code,course_name,class_time,location,description,instructor
COMP7940,AI and Chatbot Development,"Monday 14:30-17:15",DLB 514,Development of AI-powered chatbots,Dr. Chan
COMP7930,Big Data Analytics,"Wednesday 09:00-11:45",ACB 302,Big data processing with Spark,Prof. Lee
```

作业数据 (`source_type=assignments`):
```csv
course_code,title,deadline,description,weight
COMP7940,Chatbot Project,2025-04-15 23:59,Develop a Telegram chatbot using LangGraph,40%
COMP7940,Quiz 1,2025-03-01 23:59,Basic concepts of NLP and chatbot design,10%
```

**JSON 文件格式**:

```json
[
  {
    "course_code": "COMP7940",
    "course_name": "AI and Chatbot Development",
    "class_time": "Monday 14:30-17:15",
    "location": "DLB 514",
    "description": "Development of AI-powered chatbots using LLMs"
  }
]
```

**请求示例**:

```bash
# 上传课程 CSV
curl -X POST http://localhost:8000/api/upload \
  -F "file=@courses.csv" \
  -F "source_type=courses"

# 上传作业 JSON
curl -X POST http://localhost:8000/api/upload \
  -F "file=@assignments.json" \
  -F "source_type=assignments"
```

**响应示例**:

```json
{
  "success": true,
  "filename": "courses.csv",
  "source_type": "courses",
  "records_parsed": 3,
  "chunks_stored": 12,
  "message": "✅ Imported 3 courses (12 chunks into Milvus)."
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | bool | 是否成功 |
| `filename` | string | 文件名 |
| `source_type` | string | 数据类型 |
| `records_parsed` | int | 解析出的记录数 |
| `chunks_stored` | int | 分块后存入 Milvus 的块数 |
| `message` | string | 友好提示信息 |

**错误响应**:

```json
{
  "detail": "No valid records found in the file. Check column names and data format."
}
```

| HTTP 状态码 | 说明 |
|-------------|------|
| 200 | 上传成功 |
| 400 | 文件格式错误或无有效数据 |
| 500 | 服务器内部错误（Milvus 连接失败等） |

---

## 3. 直接注入 JSON

通过请求 Body 直接提交 JSON 数据，无需文件上传。

```
POST /api/ingest
Content-Type: application/json
```

**请求 Body**:

```json
{
  "source_type": "courses",
  "data": [
    {
      "course_code": "COMP7940",
      "course_name": "AI and Chatbot Development",
      "class_time": "Monday 14:30-17:15",
      "location": "DLB 514",
      "description": "Development of AI-powered chatbots using LLMs and LangGraph"
    }
  ]
}
```

**字段说明**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `source_type` | string | 否 | `"courses"`（默认）或 `"assignments"` |
| `data` | array | 是 | 课程/作业对象数组，或单个对象 |

**课程对象字段**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `course_code` | string | 是 | 课程代码（如 `COMP7940`） |
| `course_name` | string | 否 | 课程名称 |
| `class_time` | string | 否 | 上课时间 |
| `location` | string | 否 | 上课地点 |
| `description` | string | 否 | 课程描述 |
| `instructor` | string | 否 | 授课教师 |

**作业对象字段**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `course_code` | string | 是 | 所属课程代码 |
| `title` | string | 否 | 作业标题 |
| `deadline` | string | 否 | 截止日期 |
| `description` | string | 否 | 作业描述 |
| `weight` | string | 否 | 权重（如 `"40%"`） |

**请求示例**:

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d @courses.json

# 或直接传入 JSON 字符串
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "courses", "data": [{"course_code": "COMP7940", "course_name": "AI and Chatbot"}]}'
```

**响应示例**:

```json
{
  "success": true,
  "source_type": "courses",
  "records_parsed": 3,
  "chunks_stored": 12,
  "message": "✅ Ingested 3 courses (12 chunks)."
}
```

---

## 4. 查看统计

查询 Milvus 中已存储的数据概览。

```
GET /api/stats
```

**请求示例**:

```bash
curl http://localhost:8000/api/stats
```

**响应示例**:

```json
{
  "total_samples_shown": 15,
  "entries": [
    {"key": "COMP7940 (courses)", "count": 5},
    {"key": "COMP7940 (assignments)", "count": 3},
    {"key": "COMP7930 (courses)", "count": 4},
    {"key": "COMP7510 (courses)", "count": 3}
  ],
  "collection": "course_documents"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_samples_shown` | int | 本次查询返回的样本数（非精确总数） |
| `entries` | array | 按课程代码+来源分组的块数统计 |
| `entries[].key` | string | 分组键，格式：`"课程代码 (来源)"` |
| `entries[].count` | int | 该组的文档块数 |
| `collection` | string | Milvus 集合名称 |

> **注意**: Milvus 不提供精确的 `COUNT(*)` API，此接口通过相似度搜索采样统计，反映的是大致分布而非精确总量。

---

## 5. 管理后台

图形化管理界面，支持拖拽上传和数据查看。

```
GET /admin
```

在浏览器中打开 [http://localhost:8000/admin](http://localhost:8000/admin) 即可访问。

**界面功能**:

| 区域 | 功能 |
|------|------|
| 上传文件 | 拖拽或点击选择 CSV/JSON 文件上传 |
| JSON 编辑 | 直接在文本框粘贴 JSON 数据提交 |
| 上传结果 | 显示最近的上传记录（成功/失败） |
| Milvus 统计 | 查看当前存储在 Milvus 中的课程数据分布 |
| 模板参考 | 查看 CSV 和 JSON 的格式示例 |

---

## 6. 错误处理

所有 API 端点使用统一的错误格式：

```json
{
  "detail": "错误描述信息"
}
```

| HTTP 状态码 | 常见原因 |
|-------------|----------|
| 400 | 请求参数错误、文件格式不正确、无有效数据 |
| 500 | Milvus 连接失败、嵌入模型调用失败、解析异常 |

**常见错误及解决方案**:

| 错误信息 | 解决方案 |
|----------|----------|
| `No valid records found in the file` | 检查 CSV 列名或 JSON 字段名是否正确 |
| `source_type must be 'courses' or 'assignments'` | 确保 `source_type` 参数值为 `courses` 或 `assignments` |
| Milvus 连接超时 | 确认 Milvus 容器是否正常运行（`docker-compose ps`） |
| Embedding API 调用失败 | 检查 `EMBEDDING_API_KEY` 配置是否正确 |

---

## 7. curl 速查表

```bash
# 健康检查
curl http://localhost:8000/api/health

# 上传课程 CSV
curl -X POST http://localhost:8000/api/upload \
  -F "file=@data/courses.csv" \
  -F "source_type=courses"

# 上传作业 CSV
curl -X POST http://localhost:8000/api/upload \
  -F "file=@data/assignments.csv" \
  -F "source_type=assignments"

# 注入 JSON（从文件）
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d @data.json

# 注入 JSON（内联）
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type":"courses","data":[{"course_code":"COMP7940","course_name":"AI"}]}'

# 查看统计
curl http://localhost:8000/api/stats
```
