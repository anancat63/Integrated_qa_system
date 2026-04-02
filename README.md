# 集成问答系统（Integrated QA System）

这是一个面向“文档问答”的本地化智能问答系统。它把多种检索能力做成一套统一的服务：

- MySQL FAQ 精准检索（BM25）
- Milvus + RAG 语义检索增强
- Redis 缓存（加速 FAQ 与查询）
- FastAPI 接口服务（REST + WebSocket，支持流式回答）
- 会话历史持久化（写入 MySQL）

适合：知识库问答、课程助教、文档检索问答等场景。

## 主要能力与工作流概览

一次完整问答大致分为以下步骤：

1. 会话处理
   服务端支持创建会话、查询会话历史、清除会话历史。
2. FAQ 直达检索（MySQL + BM25）
   当 FAQ 置信度足够高时，系统直接返回 MySQL 的答案（走低延迟路径）。
3. 语义检索 + 生成（Milvus + RAG + LLM）
   当 FAQ 未命中或不可靠时，系统回退到 RAG：对文档进行向量检索（hybrid dense/sparse + rerank），再把上下文拼到提示词中交给 LLM 生成答案，并以流式方式返回。
4. 可选查询路由（BERT 分类）
   `rag_qa` 内置了一个 BERT 分类器，用于区分“通用知识/专业咨询”等类别（用于决定是否走检索上下文）。

## 项目结构（你需要关心的部分）

```text
.
├─ app.py                          # FastAPI 主服务（REST + WebSocket + 静态页面）
├─ api.py                          # 轻量 SSE 流式接口（/query）
├─ main.py                         # 集成核心：把 MySQL FAQ + RAG 组织成一个系统
├─ base/                           # 配置读取与日志封装
│  ├─ config.py                    # 读取 config.ini -> Config 对象
│  └─ logger.py                    # 日志输出
├─ mysql_qa/                       # FAQ 检索与缓存（MySQL + Redis + BM25）
│  ├─ db/mysql_client.py          # jpkb 表结构、CSV 导入
│  ├─ cache/redis_client.py       # Redis 缓存键与 get/set 逻辑
│  └─ retrieval/bm25_search.py    # BM25 搜索与阈值判断
├─ rag_qa/                         # 知识库 RAG 与 Milvus 向量检索
│  ├─ rag_main.py                 # 文档入库（--data-processing）/交互式查询
│  ├─ core/                        # 文档处理、Milvus 向量存储、RAG 主逻辑
│  ├─ document_loaders/          # pdf/doc/ppt/图片等加载器（OCR）
│  └─ text_splitter/             # 中文文本切分器（父块/子块）
└─ static/                         # 可选：前端页面资源
```

## 环境要求

- Python 3.10+
- MySQL 8+
- Redis 6+
- Milvus 2.x
- LLM：当前代码以 DashScope-compatible 的方式调用（`openai` SDK + 配置 `base_url`）

## 快速开始（从零到可用）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备配置文件 `config.ini`

`base/config.py` 会读取项目根目录下的 `config.ini`，并把配置映射成 `Config()` 对象的属性。

步骤：

1. 复制 `config.ini.example` 到 `config.ini`
2. 修改其中的 MySQL / Redis / Milvus / LLM 配置

配置段说明：

- `[mysql]`：`host/user/password/database`，FAQ 数据写入/读取使用
- `[redis]`：`host/port/password/db`，FAQ 分词与答案缓存使用
- `[milvus]`：`host/port/database_name/collection_name`，RAG 向量索引存储使用
- `[llm]`：`model`、`dashscope_api_key`、`dashscope_base_url`，用于生成答案
- `[retrieval]`：父块/子块大小、chunk 重叠、检索数量等
- `[app]`：`valid_sources`（RAG 数据目录命名约定与筛选用）、`customer_service_phone`

> 注意：请不要把真实密钥直接提交到 GitHub。建议只提交 `config.ini.example`，本地用 `config.ini`。

## 数据准备（分两部分：FAQ + 知识文档）

### 1) FAQ（MySQL + BM25）

MySQL FAQ 的表名是 `jpkb`，其字段约定为：

- `subject_name`：学科名称（字符串）
- `question`：问题文本（字符串）
- `answer`：答案文本（字符串）

系统提供了 `mysql_qa/db/mysql_client.py` 的导入能力：从 CSV 导入到 `jpkb`。

CSV 列名必须与代码一致：

- `学科名称`
- `问题`
- `答案`

推荐你按数据量“分批导入”（例如一次导入 500~5000 条），避免一次性导入太慢或锁表时间过长。

导入示例（在项目根目录执行）：

```bash
python - <<'PY'
from mysql_qa.db.mysql_client import MySQLClient

client = MySQLClient()
client.create_table()
client.insert_data("path/to/your_faq.csv")  # CSV 列名必须是：学科名称/问题/答案
client.close()
PY
```

### 2) 知识文档（RAG + Milvus）

RAG 文档入库由 `rag_qa/rag_main.py` 完成。

目录约定：

- 在 `rag_qa/data/{source}_data/` 下放文档文件
- `source` 来自 `config.ini` 的 `app.valid_sources`
- 例如：`rag_qa/data/ai_data/`、`rag_qa/data/java_data/`

支持的文件类型（会自动选择对应加载器）：

- `.txt`（纯文本）
- `.pdf`（OCR 解析）
- `.docx`（OCR 解析）
- `.ppt/.pptx`（OCR 解析）
- `.jpg/.png`（图片 OCR）
- `.md`（Markdown）

入库命令（建议从一个 source 目录开始，逐步扩充数据）：

```bash
python rag_qa/rag_main.py --data-processing --data-dir rag_qa/data
```

入库逻辑会把文档切成“父块/子块”两级结构后写入 Milvus，并在查询时回溯父块组装上下文。

## 模型资源（重要）

本仓库的 `.gitignore` 会忽略一些模型目录（为了让 GitHub 体积可控）。你需要在本地准备下面资源，否则对应功能可能失败或效果较差：

- 向量检索与重排序模型（Milvus 端）：`rag_qa/models/bge-m3`、`rag_qa/models/bge-reranker-large`
- 查询分类器模型（BERT Router）：基础 BERT `rag_qa/models/bert-base-chinese`
- （可选）微调后的分类器权重目录：`rag_qa/core/bert_query_classifier_1`

当前代码的行为是：

- 若 `bert_query_classifier_1` 不存在，会用 `bert-base-chinese` 初始化一个新模型（但未必已训练），此时路由效果可能不佳。

## 启动服务

### 方式 A：推荐（FastAPI 主服务）

```bash
python app.py
```

默认监听地址与端口：

- `http://localhost:8000`

健康检查：

- `GET /health`

会话与问答接口：

- `POST /api/create_session`：创建会话，返回 `session_id`
- `GET /api/sources`：返回 `valid_sources`
- `POST /api/query`：非 WebSocket 查询（FAQ 命中则返回答案，未命中可能提示改用 WebSocket 流式）
- `WS /api/stream`：WebSocket 流式回答（用于 RAG 回退场景）
- `GET /api/history/{session_id}`：获取会话历史
- `DELETE /api/history/{session_id}`：清除会话历史

### 方式 B：轻量 SSE 接口

```bash
python api.py
```

接口：

- `POST /query`（SSE，流式输出）

你可以直接运行 `use_api.py` 做接口演示（它会以 SSE 方式持续打印输出）。

## 使用示例（快速验证）

1. 健康检查

```bash
curl http://localhost:8000/health
```

2. 非流式问答（MySQL FAQ）

```bash
curl -X POST "http://localhost:8000/api/query" -H "Content-Type: application/json" -d "{\"query\":\"你的问题是什么？\",\"source_filter\":\"ai\"}"
```

3. 流式问答

流式接口在 `WS /api/stream`。你也可以打开 `static` 下的页面资源体验交互式聊天（需要确认前端与后端的接口路径一致）。

## BERT Query Router（可选）

`rag_qa/core/query_classifier.py` 里提供了一个 `train_model()` 方法用于训练二分类模型。

训练数据格式约定（`train_model()` 会逐行读取 JSON）：

- 每行一个 JSON 对象
- 字段包括：`query`（问题文本）、`label`（类别字符串，代码里使用 `通用知识` 与 `专业咨询`）

如果你需要训练：

1. 准备你的数据文件（例如 `training_dataset_hybrid_5000.json`）
2. 跑一段训练脚本（示例）：

```bash
python - <<'PY'
import os
from rag_qa.core.query_classifier import QueryClassifier

qc = QueryClassifier(model_path=os.path.join("rag_qa","core","bert_query_classifier_1"))
qc.train_model(data_file="path/to/training_dataset_hybrid_5000.json")
PY
```

## 常见问题排查

- `MySQL 连接失败`：检查 `config.ini` 的 `[mysql]` 配置项是否正确，并确认 FAQ 表 `jpkb` 已创建并导入数据
- `Redis 连接失败`：检查 `config.ini` 的 `[redis]` 配置项、端口、密码
- `Milvus 检索异常`：确认 Milvus 可连接，且指定的 `collection_name` 已完成入库
- `LLM 调用失败`：检查 `config.ini` 的 `[llm]`：`dashscope_api_key` 与 `dashscope_base_url`


