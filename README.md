# AI PM 求职助手

基于 RAG + DeepSeek 的 JD 定制简历优化 Agent：把目标岗位 JD 和你自己的课程知识库结合起来，帮你把简历优化成招聘方想要的样子。

> 📍 项目地址：https://github.com/wbbdsgzs-hub/ai-pm-resume-agent

## ✨ 功能

### 📝 简历分析（9 步专业流程）

| 步骤 | 内容 |
|------|------|
| 1 | 解析目标岗位 JD |
| 2 | 诊断原始简历（10 维度 + 评分） |
| 3 | JD-简历匹配表（证据强度分级） |
| 4 | 经历追问清单 |
| 5 | 重构简历策略 |
| 6 | 逐段优化（修改前后对照） |
| 7 | 输出最终优化版简历 |
| 8 | 面试风险提示（Top 10 追问） |
| 9 | 进一步补强建议 |

每一步独立执行、流式输出，完成后可在聊天框**追问、补充信息、多轮对话**，再进入下一步。

### 💬 AI PM 知识问答

基于课程知识库的 AI PM 专属问答助手，涵盖：
- AI PM 基础知识（Prompt / RAG / Agent / Workflow / MCP）
- 面试真题及回答思路
- AI 产品案例分析
- 职业发展路径

### 📚 RAG 知识注入

简历分析的每一步、知识问答的每次提问，都会自动检索课程知识库（当前构建 306 个知识块），把相关知识注入提示词——建议基于方法论而非模型记忆，缓解幻觉与知识时效问题。

### 🛠 其他能力

- **文件上传**：支持 PDF / Word(.docx) / Markdown / TXT，上传后自动解析文本（简历页和每步对话均可上传）
- **历史记录**：本地自动保存，支持重命名、删除、重新查看
- **导出**：单份导出、历史记录批量导出（txt 文件）
- **长回答自动续接**：检测 `finish_reason: length` 自动续写，防止回答被截断

## 🚀 快速开始

### 0. 环境要求

- Python 3.9+（开发环境为 3.14，Windows / macOS / Linux 均可）
- 一个 DeepSeek API Key（[platform.deepseek.com](https://platform.deepseek.com) 注册获取）

### 1. 准备课程笔记（关键步骤）

> ⚠️ 课程笔记**不在本仓库中**（内容属于你的课程版权），需要自己准备。

在项目文件夹的**上一级目录**放置一个 `AI PM 课程笔记` 文件夹，结构如下：

```
新建文件夹 (2)/
├── ai-pm-rag-share/          # 本项目
└── AI PM 课程笔记/            # 你的笔记（默认路径）
    ├── 课件一/
    │   ├── 内容.md            # 课程内容精要
    │   └── 案例.md            # 案例复盘
    ├── 课件二/
    │   ├── 内容.md
    │   └── 案例.md
    ├── ...
    └── 00_AI PM 知识谱系.md    # 可选：根目录下的总览文件
```

- 每个 Markdown 文件建议以 H1 标题开头，格式：`# 09 结构设计…… — 内容精要`（课程编号 + 主题），分块器会用它识别课程编号和主题
- 文件类型自动识别：`内容.md` → 内容精要、`案例.md` → 案例复盘、根目录 `00_*.md` → 知识谱系
- 笔记放别处？改 [config.py](config.py) 里的 `NOTES_DIR` 即可

### 2. 安装依赖

```bash
cd ai-pm-rag-share
pip install -r requirements.txt
```

> 首次构建向量库时需要下载 Embedding 模型（all-MiniLM-L6-v2，约 90MB），国内网络可能较慢，耐心等待即可。

### 3. 配置 API Key

```bash
cp .env.example .env
# 用编辑器打开 .env，填入 DEEPSEEK_API_KEY=sk-你的key
```

Windows 用户也可直接新建一个 `.env` 文件，内容参考 [.env.example](.env.example)。

### 4. 构建向量库（首次使用必须）

```bash
python build_vectorstore.py
```

构建完成后会自动跑 3 个测试查询，输出类似 `✅ 构建完成！向量库共 306 条记录` 即成功。

之后笔记有增删改，重新全量构建即可（旧数据会被清除）：

```bash
python build_vectorstore.py            # 全量重建（推荐）
python build_vectorstore.py --incremental  # 增量更新（不删除旧块）
```

### 5. 启动服务

```bash
python server.py
```

Windows 用户也可以直接双击 `start.bat`。

终端会打印访问地址（端口每次随机分配），例如：

```
AI PM Agent backend: http://localhost:64257
```

用浏览器打开这个地址（推荐 Chrome / Edge）。**浏览器不会自动打开，请手动访问终端显示的地址。**

### 6. 使用流程

**简历分析**：
1. 首页点击「简历分析」，填写求职目标、粘贴目标岗位 JD、粘贴简历（或上传 PDF/Word 文件）
2. 点击「开始优化」，Agent 依次执行 9 步（每一步约 10-30 秒，实时流式输出）
3. 每步完成后可追问、补充信息、上传补充文件，再点「进入下一步」
4. 全部完成后点右上角「导出」保存结果；中途返回首页会自动保存历史记录

**知识问答**：
1. 首页点击「AI PM 知识问答」
2. 直接提问（页面有推荐问题），回答会附上检索到的课程知识
3. 问答记录同样自动保存，可在首页查看、重命名、删除、批量导出

**命令行检索（可选）**：

```bash
python query.py
# 交互式输入问题，支持内联过滤参数：
#   RAG 是什么 --course=09 --type=案例复盘 --top=3
# 输入 quit / exit / q 退出
```

## 📁 项目结构

```
ai-pm-rag-share/
├── server.py              # Flask Web 后端（主入口，随机端口启动）
├── web_ui.html            # 前端界面（原生 HTML/CSS/JS，无框架）
├── config.py              # RAG 配置（笔记路径、Embedding、分块参数）
├── chunker.py             # Markdown 智能分块器（按标题层级）
├── build_vectorstore.py   # 向量库构建脚本
├── query.py               # RAG 查询接口（命令行 / Python API）
├── agent/
│   ├── prompt.py          # 系统提示词（CRISPE 框架：简历分析 + 知识问答）
│   ├── llm.py             # DeepSeek API 封装（流式 + 自动续接）
│   └── workflow.py        # 9 步工作流编排
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量模板（复制为 .env 后填写）
├── start.bat              # Windows 一键启动脚本
├── chroma_db/             # ChromaDB 持久化目录（构建后自动生成）
└── README.md
```

## 🔧 配置说明

### 环境变量（.env）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_API_KEY` | - | DeepSeek API Key（**必填**） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 对话模型（可选 `deepseek-reasoner` 深度思考模式） |
| `EMBEDDING_PROVIDER` | `default` | RAG Embedding 方案（见下） |
| `OPENAI_API_KEY` | - | 仅当 `EMBEDDING_PROVIDER=openai` 时需要 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI Embedding 模型 |

### Embedding 方案选择

| 方案 | 设置 | 特点 |
|------|------|------|
| 内置模型（默认） | `EMBEDDING_PROVIDER=default` | all-MiniLM-L6-v2，免费、开箱即用，中文效果一般 |
| OpenAI | `EMBEDDING_PROVIDER=openai` + 填 `OPENAI_API_KEY` | text-embedding-3-small，中文效果好，有 API 费用 |

⚠️ 切换 Embedding 方案后需要**重新构建向量库**。

### 分块参数（config.py）

- `MAX_CHUNK_CHARS = 4500`：单个知识块最大字符数，超过按子标题拆分
- `MIN_CHUNK_CHARS = 100`：小于该值的块与相邻块合并

## 💰 费用估算

- `deepseek-chat`：输入 ¥2/百万 token，输出 ¥8/百万 token
- 一次完整 9 步流程约 ¥0.5-1 元
- 知识问答每次对话约 ¥0.01-0.05 元

## ⚠️ 注意事项

- **API Key 安全**：`.env` 已加入 `.gitignore`，不会被提交到 Git；分享代码时不要带上 `.env`
- **先构建向量库**：不构建也能启动服务，但 RAG 不生效、答案质量明显下降
- **隐私**：简历和 JD 会发送到 DeepSeek API，请勿粘贴敏感信息
- **笔记更新**：修改课程笔记后需重新运行 `build_vectorstore.py`，否则检索到的还是旧内容
- **浏览器**：推荐 Chrome / Edge，需要支持 ES6+

## 📦 部署到公网（可选）

### 方案一：Railway

1. 把代码推到 GitHub（本仓库已经是）
2. 在 [Railway](https://railway.app) 创建项目，连接仓库
3. 添加环境变量 `DEEPSEEK_API_KEY`
4. ⚠️ 需要把课程笔记目录一并传入部署环境，并在部署时构建向量库（建议用 Docker 把构建步骤固化）

### 方案二：云服务器（阿里云/腾讯云）

1. 购买轻量服务器（~30 元/月）
2. 上传代码和课程笔记，安装依赖
3. `nohup python server.py &` 后台运行
4. 配置 Nginx 反向代理

### 方案三：直接分享（适合熟人）

把整个 `ai-pm-rag-share` 文件夹连同 `AI PM 课程笔记` 一起发给对方，按「快速开始」步骤操作即可。
