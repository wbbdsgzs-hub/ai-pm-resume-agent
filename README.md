# AI PM 求职助手

基于 24 份 AI PM 课程笔记，用 RAG + DeepSeek 帮你把简历优化成招聘方想要的样子。

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

每步完成后可追问、补充信息、多轮对话，再进入下一步。

###  AI PM 知识问答

基于课程知识库 + 大厂实战经验的 AI PM 专属问答助手，涵盖：
- AI PM 基础知识（Prompt / RAG / Agent / Workflow / MCP）
- 面试真题及回答思路
- AI 产品案例分析
- 职业发展路径

### 📚 RAG 知识注入

每步自动检索 24 份 AI PM 课程知识库（500+ 知识块），建议基于方法论而非模型记忆。

## 🚀 快速开始

### 1. 安装依赖

```bash
cd ai-pm-rag
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

**DeepSeek API Key 获取：** 访问 https://platform.deepseek.com 注册并获取。

### 3. 构建向量库（首次使用）

```bash
python build_vectorstore.py
```

### 4. 启动 Web 服务

```bash
python server.py
```

浏览器会自动打开，或手动访问终端显示的地址（如 `http://localhost:50977`）。

## 📁 项目结构

```
ai-pm-rag/
├── server.py              # Flask Web 后端（主入口）
├── web_ui.html            # 前端界面
├── config.py              # RAG 配置
├── chunker.py             # Markdown 智能分块器
├── build_vectorstore.py   # 向量库构建脚本
── query.py               # RAG 查询接口
├── run_agent.py           # 命令行入口（可选）
├── agent/
│   ├── __init__.py
│   ├── prompt.py          # 系统提示词（简历分析 + 知识问答）
│   ├── llm.py             # DeepSeek API 封装（自动续接）
│   └── workflow.py        # 9 步工作流编排
├── requirements.txt
├── .env.example
├── chroma_db/             # ChromaDB 持久化目录（自动生成）
── README.md
```

## 🔧 配置说明

### 环境变量（.env）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_API_KEY` | - | DeepSeek API Key（必填） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型 |
| `EMBEDDING_PROVIDER` | `default` | RAG Embedding |
| `OPENAI_API_KEY` | - | OpenAI API Key（仅 Embedding 用） |

### 模型选择

- **`deepseek-chat`**：通用对话，速度快，适合大部分场景（推荐）
- **`deepseek-reasoner`**：深度思考模式，适合复杂分析

## 💰 费用估算

- `deepseek-chat`：输入 ¥2/百万 token，输出 ¥8/百万 token
- 一次完整 9 步流程约 ¥0.5-1 元
- 知识问答每次对话约 ¥0.01-0.05 元

## ⚠️ 注意事项

- **API Key**：必须在 `.env` 中配置，否则无法运行
- **向量库**：需要先运行 `python build_vectorstore.py`，否则 RAG 不生效
- **隐私**：简历和 JD 会发送到 DeepSeek API，请确保不包含敏感信息
- **浏览器**：推荐使用 Chrome / Edge，需要支持 ES6+

## 📦 部署到公网（可选）

### 方案一：Railway（推荐，最简单）

1. 代码推到 GitHub
2. 在 [Railway](https://railway.app) 创建项目，连接仓库
3. 添加环境变量 `DEEPSEEK_API_KEY`
4. 自动部署，获得公网 URL

### 方案二：阿里云/腾讯云轻量服务器

1. 购买轻量服务器（~30 元/月）
2. 上传代码，安装依赖
3. 用 `nohup python server.py &` 后台运行
4. 配置 Nginx 反向代理

### 方案三：直接分享（适合熟人）

把整个 `ai-pm-rag` 文件夹发给对方，按「快速开始」步骤操作即可。
