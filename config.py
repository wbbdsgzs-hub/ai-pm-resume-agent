"""
RAG 系统配置文件
通过环境变量或 .env 文件配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============ 路径配置 ============
# 项目根目录
PROJECT_DIR = Path(__file__).parent
# 课程笔记根目录（默认与项目同级；如位置不同请改这里）
NOTES_DIR = PROJECT_DIR.parent / "AI PM 课程笔记"
# ChromaDB 持久化目录
CHROMA_DIR = PROJECT_DIR / "chroma_db"
# ChromaDB collection 名称
COLLECTION_NAME = "ai_pm_knowledge"

# ============ Embedding 配置 ============
# 可选: "openai" | "default"
# "openai"  → 用 OpenAI text-embedding-3-small（需设 OPENAI_API_KEY，中文效果好）
# "default" → ChromaDB 内置 all-MiniLM-L6-v2（无需 API key，开箱即用，中文效果一般）
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "default")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ============ 分块配置 ============
# 单个 chunk 最大字符数（超过则按子标题拆分）
MAX_CHUNK_CHARS = 4500
# 最小 chunk 字符数（小于此值的 chunk 会与相邻 chunk 合并）
MIN_CHUNK_CHARS = 100
