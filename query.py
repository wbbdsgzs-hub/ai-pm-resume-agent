"""
RAG 查询接口

提供两种使用方式：
  1. 命令行交互：python query.py
  2. Python API：from query import query_rag

支持过滤：
  - course_id: 指定课程编号（如 "02"）
  - file_type: 指定文件类型（"内容精要" / "案例复盘" / "知识谱系"）
  - n_results: 返回 top-k 个结果
"""
import sys
import chromadb
from typing import List, Dict, Optional
from dataclasses import dataclass

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_PROVIDER, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


@dataclass
class QueryResult:
    """单条查询结果"""
    content: str
    score: float  # cosine similarity, 越高越相关
    course_id: str
    course_topic: str
    file_type: str
    section_path: str
    heading: str
    source_file: str


def get_embedding_function():
    """根据配置返回 embedding function"""
    if EMBEDDING_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("EMBEDDING_PROVIDER=openai 但未设置 OPENAI_API_KEY")
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        return OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=OPENAI_EMBEDDING_MODEL,
        )
    else:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        return DefaultEmbeddingFunction()


def _get_collection():
    """获取 ChromaDB collection"""
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"向量库目录不存在: {CHROMA_DIR}\n"
            "请先运行 python build_vectorstore.py 构建向量库"
        )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )
    except Exception:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' 不存在\n"
            "请先运行 python build_vectorstore.py 构建向量库"
        )
    return collection


def query_rag(
    question: str,
    n_results: int = 5,
    course_id: Optional[str] = None,
    file_type: Optional[str] = None,
) -> List[QueryResult]:
    """
    查询 RAG 知识库

    Args:
        question: 查询问题
        n_results: 返回 top-k 个结果（默认 5）
        course_id: 过滤课程编号（如 "02"），None 表示不过滤
        file_type: 过滤文件类型（"内容精要" / "案例复盘" / "知识谱系"），None 表示不过滤

    Returns:
        QueryResult 列表，按相关性降序排列
    """
    collection = _get_collection()

    # 构建 where 过滤条件
    where = {}
    if course_id:
        where["course_id"] = course_id
    if file_type:
        where["file_type"] = file_type

    # 查询
    kwargs = {
        "query_texts": [question],
        "n_results": n_results,
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    # 解析结果
    query_results = []
    for i in range(len(results["ids"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        # ChromaDB cosine distance → similarity: score = 1 - distance
        distance = results["distances"][0][i] if "distances" in results else 0
        score = 1 - distance

        query_results.append(QueryResult(
            content=doc,
            score=score,
            course_id=meta.get("course_id", ""),
            course_topic=meta.get("course_topic", ""),
            file_type=meta.get("file_type", ""),
            section_path=meta.get("section_path", ""),
            heading=meta.get("heading", ""),
            source_file=meta.get("source_file", ""),
        ))

    return query_results


def format_results(results: List[QueryResult], max_content_len: int = 500) -> str:
    """将查询结果格式化为可读文本"""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"\n{'=' * 60}")
        lines.append(f"📄 结果 {i} (相关度: {r.score:.3f})")
        lines.append(f"📚 课程: {r.course_id} - {r.course_topic}")
        lines.append(f"📁 类型: {r.file_type}")
        lines.append(f"📍 路径: {r.section_path}")
        lines.append(f"📝 标题: {r.heading}")
        lines.append(f"📎 来源: {r.source_file}")
        lines.append(f"{'─' * 60}")
        content_preview = r.content[:max_content_len]
        if len(r.content) > max_content_len:
            content_preview += "..."
        lines.append(content_preview)
    return "\n".join(lines)


def interactive_query():
    """命令行交互查询"""
    print("=" * 60)
    print("🔍 AI PM 课程笔记 RAG 查询系统")
    print("=" * 60)
    print("输入问题查询知识库，输入 'quit' 或 'exit' 退出")
    print("可选参数：")
    print("  --course=02     限定课程编号")
    print("  --type=内容精要  限定文件类型")
    print("  --top=5         返回结果数量")
    print()

    # 先验证 collection 可用
    try:
        collection = _get_collection()
        count = collection.count()
        print(f"✅ 向量库已加载，共 {count} 条知识\n")
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("🧑 你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见 👋")
            break

        # 解析参数
        course_id = None
        file_type = None
        n_results = 5
        question = user_input

        for part in user_input.split():
            if part.startswith("--course="):
                course_id = part.split("=", 1)[1]
                question = question.replace(part, "").strip()
            elif part.startswith("--type="):
                file_type = part.split("=", 1)[1]
                question = question.replace(part, "").strip()
            elif part.startswith("--top="):
                n_results = int(part.split("=", 1)[1])
                question = question.replace(part, "").strip()

        if not question:
            print("⚠️  请输入查询问题")
            continue

        # 查询
        try:
            results = query_rag(
                question=question,
                n_results=n_results,
                course_id=course_id,
                file_type=file_type,
            )
            print(format_results(results))
            print()
        except Exception as e:
            print(f"❌ 查询失败: {e}\n")


if __name__ == "__main__":
    interactive_query()
