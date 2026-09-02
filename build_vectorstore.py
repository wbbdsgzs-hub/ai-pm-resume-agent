"""
向量库构建脚本

读取分块结果 → 调用 Embedding 模型 → 存入 ChromaDB 持久化
支持两种 Embedding：
  - "openai"：OpenAI text-embedding-3-small（需 OPENAI_API_KEY）
  - "default"：ChromaDB 内置 all-MiniLM-L6-v2（无需 API key，开箱即用）

用法：
  python build_vectorstore.py              # 构建/重建向量库
  python build_vectorstore.py --incremental # 增量更新（仅处理新增/修改的文件）
"""
import sys
import time
import chromadb
from pathlib import Path

# Windows 控制台默认 GBK，无法打印 emoji，强制 stdout/stderr 使用 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import (
    CHROMA_DIR, COLLECTION_NAME,
    EMBEDDING_PROVIDER, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL,
)
from chunker import chunk_all_notes, Chunk


def get_embedding_function():
    """根据配置返回 embedding function"""
    if EMBEDDING_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            print("❌ EMBEDDING_PROVIDER=openai 但未设置 OPENAI_API_KEY")
            print("   请在 .env 文件中设置，或改用 EMBEDDING_PROVIDER=default")
            sys.exit(1)
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        return OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=OPENAI_EMBEDDING_MODEL,
        )
    else:
        # ChromaDB 默认 embedding（all-MiniLM-L6-v2）
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        return DefaultEmbeddingFunction()


def chunks_to_batch(chunks: list[Chunk]) -> dict:
    """将 Chunk 列表转为 ChromaDB upsert 所需的 batch 格式"""
    ids = []
    documents = []
    metadatas = []

    for c in chunks:
        ids.append(c.id)
        # 给文档加上元信息前缀，提升检索质量
        prefix = f"[{c.course_id}课-{c.file_type}] {c.course_topic} / {c.section_path}\n\n"
        documents.append(prefix + c.content)
        metadatas.append({
            "course_id": c.course_id,
            "course_topic": c.course_topic[:100],  # ChromaDB metadata value 长度限制
            "file_type": c.file_type,
            "section_path": c.section_path[:200],
            "heading": c.heading[:100],
            "source_file": c.source_file,
        })

    return {"ids": ids, "documents": documents, "metadatas": metadatas}


def build_vectorstore(incremental: bool = False):
    """构建向量库"""
    print("=" * 60)
    print("📚 AI PM 课程笔记 RAG — 向量库构建")
    print("=" * 60)
    print(f"Embedding: {EMBEDDING_PROVIDER}")
    print(f"存储路径: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print()

    # 1. 分块
    print("📝 Step 1: 分块 Markdown 文件...")
    t0 = time.time()
    chunks = chunk_all_notes()
    t1 = time.time()
    print(f"   ✅ 生成 {len(chunks)} 个知识块 ({t1 - t0:.1f}s)")

    if not chunks:
        print("❌ 没有生成任何知识块，请检查 NOTES_DIR 配置")
        sys.exit(1)

    # 2. 初始化 ChromaDB
    print("\n💾 Step 2: 初始化 ChromaDB...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embedding_fn = get_embedding_function()

    # 如果增量模式，先获取已有 collection
    if incremental:
        try:
            collection = client.get_collection(COLLECTION_NAME)
            existing_count = collection.count()
            print(f"   增量模式: 已有 {existing_count} 条记录")
        except Exception:
            print("   增量模式: 无已有 collection，切换为全量构建")
            incremental = False

    if not incremental:
        # 全量：删除旧 collection 重建
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"   已删除旧 collection: {COLLECTION_NAME}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # 3. 分批 upsert（ChromaDB 单次 upsert 有大小限制）
    print("\n🔢 Step 3: 生成 Embedding 并写入 ChromaDB...")
    BATCH_SIZE = 100
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch = chunks_to_batch(batch_chunks)
        batch_num = i // BATCH_SIZE + 1

        t0 = time.time()
        collection.upsert(
            ids=batch["ids"],
            documents=batch["documents"],
            metadatas=batch["metadatas"],
        )
        t1 = time.time()

        print(f"   [{batch_num}/{total_batches}] 写入 {len(batch_chunks)} 条 ({t1 - t0:.1f}s)")

    # 4. 验证
    final_count = collection.count()
    print(f"\n✅ 构建完成！向量库共 {final_count} 条记录")
    print(f"   存储路径: {CHROMA_DIR}")
    print(f"   Collection: {COLLECTION_NAME}")

    # 5. 简单测试
    print("\n🧪 Step 4: 快速测试检索...")
    test_queries = [
        "RAG 是什么，怎么解决幻觉问题？",
        "Agent 的七大工程要素有哪些？",
        "CRISPE 提示词框架怎么用？",
    ]
    for q in test_queries:
        results = collection.query(query_texts=[q], n_results=3)
        top = results["metadatas"][0][0]
        doc_preview = results["documents"][0][0][:80].replace("\n", " ")
        print(f"\n   Q: {q}")
        print(f"   → [{top['course_id']}课] {top['section_path'][:50]}")
        print(f"     {doc_preview}...")

    print("\n" + "=" * 60)
    print("🎉 全部完成！可以用 query.py 进行查询了")
    print("=" * 60)


if __name__ == "__main__":
    incremental = "--incremental" in sys.argv
    build_vectorstore(incremental=incremental)
