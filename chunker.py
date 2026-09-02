"""
Markdown 智能分块器

按 H2/H3 标题层级分块，保留课程编号、主题、章节路径等元数据。
分块策略：
  1. 先按 H2（##）拆分大章节
  2. 大章节内再按 H3（###）拆分子节
  3. 如果单节超过 MAX_CHUNK_CHARS，按 H4（####）继续拆
  4. 小于 MIN_CHUNK_CHARS 的碎片与前一节合并
  5. 每个 chunk 附带完整元数据（课程编号、主题、章节路径、文件类型）
"""
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from config import NOTES_DIR, MAX_CHUNK_CHARS, MIN_CHUNK_CHARS


@dataclass
class Chunk:
    """一个知识块"""
    id: str                          # 唯一 ID
    content: str                     # 文本内容
    course_id: str                   # 课程编号，如 "02"
    course_topic: str                # 课程主题，如 "整体框架，AI agent产品主要框架和知识点"
    file_type: str                   # 文件类型："内容精要" | "案例复盘" | "知识谱系"
    section_path: str                # 章节路径，如 "Part One > 1.2 五层结构框架"
    heading: str                     # 当前块的直接标题
    source_file: str                 # 源文件路径


# 文件名 → file_type 映射（兼容 内容.md / 案例.md 这类简称）
FILE_TYPE_MAP = {
    "内容": "内容精要",
    "内容精要": "内容精要",
    "案例": "案例复盘",
    "案例复盘": "案例复盘",
}

# file_type → chunk ID 使用的 ASCII 后缀（ChromaDB ID 不允许非 ASCII 字符）
TYPE_SLUG = {"内容精要": "content", "案例复盘": "case", "知识谱系": "spectrum"}


def _parse_title(file_path: Path) -> tuple:
    """从文件 H1 标题解析 (课程编号, 课程主题)

    兼容两种标题格式：
      "01 AI产品经理的核心能力和时代机遇 — 内容精要" → ("01", "AI产品经理的核心能力和时代机遇")
      "课件12：测试与数据，大模型产品开发中的数据准备与反馈闭环 — 内容精要" → ("12", "测试与数据，…")
    解析失败返回 (None, None)
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return None, None
    h1 = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if not h1:
        return None, None
    title = h1.group(1).strip()
    # 去掉末尾 " — 内容精要 / 案例复盘" 后缀（要求长横线两侧有空格，避免误伤主题里的连字符）
    title = re.sub(r"\s+[—–]\s+[^—–]+$", "", title).strip()
    m = re.match(r"^(?:课件)?(\d{1,2})[.、．:：\s]+(.+)$", title)
    if m:
        return m.group(1).zfill(2), m.group(2).strip()
    return None, None


def parse_course_info(file_path: Path) -> tuple:
    """从文件路径解析课程编号和主题"""
    # 00_AI PM 知识谱系.md → course_id="00", topic="AI PM 知识谱系"
    # 02_xxx/内容精要.md → course_id="02", topic="xxx", file_type="内容精要"
    # 课件五/内容.md → 文件夹名匹配不上时，从 H1 标题解析 course_id 和 topic
    rel = file_path.relative_to(NOTES_DIR)
    parts = rel.parts

    if len(parts) == 1:
        # 根目录文件（如 00_AI PM 知识谱系.md）
        name = parts[0]
        m = re.match(r"^(\d+)_(.+)\.md$", name)
        if m:
            return m.group(1), m.group(2), "知识谱系"
        return "00", name, "知识谱系"

    # 子目录文件
    dir_name = parts[0]
    file_name = parts[1].replace(".md", "")
    file_type = FILE_TYPE_MAP.get(file_name, file_name)
    m = re.match(r"^(\d+)_(.+)$", dir_name)
    if m:
        course_id = m.group(1)
        course_topic = m.group(2)
    else:
        # 文件夹名不带编号（如 课件五）→ 从 H1 标题解析
        course_id, course_topic = _parse_title(file_path)
        if course_id is None:
            course_id, course_topic = "00", dir_name

    return course_id, course_topic, file_type


def split_by_heading(text: str, level: int) -> List[Dict]:
    """按指定标题级别拆分文本，返回 [{heading, content, level}, ...]"""
    pattern = r"^(#{" + str(level) + r"}\s+.+)$"
    lines = text.split("\n")
    sections = []
    current_heading = ""
    current_lines = []

    for line in lines:
        if re.match(pattern, line):
            if current_lines or current_heading:
                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_lines).strip(),
                    "level": level,
                })
            current_heading = line.strip("# ").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines or current_heading:
        sections.append({
            "heading": current_heading,
            "content": "\n".join(current_lines).strip(),
            "level": level,
        })

    return [s for s in sections if s["content"].strip()]


def extract_section_path(content: str) -> str:
    """从内容中提取章节路径（向上追溯父标题）"""
    # 简化：取第一个 H2 标题作为 section_path
    lines = content.split("\n")
    for line in lines:
        if line.startswith("## ") and not line.startswith("### "):
            return line.strip("# ").strip()
    return ""


def chunk_markdown_file(file_path: Path) -> List[Chunk]:
    """将单个 Markdown 文件分块"""
    course_id, course_topic, file_type = parse_course_info(file_path)
    text = file_path.read_text(encoding="utf-8")

    # 跳过文件开头的元信息（> 来源：xxx）
    # 保留第一个 H1 作为文件标题
    h1_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    file_title = h1_match.group(1).strip() if h1_match else course_topic

    chunks = []
    chunk_counter = 0

    # 第一层：按 H2 拆分
    h2_sections = split_by_heading(text, 2)

    for h2 in h2_sections:
        h2_content = h2["content"]
        h2_heading = h2["heading"]

        # 如果 H2 节内容不大，直接作为一个 chunk
        if len(h2_content) <= MAX_CHUNK_CHARS:
            if len(h2_content) >= MIN_CHUNK_CHARS:
                chunk_counter += 1
                chunks.append(Chunk(
                    id=f"{course_id}_{TYPE_SLUG.get(file_type, 'doc')}_{chunk_counter:03d}",
                    content=h2_content,
                    course_id=course_id,
                    course_topic=course_topic,
                    file_type=file_type,
                    section_path=h2_heading,
                    heading=h2_heading,
                    source_file=str(file_path),
                ))
            continue

        # 大节：按 H3 继续拆
        h3_sections = split_by_heading(h2_content, 3)
        for h3 in h3_sections:
            h3_content = h3["content"]
            h3_heading = h3["heading"]

            if len(h3_content) <= MAX_CHUNK_CHARS:
                if len(h3_content) >= MIN_CHUNK_CHARS:
                    chunk_counter += 1
                    chunks.append(Chunk(
                        id=f"{course_id}_{TYPE_SLUG.get(file_type, 'doc')}_{chunk_counter:03d}",
                        content=h3_content,
                        course_id=course_id,
                        course_topic=course_topic,
                        file_type=file_type,
                        section_path=f"{h2_heading} > {h3_heading}",
                        heading=h3_heading,
                        source_file=str(file_path),
                    ))
                continue

            # 超大节：按 H4 继续拆
            h4_sections = split_by_heading(h3_content, 4)
            for h4 in h4_sections:
                h4_content = h4["content"]
                h4_heading = h4["heading"]

                if len(h4_content) >= MIN_CHUNK_CHARS:
                    chunk_counter += 1
                    chunks.append(Chunk(
                        id=f"{course_id}_{TYPE_SLUG.get(file_type, 'doc')}_{chunk_counter:03d}",
                        content=h4_content,
                        course_id=course_id,
                        course_topic=course_topic,
                        file_type=file_type,
                        section_path=f"{h2_heading} > {h3_heading} > {h4_heading}",
                        heading=h4_heading,
                        source_file=str(file_path),
                    ))

    return chunks


def chunk_all_notes() -> List[Chunk]:
    """分块所有课程笔记"""
    all_chunks = []
    md_files = sorted(NOTES_DIR.rglob("*.md"))

    for f in md_files:
        try:
            chunks = chunk_markdown_file(f)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ⚠️  分块失败: {f} — {e}")

    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all_notes()
    print(f"共生成 {len(chunks)} 个知识块")

    # 统计
    by_course = {}
    for c in chunks:
        key = f"{c.course_id}_{c.course_topic[:20]}"
        by_course.setdefault(key, []).append(c)

    for k, v in sorted(by_course.items()):
        total_chars = sum(len(c.content) for c in v)
        print(f"  {k}: {len(v)} chunks, {total_chars:,} chars")

    # 示例
    print("\n--- 示例 chunk ---")
    sample = chunks[5]
    print(f"ID: {sample.id}")
    print(f"课程: {sample.course_id} {sample.course_topic}")
    print(f"类型: {sample.file_type}")
    print(f"路径: {sample.section_path}")
    print(f"内容前200字: {sample.content[:200]}...")
