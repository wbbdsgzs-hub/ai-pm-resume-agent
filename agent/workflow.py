"""
AI PM 求职 Agent — 9 步工作流

整合 RAG 知识检索 + DeepSeek LLM 调用，按课件 CRISPE 框架的 9 步流程执行。
"""
import sys
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# 将项目根目录加入 path，以便导入 query.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.prompt import build_system_prompt, build_user_prompt, CAPACITY_AND_ROLE, PERSONALITY
from agent.llm import DeepSeekLLM

# RAG 查询（延迟导入，避免未构建向量库时报错）
_rag_module = None


def _get_rag():
    global _rag_module
    if _rag_module is None:
        try:
            from query import query_rag
            _rag_module = query_rag
        except ImportError:
            print("⚠️  RAG 模块未找到，将不使用知识库检索")
            _rag_module = None
    return _rag_module


@dataclass
class StepResult:
    """单步执行结果"""
    step: int
    title: str
    content: str
    rag_sources: List[str] = field(default_factory=list)  # 引用的知识来源


class AIPMAgent:
    """AI PM 求职 Agent"""

    STEP_TITLES = {
        1: "解析目标岗位 JD",
        2: "诊断原始简历",
        3: "建立 JD-简历匹配表",
        4: "追问清单",
        5: "重构简历策略",
        6: "逐段优化简历内容",
        7: "输出最终优化版简历",
        8: "面试风险提示",
        9: "进一步补强建议",
    }

    def __init__(self, use_rag: bool = True, rag_top_k: int = 3):
        self.llm = DeepSeekLLM()
        self.use_rag = use_rag
        self.rag_top_k = rag_top_k
        self.step_results: Dict[int, StepResult] = {}
        self.conversation_history: List[Dict[str, str]] = []

    def _search_rag(self, query: str) -> tuple:
        """检索 RAG 知识库，返回 (格式化上下文, 来源列表)"""
        if not self.use_rag:
            return "", []

        query_fn = _get_rag()
        if query_fn is None:
            return "", []

        try:
            results = query_fn(query, n_results=self.rag_top_k)
            if not results:
                return "", []

            # 格式化 RAG 结果
            context_parts = []
            sources = []
            for r in results:
                context_parts.append(
                    f"[{r.course_id}课-{r.file_type}] {r.section_path}\n{r.content}"
                )
                sources.append(f"[{r.course_id}课] {r.section_path} (相关度: {r.score:.3f})")

            context = "\n\n---\n\n".join(context_parts)
            return context, sources
        except Exception as e:
            print(f"  ⚠️  RAG 检索失败: {e}")
            return "", []

    # 每步的详细指令（包含"不要做什么"的负向约束 + 固定输出模板）
    STEP_INSTRUCTIONS = {
        1: """## 当前任务：第1步 - 解析目标岗位 JD

你是一位资深互联网招聘专家。你的**唯一任务**是解析以下岗位 JD。

### 输入数据

{input_data}

### 输出要求（严格按以下 6 个模块输出，不要增加任何其他模块）

**模块一：岗位职责拆解**

| 职责维度 | 关键词提取 | 能力层级（执行/主导/规划） |
|---|---|---|

**模块二：任职要求分类**

- 硬性门槛：（只列 JD 中明确写出的学历、年限、工具技能要求）
- 优先加分项：（只列 JD 中写"优先""加分"的条件）
- 隐性要求：（从措辞、业务背景中推断的软性要求，需注明推断依据）

**模块三：高频关键词**

（提取 JD 中的 AI 术语、产品方法论术语、行业黑话，按频次排序）

**模块四：目标人才画像**

（2-3 句话描述招聘方真正想找的人）

**模块五：岗位最看重的 5 项能力**

（按重要性排序，每项附一句说明）

**模块六：岗位特征判断**

- 业务属性：toC / toB / 平台 / 增长 / 商业化
- 行业方向：
- 技术深度要求：浅 / 中 / 深
- 团队定位：执行层 / 骨干层 / 管理层

### 输出边界（极其重要）

你只能输出以上 6 个模块的内容。以下行为严格禁止：
- 不要对任何简历进行诊断或评价
- 不要给出简历优化建议
- 不要做 JD 与简历的匹配分析
- 不要输出"你的简历存在问题"之类的话
- 不要输出"建议补充"之类的话
- 不要输出任何关于求职者个人的评价

你看到的输入只有 JD 和求职目标，没有简历。你的任务仅仅是解析 JD，仅此而已。

如果你发现自己正在写任何关于简历的内容，请立即停止并删除。""",

        2: """## 当前任务：第2步 - 诊断原始简历

你是一位资深简历诊断专家。你的**唯一任务**是对原始简历进行诊断。

### 输入数据

{input_data}

### 输出要求（严格按以下模块输出）

**模块一：总体评分**

- 匹配度评分：XX/100
- 一句话诊断结论：

**模块二：10 维度诊断**

每个维度给出：评分（0-10）、问题描述、风险等级（高/中/低）

1. 岗位匹配度：
2. 简历结构：
3. 职业定位：
4. 工作经历表达：
5. AI 产品项目深度：
6. 成果量化：
7. AI 技术关键词覆盖：
8. 差异化亮点：
9. 可信度与面试风险：
10. 最需要优先修改的地方：

**模块三：致命伤清单**

（列出足以让 HR 在 10 秒内 Pass 的问题，每个附诊断+处方）

### 输出边界

你只能输出诊断结果，指出问题。以下行为严格禁止：
- 不要给出优化后的简历内容
- 不要做 JD-简历匹配表
- 不要给出重构策略
- 不要写"建议改为XXX"的具体改写内容
- 只做诊断，不解决问题""",

        3: """## 当前任务：第3步 - 建立 JD-简历匹配表

你是一位招聘匹配分析专家。你的**唯一任务**是建立 JD 要求与简历证据的匹配表。

### 输入数据

{input_data}

### 输出要求

**模块一：匹配总览表**

| JD 要求 | 简历中已有证据（引用原文） | 证据强度 | 是否需要补充 | 补充建议 |
|---|---|---|---|---|

证据强度：强 / 中 / 弱 / 无

**模块二：匹配度热力图**

- 绿色区（匹配度高）：
- 黄色区（勉强匹配）：
- 红色区（完全空白）：

**模块三：关键缺口分析**

（JD 中最重要的 3-5 项要求中，简历完全没有对应的"硬缺口"）

### 输出边界

你只能输出匹配表和分析。以下行为严格禁止：
- 不要重新诊断简历
- 不要给出优化后的简历
- 不要做重构策略""",

        4: """## 当前任务：第4步 - 追问清单

你是一位面试辅导专家。你的**唯一任务**是根据简历与 JD 的差距提出追问。

### 输入数据

{input_data}

### 输出要求

**核心追问（5-10 个）**

每个追问包含：
- **问题**：（具体、可回答）
- **为什么问**：（这个信息对简历优化或面试的重要性）
- **理想回答要素**：（一个好的回答应包含哪些信息）

**追问优先级**

- 必须回答：
- 最好有：

### 输出边界

你只能输出追问。以下行为严格禁止：
- 不要给出优化建议
- 不要输出简历内容
- 每个追问要具体到能让用户直接回答""",

        5: """## 当前任务：第5步 - 重构简历策略

你是一位简历策略顾问。你的**唯一任务**是制定简历重构策略。

### 输入数据

{input_data}

### 输出要求

**模块一：职业定位策略**

（明确简历应突出的核心定位 + 理由）

**模块二：经历排序策略**

（最优排列顺序 + 排序逻辑）

**模块三：关键词嵌入策略**

（应嵌入的 AI 关键词 + 应出现在哪个模块）

**模块四：代表项目选择**

（选出 1-2 个代表项目 + 选择理由）

**模块五：差异化策略**

（用户相对于同类候选人的差异化优势 + 如何放大）

**模块六：风险提示**

（重构中需要注意的风险点）

### 输出边界

你只能给策略方向。以下行为严格禁止：
- 不要输出优化后的简历内容
- 不要逐段改写经历
- 不写具体简历文字""",

        6: """## 当前任务：第6步 - 逐段优化简历内容

你是一位资深简历优化专家。你的**唯一任务**是对简历逐段进行优化改写。

### 输入数据

{input_data}

### 输出要求

**逐段优化对照表**

| 模块 | 修改前（原文引用） | 修改后（优化版本） | 修改理由 |
|---|---|---|---|

优化原则：
1. "动作 + 对象 + 方法 + 结果"结构
2. 删除空话
3. AI 技术描述具体化
4. 没有数据时不编造
5. 不把"参与"强行改成"主导"
6. 每段末尾标注体现的 AI PM 核心能力

信息不足处用【待补充】标记。

### 输出边界

你只能输出逐段优化对照表。以下行为严格禁止：
- 不要输出完整简历（那是第 7 步的工作）
- 不要做面试风险分析""",

        7: """## 当前任务：第7步 - 输出最终优化版简历

你是一位专业简历撰写专家。你的**唯一任务**是输出一版完整的优化后简历。

### 输入数据

{input_data}

### 输出要求

按以下结构输出完整简历：

1. 个人信息
2. 求职意向
3. 职业摘要（3-4 句）
4. 核心能力（AI 技术能力 + 产品能力）
5. 工作经历
6. AI 产品项目经历（重点展示）
7. 技能工具
8. 教育背景
9. 其他加分项

写作标准：专业、简洁、有信息密度、突出 AI 产品能力、保持真实可信。
所有【待补充】标记保留。

### 输出边界

你只能输出最终简历。以下行为严格禁止：
- 不要做面试风险分析
- 不要给补强建议""",

        8: """## 当前任务：第8步 - 面试风险提示

你是一位面试辅导专家。你的**唯一任务**是进行面试风险预判。

### 输入数据

{input_data}

### 输出要求

**模块一：高频追问 Top 10**

每个问题包含：问题 + 面试官想听什么 + 准备建议

**模块二：表述风险清单**

- 高风险：
- 中风险：
- 低风险：

**模块三：数据/项目细节补充清单**

（按紧急程度排序）

**模块四：回答策略建议**

（针对 Top 3 最高风险问题的完整回答框架）

### 输出边界

你只能做面试风险分析。以下行为严格禁止：
- 不要修改简历
- 不要给补强建议（作品集、额外项目）""",

        9: """## 当前任务：第9步 - 进一步补强建议

你是一位职业规划顾问。你的**唯一任务**是给出进一步提升竞争力的建议。

### 输入数据

{input_data}

### 输出要求

**模块一：项目补强建议**

（2-3 个项目方向，每个含：方向 + 为什么能提高命中率 + 预计投入时间 + 预期产出）

**模块二：作品集建议**

（是否需要 + 应包含什么 + 呈现方式）

**模块三：简历版本策略**

（可衍生版本 + 每个版本突出什么 + 优先投递建议）

**模块四：长期竞争力提升**

（3-6 个月路线图：技术方向 + 关注产品 + 行业认知）

### 输出边界

你只能给补强建议和规划。以下行为严格禁止：
- 不要修改简历
- 不要做面试风险分析""",
    }


    def _build_messages(self, step: int, user_input: str, rag_context: str = "") -> List[Dict[str, str]]:
        """构建 LLM 消息列表"""
        system_prompt = build_system_prompt(rag_context)

        # 构建用户消息
        user_msg_parts = []

        # 加入之前的步骤输出作为上下文（摘要形式，避免模型重复执行）
        if self.step_results:
            user_msg_parts.append("## 之前步骤的摘要（仅供参考，不要重复输出）")
            for step_num in sorted(self.step_results.keys()):
                sr = self.step_results[step_num]
                # 大幅截断，只保留前 500 字作为上下文提示
                content = sr.content
                if len(content) > 500:
                    content = content[:500] + "\n...（省略）"
                user_msg_parts.append(f"\n### 第{step_num}步已完成：{sr.title}\n{content}")

        # 加入当前步骤的详细指令
        step_instruction = self.STEP_INSTRUCTIONS.get(step, f"## 当前任务：第{step}步 - {self.STEP_TITLES.get(step, chr(39)+chr(39))}")
        step_instruction = step_instruction.replace("{input_data}", user_input)
        user_msg_parts.append(f"\n\n---\n\n{step_instruction}")

        user_msg = "\n".join(user_msg_parts)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]


    def _truncate_to_step(self, step: int, text: str) -> str:
        """按步骤截断模型越界输出的内容"""
        # 定义每步的"终止标记"——模型如果开始写下一步的内容，从这里截断
        step_stop_markers = {
            1: [
                "二、简历逐项诊断", "二、简历诊断", "二、现有简历诊断",
                "三、改写方案", "三、逐项优化建议", "三、逐段优化",
                "四、必须补充", "四、与目标岗位",
                "五、容易被面试官", "五、行动建议",
                "六、方法论引用", "六、总结", "六、优先级排序",
                "七、最终结论", "七、总结",
                "简历诊断与优化报告", "诊断原始简历", "简历整体诊断",
                "你的简历目前", "你的简历存在", "你的简历底子",
                "改写后版本", "修改后版本", "优化后版本",
            ],
            2: [
                "三、建立", "三、JD-简历匹配", "三、匹配表",
                "四、追问", "五、重构", "六、逐段",
                "改写方案", "优化建议", "修改后",
            ],
            3: [
                "四、追问", "五、重构策略", "六、逐段优化",
            ],
            4: [
                "五、重构策略", "六、逐段优化", "七、最终简历",
            ],
            5: [
                "六、逐段优化", "七、最终简历", "八、面试风险",
            ],
            6: [
                "七、最终简历", "八、面试风险", "九、补强",
            ],
            7: [
                "八、面试风险", "九、补强建议",
            ],
            8: [
                "九、补强建议", "十、",
            ],
        }

        markers = step_stop_markers.get(step, [])
        if not markers:
            return text

        # 找到最早的终止标记位置
        earliest_pos = len(text)
        for marker in markers:
            pos = text.find(marker)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos

        if earliest_pos < len(text):
            # 截断到终止标记之前，去掉末尾的 --- 分隔符
            truncated = text[:earliest_pos].rstrip()
            # 去掉末尾的 --- 或 ---
            while truncated.endswith("---") or truncated.endswith("———"):
                truncated = truncated[:-3].rstrip()
            return truncated

        return text

    def run_step(self, step: int, user_input: str, stream: bool = True) -> StepResult:
        """
        执行单步

        Args:
            step: 步骤编号（1-9）
            user_input: 用户输入（如 JD、简历、追问回答等）
            stream: 是否流式输出

        Returns:
            StepResult
        """
        title = self.STEP_TITLES.get(step, f"步骤{step}")
        print(f"\n{'=' * 60}")
        print(f"📋 第{step}步：{title}")
        print(f"{'=' * 60}")

        # 1. RAG 检索
        rag_context = ""
        rag_sources = []
        if self.use_rag:
            print("\n🔍 正在检索 AI PM 课程知识库...")
            rag_context, rag_sources = self._search_rag(user_input)
            if rag_sources:
                print(f"   找到 {len(rag_sources)} 条相关知识：")
                for s in rag_sources:
                    print(f"   - {s}")
            else:
                print("   未找到相关知识")

        # 2. 构建消息
        messages = self._build_messages(step, user_input, rag_context)

        # 3. 调用 LLM
        print(f"\n 正在调用 DeepSeek ({self.llm.model})...\n")

        if stream:
            full_response = ""
            for chunk in self.llm.chat_stream(messages, temperature=0.7, max_tokens=8192):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()  # 换行
        else:
            full_response = self.llm.chat(messages, temperature=0.7, max_tokens=8192)
            print(full_response)

        # 4. 后处理：按步骤截断越界内容
        full_response = self._truncate_to_step(step, full_response)

        # 5. 保存结果
        result = StepResult(
            step=step,
            title=title,
            content=full_response,
            rag_sources=rag_sources,
        )
        self.step_results[step] = result

        return result

    def run_full(self, jd: str, resume: str, job_target: str = "", stream: bool = True) -> Dict[int, StepResult]:
        """
        执行完整 9 步流程

        Args:
            jd: 目标岗位 JD
            resume: 原始简历
            job_target: 求职目标描述
            stream: 是否流式输出

        Returns:
            所有步骤的结果字典
        """
        # 构建初始输入
        initial_input = f"""## 1. 我的求职目标
{job_target or 'AI 产品经理'}

## 2. 目标岗位 JD
{jd}

## 3. 我的原始简历
{resume}"""

        # 第 1-3 步：分析阶段（基于 JD + 简历）
        for step in range(1, 4):
            self.run_step(step, initial_input, stream=stream)

            # 步骤间暂停，让用户确认
            if step < 3:
                print(f"\n{'─' * 60}")
                try:
                    user_confirm = input(f"✅ 第{step}步完成。按 Enter 继续第{step + 1}步，输入 'skip' 跳过后续步骤: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if user_confirm.lower() == "skip":
                    break

        # 第 4 步：追问（需要用户回答）
        if 4 not in self.step_results:
            print(f"\n{'─' * 60}")
            print("📋 第4步：追问清单")
            print(f"{'─' * 60}")
            self.run_step(4, initial_input, stream=stream)

            # 等待用户回答追问
            print(f"\n{'─' * 60}")
            try:
                user_answers = input("请回答上述追问（直接粘贴回答，或按 Enter 跳过）: ").strip()
            except (EOFError, KeyboardInterrupt):
                user_answers = ""

            if user_answers:
                # 将用户回答作为第 5 步的输入
                step5_input = f"## 我对追问的回答\n{user_answers}"
            else:
                step5_input = "用户未回答追问，请基于现有信息做保守优化。"

            # 第 5-9 步：优化阶段
            for step in range(5, 10):
                print(f"\n{'─' * 60}")
                try:
                    user_confirm = input(f"按 Enter 继续第{step}步，输入 'skip' 跳过: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if user_confirm.lower() == "skip":
                    break

                self.run_step(step, step5_input if step == 5 else "请继续下一步", stream=stream)

        return self.step_results

    def get_all_results_text(self) -> str:
        """获取所有步骤结果的完整文本"""
        parts = []
        for step in sorted(self.step_results.keys()):
            sr = self.step_results[step]
            parts.append(f"# 第{step}步：{sr.title}\n\n{sr.content}")
            if sr.rag_sources:
                parts.append(f"\n**参考知识来源：**\n" + "\n".join(f"- {s}" for s in sr.rag_sources))
            parts.append("\n---\n")
        return "\n".join(parts)

    def export_results(self, filepath: str):
        """导出所有结果到文件"""
        content = self.get_all_results_text()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n📄 结果已导出到: {filepath}")
