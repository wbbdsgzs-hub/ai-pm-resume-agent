"""
AI PM 求职 Agent — Flask 后端（逐步骤 + 多轮对话版）

每个步骤执行完后，用户可以：
1. 在聊天框中追问、补充信息、多轮对话
2. 点击"进入下一步"继续
"""
import os
import sys
import json
import socket
import tempfile

# Windows 控制台默认 GBK，无法打印 emoji，强制 stdout/stderr 使用 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.workflow import AIPMAgent
from agent.llm import DeepSeekLLM
from agent.prompt import CAPACITY_AND_ROLE, PERSONALITY

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max
CORS(app)

agent = None
llm = None

# 每步的对话历史 {step: [{"role": "user/assistant", "content": "..."}]}
chat_histories = {}

# 会话级文件存储 {session_id: {file_id: {filename, text}}}
session_files = {}
import uuid

STEP_TITLES = {
    1: "JD 解析", 2: "简历诊断", 3: "匹配分析",
    4: "经历追问", 5: "优化策略", 6: "逐段优化",
    7: "最终简历", 8: "面试风险", 9: "补强建议",
}


def extract_text_from_file(filepath, filename):
    """从上传的文件中提取文本内容"""
    ext = filename.lower().split('.')[-1]

    if ext == 'pdf':
        import pdfplumber
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    elif ext in ('docx', 'doc'):
        import docx
        doc = docx.Document(filepath)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return text

    elif ext == 'md':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    elif ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    else:
        return None


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传文件，后端解析后存储，只返回文件名和ID（不返回内容给前端）"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': '文件名为空'}), 400

    filename = file.filename
    ext = filename.lower().split('.')[-1]

    if ext not in ('pdf', 'docx', 'doc', 'md', 'txt'):
        return jsonify({'error': '仅支持 PDF、Word（.docx/.doc）、Markdown（.md）、TXT 文件'}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + ext) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        text = extract_text_from_file(tmp_path, filename)
        os.unlink(tmp_path)

        if text is None:
            return jsonify({'error': '无法解析该文件格式'}), 400

        if not text.strip():
            return jsonify({'error': '文件中没有提取到文本内容'}), 400

        # 存储到会话级文件库（前端只拿到 ID 和文件名）
        file_id = str(uuid.uuid4())[:8]
        session_files[file_id] = {'filename': filename, 'text': text}

        return jsonify({
            'status': 'ok',
            'file_id': file_id,
            'filename': filename,
            'char_count': len(text),
        })
    except Exception as e:
        return jsonify({'error': f'文件解析失败：{str(e)}'}), 500


@app.route('/')
def index():
    return send_from_directory('.', 'web_ui.html')


@app.route('/api/start', methods=['POST'])
def start_session():
    """创建 Agent 会话"""
    global agent, llm, chat_histories
    data = request.json
    job_target = data.get('job_target', '')
    jd = data.get('jd', '')
    resume = data.get('resume', '')
    resume_file_ids = data.get('resume_file_ids', [])
    use_rag = data.get('use_rag', True)
    rag_top_k = data.get('rag_top_k', 3)

    if not jd:
        return jsonify({'error': 'JD 是必填项'}), 400
    if not resume and not resume_file_ids:
        return jsonify({'error': '请粘贴简历内容或上传简历文件'}), 400

    # 注入上传文件的解析内容到简历中
    resume_extra = ""
    for fid in resume_file_ids:
        if fid in session_files:
            f = session_files[fid]
            resume_extra += f"\n\n---\n\n【附件文件：{f['filename']}】\n{f['text']}"

    agent = AIPMAgent(use_rag=use_rag, rag_top_k=rag_top_k)
    agent._job_target = job_target
    agent._jd = jd
    agent._resume = resume + resume_extra
    llm = DeepSeekLLM()
    chat_histories = {}

    return jsonify({'status': 'ok'})


@app.route('/api/run-step/<int:step>', methods=['POST'])
def run_step(step):
    """执行单个步骤，流式返回结果"""
    global agent
    if agent is None:
        return jsonify({'error': '请先调用 /api/start'}), 400
    if step < 1 or step > 9:
        return jsonify({'error': '步骤编号必须在 1-9 之间'}), 400

    # 按步骤传入不同的数据，避免模型"越界"
    if step == 1:
        # Step 1 只看 JD，不看简历
        user_input = f"求职目标：{agent._job_target}\n\n目标岗位 JD：\n{agent._jd}"
    else:
        # Step 2+ 看 JD + 简历
        user_input = f"求职目标：{agent._job_target}\n\n目标岗位 JD：\n{agent._jd}\n\n原始简历：\n{agent._resume}"

    def generate():
        try:
            result = agent.run_step(step, user_input, stream=False)
            chunk = json.dumps({
                'type': 'step_complete',
                'step': result.step,
                'title': result.title,
                'content': result.content,
                'rag_sources': result.rag_sources,
            }, ensure_ascii=False)
            yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'step': step, 'error': str(e)}, ensure_ascii=False)}\n\n"

        yield "data: [STEP_DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/chat/<int:step>', methods=['POST'])
def chat_step(step):
    """在某个步骤内多轮对话，流式返回"""
    global agent, llm
    if agent is None or llm is None:
        return jsonify({'error': '会话不存在'}), 400

    data = request.json
    user_message = data.get('message', '')
    chat_file_ids = data.get('file_ids', [])
    if not user_message:
        return jsonify({'error': '消息不能为空'}), 400

    # 注入上传文件内容到消息中
    for fid in chat_file_ids:
        if fid in session_files:
            f = session_files[fid]
            user_message = f"【附件文件：{f['filename']}】\n{f['text']}\n\n---\n\n{user_message}"

    # 获取该步骤的对话历史
    history = chat_histories.get(step, [])

    # 构建消息：系统提示 + 该步骤的结果 + 对话历史 + 当前消息
    step_result = ""
    if step in agent.step_results:
        step_result = agent.step_results[step].content

    system_msg = f"""{CAPACITY_AND_ROLE}

{PERSONALITY}

## 当前步骤：第{step}步 {STEP_TITLES.get(step, '')}

以下是你刚才对第{step}步的分析结果：
{step_result}

请基于以上分析结果，回答用户的问题。如果用户提供了补充信息，请更新你的分析。
保持专业、直接、严格的风格。"""

    messages = [{"role": "system", "content": system_msg}]

    # 添加对话历史
    for msg in history:
        messages.append(msg)

    # 添加当前用户消息
    messages.append({"role": "user", "content": user_message})

    # 保存到历史
    history.append({"role": "user", "content": user_message})

    def generate():
        try:
            full_response = ""
            for chunk_text in llm.chat_stream(messages, temperature=0.7, max_tokens=8192):
                full_response += chunk_text
                yield f"data: {json.dumps({'type': 'chat_chunk', 'content': chunk_text}, ensure_ascii=False)}\n\n"

            history.append({"role": "assistant", "content": full_response})
            chat_histories[step] = history

            yield f"data: {json.dumps({'type': 'chat_done', 'content': full_response}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/next-step', methods=['POST'])
def next_step():
    """用户确认进入下一步"""
    global agent
    if agent is None:
        return jsonify({'error': '会话不存在'}), 400

    data = request.json
    current_step = data.get('current_step', 0)
    next_step_num = current_step + 1

    if next_step_num > 9:
        return jsonify({'status': 'all_done', 'message': '全部步骤已完成'})

    return jsonify({'status': 'ok', 'next_step': next_step_num, 'title': STEP_TITLES.get(next_step_num, '')})


@app.route('/api/export', methods=['GET'])
def export_results():
    global agent
    if agent is None or not agent.step_results:
        return jsonify({'error': '没有可导出的结果'}), 400

    content = agent.get_all_results_text()

    # 附加对话历史
    if chat_histories:
        content += "\n\n---\n\n## 对话记录\n\n"
        for step in sorted(chat_histories.keys()):
            content += f"\n### 第{step}步对话\n\n"
            for msg in chat_histories[step]:
                role = "用户" if msg["role"] == "user" else "Agent"
                content += f"**{role}**: {msg['content']}\n\n"

    return jsonify({'status': 'ok', 'content': content})




# ===== AI PM 知识问答相关 =====
qa_chat_histories = {}  # {session_id: [messages]}

@app.route('/api/qa-chat', methods=['POST'])
def qa_chat():
    """AI PM 知识问答对话"""
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', '')
    use_rag = data.get('use_rag', True)

    if not message:
        return jsonify({'error': '消息不能为空'}), 400

    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())[:8]

    # 构建消息
    rag_context = ""
    if use_rag:
        try:
            from query import query_rag
            results = query_rag(message, n_results=3)
            if results:
                parts = []
                for r in results:
                    parts.append(f"[{r.course_id}课] {r.section_path}\n{r.content}")
                rag_context = "\n\n---\n\n".join(parts)
        except Exception as e:
            print(f"RAG 检索失败: {e}")

    from agent.prompt import QA_CAPACITY_AND_ROLE, QA_PERSONALITY
    system_msg = QA_CAPACITY_AND_ROLE + "\n\n---\n\n" + QA_PERSONALITY
    if rag_context:
        system_msg += "\n\n---\n\n## 参考知识（来自 AI PM 课程笔记）\n\n" + rag_context

    # 获取历史消息
    history = qa_chat_histories.get(session_id, [])

    messages = [{"role": "system", "content": system_msg}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": message})

    def generate():
        try:
            full_response = ""
            from agent.llm import DeepSeekLLM
            llm = DeepSeekLLM()
            for chunk_text in llm.chat_stream(messages, temperature=0.7, max_tokens=8192):
                full_response += chunk_text
                chunk_json = json.dumps({'type': 'qa_chunk', 'content': chunk_text}, ensure_ascii=False)
                yield f"data: {chunk_json}\n\n"

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": full_response})
            qa_chat_histories[session_id] = history

            done_json = json.dumps({'type': 'qa_done', 'content': full_response, 'session_id': session_id}, ensure_ascii=False)
            yield f"data: {done_json}\n\n"
        except Exception as e:
            err_json = json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)
            yield f"data: {err_json}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/qa-clear', methods=['POST'])
def qa_clear():
    """清空问答会话"""
    data = request.json
    session_id = data.get('session_id', '')
    if session_id in qa_chat_histories:
        del qa_chat_histories[session_id]
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    print(f"AI PM Agent backend: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
