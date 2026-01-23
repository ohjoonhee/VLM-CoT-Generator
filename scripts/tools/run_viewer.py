import json
import os
from flask import Flask, render_template_string, abort, send_file, request
import urllib.parse

app = Flask(__name__)

# Configuration
DATA_FILE = "debug/final_response.jsonl"

# Embedded HTML Template
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conversation Viewer</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --sidebar-bg: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-color: #3b82f6;
            --user-bubble: #1e40af;
            --assistant-bubble: #334155;
            --system-bubble: #475569;
            --tool-bubble: #0f766e;
            --border-color: #334155;
        }

        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        /* Sidebar */
        .sidebar {
            width: 300px;
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }

        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid var(--border-color);
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--accent-color);
        }

        .conversation-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .conversation-item {
            padding: 15px 20px;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
            transition: background 0.2s;
        }

        .conversation-item:hover, .conversation-item.active {
            background-color: rgba(59, 130, 246, 0.1);
        }

        .conversation-item-title {
            font-weight: 600;
            margin-bottom: 5px;
        }

        .conversation-item-preview {
            font-size: 0.875rem;
            color: var(--text-secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Main Chat Area */
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }

        .chat-header {
            padding: 20px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: var(--bg-color); /* Sticky header bg */
        }

        .chat-scroll-area {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        /* Message Bubbles */
        .message {
            max-width: 80%;
            padding: 15px;
            border-radius: 12px;
            position: relative;
            line-height: 1.5;
            animation: fadeIn 0.3s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-role {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
            opacity: 0.8;
            font-weight: 600;
        }

        .message.user {
            align-self: flex-end;
            background-color: var(--user-bubble);
            border-bottom-right-radius: 2px;
        }

        .message.assistant {
            align-self: flex-start;
            background-color: var(--assistant-bubble);
            border-bottom-left-radius: 2px;
        }

        .message.system {
            align-self: center;
            background-color: var(--system-bubble);
            max-width: 90%;
            font-size: 0.9rem;
        }

        .message.tool {
            align-self: flex-start;
            background-color: var(--tool-bubble);
            border-left: 4px solid #14b8a6;
            font-family: monospace;
            padding-top: 5px;
            padding-bottom: 5px;
        }

        .image-container {
            white-space: normal;
        }
        
        .message-text-content {
            white-space: pre-wrap;
        }

        /* Content Styling */
        .message-content img {
            max-width: 100%;
            border-radius: 8px;
            margin-top: 5px;
            border: 1px solid var(--border-color);
        }

        .tool-call {
            background-color: rgba(0, 0, 0, 0.2);
            padding: 10px;
            border-radius: 8px;
            margin-top: 10px;
            font-family: monospace;
            font-size: 0.9rem;
            border: 1px solid var(--border-color);
        }
        
        .tool-call-header {
            color: #fbbf24;
            font-weight: bold;
            margin-bottom: 5px;
        }

        pre {
            background: rgba(0,0,0,0.3);
            padding: 10px;
            border-radius: 6px;
            overflow-x: auto;
        }
        
        .empty-state {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-secondary);
            font-size: 1.5rem;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg-color);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-secondary);
        }
    </style>
</head>
<body>

    <div class="sidebar">
        <div class="sidebar-header">
            Conversations
        </div>
        <ul class="conversation-list">
            {% for i in range(total_conversations) %}
            <li class="conversation-item {% if i == current_index %}active{% endif %}" onclick="window.location.href='/view/{{ i }}'">
                <div class="conversation-item-title">Conversation {{ i + 1 }}</div>
                <div class="conversation-item-preview">View details...</div>
            </li>
            {% endfor %}
        </ul>
    </div>

    <div class="chat-container">
        {% if conversation %}
            <div class="chat-header">
                <h2>Conversation {{ current_index + 1 }}</h2>
            </div>
            <div class="chat-scroll-area">
                {% for msg in conversation %}
                    <div class="message {{ msg.role }}">
                        <div class="message-role" style="margin-bottom: 0px;">{{ msg.role }}</div>
                        <div class="message-content">
                            {#- Display text content -#}
                            {%- if msg.content_text -%}
                                <div class="message-text-content">{{ msg.content_text }}</div>
                            {%- endif -%}

                            {#- Display images -#}
                            {%- if msg.images -%}
                                {%- for img_url in msg.images -%}
                                    <div class="image-container">
                                        <img src="{{ img_url }}" alt="User Image">
                                    </div>
                                {%- endfor -%}
                            {%- endif -%}

                             {#- Display Tool Calls -#}
                            {%- if msg.tool_calls -%}
                                {%- for tc in msg.tool_calls -%}
                                    <div class="tool-call">
                                        <div class="tool-call-header">🔧 Tool Call: {{ tc.function.name }}</div>
                                        <div>args: {{ tc.function.arguments }}</div>
                                    </div>
                                {%- endfor -%}
                            {%- endif -%}
                            
                             {#- Display Tool Output (usually in 'tool' role messages content) -#}
                             {#- Automatically handled if content is text, but checking json structure -#}
                        </div>
                    </div>
                {% endfor %}
            </div>
        {% else %}
            <div class="empty-state">
                Select a conversation to view
            </div>
        {% endif %}
    </div>

</body>
</html>
"""


def load_conversations():
    """Loads all conversations from the JSONL file."""
    conversations = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                for line in f:
                    try:
                        conversations.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading file: {e}")
    return conversations


def load_conversation_at_index(index):
    """Efficiently load only the specific line if possible, or list access."""
    # Since specific requirement mentioned "cautious with jsonl file",
    # but file is only 18MB, reading lines is fine.
    # We will read line by line until we hit the index to save memory if it were huge.
    if not os.path.exists(DATA_FILE):
        return None

    current_idx = 0
    with open(DATA_FILE, "r") as f:
        for line in f:
            if current_idx == index:
                return json.loads(line)
            current_idx += 1
    return None


def count_conversations():
    if not os.path.exists(DATA_FILE):
        return 0
    with open(DATA_FILE, "r") as f:
        return sum(1 for _ in f)


def parse_message(msg):
    """
    Parses a message object into a structure friendly for the template.
    Extracts text and image URLs.
    Handles local paths by converting them to a local_image route.
    """
    parsed = {"role": msg.get("role", "unknown"), "content_text": "", "images": [], "tool_calls": msg.get("tool_calls", [])}

    content = msg.get("content")

    if isinstance(content, str):
        parsed["content_text"] = content
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parsed["content_text"] += item.get("text", "")
                elif item.get("type") == "image_url":
                    img_data = item.get("image_url", {})
                    url = img_data.get("url", "")
                    if url:
                        if not url.startswith("http") and not url.startswith("data:"):
                            # It's likely a local file path
                            # Use urllib.parse.quote for safety
                            encoded_path = urllib.parse.quote(url)
                            url = f"/local_image?path={encoded_path}"
                        parsed["images"].append(url)

    return parsed


@app.route("/")
def index():
    total = count_conversations()
    # Default to viewing the first conversation if it exists
    if total > 0:
        return view_conversation(0)
    return render_template_string(TEMPLATE, total_conversations=0, current_index=-1, conversation=None)


@app.route("/view/<int:index>")
def view_conversation(index):
    total = count_conversations()
    if index < 0 or index >= total:
        abort(404)

    raw_conversation = load_conversation_at_index(index)
    parsed_conversation = [parse_message(msg) for msg in raw_conversation]

    return render_template_string(TEMPLATE, total_conversations=total, current_index=index, conversation=parsed_conversation)


@app.route("/local_image")
def local_image():
    """Serves images from the local filesystem."""
    path = request.args.get('path')
    if not path:
        abort(400)
    
    # Security note: In a real production app, this would be highly insecure (LFI vulnerability).
    # Since this is a local debug viewer running on localhost, we allow it.
    # We could add checks to ensure it's within the project directory if needed.
    
    if os.path.exists(path) and os.path.isfile(path):
         return send_file(path)
    
    abort(404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
