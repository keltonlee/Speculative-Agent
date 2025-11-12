"""LLM adapter using proper LangGraph tool calling pattern."""
from typing import List
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .models import Msg
from .config import config


# System prompt
SYSTEM_PROMPT = (
    "You are an AI assistant that helps answer questions by using tools.\n"
    "\n## Available Tools:\n"
    "\n**Search & Web:**\n"
    "- search_web: Quick web search (3 results default, 10 with expand_search=True)\n"
    "\n**Files:**\n"
    "- file_read: Read files (supports CSV, XLSX, PDF, DOCX, TXT, JSON, YAML, XML, HTML)\n"
    "\n**Computation:**\n"
    "- calculate: Evaluate math expressions (use Python syntax like 2**10, math.sqrt(16))\n"
    "- code_exec: Execute Python code in a sandbox\n"
    "- code_generate: Generate Python code for a task\n"
    "\n**Vision:**\n"
    "- vision_analyze: Analyze images\n"
    "- vision_ocr: Extract text from images\n"
    "\n## Important Rules:\n"
    "1. Use tools when you need information (search, calculate, read files, etc.)\n"
    "2. After getting tool results, review them and decide if you have enough information\n"
    "3. When you have sufficient information to answer, YOU MUST provide the final answer\n"
    "4. Format your final answer EXACTLY as: FINAL ANSWER: <your answer here>\n"
    "5. Do NOT continue asking for more information if you already have the answer\n"
    "6. Be concise and accurate in your responses\n"
)


# Cache for models (initialized on first use, after .env is loaded)
_actor_model = None
_spec_model = None


def get_actor_model():
    """Get the actor model with tools bound (lazy initialization)."""
    global _actor_model
    if _actor_model is None:
        from .tools_langchain import ALL_TOOLS
        
        model = init_chat_model(config.actor_model, model_provider=config.model_provider)
        _actor_model = model.bind_tools(ALL_TOOLS)
    return _actor_model


def get_spec_model():
    """Get the speculator model with tools bound (lazy initialization)."""
    global _spec_model
    if _spec_model is None:
        from .tools_langchain import READ_ONLY_TOOLS
        
        model = init_chat_model(config.spec_model, model_provider=config.model_provider)
        _spec_model = model.bind_tools(READ_ONLY_TOOLS)
    return _spec_model


def convert_msg_to_langchain(messages: List[Msg]) -> List:
    """Convert our Msg objects to LangChain messages."""
    lc_messages = []
    for m in messages:
        if m.role == "system":
            lc_messages.append(SystemMessage(content=m.content))
        elif m.role == "user":
            lc_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=m.content))
        elif m.role == "tool":
            # Convert tool results to user messages for simplicity
            tool_result = f"[TOOL: {m.name}]\n{m.content}"
            lc_messages.append(HumanMessage(content=tool_result))
    return lc_messages
