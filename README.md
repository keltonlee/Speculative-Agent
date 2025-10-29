# Speculative Agent

Research on Agent Tool Call Speculation - comparing target model (with all tools) and draft model (with simple tools).

## Overview

This project implements a **real LLM agent** using LangChain with native tool calling capabilities. The agent demonstrates how an LLM can dynamically select and call tools based on user queries without manual regex parsing.

## Features

- ✅ **Native Tool Calling**: Uses `ChatOllama` with Llama 3.1 that supports `bind_tools()`
- ✅ **No Manual Parsing**: LangChain's `create_agent()` handles everything automatically
- ✅ **Real Agent Loop**: LLM sees tools, decides which to call, generates structured output, system executes
- ✅ **Fully Transparent**: Shows detailed tool call information at each step

## Architecture

```
User Query
    ↓
LLM Sees Tools (via bind_tools())
    ↓
LLM Generates Tool Calls (structured format)
    ↓
System Executes Tools
    ↓
Results Returned to LLM
    ↓
LLM Generates Final Answer
```

## Setup

### 1. Install Dependencies

```bash
conda activate 256HW  # or your environment
pip install langchain langchain-ollama langchain-core
```

### 2. Install and Run Ollama

```bash
# Start Ollama service
ollama serve

# Pull Llama 3.1 model (supports tool calling)
ollama pull llama3.1
```

### 3. Run the Agent

```bash
python test_simple_agent.py
```

## How It Works

### Tool Definitions

The agent has access to multiple tools:

- **`calculator`**: Performs mathematical calculations
- **`get_current_weather`**: Gets weather information
- **`get_current_time`**: Gets current time
- **`search_information`**: Searches for information

### LangChain Agent Flow

1. **Tool Binding**: `ChatOllama` uses `bind_tools()` to expose tools to the LLM
2. **LLM Decision**: LLM (Llama 3.1) sees available tools and decides which to call
3. **Structured Output**: LLM generates `tool_calls` in structured format
4. **Execution**: LangChain executes the tools automatically
5. **Integration**: Tool results are fed back to LLM
6. **Final Answer**: LLM generates final response based on tool results

### Example Interaction

```
User: "What is 127 multiplied by 34?"

Agent Flow:
1. LLM sees query and available tools
2. LLM decides to call calculator tool
3. Generates: tool_calls=[{'name': 'calculator', 'args': {'expression': '127 * 34'}}]
4. System executes: calculator('127 * 34') → Result: 4318
5. LLM receives result and generates: "The result of multiplying 127 by 34 is 4318."
```

## File Structure

```
.
├── README.md
├── test_simple_agent.py      # Main agent implementation
├── requirements.txt          # Python dependencies
└── AGENT_DEMO_SUCCESS.md    # Detailed execution logs
```

## Tool Call Details

The agent shows detailed information for each step:

- **Human Query**: The user's question
- **AIMessage with tool_calls**: LLM's decision to call a tool
- **ToolMessage**: Result from tool execution
- **AIMessage with content**: Final answer from LLM

## For Speculative Decoding Research

This framework can be extended to test:
- **Draft Model**: Only sees simple/atomic tools
- **Target Model**: Sees complex tools (combinations of simple tools)
- **Verification**: Check if draft's tool sequence equals target's combined call

## Requirements

- Python 3.12+
- LangChain
- LangChain Ollama
- Ollama + Llama 3.1 (8B)
- Conda environment

## Key Differences from Previous Approaches

### ❌ Old Approach (Manual Parsing)
- Used regex to parse LLM text output
- Required manual tool call extraction
- Fragile and error-prone
- Example: `real_agent_tool_calling.py`

### ✅ Current Approach (Native Tool Calling)
- Uses `bind_tools()` method
- Structured tool calls automatically parsed
- Fully automatic tool execution
- Example: `test_simple_agent.py`

## References

- [LangChain Agents Documentation](https://python.langchain.com/docs/modules/agents/)
- [LangChain Ollama Integration](https://python.langchain.com/docs/integrations/chat/ollama/)
- [Llama 3.1 Models](https://llama.meta.com/llama3.1/)

## Author

Kelton Lee - Research on Agent Tool Call Speculation

