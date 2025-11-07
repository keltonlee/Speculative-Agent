# GAIA Agent with LangGraph & Tool Calling

A powerful ReAct agent system built with **LangGraph** for the GAIA benchmark, featuring native tool calling, comprehensive file support, and multimodal capabilities.

## 🎯 Key Features

- **ReAct Pattern**: Implements Reason → Act → Observe loop for systematic problem-solving
- **8 Specialized Tools**: Web search, file reading (10+ formats), code execution, vision analysis
- **LangGraph Architecture**: Uses LangGraph's native tool calling with `bind_tools()`
- **GPT-5 Support**: Compatible with latest OpenAI models (GPT-5, GPT-4o, GPT-4o-mini)
- **Rich Output**: Detailed execution traces with timing, tool arguments, and results
- **Easy Configuration**: Run with different models via environment variables

## 🏗️ Architecture

```
┌────────────────────────────────────────┐
│         LangGraph Workflow             │
├────────────────────────────────────────┤
│                                        │
│        ┌──────────────┐                │
│    ┌───│  LLM Node    │───┐            │
│    │   │  (Reason)    │   │            │
│    │   └──────────────┘   │            │
│    │                      │            │
│    │   Decides:           │            │
│    │   - Call tools       │            │
│    │   - Think more       │            │
│    │   - Final answer     │            │
│    │                      │            │
│    ▼                      ▼            │
│  ┌──────────────┐    ┌──────────────┐  │
│  │ Tools Node   │    │     END      │  │
│  │ (Observe)    │    │              │  │
│  └──────┬───────┘    └──────────────┘  │
│         │                              │
│         │ Returns results              │
│         └────────┐                     │
│                  ▼                     │
│         Back to LLM Node               │
│                                        │
└────────────────────────────────────────┘
```

## 🛠️ Available Tools (8 Total)

### 🔍 Web Search & Browsing
- **`search_web`** - Quick web search (3 results default, 10 with `expand_search=True`)
  - Uses Google Serper API
  - Returns titles, snippets, and URLs
- **`search_with_content`** - Deep search with full content extraction (1 result default, 3 with `expand_search=True`)
  - Fetches and parses actual webpage content
  - More thorough but slower

### 📄 File Reading (10+ formats)
- **`file_read`** - Intelligent file reader with auto-detection:
  - **Text**: `.txt`, `.md`, `.log`
  - **Data**: `.csv`, `.json`, `.jsonl`, `.yaml`, `.yml`
  - **Spreadsheets**: `.xlsx`, `.xls` (with pandas)
  - **Documents**: `.pdf` (with pdfplumber), `.docx`
  - **Markup**: `.xml`, `.html`

### 💻 Code & Computation
- **`calculate`** - Evaluate mathematical expressions
  - Supports Python syntax: `2**10`, `math.sqrt(16)`
  - Safe execution environment
- **`code_exec`** - Execute Python code in sandbox
  - Timeout protection
  - Isolated execution
- **`code_generate`** - Generate Python code using GPT-4o
  - Takes task description and optional context
  - Returns production-ready code

### 👁️ Vision & Multimodality
- **`vision_analyze`** - Analyze images with vision model
  - Can answer specific questions about images
  - Supports common image formats
- **`vision_ocr`** - Extract text from images
  - OCR functionality
  - Useful for diagrams, screenshots, documents

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file:

```bash
# Required
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key  # For web search

# Optional: Customize models
GAIA_ACTOR_MODEL=gpt-5           # or gpt-4o, gpt-4o-mini
GAIA_SPEC_MODEL=gpt-5-mini       # For future speculation features
GAIA_MAX_STEPS=12                # Max reasoning steps
```

### 3. Download GAIA Dataset

```bash
python download_gaia.py
```

This creates:
```
gaia_dataset/
├── level1/  # Easy questions (165 examples)
│   ├── example_000/
│   │   ├── metadata.json
│   │   └── [attached files]
│   └── ...
├── level2/  # Medium questions
└── level3/  # Hard questions
```

### 4. Run Examples

**Run a specific example:**
```bash
python run_gaia_example.py gaia_dataset/level1/example_000
```

**Run with different models:**
```bash
# With GPT-5
GAIA_ACTOR_MODEL=gpt-5 python run_gaia_example.py gaia_dataset/level3/example_000

# With GPT-4o
GAIA_ACTOR_MODEL=gpt-4o python run_gaia_example.py gaia_dataset/level2/example_005

# With GPT-4o-mini (faster, cheaper)
GAIA_ACTOR_MODEL=gpt-4o-mini python run_gaia_example.py
```

**Run full evaluation:**
```bash
# Evaluate level 1
python main.py --mode eval --level 1

# Evaluate first 5 examples from all levels
python main.py --mode eval --max-examples 5

# Evaluate specific level with limit
python main.py --mode eval --level 2 --max-examples 10
```

## 📊 Example Output

```
================================================================================
GAIA Example: e1fc63a2-da7a-432f-be78-7c4a95598703
Level: 1
================================================================================

Question:
If Eliud Kipchoge could maintain his record-making marathon pace indefinitely...

Ground Truth Answer: 17

================================================================================
Running Actor Model: gpt-5
================================================================================

================================================================================
[Step 1] LLM
================================================================================
⏱️  LLM call: 10.23s
🔧 Decision: Call tool
   Tool: search_web
   Args:
      query = Eliud Kipchoge marathon record pace

================================================================================
[Step 2] TOOLS
================================================================================
⏱️  Execution: 1.52s
📤 Output from 'search_web':
   Found 3 relevant web results:
   1. Eliud Kipchoge - Wikipedia...

================================================================================
[Step 3] LLM
================================================================================
⏱️  LLM call: 9.09s
🔧 Decision: Call tool
   Tool: calculate
   Args:
      expression = (356400 * 1000) / (42195 / (2 * 3600 + 1 * 60 + 39))

...

================================================================================
✅ EXECUTION COMPLETE
================================================================================

Predicted Answer: 17000
Ground Truth:     17

Total Steps: 5
Total Messages: 6
```

## 📁 Project Structure

```
spec_tool_call/
├── spec_tool_call/                # Main package
│   ├── tools/                     # Tool implementations
│   │   ├── search_tool.py         # Web search (Serper API)
│   │   ├── file_tool.py           # Enhanced file reading
│   │   ├── code_exec_tool.py      # Code execution & generation
│   │   └── vision_tool.py         # Image analysis & OCR
│   ├── config.py                  # Configuration from env vars
│   ├── models.py                  # Pydantic models (Msg, RunState)
│   ├── tool_registry.py           # Legacy tool registry
│   ├── tools_langchain.py         # LangChain tool definitions
│   ├── llm_adapter.py             # LLM initialization & prompts
│   ├── graph.py                   # LangGraph workflow (ReAct loop)
│   └── gaia_eval.py               # GAIA dataset & evaluation
├── main.py                        # Main CLI for batch evaluation
├── run_gaia_example.py            # Run single example with rich output
├── download_gaia.py               # Download GAIA dataset
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🔧 Configuration Options

Environment variables (set in `.env` or command line):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `SERPER_API_KEY` | *(required)* | Google Serper API key for search |
| `GAIA_ACTOR_MODEL` | `gpt-5` | Main reasoning model |
| `GAIA_SPEC_MODEL` | `gpt-5-mini` | Speculator model (future use) |
| `GAIA_MAX_STEPS` | `12` | Maximum reasoning steps |
| `GAIA_TOPK` | `3` | Top-k for speculation (future) |
| `GAIA_CONF_TH` | `0.35` | Confidence threshold (future) |
| `DISABLE_SPECULATION` | `0` | Set to `1` to disable speculation |

## 🎓 Code Examples

### Example 1: Basic Usage

```python
from spec_tool_call import build_graph, Msg
from spec_tool_call.models import RunState

# Build the agent graph
app = build_graph()

# Create initial state
state = RunState(messages=[
    Msg(role="user", content="What is the capital of France?")
])

# Run the agent
async for event in app.astream(state):
    for node_name, state in event.items():
        print(f"{node_name}: {state}")
```

### Example 2: With File Reading

```python
state = RunState(messages=[
    Msg(role="user", content="""
    Read the file data.csv and tell me the average of the 'sales' column.
    """)
])

async for event in app.astream(state):
    # Agent will use file_read and calculate tools
    pass
```

### Example 3: Vision Analysis

```python
state = RunState(messages=[
    Msg(role="user", content="""
    Analyze the image chart.png and extract the data points.
    Then calculate the trend.
    """)
])

# Agent will use vision_analyze or vision_ocr, then calculate
```

## 📈 ReAct Pattern Explained

The agent follows a **Reason-Act-Observe** loop:

1. **Reason (LLM Node)**: 
   - Analyzes the question and conversation history
   - Decides whether to call tools, think more, or provide final answer
   - If calling tools, stores tool calls in state

2. **Act & Observe (Tools Node)**:
   - Executes the requested tools
   - Adds results back to conversation history
   - Returns control to LLM node

3. **Loop**:
   - Continues until final answer or max steps reached
   - Each step shows timing and detailed output

**Key Features**:
- One tool execution per reasoning step (prevents runaway calls)
- Full visibility into agent's decision-making
- Timing data for performance analysis
- Graceful error handling with full stack traces

## 🧪 Testing

### Test Individual Tools

```bash
# Test search
python -m spec_tool_call.tools.search_tool

# Test file reading
python -m spec_tool_call.tools.file_tool

# Test code execution
python -m spec_tool_call.tools.code_exec_tool

# Test vision
python -m spec_tool_call.tools.vision_tool
```

### Test Single Example

```bash
# Test level 1 example
python run_gaia_example.py gaia_dataset/level1/example_000

# Test with GPT-4o instead
GAIA_ACTOR_MODEL=gpt-4o python run_gaia_example.py gaia_dataset/level1/example_000
```

## 🐛 Troubleshooting

**"Unsupported value: 'temperature' does not support X with this model"**
- GPT-5 models only support default temperature (1.0)
- This is now handled automatically in the code

**Import errors**:
- Ensure you're running from project root
- Check Python version: Python 3.9+ required

**Missing dependencies**:
```bash
pip install -r requirements.txt
```

**GAIA dataset not found**:
```bash
python download_gaia.py
```

**API key errors**:
- Check your `.env` file has `OPENAI_API_KEY` and `SERPER_API_KEY`
- Make sure `python-dotenv` is installed

## 📚 Key Dependencies

- **LangGraph**: Workflow orchestration
- **LangChain**: LLM abstractions and tool calling
- **OpenAI**: GPT-5, GPT-4o, GPT-4o-mini models
- **httpx**: Async HTTP requests
- **BeautifulSoup4**: HTML parsing
- **pandas**: Data file handling
- **pdfplumber**: PDF reading
- **python-docx**: DOCX reading
- **Pillow**: Image handling

## 🎯 GAIA Benchmark

The [GAIA benchmark](https://huggingface.co/gaia-benchmark/GAIA) tests real-world assistant capabilities:
- **Level 1**: Simple questions requiring 1-2 tools
- **Level 2**: Multi-step reasoning with 3-5 tools
- **Level 3**: Complex questions requiring planning and diverse tools

Our system achieves competitive performance through:
- Comprehensive tool coverage
- Robust file format support
- Effective ReAct reasoning loop
- Error recovery and retry logic

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional file format support
- More specialized tools
- Better error recovery
- Performance optimizations

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built on [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [GAIA benchmark](https://huggingface.co/gaia-benchmark/GAIA)
- Powered by OpenAI models
