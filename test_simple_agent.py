from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent  # type: ignore[reportDeprecated]
import time
import os
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

# Fix for langchain version compatibility issue
import langchain
if not hasattr(langchain, 'verbose'):
    langchain.verbose = False
if not hasattr(langchain, 'debug'):
    langchain.debug = False
if not hasattr(langchain, 'llm_cache'):
    langchain.llm_cache = None

# Import tools registry from speculation_eval
import sys
sys.path.insert(0, 'speculation_eval')

try:
    from speculation_eval.tools_registry import get_all_available_tools
    REGISTRY_AVAILABLE = True
except ImportError:
    print("Warning: tools_registry not available, using fallback")
    REGISTRY_AVAILABLE = False


# ==================== Create Agent with LangGraph ====================

def create_gemini_agent():
    """Create a ReAct agent using LangGraph with all available search tools"""

    # Get all available tools from registry
    if REGISTRY_AVAILABLE:
        tools = get_all_available_tools(verbose=True)
    else:
        # Fallback: just use a simple calculator
        @tool
        def calculator(expression: str) -> str:
            """Perform calculations."""
            try:
                result = eval(expression, {"__builtins__": {}}, {})
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"
        tools = [calculator]
        print("⚠️  Using minimal toolset (calculator only)")

    if len(tools) == 0:
        print("❌ No tools available!")
        return None

    # Initialize LLM with Gemini
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )

    agent = create_react_agent(model, tools)

    return agent


# ==================== Test the Agent ====================

def test_agent():
    """Test the Gemini agent with real tools"""

    # Create agent
    agent = create_gemini_agent()

    if not agent:
        print("❌ Failed to create agent!")
        return

    # Test queries - diverse to exercise different tool types
    test_queries = [
        "What is 127 multiplied by 34?",                          # Calculator
        "Search for recent news about quantum computing",         # General search tools
        "Find academic papers about neural networks on arxiv",    # Arxiv (if available)
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {query}")
        print(f"{'='*80}\n")

        try:
            # Invoke the agent
            result = agent.invoke({"messages": [("user", query)]})

            print(f"\n{'='*80}")
            print(f"AGENT EXECUTION TRACE:")
            print(f"{'='*80}")

            # Show all messages in the conversation
            messages = result.get("messages", [])
            for j, msg in enumerate(messages):
                print(f"\n--- Step {j+1} ---")
                msg_type = type(msg).__name__
                print(f"Type: {msg_type}")

                if hasattr(msg, 'content') and msg.content:
                    content = str(msg.content)
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"Content: {content}")

                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"Tool Calls:")
                    for tc in msg.tool_calls:
                        print(f"  - {tc['name']}({tc['args']})")

                if hasattr(msg, 'name') and msg.name:
                    print(f"Tool: {msg.name}")

            # Extract final answer
            final_msg = messages[-1]
            print(f"\n{'='*80}")
            print(f"FINAL ANSWER:")
            print(f"{'='*80}")
            final_content = str(final_msg.content)
            if len(final_content) > 500:
                final_content = final_content[:500] + "..."
            print(final_content)
            print(f"{'='*80}\n")

            time.sleep(2)

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print("\nContinuing to next query...\n")


# ==================== Main ====================

if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("🤖 GEMINI AGENT WITH REAL EXA/TAVILY TOOLS")
        print("="*80)

        test_agent()

        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED!")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
