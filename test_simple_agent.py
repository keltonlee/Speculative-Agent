from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent # type: ignore[reportDeprecated]
import time
import os

# Fix for langchain version compatibility issue
import langchain
if not hasattr(langchain, 'verbose'):
    langchain.verbose = False
if not hasattr(langchain, 'debug'):
    langchain.debug = False
if not hasattr(langchain, 'llm_cache'):
    langchain.llm_cache = None

# ==================== Define Tools ====================

@tool
def search_information(query: str) -> str:
    """Search for information on the internet. Use this when you need to find information about any topic.
    
    Args:
        query: The search query string
    """
    print(f"\n🔧 [TOOL CALLED] search_information(query='{query}')")
    time.sleep(0.5)
    return f"Search results for '{query}': This topic involves multiple aspects including definitions, applications, and current trends in the field."


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations. Use this for any arithmetic operations.
    
    Args:
        expression: A mathematical expression to evaluate, e.g., '25 * 4' or '(10 + 5) / 3'
    """
    print(f"\n🔧 [TOOL CALLED] calculator(expression='{expression}')")
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_weather(location: str) -> str:
    """Get the current weather information for a specific location. Use this when asked about weather conditions.
    
    Args:
        location: The city or location name
    """
    print(f"\n🔧 [TOOL CALLED] get_current_weather(location='{location}')")
    time.sleep(0.3)
    import random
    temps = [18, 20, 22, 24, 26]
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy"]
    return f"Weather in {location}: {random.choice(conditions)}, {random.choice(temps)}°C"


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in a specific timezone. Use this when asked about the current time.
    
    Args:
        timezone: The timezone, e.g., 'UTC', 'PST', 'EST' (default: UTC)
    """
    print(f"\n🔧 [TOOL CALLED] get_current_time(timezone='{timezone}')")
    from datetime import datetime
    return f"Current time in {timezone}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


# ==================== Create Agent with LangGraph ====================

def create_gemini_agent():
    """Create a ReAct agent using LangGraph - fully automatic tool calling!"""
    
    # Define tools
    tools = [
        search_information,
        calculator,
        get_current_weather,
        get_current_time,
    ]
    
    print(f"\n📋 Available Tools: {[t.name for t in tools]}")
    
    # Initialize LLM with Gemini
    # Try gemini-2.0-flash-exp for tool calling support
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    
    agent = create_react_agent(model, tools)
    
    return agent


# ==================== Test the Agent ====================

def test_agent():
    """Test the Gemini agent"""
    
    # Create agent
    agent = create_gemini_agent()
    
    # Test queries
    test_queries = [
        "What is 127 multiplied by 34?",
        "What's the weather in Tokyo?",
        "What time is it now in UTC?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {query}")
        print(f"{'='*80}\n")
        
        try:
            # Invoke the agent - LangGraph handles EVERYTHING automatically!
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
                    print(f"Content: {msg.content}")
                
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
            print(final_msg.content)
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
        print("🤖 GEMINI AGENT WITH LANGGRAPH")
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
