from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
import time

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


# ==================== Create Standard LangChain Agent ====================

def create_standard_agent():
    """Create a standard LangChain agent - fully automatic tool calling!"""
    
    # Define tools
    tools = [
        search_information,
        calculator,
        get_current_weather,
        get_current_time,
    ]
    
    print(f"\n📋 Available Tools: {[t.name for t in tools]}")
    
    # Initialize LLM with Llama 3.1 (supports tool calling!)
    model = ChatOllama(model="llama3.1", temperature=0)
    
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="You are a helpful assistant with access to various tools. Use the tools to answer user questions accurately.",
    )
    
    return agent


# ==================== Test the Agent ====================

def test_standard_agent():
    """Test the standard LangChain agent"""
    # Create agent
    agent = create_standard_agent()
    
    # Test queries
    test_queries = [
        "What is 127 multiplied by 34?",
        "What's the weather in Tokyo?",
        "What time is it now in UTC?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*80}\n")
        
        try:
            # Invoke the agent - LangChain handles EVERYTHING automatically!
            # The LLM sees tools via bind_tools(), decides which to call, 
            # generates structured tool_calls output,
            # LangChain parses it, executes the tool, feeds back to LLM, repeats as needed
            result = agent.invoke({"messages": [("human", query)]})
            
            print(f"\n{'='*80}")
            print(f"RESPONSE:")
            print(f"{'='*80}")
            
            # Show detailed tool call information
            messages = result.get("messages", [])
            for i, msg in enumerate(messages):
                print(f"\n--- Message {i+1} ---")
                if hasattr(msg, 'content') and msg.content:
                    print(f"Content: {msg.content}")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"Tool Calls: {msg.tool_calls}")
                if hasattr(msg, 'name') and msg.name:
                    print(f"Tool Name: {msg.name}")
                    print(f"Tool Result: {msg.content}")
            
            # Extract the final answer from the messages
            final_msg = messages[-1]
            if hasattr(final_msg, 'content') and final_msg.content:
                print(f"\n{'='*80}")
                print(f"FINAL ANSWER:")
                print(f"{'='*80}")
                print(final_msg.content)
            else:
                print(result)
            print(f"{'='*80}\n")
            
            time.sleep(1)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing to next query...\n")

# ==================== Main ====================

if __name__ == "__main__":
    try:
        test_standard_agent()
        
        print("\n" + "="*80)
        print("✅ DEMO COMPLETED!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

