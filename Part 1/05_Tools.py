# pip install langchain langchain-openai langchain-core langgraph
import os
import logging
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# --- Best Practice: Configure Logging ---
# A basic logging setup helps in debugging and tracking the agent's execution.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Set up your API Key ---
# For production, it's recommended to use a more secure method for key management
# like environment variables loaded at runtime or a secret manager.
#
# Set the environment variable for your chosen LLM provider (e.g., OPENAI_API_KEY)
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
# os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"

# --- 1. Refactored Tool: Returns Clean Data ---
# The tool now returns raw data (a float) or raises a standard Python error.
# This makes it more reusable and forces the agent to handle outcomes properly.
@tool
def get_stock_price(ticker: str) -> float:
    """
    Fetches the latest simulated stock price for a given stock ticker symbol.
    Returns the price as a float. Raises a ValueError if the ticker is not found.

    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    """
    logging.info(f"Tool Call: get_stock_price for ticker '{ticker}'")

    simulated_prices = {
        "AAPL": 178.15,
        "GOOGL": 1750.30,
        "MSFT": 425.50,
    }

    price = simulated_prices.get(ticker.upper())

    if price is not None:
        return price
    else:
        # Raising a specific error is better than returning a string.
        # The agent is equipped to handle exceptions and can decide on the next action.
        raise ValueError(f"Simulated price for ticker '{ticker.upper()}' not found.")

# --- 2. Define the LLM ---
# Initialize the language model that will power the agent
def create_llm():
    """Create and configure the LLM."""
    model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
    return ChatOpenAI(
        model=model_name,
        temperature=0,  # Lower temperature for more deterministic responses
        verbose=True
    )

# --- 3. Create the System Message ---
# Define the agent's role and instructions using the modern approach
def create_system_message():
    """Create the system message for the agent."""
    return """You are a Senior Financial Analyst. Your role is to analyze stock data using provided tools and report key prices.

You are an experienced financial analyst adept at using data sources to find stock information. You provide clear, direct answers.

When a tool returns an error or exception, acknowledge it and report that you were unable to retrieve the requested information."""

# --- 4. Create the Agent Graph (LangChain 1.0 Paradigm) ---
def create_financial_agent():
    """Create the financial analyst agent with tools using LangGraph."""
    # Initialize the LLM
    llm = create_llm()

    # Define the tools available to the agent
    tools = [get_stock_price]

    # Get the system message
    system_message = create_system_message()

    # Create the ReAct agent graph using LangGraph's prebuilt function
    # This returns an executable graph, not an executor
    agent_graph = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_message  # System message defines agent behavior
    )

    return agent_graph

# --- 5. Run the Agent within a Main Execution Block ---
# Using a "if __name__ == '__main__':" block is a standard Python best practice.
def main():
    """Main function to run the agent."""
    # Check for API key before starting to avoid runtime errors.
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: The OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the script.")
        return

    print("\n## Starting the Financial Agent...")
    print("---------------------------------")

    # Create the agent graph
    agent_graph = create_financial_agent()

    # Define the task (equivalent to CrewAI's Task)
    task_description = (
        "What is the current simulated stock price for Apple (ticker: AAPL)? "
        "Use the 'get_stock_price' tool to find it. "
        "If the ticker is not found, you must report that you were unable to retrieve the price."
    )

    # Execute the task using the graph's invoke method
    # The new API uses "messages" instead of "input"
    result = agent_graph.invoke(
        {"messages": [("user", task_description)]}
    )

    print("\n---------------------------------")
    print("## Agent execution finished.")

    # Extract the final message from the graph output
    final_message = result["messages"][-1]
    print("\nFinal Result:\n", final_message.content)

if __name__ == "__main__":
    main()
