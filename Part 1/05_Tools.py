# pip install langchain langchain-openai langchain-core
import os
import logging
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

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

# --- 3. Create the Agent Prompt ---
# Define the agent's role, goal, and instructions
# This is the LangChain equivalent of CrewAI's Agent definition
def create_agent_prompt():
    """Create the prompt template for the agent."""
    template = """You are a Senior Financial Analyst. Your role is to analyze stock data using provided tools and report key prices.

You are an experienced financial analyst adept at using data sources to find stock information. You provide clear, direct answers.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    return PromptTemplate.from_template(template)

# --- 4. Create the Agent ---
def create_financial_agent():
    """Create the financial analyst agent with tools."""
    # Initialize the LLM
    llm = create_llm()

    # Define the tools available to the agent
    tools = [get_stock_price]

    # Create the prompt
    prompt = create_agent_prompt()

    # Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Create the agent executor (equivalent to CrewAI's Crew)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,  # Handle errors gracefully
        max_iterations=5  # Limit the number of iterations
    )

    return agent_executor

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

    # Create the agent executor
    agent_executor = create_financial_agent()

    # Define the task (equivalent to CrewAI's Task)
    task_description = (
        "What is the current simulated stock price for Apple (ticker: AAPL)? "
        "Use the 'get_stock_price' tool to find it. "
        "If the ticker is not found, you must report that you were unable to retrieve the price."
    )

    # Execute the task
    result = agent_executor.invoke({"input": task_description})

    print("\n---------------------------------")
    print("## Agent execution finished.")
    print("\nFinal Result:\n", result["output"])

if __name__ == "__main__":
    main()
