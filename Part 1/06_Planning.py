# pip install langchain langchain-openai langchain-core langgraph python-dotenv
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Load environment variables from .env file for security
load_dotenv()

# --- 1. Define State Schema ---
# The state holds the topic, plan, and final summary
class PlannerState(TypedDict):
    """State for the planning and writing workflow."""
    topic: str
    plan: str
    summary: str

# --- 2. Explicitly define the language model ---
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)

# --- 3. Define Node Functions ---
# Each node is a function that takes state and returns state updates

def create_plan(state: PlannerState) -> dict:
    """
    First node: Creates a bullet-point plan for the summary.
    This demonstrates the 'Planning' design pattern step 1.
    """
    topic = state["topic"]

    system_prompt = SystemMessage(content="""You are an expert technical writer and content strategist.
Your strength lies in creating clear, actionable plans before writing.
You break down complex topics into structured, logical outlines.""")

    user_prompt = HumanMessage(content=f"""Create a bullet-point plan for writing a summary on the following topic:
"{topic}"

The plan should outline the main points that will be covered in approximately 200 words.
Return ONLY the bullet-point plan, nothing else.""")

    # Invoke the LLM to create the plan
    response = llm.invoke([system_prompt, user_prompt])

    print("\n### Plan Created ###")
    print(response.content)

    # Return state update
    return {"plan": response.content}


def write_summary(state: PlannerState) -> dict:
    """
    Second node: Writes the summary based on the plan.
    This demonstrates the 'Planning' design pattern step 2.
    """
    topic = state["topic"]
    plan = state["plan"]

    system_prompt = SystemMessage(content="""You are an expert technical writer.
You excel at transforming plans into clear, concise, and engaging content.
You write informative summaries that are easy to digest.""")

    user_prompt = HumanMessage(content=f"""Based on the following plan, write a concise summary about "{topic}".

PLAN:
{plan}

Write a well-structured summary of approximately 200 words that follows this plan.
Keep it informative and engaging.""")

    # Invoke the LLM to write the summary
    response = llm.invoke([system_prompt, user_prompt])

    print("\n### Summary Written ###")
    print(response.content)

    # Return state update
    return {"summary": response.content}


# --- 4. Build the Planning Workflow Graph ---
def create_planning_workflow():
    """
    Creates a LangGraph StateGraph that implements the Planning design pattern.
    The graph has two sequential nodes: plan creation → summary writing.
    """
    # Initialize the StateGraph with our state schema
    workflow = StateGraph(PlannerState)

    # Add nodes to the graph
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("write_summary", write_summary)

    # Define the sequential flow: START → create_plan → write_summary → END
    workflow.add_edge(START, "create_plan")
    workflow.add_edge("create_plan", "write_summary")
    workflow.add_edge("write_summary", END)

    # Compile the graph into an executable application
    return workflow.compile()


# --- 5. Execute the Planning Workflow ---
def main():
    """Main function to run the planning and writing workflow."""
    # Define the topic
    topic = "The importance of Reinforcement Learning in AI"

    print("## Running the Planning and Writing Task ##")
    print(f"Topic: {topic}\n")

    # Create the workflow graph
    app = create_planning_workflow()

    # Initialize the state with the topic
    initial_state = {
        "topic": topic,
        "plan": "",
        "summary": ""
    }

    # Execute the workflow
    result = app.invoke(initial_state)

    # Display the final result
    print("\n\n---")
    print("## Task Result ##")
    print("---\n")
    print("### Plan")
    print(result["plan"])
    print("\n### Summary")
    print(result["summary"])


if __name__ == "__main__":
    main()
