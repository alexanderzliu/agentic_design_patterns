"""
Reflection Loop Agent - Refactored for LangChain 1.0

REFACTORING OVERVIEW:
=====================
This is a modern refactoring of a reflection loop agent that iteratively improves
generated code through critique and refinement cycles.

KEY CHANGES FROM ORIGINAL:
--------------------------
1. LangGraph StateGraph replaces manual for-loop iteration
   OLD: for i in range(max_iterations): ...
   NEW: StateGraph with nodes and conditional edges

   Benefits:
   - Built-in state management and persistence
   - Visual workflow representation
   - Support for streaming and human-in-the-loop
   - Automatic checkpointing for resumable workflows

2. Pydantic models replace string parsing for LLM outputs
   OLD: if "CODE IS PERFECT" in critique: ...
   NEW: critique_llm.with_structured_output(CodeCritique)

   Benefits:
   - Type-safe, validated responses
   - No fragile string matching
   - Self-documenting schemas
   - IDE autocomplete support

3. TypedDict for centralized state management
   OLD: current_code = "", message_history = [...], ...
   NEW: class ReflectionState(TypedDict): ...

   Benefits:
   - All state in one place
   - Type hints for better IDE support
   - Clear state schema

4. ChatPromptTemplate (LCEL) replaces f-string formatting
   OLD: f"You are a senior engineer...{task_prompt}..."
   NEW: ChatPromptTemplate.from_messages([...])

   Benefits:
   - Reusable, composable prompt templates
   - Better organization and testability
   - Support for message placeholders

5. Conditional edges for control flow
   OLD: if/else logic inside the loop
   NEW: should_continue() function with conditional_edges

   Benefits:
   - Declarative routing logic
   - Easier to visualize and debug
   - Better separation of concerns

6. MemorySaver checkpointer enables state persistence
   NEW: graph.compile(checkpointer=MemorySaver())

   Benefits:
   - Can resume workflows after interruption
   - Foundation for time-travel debugging
   - Enables human-in-the-loop patterns

USAGE:
------
    python reflection_loop_refactored.py

REQUIREMENTS:
-------------
    pip install langchain langchain-openai langgraph python-dotenv pydantic

MIGRATION NOTES:
----------------
The original code used a manual message history with appending, which has been
replaced with structured state management. The LangGraph approach makes the code
more maintainable and opens up new capabilities like streaming, HITL, and
distributed execution.
"""

import os
from typing import Literal, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# --- Configuration ---
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not found in .env file. Please add it."
    )


# --- Structured Output Models ---
# MODERN PATTERN: Using Pydantic for structured outputs instead of string parsing
# This replaces the old pattern of checking 'if "CODE IS PERFECT" in critique'
# with type-safe, validated model access

class CodeCritique(BaseModel):
    """
    Structured output for code review critique.

    REPLACES: Manual string parsing like 'if "CODE IS PERFECT" in critique'

    Benefits:
    - Type safety and automatic validation
    - No manual string manipulation
    - Clear schema definition
    - Better error messages
    """

    is_perfect: bool = Field(
        description="Whether the code meets all requirements and has no issues"
    )
    critiques: list[str] = Field(
        default_factory=list,
        description="List of specific critiques and improvement suggestions"
    )
    severity: Optional[Literal["minor", "major", "critical"]] = Field(
        default=None,
        description="Overall severity of issues found"
    )


# --- State Definition ---
# MODERN PATTERN: TypedDict for centralized state management
# REPLACES: Scattered variables (current_code = "", message_history = [], etc.)

class ReflectionState(TypedDict):
    """
    State for the reflection loop workflow.

    In LangGraph, state is the single source of truth passed between nodes.
    This replaces the old pattern of maintaining separate variables.

    OLD PATTERN:
        current_code = ""
        message_history = []
        max_iterations = 3

    NEW PATTERN:
        All state in one TypedDict that flows through the graph
    """

    task_description: str          # The original task prompt
    current_code: str              # Generated/refined code
    iteration: int                 # Current iteration number
    max_iterations: int            # Maximum iterations allowed
    critique: Optional[CodeCritique]  # Latest critique (structured)
    is_complete: bool              # Whether code is approved


# --- Initialize LLM ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# MODERN PATTERN: Structured output using with_structured_output()
# This is a LangChain 1.0 feature that returns Pydantic models instead of strings
critique_llm = llm.with_structured_output(CodeCritique)


# --- Prompt Templates ---
# MODERN PATTERN: ChatPromptTemplate for reusable, composable prompts
# REPLACES: f-strings scattered throughout the code

GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Python developer.
Your task is to write clean, well-documented Python code that meets the given requirements.

If this is a refinement iteration, you will receive previous critiques.
Apply all the critiques to improve the code.

Return ONLY the Python code, properly formatted with docstrings."""),
    ("human", "{task_description}"),
    # MessagesPlaceholder allows optional message insertion (for refinements)
    MessagesPlaceholder(variable_name="critique_message", optional=True)
])

REFLECTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior software engineer and Python expert.
Your role is to perform meticulous code review.

Critically evaluate the provided code based on:
- Correctness and adherence to requirements
- Edge case handling
- Code quality and best practices
- Documentation quality
- Potential bugs or issues

Be specific and constructive in your critiques."""),
    ("human", """Original Task:
{task_description}

Code to Review:
```python
{code}
```

Provide your detailed code review.""")
])


# --- Node Functions ---
# MODERN PATTERN: Nodes are pure functions that receive state and return updates
# Each node does ONE thing and returns a dict of state updates

def generate_or_refine_code(state: ReflectionState) -> dict:
    """
    Generate initial code or refine based on critiques.

    REPLACES: The if/else block inside the original for-loop:
        if i == 0:
            response = llm.invoke(message_history)
        else:
            message_history.append(...)
            response = llm.invoke(message_history)

    MODERN APPROACH:
    - Uses LCEL chains (prompt | llm)
    - Returns state updates as dict
    - Single responsibility: generate/refine code
    """

    print(f"\n{'='*25} ITERATION {state['iteration']} {'='*25}")

    if state["iteration"] == 1:
        print("\n>>> STAGE: GENERATING initial code...")

        # LCEL Pattern: chain prompt and LLM with | operator
        chain = GENERATOR_PROMPT | llm
        response = chain.invoke({
            "task_description": state["task_description"]
        })
    else:
        print("\n>>> STAGE: REFINING code based on previous critique...")

        # Format critiques from structured output
        critique_text = "\n".join([
            f"- {c}" for c in state["critique"].critiques
        ])

        chain = GENERATOR_PROMPT | llm
        response = chain.invoke({
            "task_description": state["task_description"],
            "critique_message": [
                HumanMessage(content=f"""Previous critique:
{critique_text}

Please refine the code addressing all these points.""")
            ]
        })

    code = response.content
    print(f"\n--- Generated Code (v{state['iteration']}) ---")
    print(code)

    # Return only the state keys that changed
    return {
        "current_code": code,
        "iteration": state["iteration"]
    }


def reflect_on_code(state: ReflectionState) -> dict:
    """
    Reflect on and critique the generated code.

    REPLACES: Manual string parsing in the original:
        critique_response = llm.invoke(reflector_prompt)
        critique = critique_response.content
        if "CODE IS PERFECT" in critique:
            print("No further critiques...")
            break

    MODERN APPROACH:
    - Uses structured output (CodeCritique model)
    - Type-safe access to critique fields
    - No string matching needed
    """

    print("\n>>> STAGE: REFLECTING on the generated code...")

    # LCEL chain with structured output
    # The critique_llm automatically parses response into CodeCritique model
    chain = REFLECTOR_PROMPT | critique_llm
    critique = chain.invoke({
        "task_description": state["task_description"],
        "code": state["current_code"]
    })

    # Type-safe access instead of string matching
    if critique.is_perfect:
        print("\n--- Critique ---")
        print("✓ Code is perfect! No issues found.")
        return {
            "critique": critique,
            "is_complete": True
        }
    else:
        print("\n--- Critique ---")
        print(f"Severity: {critique.severity}")
        for i, c in enumerate(critique.critiques, 1):
            print(f"{i}. {c}")

        return {
            "critique": critique,
            "is_complete": False
        }


def increment_iteration(state: ReflectionState) -> dict:
    """
    Increment the iteration counter.

    Simple node that updates state. In the original code, this was:
        (implicit at the end of the for-loop)

    In LangGraph, state updates are explicit and traceable.
    """
    return {"iteration": state["iteration"] + 1}


# --- Conditional Edge ---
# MODERN PATTERN: Routing logic as a pure function
# REPLACES: if/break statements inside the loop

def should_continue(state: ReflectionState) -> Literal["continue", "end"]:
    """
    Determine whether to continue the reflection loop or end.

    REPLACES: The break/continue logic in the original loop:
        if "CODE IS PERFECT" in critique:
            break
        (implicit continue at loop end)

    MODERN APPROACH:
    - Declarative routing based on state
    - Returns edge name to follow
    - Easy to visualize in graph
    """

    # Check if code was approved
    if state["is_complete"]:
        return "end"

    # Check if max iterations reached
    if state["iteration"] >= state["max_iterations"]:
        print(f"\n⚠ Maximum iterations ({state['max_iterations']}) reached.")
        return "end"

    # Continue refining
    return "continue"


# --- Build the Graph ---
# MODERN PATTERN: Declarative workflow definition with LangGraph
# REPLACES: Imperative for-loop with manual state management

def create_reflection_graph() -> StateGraph:
    """
    Create and compile the reflection loop graph.

    REPLACES: The entire for-loop structure:
        for i in range(max_iterations):
            if i == 0: ...
            else: ...
            critique = ...
            if "CODE IS PERFECT" in critique:
                break

    MODERN APPROACH:
    - Declarative graph structure
    - Nodes for each operation
    - Conditional edges for routing
    - Built-in checkpointing

    Benefits:
    - Visual representation available
    - State persisted automatically
    - Easier to extend and modify
    - Can add streaming, HITL, etc.
    """

    # Initialize the graph with our state schema
    workflow = StateGraph(ReflectionState)

    # Add nodes (operations)
    workflow.add_node("generate", generate_or_refine_code)
    workflow.add_node("reflect", reflect_on_code)
    workflow.add_node("increment", increment_iteration)

    # Add edges (connections between nodes)
    workflow.add_edge(START, "generate")        # Entry point
    workflow.add_edge("generate", "reflect")    # Always reflect after generating

    # Conditional edge: decide whether to continue or end
    # This replaces the if/break logic in the original loop
    workflow.add_conditional_edges(
        "reflect",
        should_continue,  # Routing function
        {
            "continue": "increment",  # If continuing, increment iteration
            "end": END                 # If done, end workflow
        }
    )

    # Close the loop: increment leads back to generate
    workflow.add_edge("increment", "generate")

    # Compile with checkpointer for state persistence
    # This enables resumable workflows and is required for HITL
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# --- Main Execution ---

def run_reflection_loop(
    task_description: str,
    max_iterations: int = 3
) -> dict:
    """
    Run the reflection loop to iteratively improve code.

    REPLACES: The old run_reflection_loop() function that used a manual for-loop

    MODERN APPROACH:
    - Initialize state as TypedDict
    - Invoke graph (replaces for-loop)
    - Graph handles all iteration logic

    Args:
        task_description: The task description for code generation
        max_iterations: Maximum number of refinement iterations

    Returns:
        Final state containing the refined code and critique history
    """

    # Create the graph
    graph = create_reflection_graph()

    # Initial state - all state in one place
    # REPLACES: current_code = "", message_history = [...], etc.
    initial_state: ReflectionState = {
        "task_description": task_description,
        "current_code": "",
        "iteration": 1,
        "max_iterations": max_iterations,
        "critique": None,
        "is_complete": False
    }

    # Execute the graph
    # Config with thread_id enables state persistence across runs
    config = {"configurable": {"thread_id": "reflection_session_1"}}

    # graph.invoke() replaces the entire for-loop
    # It will:
    # 1. Generate code
    # 2. Reflect on it
    # 3. Decide whether to continue or end
    # 4. If continue, increment and loop back to generate
    # 5. Repeat until complete or max iterations
    final_state = graph.invoke(initial_state, config=config)

    # Print final result
    print("\n" + "="*30 + " FINAL RESULT " + "="*30)
    print("\nFinal refined code after the reflection process:\n")
    print(final_state["current_code"])

    return final_state


if __name__ == "__main__":
    # Task description (same as original)
    TASK_PROMPT = """
Your task is to create a Python function named `calculate_factorial`.

This function should do the following:
1. Accept a single integer `n` as input.
2. Calculate its factorial (n!).
3. Include a clear docstring explaining what the function does.
4. Handle edge cases: The factorial of 0 is 1.
5. Handle invalid input: Raise a ValueError if the input is a negative number.
"""

    # Run the reflection loop
    # Same interface as original, but with modern internals
    run_reflection_loop(
        task_description=TASK_PROMPT,
        max_iterations=3
    )

    # ADDITIONAL CAPABILITIES NOW AVAILABLE:
    # ======================================
    #
    # 1. Streaming:
    #    for event in graph.stream(initial_state, config):
    #        print(event)
    #
    # 2. Human-in-the-loop:
    #    Add interrupt_before=["reflect"] to see state before reflecting
    #
    # 3. Time travel:
    #    state_history = graph.get_state_history(config)
    #
    # 4. Visualization:
    #    graph.get_graph().draw_mermaid_png()
    #
    # 5. Async execution:
    #    await graph.ainvoke(initial_state, config)
