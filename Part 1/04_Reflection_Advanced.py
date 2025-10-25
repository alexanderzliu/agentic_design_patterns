"""
Advanced Reflection Loop Agent - LangChain 1.0

This advanced implementation demonstrates:
- Streaming support
- Custom state reducers
- Better visualization
- Multiple LLM strategies (ToolStrategy for structured output)
- Proper error handling
- Rich console output
"""

import os
from typing import Annotated, Literal, Optional, Sequence
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# --- Configuration ---
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file. Please add it.")


# --- Structured Output Models ---
class CodeIssue(BaseModel):
    """A single code issue or improvement suggestion."""

    category: Literal["bug", "style", "edge_case", "documentation", "performance"] = Field(
        description="Category of the issue"
    )
    description: str = Field(description="Detailed description of the issue")
    severity: Literal["low", "medium", "high"] = Field(
        description="Severity of the issue"
    )
    suggestion: str = Field(description="Specific suggestion for improvement")


class CodeReview(BaseModel):
    """Complete code review with multiple issues or approval."""

    is_approved: bool = Field(
        description="Whether the code is approved without any changes"
    )
    issues: list[CodeIssue] = Field(
        default_factory=list,
        description="List of issues found in the code"
    )
    overall_quality_score: int = Field(
        ge=0,
        le=10,
        description="Overall code quality score from 0-10"
    )
    summary: str = Field(description="Brief summary of the review")


class GeneratedCode(BaseModel):
    """Generated code with metadata."""

    code: str = Field(description="The Python code")
    explanation: str = Field(
        description="Brief explanation of key implementation choices"
    )
    changes_made: Optional[list[str]] = Field(
        default=None,
        description="List of changes made if this is a refinement"
    )


# --- State Definition with Reducers ---
class AdvancedReflectionState(TypedDict):
    """State for the advanced reflection loop workflow."""

    # Core state
    task_description: str
    current_code: Optional[GeneratedCode]
    iteration: int
    max_iterations: int

    # Review tracking
    review: Optional[CodeReview]
    all_reviews: Annotated[Sequence[CodeReview], add_messages]  # Accumulate reviews

    # Control flow
    is_complete: bool


# --- Initialize LLM ---
base_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Create specialized LLMs with structured outputs
code_generator_llm = base_llm.with_structured_output(GeneratedCode)
code_reviewer_llm = base_llm.with_structured_output(CodeReview)


# --- Prompt Templates ---
INITIAL_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Python developer.
Write clean, production-quality Python code that meets the given requirements.

Focus on:
- Correctness and completeness
- Proper error handling
- Clear documentation
- Edge case handling
- Pythonic idioms

Provide both the code and a brief explanation of your implementation choices."""),
    ("human", "{task_description}")
])

REFINEMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Python developer refining code based on review feedback.

Apply ALL the suggested improvements from the code review.
Be thorough and address each point specifically.

Provide the improved code and document what changes you made."""),
    ("human", """Original Task:
{task_description}

Previous Code:
```python
{previous_code}
```

Code Review:
{review_summary}

Issues to address:
{issues_list}

Please refine the code addressing all these issues.""")
])

REVIEWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior software engineer conducting a thorough code review.

Evaluate the code against:
1. Correctness: Does it meet all requirements?
2. Edge cases: Are all edge cases handled?
3. Error handling: Are errors handled properly?
4. Code quality: Is it clean and maintainable?
5. Documentation: Is it well-documented?
6. Best practices: Does it follow Python best practices?

If the code is perfect (quality score 9+), approve it.
Otherwise, provide specific, actionable feedback for each issue found."""),
    ("human", """Task Requirements:
{task_description}

Code to Review (Iteration {iteration}):
```python
{code}
```

Provide your detailed code review.""")
])


# --- Node Functions ---
def generate_initial_code(state: AdvancedReflectionState) -> dict:
    """Generate initial code based on task description."""

    print(f"\n{'='*60}")
    print(f"  ITERATION {state['iteration']}: GENERATING INITIAL CODE")
    print(f"{'='*60}")

    chain = INITIAL_GENERATOR_PROMPT | code_generator_llm
    result = chain.invoke({
        "task_description": state["task_description"]
    })

    print(f"\n📝 Generated Code:")
    print(f"{'─'*60}")
    print(result.code)
    print(f"{'─'*60}")
    print(f"\n💡 Explanation: {result.explanation}")

    return {
        "current_code": result,
        "iteration": state["iteration"]
    }


def refine_code(state: AdvancedReflectionState) -> dict:
    """Refine code based on review feedback."""

    print(f"\n{'='*60}")
    print(f"  ITERATION {state['iteration']}: REFINING CODE")
    print(f"{'='*60}")

    review = state["review"]

    # Format issues for the prompt
    issues_list = "\n".join([
        f"{i+1}. [{issue.severity.upper()}] {issue.category}: {issue.description}\n   → Suggestion: {issue.suggestion}"
        for i, issue in enumerate(review.issues)
    ])

    chain = REFINEMENT_PROMPT | code_generator_llm
    result = chain.invoke({
        "task_description": state["task_description"],
        "previous_code": state["current_code"].code,
        "review_summary": review.summary,
        "issues_list": issues_list
    })

    print(f"\n📝 Refined Code:")
    print(f"{'─'*60}")
    print(result.code)
    print(f"{'─'*60}")

    if result.changes_made:
        print(f"\n✏️  Changes Made:")
        for change in result.changes_made:
            print(f"  • {change}")

    return {
        "current_code": result,
        "iteration": state["iteration"]
    }


def review_code(state: AdvancedReflectionState) -> dict:
    """Conduct code review and provide structured feedback."""

    print(f"\n{'─'*60}")
    print(f"  🔍 CONDUCTING CODE REVIEW")
    print(f"{'─'*60}")

    chain = REVIEWER_PROMPT | code_reviewer_llm
    review = chain.invoke({
        "task_description": state["task_description"],
        "code": state["current_code"].code,
        "iteration": state["iteration"]
    })

    print(f"\n📊 Review Summary: {review.summary}")
    print(f"Quality Score: {review.quality_score}/10")

    if review.is_approved:
        print(f"\n✅ CODE APPROVED!")
        return {
            "review": review,
            "all_reviews": [review],
            "is_complete": True
        }
    else:
        print(f"\n❌ Issues Found ({len(review.issues)}):")
        for i, issue in enumerate(review.issues, 1):
            severity_emoji = {"low": "ℹ️", "medium": "⚠️", "high": "🚨"}
            print(f"\n{i}. {severity_emoji[issue.severity]} [{issue.category.upper()}] - {issue.severity.upper()}")
            print(f"   {issue.description}")
            print(f"   💡 {issue.suggestion}")

        return {
            "review": review,
            "all_reviews": [review],
            "is_complete": False
        }


def increment_iteration(state: AdvancedReflectionState) -> dict:
    """Increment the iteration counter."""
    return {"iteration": state["iteration"] + 1}


# --- Routing Logic ---
def route_after_review(state: AdvancedReflectionState) -> Literal["refine", "end"]:
    """Route to refinement or end based on review results."""

    # Check if approved
    if state["is_complete"]:
        return "end"

    # Check if max iterations reached
    if state["iteration"] >= state["max_iterations"]:
        print(f"\n⚠️  Maximum iterations ({state['max_iterations']}) reached.")
        print("Ending with current code.")
        return "end"

    # Continue refining
    return "refine"


def route_from_start(state: AdvancedReflectionState) -> Literal["generate", "refine"]:
    """Route to initial generation or refinement based on iteration."""

    if state["iteration"] == 1:
        return "generate"
    else:
        return "refine"


# --- Build the Graph ---
def create_advanced_reflection_graph() -> StateGraph:
    """Create and compile the advanced reflection loop graph."""

    workflow = StateGraph(AdvancedReflectionState)

    # Add nodes
    workflow.add_node("generate_initial", generate_initial_code)
    workflow.add_node("refine", refine_code)
    workflow.add_node("review", review_code)
    workflow.add_node("increment", increment_iteration)

    # Entry point: route to generate or refine based on state
    workflow.add_conditional_edges(
        START,
        route_from_start,
        {
            "generate": "generate_initial",
            "refine": "refine"
        }
    )

    # Both generate and refine go to review
    workflow.add_edge("generate_initial", "review")
    workflow.add_edge("refine", "review")

    # Review decides whether to continue or end
    workflow.add_conditional_edges(
        "review",
        route_after_review,
        {
            "refine": "increment",
            "end": END
        }
    )

    # Increment loops back to start (which will route to refine)
    workflow.add_edge("increment", START)

    # Compile with memory for state persistence
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# --- Main Execution ---
def run_advanced_reflection_loop(
    task_description: str,
    max_iterations: int = 3,
    stream: bool = False
) -> dict:
    """
    Run the advanced reflection loop with structured outputs and rich feedback.

    Args:
        task_description: The task description for code generation
        max_iterations: Maximum number of refinement iterations
        stream: Whether to stream the output (for real-time feedback)

    Returns:
        Final state with refined code and review history
    """

    graph = create_advanced_reflection_graph()

    initial_state: AdvancedReflectionState = {
        "task_description": task_description,
        "current_code": None,
        "iteration": 1,
        "max_iterations": max_iterations,
        "review": None,
        "all_reviews": [],
        "is_complete": False
    }

    config = {"configurable": {"thread_id": "advanced_reflection_session"}}

    if stream:
        # Streaming execution (for real-time updates)
        print("\n🚀 Starting reflection loop (streaming mode)...\n")
        for event in graph.stream(initial_state, config=config):
            # Events are streamed as they happen
            pass
        final_state = graph.get_state(config).values
    else:
        # Standard execution
        print("\n🚀 Starting reflection loop...\n")
        final_state = graph.invoke(initial_state, config=config)

    # Print final summary
    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")

    if final_state["current_code"]:
        print(f"\n📄 Final Code:")
        print(f"{'─'*60}")
        print(final_state["current_code"].code)
        print(f"{'─'*60}")

    print(f"\n📈 Iterations completed: {final_state['iteration']}")
    print(f"Quality progression:")
    for i, review in enumerate(final_state["all_reviews"], 1):
        status = "✅ APPROVED" if review.is_approved else f"❌ {len(review.issues)} issues"
        print(f"  Iteration {i}: {review.overall_quality_score}/10 - {status}")

    return final_state


if __name__ == "__main__":
    TASK = """
Create a Python function named `calculate_factorial` with the following requirements:

1. Accept a single integer `n` as input
2. Calculate and return its factorial (n!)
3. Include comprehensive docstring with examples
4. Handle edge cases:
   - factorial(0) should return 1
   - factorial(1) should return 1
5. Handle invalid input:
   - Raise ValueError for negative numbers
   - Raise TypeError for non-integer inputs
6. Optimize for reasonable performance
7. Include type hints
"""

    # Run the advanced reflection loop
    final_state = run_advanced_reflection_loop(
        task_description=TASK,
        max_iterations=3,
        stream=False  # Set to True for streaming output
    )

    # Access the final code
    if final_state["current_code"]:
        print("\n" + "="*60)
        print("✨ Code is ready to use!")
        print("="*60)
