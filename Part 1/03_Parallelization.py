"""
LangChain 1.0 Parallel Processing Example

This example demonstrates how to use LCEL (LangChain Expression Language) and
RunnableParallel to execute multiple independent tasks concurrently and synthesize
their results into a comprehensive response.
"""

import os
import asyncio
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

# --- Configuration ---
# Ensure your API key environment variable is set (e.g., OPENAI_API_KEY)

try:
    llm: Optional[ChatOpenAI] = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None

# --- Define Independent Chains ---
# These three chains represent distinct tasks that can be executed in parallel.

summarize_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarize the following topic concisely:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

questions_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Generate three interesting questions about the following topic:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

terms_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Identify 5-10 key terms from the following topic, separated by commas:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

# --- Build the Parallel + Synthesis Chain ---

# 1. Define the block of tasks to run in parallel. The results of these,
#    along with the original topic, will be fed into the next step.
map_chain = RunnableParallel(
    {
        "summary": summarize_chain,
        "questions": questions_chain,
        "key_terms": terms_chain,
        "topic": RunnablePassthrough(),  # Pass the original topic through
    }
)

# 2. Define the final synthesis prompt which will combine the parallel results.
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", """Based on the following information:

Summary: {summary}

Related Questions: {questions}

Key Terms: {key_terms}

Synthesize a comprehensive answer."""),
    ("user", "Original topic: {topic}")
])

# 3. Construct the full chain by piping the parallel results directly
#    into the synthesis prompt, followed by the LLM and output parser.
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()

# --- Run the Chain ---

async def run_parallel_example(topic: str) -> None:
    """
    Asynchronously invokes the parallel processing chain with a specific topic
    and prints the synthesized result.

    Args:
        topic: The input topic to be processed by the LangChain chains.
    """
    if not llm:
        print("LLM not initialized. Cannot run example.")
        return

    print(f"\n--- Running Parallel LangChain Example for Topic: '{topic}' ---\n")

    try:
        # The input to `ainvoke` is the single 'topic' string,
        # then passed to each runnable in the `map_chain`.
        response = await full_parallel_chain.ainvoke(topic)

        print("\n--- Final Response ---")
        print(response)
    except Exception as e:
        print(f"\nAn error occurred during chain execution: {e}")


if __name__ == "__main__":
    test_topic = "The history of space exploration"

    # In Python 3.7+, asyncio.run is the standard way to run an async function.
    asyncio.run(run_parallel_example(test_topic))


# ============================================================================
# Alternative Implementation: LangGraph Functional API (No LCEL/Runnables)
# ============================================================================

"""
The code below demonstrates a more Pythonic alternative to LCEL using LangGraph's
Functional API with @task and @entrypoint decorators.
"""

from langgraph.func import task, entrypoint

# --- Define Parallel Tasks ---
# Each @task decorator creates an independent task that can run in parallel

@task
def summarize_topic(topic: str) -> str:
    """Generate a concise summary of the topic."""
    response = llm.invoke([
        {"role": "system", "content": "Summarize the following topic concisely:"},
        {"role": "user", "content": topic}
    ])
    return response.content


@task
def generate_questions(topic: str) -> str:
    """Generate three interesting questions about the topic."""
    response = llm.invoke([
        {"role": "system", "content": "Generate three interesting questions about the following topic:"},
        {"role": "user", "content": topic}
    ])
    return response.content


@task
def identify_key_terms(topic: str) -> str:
    """Identify 5-10 key terms from the topic."""
    response = llm.invoke([
        {"role": "system", "content": "Identify 5-10 key terms from the following topic, separated by commas:"},
        {"role": "user", "content": topic}
    ])
    return response.content


@task
def synthesize_results(topic: str, summary: str, questions: str, key_terms: str) -> str:
    """Combine all parallel results into a comprehensive answer."""
    response = llm.invoke([
        {"role": "system", "content": f"""Based on the following information:

Summary: {summary}

Related Questions: {questions}

Key Terms: {key_terms}

Synthesize a comprehensive answer."""},
        {"role": "user", "content": f"Original topic: {topic}"}
    ])
    return response.content


# --- Build Workflow ---

@entrypoint()
def parallel_workflow(topic: str) -> str:
    """
    Main workflow that executes tasks in parallel and synthesizes results.

    Tasks are invoked and return futures that can be resolved with .result()
    """
    # Launch all three tasks in parallel
    summary_future = summarize_topic(topic)
    questions_future = generate_questions(topic)
    terms_future = identify_key_terms(topic)

    # Wait for all results and synthesize
    return synthesize_results(
        topic,
        summary_future.result(),
        questions_future.result(),
        terms_future.result()
    ).result()


# Uncomment to run the LangGraph functional API example:
# if __name__ == "__main__":
#     test_topic = "The history of space exploration"
#     print(f"\n--- Running LangGraph Functional API Example for Topic: '{test_topic}' ---\n")
#     try:
#         result = parallel_workflow.invoke(test_topic)
#         print("\n--- Final Response ---")
#         print(result)
#     except Exception as e:
#         print(f"\nAn error occurred during workflow execution: {e}")


# ============================================================================
# Alternative Implementation: LangGraph StateGraph API (Declarative)
# ============================================================================

"""
The StateGraph API is a declarative approach where you define:
1. A shared state schema (what data flows through the graph)
2. Nodes (functions that transform state)
3. Edges (the execution flow between nodes)

This is useful for complex workflows with conditional branching, loops, and
when you need to inspect/debug intermediate states.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# --- Define State Schema ---
# The state is a shared data structure that flows through all nodes

class ParallelState(TypedDict):
    """State schema for the parallel processing workflow."""
    topic: str
    summary: str
    questions: str
    key_terms: str
    combined_output: str


# --- Define Node Functions ---
# Each node receives the current state and returns a partial state update

def generate_summary(state: ParallelState) -> dict:
    """Node that generates a summary of the topic."""
    response = llm.invoke([
        {"role": "system", "content": "Summarize the following topic concisely:"},
        {"role": "user", "content": state["topic"]}
    ])
    return {"summary": response.content}


def generate_questions_node(state: ParallelState) -> dict:
    """Node that generates questions about the topic."""
    response = llm.invoke([
        {"role": "system", "content": "Generate three interesting questions about the following topic:"},
        {"role": "user", "content": state["topic"]}
    ])
    return {"questions": response.content}


def generate_key_terms(state: ParallelState) -> dict:
    """Node that identifies key terms from the topic."""
    response = llm.invoke([
        {"role": "system", "content": "Identify 5-10 key terms from the following topic, separated by commas:"},
        {"role": "user", "content": state["topic"]}
    ])
    return {"key_terms": response.content}


def synthesize_node(state: ParallelState) -> dict:
    """Node that synthesizes all parallel results into a final answer."""
    response = llm.invoke([
        {"role": "system", "content": f"""Based on the following information:

Summary: {state['summary']}

Related Questions: {state['questions']}

Key Terms: {state['key_terms']}

Synthesize a comprehensive answer."""},
        {"role": "user", "content": f"Original topic: {state['topic']}"}
    ])
    return {"combined_output": response.content}


# --- Build the Graph ---

# Create a StateGraph with our state schema
graph_builder = StateGraph(ParallelState)

# Add nodes to the graph
graph_builder.add_node("generate_summary", generate_summary)
graph_builder.add_node("generate_questions", generate_questions_node)
graph_builder.add_node("generate_key_terms", generate_key_terms)
graph_builder.add_node("synthesize", synthesize_node)

# Add edges to define execution flow
# All three generation nodes start in parallel from START
graph_builder.add_edge(START, "generate_summary")
graph_builder.add_edge(START, "generate_questions")
graph_builder.add_edge(START, "generate_key_terms")

# All three converge to the synthesis node
graph_builder.add_edge("generate_summary", "synthesize")
graph_builder.add_edge("generate_questions", "synthesize")
graph_builder.add_edge("generate_key_terms", "synthesize")

# Synthesis node leads to END
graph_builder.add_edge("synthesize", END)

# Compile the graph into a runnable
graph_workflow = graph_builder.compile()


# Uncomment to run the LangGraph StateGraph API example:
# if __name__ == "__main__":
#     test_topic = "The history of space exploration"
#     print(f"\n--- Running LangGraph StateGraph API Example for Topic: '{test_topic}' ---\n")
#     try:
#         result = graph_workflow.invoke({"topic": test_topic})
#         print("\n--- Final Response ---")
#         print(result["combined_output"])
#     except Exception as e:
#         print(f"\nAn error occurred during graph execution: {e}")


# ============================================================================
# PARADIGM COMPARISON: Functional vs. Declarative vs. LCEL
# ============================================================================

"""
THREE APPROACHES TO PARALLELIZATION IN LANGCHAIN 1.0:

┌─────────────────────────────────────────────────────────────────────────┐
│ 1. LCEL (LangChain Expression Language) - DECLARATIVE COMPOSITION      │
└─────────────────────────────────────────────────────────────────────────┘

Paradigm: Declarative chaining with pipe operators

Best for:
  - Simple linear or parallel chains
  - When you want minimal boilerplate
  - Quick prototyping with standard components

Pros:
  ✓ Concise syntax with | pipe operator
  ✓ Built-in streaming, batching, and async support
  ✓ Automatic error handling and retries
  ✓ Works seamlessly with existing LangChain components

Cons:
  ✗ Less intuitive for complex control flow
  ✗ Harder to debug (abstracted execution)
  ✗ Limited flexibility for conditional logic
  ✗ Can feel "magical" or implicit

Example structure:
    chain = prompt | llm | parser
    parallel_chain = RunnableParallel({"a": chain1, "b": chain2})


┌─────────────────────────────────────────────────────────────────────────┐
│ 2. LangGraph Functional API (@task/@entrypoint) - IMPERATIVE           │
└─────────────────────────────────────────────────────────────────────────┘

Paradigm: Imperative programming with decorator-based concurrency

Best for:
  - Python developers who prefer explicit control flow
  - When you want to see exactly what runs when
  - Workflows that benefit from standard Python patterns
  - Simple to moderate complexity tasks

Pros:
  ✓ Most Pythonic - looks like regular Python code
  ✓ Explicit control flow - easy to read and understand
  ✓ Familiar futures/async patterns
  ✓ Simple mental model (decorate, invoke, get result)
  ✓ Less ceremony than StateGraph

Cons:
  ✗ Limited to simpler workflows
  ✗ No built-in state persistence/checkpointing
  ✗ Harder to add conditional branching mid-workflow
  ✗ Less tooling for visualization

Example structure:
    @task
    def my_task(input):
        return result

    @entrypoint()
    def workflow(input):
        future = my_task(input)
        return future.result()


┌─────────────────────────────────────────────────────────────────────────┐
│ 3. LangGraph StateGraph API - DECLARATIVE GRAPH                        │
└─────────────────────────────────────────────────────────────────────────┘

Paradigm: Declarative state machine / workflow orchestration

Best for:
  - Complex workflows with branching, loops, or conditional logic
  - Multi-agent systems
  - When you need to inspect/debug intermediate states
  - Workflows that require checkpointing and state persistence
  - When you want to visualize the execution graph

Pros:
  ✓ Handles complex workflows (conditionals, loops, human-in-the-loop)
  ✓ Built-in state management and persistence
  ✓ Excellent debugging - can inspect state at any node
  ✓ Graph visualization tools available
  ✓ Supports streaming updates
  ✓ Can checkpoint and resume execution

Cons:
  ✗ More boilerplate (state schema, node functions, edges)
  ✗ Steeper learning curve
  ✗ Overkill for simple linear/parallel tasks
  ✗ More verbose than other approaches

Example structure:
    class State(TypedDict):
        field1: str
        field2: str

    def node_fn(state: State) -> dict:
        return {"field1": updated_value}

    graph = StateGraph(State)
    graph.add_node("node_name", node_fn)
    graph.add_edge(START, "node_name")
    graph.add_edge("node_name", END)


┌─────────────────────────────────────────────────────────────────────────┐
│ DECISION MATRIX: Which Approach to Use?                                │
└─────────────────────────────────────────────────────────────────────────┘

Use LCEL when:
  • Building simple chains or parallel operations
  • Integrating existing LangChain components
  • Prototyping quickly
  • You're comfortable with functional composition

Use Functional API (@task/@entrypoint) when:
  • You prefer explicit, Pythonic code
  • Workflow is straightforward (parallel tasks + aggregation)
  • You don't need state persistence or complex branching
  • Team is more familiar with imperative programming

Use StateGraph API when:
  • Complex workflows with conditional branching or loops
  • Building multi-agent systems
  • Need to inspect/debug intermediate states
  • Require checkpointing and state persistence
  • Want to visualize execution flow
  • Building production-grade agents


┌─────────────────────────────────────────────────────────────────────────┐
│ MIGRATION PATH                                                          │
└─────────────────────────────────────────────────────────────────────────┘

Simple → Complex:

  Start with LCEL
       ↓
  If control flow becomes unclear → Functional API
       ↓
  If you need branching/state/checkpointing → StateGraph API

All three approaches can coexist in the same codebase, and you can even
mix them (e.g., use LCEL chains as nodes in a StateGraph).
"""


# ============================================================================
# ASYNC/AWAIT: Concurrency vs Parallelism
# ============================================================================

"""
IMPORTANT CONCEPT: CONCURRENCY ≠ PARALLELISM

┌─────────────────────────────────────────────────────────────────────────┐
│ CONCURRENCY                                                             │
└─────────────────────────────────────────────────────────────────────────┘

Definition: Multiple tasks making progress during overlapping time periods
            (but not necessarily at the exact same instant)

How it works:
  - Single CPU core switches between tasks
  - While Task A waits for I/O (API call, disk read), Task B runs
  - Tasks take turns using the CPU
  - Cooperative multitasking (async/await) or preemptive (threading)

Python Implementation: async/await (asyncio)

Example Timeline:
  Task A: [===API_CALL====================][process]
  Task B:             [===API_CALL====================][process]
  Task C:                         [===API_CALL====================][process]
  CPU:    [A][B][C][wait][A][B][C][wait][A][B][C]

Best for: I/O-bound operations (API calls, database queries, file I/O)


┌─────────────────────────────────────────────────────────────────────────┐
│ PARALLELISM                                                             │
└─────────────────────────────────────────────────────────────────────────┘

Definition: Multiple tasks executing simultaneously on different CPU cores

How it works:
  - Multiple CPU cores each running a separate task
  - True simultaneous execution
  - Each task has its own memory space (in Python multiprocessing)

Python Implementation: multiprocessing (due to GIL*)

Example Timeline:
  CPU 1: [===========Task A===========]
  CPU 2: [===========Task B===========]
  CPU 3: [===========Task C===========]
  (All running at the EXACT same time)

Best for: CPU-bound operations (heavy computation, data processing)

*GIL (Global Interpreter Lock): Python's limitation that prevents true
 parallelism with threads. Only one thread can execute Python bytecode
 at a time, even on multi-core systems.


┌─────────────────────────────────────────────────────────────────────────┐
│ WHY ASYNC/AWAIT MATTERS FOR LLM APPLICATIONS                           │
└─────────────────────────────────────────────────────────────────────────┘

LLM API calls are I/O-bound:
  1. Send request to API (fast)
  2. Wait for LLM to process (1-10+ seconds) ← BLOCKING
  3. Receive response (fast)

Without async:
  [Task 1: API call ====== wait ======= response]
                                                  [Task 2: API call ====== wait =======]
  Total time: 10s + 10s = 20 seconds

With async:
  [Task 1: API call ====== wait ====== response]
  [Task 2: API call ====== wait ====== response]
  Total time: ~10 seconds (both wait concurrently)

Async is ideal because:
  ✓ LLM calls spend 99% of time waiting for network I/O
  ✓ No CPU-intensive work (no need for true parallelism)
  ✓ Can handle 100+ concurrent requests on a single core
  ✓ Lower memory overhead than multiprocessing
  ✓ More efficient resource usage


┌─────────────────────────────────────────────────────────────────────────┐
│ WHY THE EXAMPLES DON'T ALWAYS SHOW ASYNC EXPLICITLY                    │
└─────────────────────────────────────────────────────────────────────────┘

1. LCEL Example (lines 93-115):
   ✓ Uses async/await explicitly
   ✓ `async def run_parallel_example()` and `await full_parallel_chain.ainvoke()`
   ✓ RunnableParallel automatically handles concurrent execution

2. Functional API Example (lines 189-207):
   ✗ Appears synchronous (uses .invoke() not .ainvoke())
   ✓ BUT: LangGraph handles async internally via futures
   ✓ When you call a @task, it returns a future and runs concurrently
   ✓ .result() blocks until the task completes

3. StateGraph API Example (lines 300-324):
   ✗ Node functions are synchronous
   ✓ BUT: LangGraph automatically runs nodes concurrently when possible
   ✓ Graph edges with multiple outgoing connections = concurrent execution

The framework abstracts away the async complexity, but it's happening
under the hood!
"""


# ============================================================================
# Explicit Async Examples
# ============================================================================

"""
Below are examples showing EXPLICIT async/await usage for each approach.
Use these when you need fine-grained control over async behavior.
"""

# --- Example 1: Explicit Async with Pure Python (no framework) ---

async def async_summarize(topic: str) -> str:
    """Async function to generate a summary."""
    response = await llm.ainvoke([
        {"role": "system", "content": "Summarize the following topic concisely:"},
        {"role": "user", "content": topic}
    ])
    return response.content


async def async_generate_questions(topic: str) -> str:
    """Async function to generate questions."""
    response = await llm.ainvoke([
        {"role": "system", "content": "Generate three interesting questions about the following topic:"},
        {"role": "user", "content": topic}
    ])
    return response.content


async def async_identify_terms(topic: str) -> str:
    """Async function to identify key terms."""
    response = await llm.ainvoke([
        {"role": "system", "content": "Identify 5-10 key terms from the following topic, separated by commas:"},
        {"role": "user", "content": topic}
    ])
    return response.content


async def async_synthesize(topic: str, summary: str, questions: str, terms: str) -> str:
    """Async function to synthesize results."""
    response = await llm.ainvoke([
        {"role": "system", "content": f"""Based on the following information:

Summary: {summary}
Related Questions: {questions}
Key Terms: {terms}

Synthesize a comprehensive answer."""},
        {"role": "user", "content": f"Original topic: {topic}"}
    ])
    return response.content


async def run_pure_async_example(topic: str) -> str:
    """
    Pure Python async/await example using asyncio.gather for concurrency.

    This is the most explicit way to handle concurrent LLM calls in Python.
    """
    # Launch all three tasks concurrently
    # asyncio.gather runs them concurrently and waits for all to complete
    summary, questions, terms = await asyncio.gather(
        async_summarize(topic),
        async_generate_questions(topic),
        async_identify_terms(topic)
    )

    # Synthesize the results
    result = await async_synthesize(topic, summary, questions, terms)
    return result


# --- Example 2: Async with StateGraph ---

async def async_generate_summary(state: ParallelState) -> dict:
    """Async node function for StateGraph."""
    response = await llm.ainvoke([
        {"role": "system", "content": "Summarize the following topic concisely:"},
        {"role": "user", "content": state["topic"]}
    ])
    return {"summary": response.content}


async def async_generate_questions_node(state: ParallelState) -> dict:
    """Async node function for StateGraph."""
    response = await llm.ainvoke([
        {"role": "system", "content": "Generate three interesting questions about the following topic:"},
        {"role": "user", "content": state["topic"]}
    ])
    return {"questions": response.content}


async def async_generate_key_terms(state: ParallelState) -> dict:
    """Async node function for StateGraph."""
    response = await llm.ainvoke([
        {"role": "system", "content": "Identify 5-10 key terms from the following topic, separated by commas:"},
        {"role": "user", "content": state["topic"]}
    ])
    return {"key_terms": response.content}


async def async_synthesize_node(state: ParallelState) -> dict:
    """Async synthesis node for StateGraph."""
    response = await llm.ainvoke([
        {"role": "system", "content": f"""Based on the following information:

Summary: {state['summary']}
Related Questions: {state['questions']}
Key Terms: {state['key_terms']}

Synthesize a comprehensive answer."""},
        {"role": "user", "content": f"Original topic: {state['topic']}"}
    ])
    return {"combined_output": response.content}


# Build async StateGraph
async_graph_builder = StateGraph(ParallelState)
async_graph_builder.add_node("generate_summary", async_generate_summary)
async_graph_builder.add_node("generate_questions", async_generate_questions_node)
async_graph_builder.add_node("generate_key_terms", async_generate_key_terms)
async_graph_builder.add_node("synthesize", async_synthesize_node)

async_graph_builder.add_edge(START, "generate_summary")
async_graph_builder.add_edge(START, "generate_questions")
async_graph_builder.add_edge(START, "generate_key_terms")
async_graph_builder.add_edge("generate_summary", "synthesize")
async_graph_builder.add_edge("generate_questions", "synthesize")
async_graph_builder.add_edge("generate_key_terms", "synthesize")
async_graph_builder.add_edge("synthesize", END)

async_graph_workflow = async_graph_builder.compile()


# Uncomment to run async examples:
# if __name__ == "__main__":
#     test_topic = "The history of space exploration"
#
#     # Example 1: Pure async/await with asyncio.gather
#     print("\n--- Pure Async/Await Example ---\n")
#     result = asyncio.run(run_pure_async_example(test_topic))
#     print(result)
#
#     # Example 2: Async StateGraph
#     print("\n--- Async StateGraph Example ---\n")
#     result = asyncio.run(async_graph_workflow.ainvoke({"topic": test_topic}))
#     print(result["combined_output"])


"""
┌─────────────────────────────────────────────────────────────────────────┐
│ PERFORMANCE COMPARISON                                                  │
└─────────────────────────────────────────────────────────────────────────┘

Assume each LLM call takes 3 seconds:

Sequential (no concurrency):
  summary: 3s → questions: 3s → terms: 3s → synthesis: 3s
  Total: 12 seconds

Concurrent (async/await or framework-managed):
  [summary: 3s]
  [questions: 3s]  ← All three run concurrently
  [terms: 3s]
  synthesis: 3s
  Total: ~6 seconds (50% faster!)

True Parallelism (multiprocessing - overkill for LLMs):
  Same result as concurrent, but with:
  - Much higher memory overhead (separate processes)
  - Process spawning overhead
  - Not recommended for I/O-bound tasks


┌─────────────────────────────────────────────────────────────────────────┐
│ KEY TAKEAWAYS                                                           │
└─────────────────────────────────────────────────────────────────────────┘

1. LLM applications are I/O-bound → Use CONCURRENCY (async/await)
2. Concurrency = tasks take turns, Parallelism = tasks run simultaneously
3. Python's asyncio is perfect for concurrent I/O operations
4. LangChain/LangGraph handle async internally, but you can be explicit
5. Use async when:
   - Making multiple independent LLM calls
   - Calling external APIs
   - Performing database queries
   - Reading/writing many files
6. Don't use parallelism (multiprocessing) for LLM calls - it's overkill

Prefer: async/await > threading > multiprocessing (for LLM applications)
"""
