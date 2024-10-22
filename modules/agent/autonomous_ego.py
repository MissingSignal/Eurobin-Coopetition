from typing import Annotated, Literal, TypedDict
import os

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

LANGSMITH_TRACING="true"
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT= os.getenv("AZURE_OPENAI_ENDPOINT")

# Define the tools for the agent to use
@tool
def go_to(query: str):
    """call to navigate to a location."""
    print("going to", query)
    return True

@tool
def pick(query: str):
    """call to grasp an object"""
    print(f"grasp {query}")
    return True

@tool
def place(query: str):
    """call to place an object"""
    print(f"placing object in {query}")
    return True

@tool
def give():
    """call to give object to human"""
    print("giving object")
    return True

@tool
def understand_scene(query: str):
    """call to analize the scene using VLLM. e.g. what is in the scene, what color is the apple, etc."""
    print(f"understandig scene {query}")
    return True

tools = [go_to, pick, place, give, understand_scene]

action_node = ToolNode(tools)

model = AzureChatOpenAI(
    openai_api_version="2023-09-15-preview",
    deployment_name="contact-Raisespoke4_gpt4omini",
    model = "gpt4o-mini",
    max_tokens=128,
).bind_tools(tools)

def should_continue(state: MessagesState) -> Literal["tools", END]:
    """Determine whether to continue or not."""
    messages = state['messages']
    last_message = messages[-1]

    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


def call_model(state: MessagesState):
    """Call the model."""
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
# 1) agent: node used when the robot is talking to the user
# 2) tools: node used when the robot is acting using tools

workflow.add_node("agent", call_model)
workflow.add_node("action", action_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `action` to `agent`.
# This means that after `action` is called, `agent` node is called next.
workflow.add_edge('action', 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
robot = workflow.compile(checkpointer=checkpointer)


### INTERACT WITH EGO ###

# Use the Runnable
current_state = robot.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config={"configurable": {"thread_id": 42}}
)

print(current_state["messages"][-1].content)
