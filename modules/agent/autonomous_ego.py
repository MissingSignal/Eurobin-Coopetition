from typing import Annotated, Literal, TypedDict
import os
import time

import click
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage

LANGSMITH_TRACING="true"
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT= os.getenv("AZURE_OPENAI_ENDPOINT")

# Define the tools for the agent to use
@tool
def go_to_location(query: str):
    """call to navigate to a location."""
    click.secho(f"ACTION: going to <{query}>", fg='green', bold=True, italic=True)
    # with click.progressbar([1, 2, 3]) as bar:
    #     for x in bar:
    #         #print(f"sleep({x})...")
    #         time.sleep(x)
    return True

@tool
def go_to_sublocation(query: str):
    """call to select from which area of the location to pick an object. It can be Dishwasher, Table or Cabinet. Call this after go_to_location."""
    click.secho(f"ACTION: selecting area <{query}>", fg='green', bold=True, italic=True)
    if query.lower() in ["dishwasher", "table", "cabinet"]:
        return True
    else:
        click.secho(f"ERROR: invalid sublocation <{query}>, choose between Dishwasher,Table or Cabinet", fg='red', bold=True, italic=True)
        return False

@tool
def pick(query: str):
    """call to grasp an object"""
    click.secho(f"ACTION: picking <{query}>", fg='green', bold=True, italic=True)
    return True

@tool
def place(query: str):
    """call to place an object"""
    click.secho("ACTION: placing object", fg='green', bold=True, italic=True)
    return True

@tool
def give(query: str):
    """call to give object to human definde by query"""
    click.secho(f"ACTION: giving object to <{query}>", fg='green', bold=True, italic=True)
    return True

@tool
def understand_scene(query: str):
    """call to analize the scene using VLLM. e.g. what is in the scene, what color is the apple, etc."""
    click.secho(f"ACTION: understanding the prompt based on: {query}]", fg='green', bold=True, italic=True)
    return True

tools = [go_to_location, go_to_sublocation, pick, place, give, understand_scene]

action_node = ToolNode(tools)

model = AzureChatOpenAI(
    openai_api_version="2023-09-15-preview",
    deployment_name="contact-Raisespoke4_gpt4omini",
    model = "gpt4o-mini",
    max_tokens=128,
).bind_tools(tools)


def should_continue(state: MessagesState) -> Literal["action", END]:
    """Determine whether to continue or not."""
    messages = state['messages']
    last_message = messages[-1]

    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "action"
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
workflow.add_edge("action", "agent")

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
robot = workflow.compile(checkpointer=checkpointer)


### INTERACT WITH EGO ###

# Use the Runnable
def chat_with_robot():
    """ main interaction loop with the robot """
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat.")
                break

            current_state = robot.invoke(
                {"messages": [SystemMessage("esegui task scomponendoli in azioni che siano di questo tipo GO_TO_LOCATION() GO_TO_SUBLOCTION() , PICK(), PLACE()/GIVE(). Ad esempio GO_TO_LOCATION(bagno) GO_TO_SUBLOCTION(vasca) , PICK(mela), PLACE()") , HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": 42}},
            )

            response = current_state["messages"][-1].content
            print(f"Robot: {response}")

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break

if __name__ == "__main__":
    chat_with_robot()
