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

objects = ["cleanser","sponge","bowl", "plate", "mug", "fork", "knife", "spoon","chocolate_jello", "strawberry_jello", "coffee_grounds", "mustard", "spam", "sugar", "tomato_soup", "tuna","bowl", "plate", "mug", "fork", "knife", "spoon","cleanser",
            "sponge","apple", "banana", "lemon", "orange", "peach", "pear", "plum", "strawberry", "cheezit", "cornflakes", "pringles","baseball", "dice", "rubiks_cube", "soccer_ball", "tennis_ball"]

# Define the tools for the agent to use
@tool
def go_to_location(location: str):
    """
    Call to navigate to a location. It can be INRIA, KIT or DLR. Call this before pick() or place() tools. call this tool only once.
    
    Args:
        location (str): INRIA, KIT or DLR
    """

    click.secho(f"ACTION: going to <{location}>", fg='green', bold=True, italic=True)
    if query.lower() in ["inria", "kit", "dlr"]:
        return True
    else:
        click.secho(f"ERROR: invalid location <{query}>, choose between INRIA, KIT or DLR", fg='red', bold=True, italic=True)
        return False

@tool
def go_to_sublocation(sublocation: str):
    """
    Call to select from which area of the location to pick an object. call this tool only once.
    Args:
        query (str): Dishwasher, Table or Cabinet
    """


    click.secho(f"ACTION: selecting area <{sublocation}>", fg='green', bold=True, italic=True)
    if query.lower() in ["dishwasher", "table", "cabinet"]:
        return True
    else:
        click.secho(f"ERROR: invalid sublocation <{sublocation}>, choose between Dishwasher,Table or Cabinet", fg='red', bold=True, italic=True)
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

@tool
def execute_plan(object: str, origin_location: str, origin_sublocation: str, target_location: str, target_sublocation: str):
    """call this tool when the plan is ready to be executed
    Args:
        object (str): object to pick referred as O
        origin_location (str): location to go (INRIA, KIT or DLR) referred K_origin
        origin_sublocation (str): sublocation to pick object from (Dishwasher, Table or Cabinet) referred as L_origin 
        target_location (str): location to place object (INRIA, KIT or DLR) referred as K_target
        target_sublocation (str): sublocation to place object (Dishwasher, Table or Cabinet, person) referred as L_target
    """
    print("\n")
    click.secho("[ACTION:] THE PLAN IS SET", fg='green', bold=True)
    click.secho("Moving", nl=False)
    click.secho(f" {object} ", fg='yellow', bold=True, nl=False)
    click.secho("from", nl=False)
    click.secho(f" {origin_location} ", fg='blue', bold=True, nl=False)
    click.secho(f" {origin_sublocation} ", fg='red', bold=True, nl=False)
    click.secho(f"to", nl=False)
    click.secho(f" {target_location} ", fg='blue', bold=True, nl=False)
    click.secho(f" {target_sublocation} ", fg='red', bold=True)
    print("\n")

    return True

#Pick the jello from the DLR cabinet and place it on the INRIA cabinet

# @tool
# def execute_plan():
#     """call this tool when the plan is ready to be executed"""
#     click.secho(f"ACTION: ready to go", fg='green', bold=True, italic=True)
#     return True 

tools = [execute_plan]

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
            
            #Pick Object Apple at Location Table in Kitchen INRIA
            current_state = robot.invoke(
                {"messages": [SystemMessage(f"Given a prompt in the format: Pick Object 'O' at Location 'L_origin' in Kitchen 'K_origin' and place it on 'K_target' 'L_target'. Where O={{{', '.join(objects)}}}, L_target={{Dishwasher, Table, Cabinet}}, L_origin={{Dishwasher, Table, Cabinet, person}} K_target={{DLR, KIT, INRIA}} and K_origin={{DLR, KIT, INRIA}}. Extract O, L_origin, L_target, K_origin and K target. if you are not sure about one component ask for confirmation"), HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": 42}},
            )

            response = current_state["messages"][-1].content
            print(f"Robot: {response}")

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break

if __name__ == "__main__":
    chat_with_robot()
