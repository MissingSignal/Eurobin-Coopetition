import os
import rospy
import click
import dotenv
import numpy as np
from std_msgs.msg import String
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import AzureChatOpenAI
from geometry_msgs.msg import Point, Quaternion
from typing import Annotated, Literal, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from eurobin_coopetition.srv import EnableAutoModeService, WhereRUService, HappyPoseService, PickService, NavService 
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from std_msgs.msg import Bool

dotenv.load_dotenv()

# Get environment variables
LANGSMITH_TRACING = "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Hard coded list of available objects
objects = [
    "cleanser", "sponge", "bowl", "plate", "mug", "fork", "knife", "spoon", "chocolate_jello", "strawberry_jello",
    "coffee_grounds", "mustard", "spam", "sugar", "tomato_soup", "tuna", "bowl", "plate", "mug", "fork", "knife",
    "spoon", "cleanser", "sponge", "apple", "banana", "lemon", "orange", "peach", "pear", "plum", "strawberry",
    "cheezit", "cornflakes", "pringles", "baseball", "dice", "rubiks_cube", "soccer_ball", "tennis_ball"
]

def call_service(service_name, service_type, **kwargs):
    rospy.wait_for_service(service_name)
    try:
        service = rospy.ServiceProxy(service_name, service_type)
        response = service(**kwargs)
        if response.success:
            rospy.loginfo(f"{service_name} action was successful.")
            return response
        else:
            rospy.logwarn(f"{service_name} action failed.")
            return None
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call to {service_name} failed: {e}")
        return None
    
# Define the tools for the agent to use
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

    
    #--------------------------------------------------------------------------------------------------------ENABLE AUTO MODE
    rospy.loginfo("Enabling the auto mode")
    response = call_service('/enable_auto_mode_service', EnableAutoModeService, enable=True)
    if not response:
        return False

    
    #--------------------------------------------------------------------------------------------------------GET THE STARTING POINT
    rospy.loginfo("Requesting the starting point")
    response = call_service('/where_are_you_service', WhereRUService, request=True)
    if not response:
        return False
    instruction_point = response.instruction_point
    rospy.loginfo("Starting from the location: " + instruction_point)

    #--------------------------------------------------------------------------------------------------------GO TO ORIGIN LOCATION
    if instruction_point == origin_location:
        rospy.loginfo("Requesting navigation to go to origin location")
        response = call_service('/navigation_service', NavService, location=origin_location, sub_location=origin_sublocation)
        if not response:
            return False
    else:
        rospy.loginfo("Already in the origin location")
        if instruction_point == "INRIA" or instruction_point == "DLR":
            if origin_location == "KIT" :
                rospy.loginfo("C'è la porta nord")
                rospy.loginfo("Requesting navigation to go to the door")
                response = call_service('/navigation_service', NavService, location="Door", sub_location="INRIA-KIT")
                if not response:
                    return False
        elif instruction_point == "KIT":
            if origin_location == "INRIA" or origin_location == "DLR":
                rospy.loginfo("C'è la porta sud")
                rospy.loginfo("Requesting navigation to go to the door")
                response = call_service('/navigation_service', NavService, location="Door", sub_location="KIT-INRIA")
                if not response:
                    return False
        elif instruction_point == "DLR":
            if origin_location == "INRIA":
                response = call_service('/navigation_service', NavService, location=origin_location, sub_location=origin_sublocation)
                if not response:
                    return False
        elif instruction_point == "INRIA":
            if origin_location == "DLR":
                response = call_service('/navigation_service', NavService, location=origin_location, sub_location=origin_sublocation)
                if not response:
                    return False


    rospy.loginfo("Moving to take the object")
    #--------------------------------------------------------------------------------------------------------DETECT OBJECT
    rospy.loginfo("Requesting happypose to find the object")
    response = call_service('/happypose_service', HappyPoseService, object=object)
    if not response:
        return False

    position = response.position
    orientation = response.orientation
    rospy.loginfo(f"Object {object} found at position ({position.x}, {position.y}, {position.z}) and orientation ({orientation.x}, {orientation.y}, {orientation.z}, {orientation.w})")
    rospy.sleep(4)
    rospy.loginfo("Pick the object")
    #--------------------------------------------------------------------------------------------------------PICK THE OBJECT
    rospy.loginfo("Requesting pick service to pick the object")
    response = call_service('/pick_service', PickService, object=object, position=position, orientation=orientation)
    if not response:
        return False

    rospy.loginfo("Moving to place the object")
    #--------------------------------------------------------------------------------------------------------GO TO THE TARGET LOCATION
    rospy.loginfo("Requesting navigation to go to target location")
    response = call_service('/navigation_service', NavService, location=target_location, sub_location=target_sublocation)
    if not response:
        return False

    rospy.loginfo("Moving to place the object")

    return True

tools = [execute_plan]

action_node = ToolNode(tools)

model = AzureChatOpenAI(
    openai_api_version="2023-09-15-preview",
    deployment_name="contact-Raisespoke4_gpt4omini",
    model="gpt4o-mini",
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
workflow.add_node("agent", call_model)
workflow.add_node("action", action_node)

# Set the entrypoint as `agent`
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# We now add a normal edge from `action` to `agent`.
workflow.add_edge("action", "agent")

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Compile the workflow into a runnable robot
robot = workflow.compile(checkpointer=checkpointer)

# Main interaction with the robot via ROS topic
class ChatWithRobot:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('chat_with_robot', anonymous=False)
        # Get robot name from environment variable
        self.robot_name = os.getenv('ROBOT_NAME', 'robot_alterego3')
        self.robot = robot
        # Subscriber for recognized speech
        rospy.Subscriber(f"/{self.robot_name}/recognized_speech", String, self.callback)

    def callback(self, msg):
        user_input = msg.data
        rospy.loginfo(f"Received input: {user_input}")
        
        # Process the input using the state graph
        current_state = self.robot.invoke(
            {"messages": [
                SystemMessage(f"Given a prompt in the format: Pick Object 'O' at Location 'L_origin' in Kitchen 'K_origin' and place it on 'K_target' 'L_target'. Where O={{{', '.join(objects)}}}, L_target={{Dishwasher, Table, Cabinet}}, L_origin={{Dishwasher, Table, Cabinet, person}}, K_target={{DLR, KIT, INRIA}}, and K_origin={{DLR, KIT, INRIA}}. If you are not sure about one component, ask for confirmation."),
                HumanMessage(content=user_input)
            ]},
            config={"configurable": {"thread_id": 42}},
        )

        response = current_state["messages"][-1].content
        rospy.loginfo(f"Robot: {response}")

if __name__ == "__main__":
    try:
        chat_with_robot = ChatWithRobot()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass