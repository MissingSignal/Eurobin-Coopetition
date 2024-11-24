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
from eurobin_coopetition.srv import EnableAutoModeService, WhereRUService, HappyPoseService, PickService, NavService, WaitPilotService
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

# Main interaction with the robot via ROS topic
class RobotAgent:
    def __init__(self):

        tools = [self.execute_plan]
        self.action_node = ToolNode(tools)
        self.llm_model = AzureChatOpenAI(
            openai_api_version="2023-09-15-preview",
            deployment_name="contact-Raisespoke4_gpt4omini",
            model="gpt4o-mini",
            max_tokens=128,
        ).bind_tools(tools)


        #############################################################################################
        # Define a new graph
        workflow = StateGraph(MessagesState)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", self.call_model)
        workflow.add_node("action", self.action_node)

        # Set the entrypoint as `agent`
        workflow.add_edge(START, "agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
        )

        # We now add a normal edge from `action` to `agent`.
        workflow.add_edge("action", "agent")

        # Initialize memory to persist state between graph runs
        checkpointer = MemorySaver()

        # Compile the workflow into a runnable robot
        robot = workflow.compile(checkpointer=checkpointer)
        #############################################################################################
        self.robot = robot

        # Initialize the ROS node
        rospy.init_node('chat_with_robot', anonymous=False)
        # Get robot name from environment variable
        self.robot_name = os.getenv('ROBOT_NAME', 'robot_alterego3')
 
        # Subscriber for recognized speech
        rospy.Subscriber(f"/{self.robot_name}/recognized_speech", String, self.callback)
        # Subscriber for the robot speaking status
        rospy.Subscriber(f"/{self.robot_name}/is_robot_speaking", Bool, self.callback_speak)
        self.text_pub = rospy.Publisher(f"/{self.robot_name}/text2speech", String, queue_size=10)

        self.robot_speaking = None

    def callback_speak(self, msg):
        self.robot_speaking = msg.data

    def callback(self, msg):
        if not self.robot_speaking:        
            user_input = msg.data
            rospy.loginfo(f"Received input: {user_input}")
            
            # Process the input using the state graph
            current_state = self.robot.invoke(
                {"messages": [
                    SystemMessage(f"Given a prompt in the format: Pick Object 'O' at Location 'L_origin' in Kitchen 'K_origin' and place it on 'K_target' 'L_target'. Where O={{{', '.join(objects)}}}, L_target={{Dishwasher, Table, Cabinet}}, L_origin={{Dishwasher, Table, Drawer, Counter, Cabinet, person}}, K_target={{DLR, KIT, INRIA}}, and K_origin={{DLR, KIT, INRIA}}. If you are not sure about one component, ask for confirmation. Never list to the user all the objects that can be grasped."),
                    HumanMessage(content=user_input)
                ]},
                config={"configurable": {"thread_id": 42}},
            )

            response = current_state["messages"][-1].content
            rospy.loginfo(f"Robot: {response}")
            self.text_pub.publish(self.clean_answer_from_markdown(response))
        
    def call_service(self, service_name, service_type, **kwargs):
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
    def execute_plan(self, object_: str, origin_location: str, origin_sublocation: str, target_location: str, target_sublocation: str) -> bool:
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
        click.secho(f" {object_} ", fg='yellow', bold=True, nl=False)
        click.secho("from", nl=False)
        click.secho(f" {origin_location} ", fg='blue', bold=True, nl=False)
        click.secho(f" {origin_sublocation} ", fg='red', bold=True, nl=False)
        click.secho(f"to", nl=False)
        click.secho(f" {target_location} ", fg='blue', bold=True, nl=False)
        click.secho(f" {target_sublocation} ", fg='red', bold=True)
        print("\n")

        text2pub = (f"Moving {object_} from {origin_location} {origin_sublocation} to {target_location} {target_sublocation}")
        
        # # #--------------------------------------------------------------------------------------------------------ENABLE AUTO MODE
        rospy.loginfo("Enabling the auto mode")
        response = self.call_service('/enable_auto_mode_service', EnableAutoModeService, enable=True)
        if not response:
            return False

        
        # # #--------------------------------------------------------------------------------------------------------GET THE STARTING POINT
        rospy.loginfo("Requesting the starting point")
        response = self.call_service('/where_are_you_service', WhereRUService, request=True)
        if not response:
            return False
        instruction_point = response.instruction_point
        rospy.loginfo("Starting from the location: " + instruction_point)

        # # #--------------------------------------------------------------------------------------------------------GO TO ORIGIN LOCATION
        if instruction_point == origin_location:
            rospy.loginfo("Already in the origin location")
            rospy.loginfo("Requesting navigation to go to origin sub location")
            response = self.call_service('/navigation_service', NavService, location=origin_location, sub_location=origin_sublocation)
            if not response:
                return False
        else:
            if instruction_point == "KIT" or instruction_point == "DLR":
                if origin_location == "INRIA" :
                    rospy.loginfo("Requesting navigation to go to the door KIT-INRIA")
                    response = self.call_service('/navigation_service', NavService, location="Door", sub_location="KIT-INRIA")
                    if not response:
                        return False
                    # # #--------------------------------------------------------------------------------------------------------DISABLE AUTO MODE
                    rospy.loginfo("Disabling the auto mode")
                    response = self.call_service('/enable_auto_mode_service', EnableAutoModeService, enable=False)
                    if not response:
                        return False
                    # # #--------------------------------------------------------------------------------------------------------SEND AUDIO REQUEST
                    rospy.loginfo("Sending audio request")
                    self.text_pub.publish("Now the pilot will help me to open the door")
                    
                    # # #--------------------------------------------------------------------------------------------------------WAIT THE PILOT IN/OUT
                    rospy.loginfo("Requesting help from the pilot")
                    response = self.call_service('/wait_help_from_pilot_service', WaitPilotService, request=True)
                    
                    while True:
                        response = self.call_service('/wait_help_from_pilot_service', WaitPilotService, request=False)
                        if response:
                            break
                        rospy.sleep(0.1)
                    # # #--------------------------------------------------------------------------------------------------------CONTINUE NAVIGATION TO THE ORIGIN LOCATION
                    response = self.call_service('/navigation_service', NavService, location=origin_location, sub_location=origin_sublocation)
                    if not response:
                        return False

                    
            elif instruction_point == "INRIA":
                if origin_location == "KIT" or origin_location == "DLR":
                    rospy.loginfo("C'è la porta sud")
                    rospy.loginfo("Requesting navigation to go to the door KIT-INRIA")
                    response = self.call_service('/navigation_service', NavService, location="Door", sub_location="INRIA-KIT")
                    if not response:
                        return False
                    # # #--------------------------------------------------------------------------------------------------------DISABLE AUTO MODE
                    rospy.loginfo("Disabling the auto mode")
                    response = self.call_service('/enable_auto_mode_service', EnableAutoModeService, enable=False)
                    if not response:
                        return False
                    # # #--------------------------------------------------------------------------------------------------------SEND AUDIO REQUEST
                    rospy.loginfo("Sending audio request")
                    self.text_pub.publish("Now the pilot will help me to open the door")                    
                    # # #--------------------------------------------------------------------------------------------------------WAIT THE PILOT IN/OUT
                    rospy.loginfo("Requesting help from the pilot")
                    response = self.call_service('/wait_help_from_pilot_service', WaitPilotService, request=True)
                    
                    while True:
                        response = self.call_service('/wait_help_from_pilot_service', WaitPilotService, request=False)
                        if response:
                            break
                        rospy.sleep(0.1)
                    # # #--------------------------------------------------------------------------------------------------------CONTINUE NAVIGATION TO THE ORIGIN LOCATION
                    response = self.call_service('/navigation_service', NavService, location=origin_location, sub_location=origin_sublocation)
                    if not response:
                        return False


            elif instruction_point == "DLR":
                if origin_location == "KIT":
                    rospy.loginfo("Requesting navigation to go to origin sub location")
                    response = self.call_service('/navigation_service', NavService, location=origin_location, sub_location=origin_sublocation)
                    if not response:
                        return False
            elif instruction_point == "KIT":
                if origin_location == "DLR":
                    rospy.loginfo("Requesting navigation to go to origin sub location")
                    response = self.call_service('/navigation_service', NavService, location=origin_location, sub_location=origin_sublocation)
                    if not response:
                        return False

        #--------------------------------------------------------------------------------------------------------WAIT THE PILOT 
        rospy.loginfo("waiting the pilot to enter and move the head")
            # # #--------------------------------------------------------------------------------------------------------SEND AUDIO REQUEST
        rospy.loginfo("Sending audio request")
        self.text_pub.publish("Now the pilot will help me to move the head")
        
            # # #--------------------------------------------------------------------------------------------------------WAIT THE PILOT TO MOVE THE HEAD
        rospy.loginfo("Requesting help from the pilot")
        response = self.call_service('/wait_help_from_pilot_service', WaitPilotService, request=True)
        
        while True:
            response = self.call_service('/wait_help_from_pilot_service', WaitPilotService, request=False)
            if response:
                break
            rospy.sleep(0.1)

        #--------------------------------------------------------------------------------------------------------DETECT OBJECT
        rospy.loginfo("Requesting happypose to find the object")
        response = self.call_service('/happypose_service', HappyPoseService, object=object)
        if not response:
            return False

        position = response.position
        orientation = response.orientation
        rospy.loginfo(f"Object {object} found at position ({position.x}, {position.y}, {position.z}) and orientation ({orientation.x}, {orientation.y}, {orientation.z}, {orientation.w})")
        rospy.sleep(4)
        rospy.loginfo("Pick the object")

        #--------------------------------------------------------------------------------------------------------WAIT THE PILOT TO PICK THE OBJECT
        rospy.loginfo("waiting the pilot to enter and pick the object")
            # # #--------------------------------------------------------------------------------------------------------SEND AUDIO REQUEST
        rospy.loginfo("Sending audio request")
        self.text_pub.publish("Now the pilot will help me to pick the object")
        
            # # #--------------------------------------------------------------------------------------------------------WAIT THE PILOT TO PICK THE OBJECT
        rospy.loginfo("Requesting help from the pilot")
        response = self.call_service('/wait_help_from_pilot_service', WaitPilotService, request=True)
        
        while True:
            response = self.call_service('/wait_help_from_pilot_service', WaitPilotService, request=False)
            if response:
                break
            rospy.sleep(0.1)
        #--------------------------------------------------------------------------------------------------------GO TO THE TARGET LOCATION
        rospy.loginfo("Requesting navigation to go to target location")
        response = self.call_service('/navigation_service', NavService, location=target_location, sub_location=target_sublocation)
        if not response:
            return False

        rospy.loginfo("Moving to place the object")

        #--------------------------------------------------------------------------------------------------------WAIT THE PILOT TO PLACE THE OBJECT
        rospy.loginfo("waiting the pilot to enter and place the object")
        # # #--------------------------------------------------------------------------------------------------------SEND AUDIO REQUEST
        rospy.loginfo("Sending audio request")
        self.text_pub.publish("And finally the pilot will help me to place the object")
        
        return True



    def should_continue(self, state: MessagesState) -> Literal["action", END]:
        """Determine whether to continue or not."""
        messages = state['messages']
        last_message = messages[-1]

        # If the LLM makes a tool call, then we route to the "tools" node
        if last_message.tool_calls:
            return "action"
        # Otherwise, we stop (reply to the user)
        return END

    def call_model(self, state: MessagesState):
        """Call the model."""
        messages = state['messages']
        response = self.llm_model.invoke(messages)

        print(response)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}
    
    def clean_answer_from_markdown(self, answer):
        """Clean the answer from markdown."""
        return answer.replace("**", "").replace("##", "").replace("###","") 

if __name__ == "__main__":
    try:
        chat_with_robot = RobotAgent()
        ########
        # DEBUG
        print("DEBUG")
        current_state = chat_with_robot.robot.invoke(
            {"messages": [
                SystemMessage(f"Given a prompt in the format: Pick Object 'O' at Location 'L_origin' in Kitchen 'K_origin' and place it on 'K_target' 'L_target'. Where O={{{', '.join(objects)}}}, L_target={{Dishwasher, Table, Cabinet}}, L_origin={{Dishwasher, Table, Drawer, Counter, Cabinet, person}}, K_target={{DLR, KIT, INRIA}}, and K_origin={{DLR, KIT, INRIA}}. If you are not sure about one component, ask for confirmation. Never list to the user all the objects that can be grasped."),
                HumanMessage(content="take the apple from the table in the INRIA kitchen and place it on the KIT table")
            ]},
            config={"configurable": {"thread_id": 42}},
        )
        ###################
        rospy.spin()
    except rospy.ROSInterruptException:
        pass