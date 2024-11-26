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
import cv2
import matplotlib.pyplot as plt
from gradio_client import Client, handle_file
import json
import datetime

dotenv.load_dotenv()

# Get environment variables
LANGSMITH_TRACING = "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Hard coded list of available objects
objects = [
    "glass_cleanser", "bleach_cleanser", "hygiene_spray", "toothpaste", "bowl", "plate", "mug",
    "cracker_box", "sugar", "chocholate_jello", "strawberry_jello", "spam", "coffee", "pringles",
    "mustard", "tomato_soup", "blue_salt", "yellow_salt", "green_salt", "mashed_potatoes", "rusk",
    "amicelli", "raviolli", "powdered_sugar", "nutella_go", "cereals", "chocolate_bars",
    "banana", "strawberry", "apple", "lemon", "peach", "pear", "orange", "plum",
    "soccer_ball", "soft_ball", "baseball", "tennis_ball", "racquetball", "rubiks_cube",
    "blue_bowl", "blu_bowl", "pink_mug", "red_mug", "glass", "blue_cup", "pink_cup", "green_cup", "purple_cup", "yellow_cup", "grey_plate", "grey_cube", "red_ball", "gray_ball" # <-- THIS LINE IS FOR UNKNOW OBJECTS
]

# Main interaction with the robot via ROS topic
class RobotAgent:
    def __init__(self):
        self.force_exit = False
        self.never_listen_again = False
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
        rospy.Subscriber(f"/{self.robot_name}/force_exit", Bool, self.callback_force_exit)

        
        self.text_pub = rospy.Publisher(f"/{self.robot_name}/text2speech", String, queue_size=10)

        self.robot_speaking = None

    def callback_speak(self, msg):
        self.robot_speaking = msg.data

    def callback_force_exit(self, msg):
        self.force_exit = msg.data
        print(f"Force exit: {self.force_exit}")

    def callback(self, msg):
        if not self.robot_speaking and not self.never_listen_again:       
            user_input = msg.data
            rospy.loginfo(f"Received input: {user_input}")
            
            # Process the input using the state graph
            current_state = self.robot.invoke(
                {"messages": [
                    SystemMessage(f"Given a prompt in the format: Pick Object 'O' at Location 'L_origin' in Kitchen 'K_origin' and place it on 'K_target' 'L_target'. Where O={{{', '.join(objects)}}}, L_target={{Dishwasher, Counter, Drawer, Table, Cabinet, Person}}, L_origin={{Dishwasher, Table, Drawer, Counter, Cabinet}}, K_target={{DLR, KIT, INRIA}}, and K_origin={{DLR, KIT, INRIA}}. \
                                 If the user ask to pick/place objects from/to Table always assume Location=INRIA, Sublocation=Table. The Table is always in the INRIA kitchen. The places might be mispelled by the user, fix in this way: NERIA=INRIA KAT=KIT, K80=KIT) Never list to the user all the objects that can be grasped."),                    HumanMessage(content=user_input)
                ]},
                config={"configurable": {"thread_id": 42}},
            )

            response = current_state["messages"][-1].content
            rospy.loginfo(f"Robot: {response}")
            #self.text_pub.publish(self.clean_answer_from_markdown(response))
            self.text_pub.publish("I'm ready to start, tell me the plan")
        
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
                
    def override_cmd_for_table_presence(self, object_, instruction_point, origin_location, origin_sublocation, target_location, target_sublocation):
        # Check origin location and sublocation
        if instruction_point == "INRIA" and origin_sublocation == "Table":
            origin_location = "INRIA"
            origin_sublocation = "Table"
            text2pub = (f"Moving {object_} from the {origin_sublocation} to {target_location} {target_sublocation}")
            rospy.loginfo(text2pub)
            self.text_pub.publish(text2pub)

        elif instruction_point in ["KIT", "DLR"]:
            if origin_sublocation == "Table":
                origin_location = instruction_point
                origin_sublocation = "Table"
                text2pub = (f"Moving {object_} from the {origin_sublocation} to {target_location} {target_sublocation}")
                rospy.loginfo(text2pub)
                self.text_pub.publish(text2pub)

        # Check target location and sublocation
        if origin_location == "INRIA" and target_sublocation == "Table":
            target_location = "INRIA"
            target_sublocation = "Table"
            text2pub = (f"Moving {object_} from {origin_location} {origin_sublocation} to the {target_sublocation}")
            self.text_pub.publish(text2pub)
            rospy.loginfo(text2pub)

        elif origin_location in ["KIT", "DLR"]:
            if target_sublocation == "Table":
                target_location = origin_location
                target_sublocation = "Table"
                text2pub = (f"Moving {object_} from {origin_location} {origin_sublocation} to the {target_sublocation}")
                self.text_pub.publish(text2pub)
                rospy.loginfo(text2pub)


        return origin_location, origin_sublocation, target_location, target_sublocation

    def open_vocabulary_detection(self, unknown_obj):
        #replace underscores with spaces in the unknown object, e.g "blue_sphere" -> "blue sphere"
        unknown_obj = unknown_obj.replace("_", " ")
        input_img_path = os.getenv('HAPPYPOSE_DATAFILES', '/home/alterego-vision/HappyPoseFiles/') + '/image_rgb.png'
        image = cv2.imread(input_img_path)

        client = Client("gokaygokay/Florence-2")
        result = client.predict(
            image=handle_file(input_img_path),
            task_prompt="Open Vocabulary Detection",
            text_input=unknown_obj,
            model_id="microsoft/Florence-2-large",
            api_name="/process_image"
        )

        # imshow with bounding boxes from results
        data_dict = json.loads(result[0].replace("'", "\""))

        bboxes = data_dict["<OPEN_VOCABULARY_DETECTION>"]["bboxes"]

        if len(bboxes) > 0:
        # draw bounding boxes
            for bbox in bboxes:
                # convert bbox to int
                bbox = [int(b) for b in bbox]
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # add label to the bounding box
                cv2.putText(image, unknown_obj, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            #write image with bounding boxes
            now = datetime.datetime.now()
            timestamp = now.strftime("%H_%M_%S")
            output_image_path = "/home/alterego-vision/HappyPoseFiles/visualizations" + f'/output-{timestamp}.png'
            cv2.imwrite(output_image_path, image)
            rospy.loginfo(f"FLORENCE saved image with bounding boxes at {output_image_path}")
        else:
            rospy.loginfo("FLORENCE did not detect any object")
            
    def start_competition(self, object_: str, origin_location: str, origin_sublocation: str, target_location: str, target_sublocation: str) -> bool:


        #stop listening to the user
        self.never_listen_again = True
        
        # # #--------------------------------------------------------------------------------------------------------ENABLE AUTO MODE
        rospy.loginfo("Enabling the auto mode")
        response = self.call_service('/enable_auto_mode_service', EnableAutoModeService, enable=True)

        
        # # #--------------------------------------------------------------------------------------------------------GET THE STARTING POINT
        rospy.loginfo("Requesting the starting point")
        #self.text_pub.publish("Hello refree, I'm ready to start the competition. Tell me the plan")
        response = self.call_service('/where_are_you_service', WhereRUService, request=True)
        instruction_point = response.instruction_point
        rospy.loginfo("Starting from the location: " + instruction_point)

        # # #--------------------------------------------------------------------------------------------------------OVERRIDE THE RECIVIED COMMAND 
        # Capisce se c'è un tavolo nel percorso e ripete ad alta voce il comando senza la location del tavolo
        origin_location, origin_sublocation, target_location, target_sublocation =  self.override_cmd_for_table_presence(object_, instruction_point, origin_location, origin_sublocation, target_location, target_sublocation)


        # Se non c'è il tavolo pronuncia bene tutto il comando
        if origin_sublocation != "Table" and target_sublocation != "Table": 
            text2pub = (f"Moving {object_} from {origin_location} {origin_sublocation} to {target_location} {target_sublocation}")
            self.text_pub.publish(text2pub)

        rospy.loginfo(f"I'm in {instruction_point} and i'm going to move the {object_} from {origin_location} {origin_sublocation} to {target_location} {target_sublocation}")
        
        # # #--------------------------------------------------------------------------------------------------------GO TO ORIGIN LOCATION
        if instruction_point == origin_location:
            rospy.loginfo("Already in the origin location")
            rospy.loginfo("Requesting navigation to go to origin sub location")
            if not self.force_exit:
                response = self.call_service('/navigation_service', NavService, location=origin_location, sub_location=origin_sublocation)
            else:
                rospy.loginfo("Force exit is True, bypassing the service call.")
        else:
            if instruction_point == "KIT" or instruction_point == "DLR":
                if origin_location == "INRIA" :
                    rospy.loginfo("Requesting navigation to go to the door KIT-INRIA")
                    response = self.call_service('/navigation_service', NavService, location="Door", sub_location="KIT-INRIA")
                    # # #--------------------------------------------------------------------------------------------------------DISABLE AUTO MODE
                    rospy.loginfo("Disabling the auto mode")
                    response = self.call_service('/enable_auto_mode_service', EnableAutoModeService, enable=False)
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
                    rospy.loginfo("Requesting navigation to go to " + origin_location + " " + origin_sublocation) 
                    response = self.call_service('/navigation_service', NavService, location=origin_location, sub_location=origin_sublocation)
            elif instruction_point == "INRIA":
                if origin_location == "KIT" or origin_location == "DLR":
                    rospy.loginfo("C'è la porta sud")
                    rospy.loginfo("Requesting navigation to go to the door KIT-INRIA")
                    response = self.call_service('/navigation_service', NavService, location="Door", sub_location="INRIA-KIT")
                    # # #--------------------------------------------------------------------------------------------------------DISABLE AUTO MODE
                    rospy.loginfo("Disabling the auto mode")
                    response = self.call_service('/enable_auto_mode_service', EnableAutoModeService, enable=False)
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
                    rospy.loginfo("Requesting navigation to go to " + origin_location + " " + origin_sublocation) 
                    response = self.call_service('/navigation_service', NavService, location=origin_location, sub_location=origin_sublocation)
            elif instruction_point == "DLR":
                if origin_location == "KIT":
                    rospy.loginfo("Requesting navigation to go to origin sub location")
                    response = self.call_service('/navigation_service', NavService, location=origin_location, sub_location=origin_sublocation)
            elif instruction_point == "KIT":
                rospy.loginfo("NO")
                
                if origin_location == "DLR":
                    rospy.loginfo("Requesting navigation to go to origin sub location")
                    response = self.call_service('/navigation_service', NavService, location=origin_location, sub_location=origin_sublocation)

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
        self.text_pub.publish("I'm looking for the object, let me see")

        response = self.call_service('/happypose_service', HappyPoseService, object=object_)
        rospy.sleep(1)
        #in paraller we run florence-2 to detect the object
        self.open_vocabulary_detection(object_)

        self.text_pub.publish("I found the object")
        if response is not None:
            position = response.position
            orientation = response.orientation
            rospy.loginfo(f"Object {object_} found at position ({position.x}, {position.y}, {position.z}) and orientation ({orientation.x}, {orientation.y}, {orientation.z}, {orientation.w})")
        
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
        
        # # #--------------------------------------------------------------------------------------------------------ENABLE AUTO MODE
        rospy.loginfo("Enabling the auto mode")
        response = self.call_service('/enable_auto_mode_service', EnableAutoModeService, enable=True)

        #--------------------------------------------------------------------------------------------------------GO TO THE TARGET LOCATION
        rospy.loginfo("Requesting navigation to go to target location")

        if origin_location == target_location:
            rospy.loginfo("Already in the target location")
            rospy.loginfo("Requesting navigation to go to target sub location")
            if not self.force_exit:
                response = self.call_service('/navigation_service', NavService, location=target_location, sub_location=target_sublocation)
            else:
                rospy.loginfo("Force exit is True, bypassing the service call.")
        else:
            if origin_location == "KIT" or origin_location == "DLR":
                if target_location == "INRIA" :
                    rospy.loginfo("Requesting navigation to go to the door KIT-INRIA")
                    response = self.call_service('/navigation_service', NavService, location="Door", sub_location="KIT-INRIA")
                    # # #--------------------------------------------------------------------------------------------------------DISABLE AUTO MODE
                    rospy.loginfo("Disabling the auto mode")
                    response = self.call_service('/enable_auto_mode_service', EnableAutoModeService, enable=False)
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
                    # # #--------------------------------------------------------------------------------------------------------ENABLE AUTO MODE
                    rospy.loginfo("Enabling the auto mode")
                    response = self.call_service('/enable_auto_mode_service', EnableAutoModeService, enable=True)

                    # # #--------------------------------------------------------------------------------------------------------CONTINUE NAVIGATION TO THE ORIGIN LOCATION
                    rospy.loginfo("Requesting navigation to go to " + target_location + " " + target_sublocation) 
                    response = self.call_service('/navigation_service', NavService, location=target_location, sub_location=target_sublocation)

                    
            elif origin_location == "INRIA":
                if target_location == "KIT" or target_location == "DLR":
                    rospy.loginfo("C'è la porta sud")
                    rospy.loginfo("Requesting navigation to go to the door KIT-INRIA")
                    response = self.call_service('/navigation_service', NavService, location="Door", sub_location="INRIA-KIT")
                    # # #--------------------------------------------------------------------------------------------------------DISABLE AUTO MODE
                    rospy.loginfo("Disabling the auto mode")
                    response = self.call_service('/enable_auto_mode_service', EnableAutoModeService, enable=False)
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

                    # # #--------------------------------------------------------------------------------------------------------ENABLE AUTO MODE
                    rospy.loginfo("Enabling the auto mode")
                    response = self.call_service('/enable_auto_mode_service', EnableAutoModeService, enable=True)

                    # # #--------------------------------------------------------------------------------------------------------CONTINUE NAVIGATION TO THE ORIGIN LOCATION
                    rospy.loginfo("Requesting navigation to go to " + target_location + " " + target_sublocation) 
                    response = self.call_service('/navigation_service', NavService, location=target_location, sub_location=target_sublocation)


            elif origin_location == "DLR":
                if target_location == "KIT":
                    rospy.loginfo("Requesting navigation to go to target sub location")
                    response = self.call_service('/navigation_service', NavService, location=target_location, sub_location=target_sublocation)
            elif origin_location == "KIT":
                if target_location == "DLR":
                    rospy.loginfo("Requesting navigation to go to target sub location")
                    response = self.call_service('/navigation_service', NavService, location=target_location, sub_location=target_sublocation)

        rospy.loginfo("Moving to place the object")

        #--------------------------------------------------------------------------------------------------------WAIT THE PILOT TO PLACE/GIVE THE OBJECT
        if target_sublocation == "Person":
            self.text_pub.publish("Hey my friend, I have something you")
            rospy.loginfo("waiting the pilot to enter and give the object")
        else: 
        
            rospy.loginfo("waiting the pilot to enter and place the object")
            rospy.loginfo("Sending audio request")
            self.text_pub.publish("And finally the pilot will help me to place the object")

        while True:
            response = self.call_service('/wait_help_from_pilot_service', WaitPilotService, request=False)
            if response:
                break
            rospy.sleep(0.1)

        while True:
            self.text_pub.publish("I did my best, I hope you are happy")
            rospy.loginfo("COOPETITION FINISHED - IF YOU SEE THIS MESSAGE, IT'S TIME TO CELEBRATE!")
            rospy.sleep(1)

    
        return True 

    @tool
    def execute_plan(object_: str, origin_location: str, origin_sublocation: str, target_location: str, target_sublocation: str) -> bool:
        """call this tool when the plan is ready to be executed
        Args:
            object (str): object to pick referred as O
            origin_location (str): location to go (INRIA, KIT or DLR) referred K_origin
            origin_sublocation (str): sublocation to pick object from (Dishwasher, Table, Drawer, Counter or Cabinet) referred as L_origin 
            target_location (str): location to place object (INRIA, KIT or DLR) referred as K_target
            target_sublocation (str): sublocation to place object (Dishwasher, Table, Drawer, Counter, Cabinet or Person) referred as L_target
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

        return True


    def should_continue(self, state: MessagesState) -> Literal["action", END]:
        """Determine whether to continue or not."""
        messages = state['messages']
        last_message = messages[-1]


        # If the LLM makes a tool call, then we route to the "tools" node
        if last_message.tool_calls:
            args = last_message.tool_calls[0]["args"]
            print(args)
            self.start_competition(args["object_"], args["origin_location"], args["origin_sublocation"], args["target_location"], args["target_sublocation"])
            return "action"
        # Otherwise, we stop (reply to the user)
        return END

    def call_model(self, state: MessagesState):
        """Call the model."""
        messages = state['messages']
        response = self.llm_model.invoke(messages)

        # print(response)
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
        # print("DEBUG")
        # current_state = chat_with_robot.robot.invoke(
        #     {"messages": [
        #         SystemMessage(f"Given a prompt in the format: Pick Object 'O' at Location 'L_origin' in Kitchen 'K_origin' and place it on 'K_target' 'L_target'. Where O={{{', '.join(objects)}}}, L_target={{Dishwasher, Table, Drawer, Counter, Cabinet}}, L_origin={{Dishwasher, Table, Drawer, Counter, Cabinet, Person}}, K_target={{DLR, KIT, INRIA}}, and K_origin={{DLR, KIT, INRIA}}. If you are not sure about one component, ask for confirmation. Never list to the user all the objects that can be grasped."),
        #         HumanMessage(content="take the apple from the table in the INRIA kitchen and place it on the KIT table")
        #     ]},
        #     config={"configurable": {"thread_id": 42}},
        # )
        ###################
        rospy.spin()
    except rospy.ROSInterruptException:
        pass