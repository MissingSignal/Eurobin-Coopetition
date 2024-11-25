
import cv2
from gradio_client import Client, handle_file
import json
import datetime
import os
import rospy

def open_vocabulary_detection(self, image, unknown_obj):
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
        output_image = "/home/alterego-vision/HappyPoseFiles/visualizations" + f'/output-{timestamp}.png'
        cv2.imwrite(output_image, image)
        rospy.loginfo(f"FLORENCE saved image with bounding boxes at {output_image}")
    else:
        print("FLORENCE did not detect any object")
        rospy.loginfo("FLORENCE did not detect any object")

open_vocabulary_detection(None, None, "box")
