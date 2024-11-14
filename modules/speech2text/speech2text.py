"""
This module is used to recognize speech from the microphone and publish it to a topic.
Written by: Luca Garello (luca.garello@iit.it)
"""

import time
import signal
import sys
import argparse
import os
import rospy
import speech_recognition as sr
from std_msgs.msg import String

class SpeechToText:
    """ Class to recognize speech from the microphone and publish it to a topic """
    def __init__(self, language="it-IT", model_name="google", mic_index=0, threshold=500, dynamic_threshold=False, robot_name='robot_alterego3'):
        assert model_name in ["tiny.en","base.en","small.en","medium.en","tiny","base","small","medium","large","turbo","google"], "Invalid model name"
        self.language = language
        self.model_name = model_name
        self.mic_index = mic_index
        self.threshold = threshold
        self.dynamic_threshold = dynamic_threshold
        self.mic_index = None

        # Initialize the ROS node
        rospy.init_node('speech2text', anonymous=False)
        signal.signal(signal.SIGINT, self.signal_handler)

        # Create a publisher for the recognized speech and a subscriber for setting the language
        self.pub = rospy.Publisher(robot_name + '/recognized_speech', String, queue_size=10)
        self.sub = rospy.Subscriber(robot_name + '/set_language', String, self.set_language_callback)

        # Initialize the speech recognizer, this segments audio from the microphone and sends it to the speech recognition service
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = self.threshold
        self.recognizer.dynamic_energy_threshold = self.dynamic_threshold
        print("Module initialized")

    def signal_handler(self):
        """ Signal handler to catch interruption and shutdown the node """
        print('Interruption received, shutting down...')
        rospy.signal_shutdown('Interruption received')
        sys.exit(0)

    def set_language_callback(self, msg):
        """ Callback function to set the language for speech recognition """
        desired_lan = msg.data
        assert desired_lan in ["english", "italian"], "Invalid language"

        print(f"Set language: {desired_lan}")
        if self.model_name == "google":
            if desired_lan == "english":
                self.language = "en-US"
            elif desired_lan == "italian":
                self.language = "it-IT"
            else:
                print("Invalid language, using default language (it-IT)")
                self.language = "it-IT"
        else:
            if desired_lan == "english":
                self.language = "en"
            elif desired_lan == "italian":
                self.language = "it"
            else:
                print("Invalid language, using default language (it)")
                self.language = "it"

    def recognize_speech(self):
        """ Recognize speech from the microphone and publish it to the specified topic """
        while not rospy.is_shutdown():
            with sr.Microphone(device_index=self.mic_index) as source:
                print("Listening ...")
                audio = self.recognizer.listen(source)
                print("Recognizing ...")

                time_start = time.time()
                if self.model_name == "google":
                    try:
                        text = self.recognizer.recognize_google(audio, language=self.language)
                    except sr.UnknownValueError:
                        print("Speech Recognition could not understand audio")
                    except sr.RequestError as e:
                        print(f"Could not request results from Speech Recognition service; {e}")
                else:
                    text = self.recognizer.recognize_whisper(audio, model=self.model_name, language=self.language)

                time_end = time.time()
                if text is not None:
                    print(f"Recognized: {text}")
                    print(f"\033[3mInference time: {time_end - time_start} seconds\033[0m")
                    self.pub.publish(text)

if __name__ == "__main__":
    # Filter arguments that are not meant for the parser
    filtered_args = [arg for arg in sys.argv if not arg.startswith('__')]

    parser = argparse.ArgumentParser(description="Speech to Text ROS Node")
    parser.add_argument('--language', type=str, default='italian', help='Language for speech recognition (default: italian)')
    parser.add_argument('--model_name', type=str, default='tiny.en', help='Model name for speech recognition (default: google) Note: google uses the google online speech recognition API')
    parser.add_argument('--mic_index', type=int, default=0, help='Microphone index (default: 0)')
    parser.add_argument('--threshold', type=int, default=500, help='Energy threshold for speech recognition (default: 2000)')
    parser.add_argument('--dynamic_threshold', type=bool, default=False, help='Dynamic energy threshold for speech recognition (default: False)')
    parser.add_argument('--robot_name', type=str, default=os.getenv('ROBOT_NAME', 'robot_alterego3'), help='Robot name for topic naming (default: robot_alterego3)')
    args = parser.parse_args(filtered_args[1:])

    stt = SpeechToText(language=args.language, model_name=args.model_name, mic_index=args.mic_index, threshold=args.threshold, dynamic_threshold=args.dynamic_threshold, robot_name=args.robot_name)
    stt.recognize_speech()