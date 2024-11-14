#!/usr/bin/env python3
"""
Simple text to speech module with mouth movement synchronization. 
Written by: Luca Garello (luca.garello@iit.it)
"""

import subprocess
import threading
import time
import sys
import os
import numpy as np
import pydub
import rospy
import argparse
import edge_tts

from std_msgs.msg import String, Float64, Bool


#OUTPUT_FILE = "/home/alterego-vision/AlterEGO_v2/catkin_ws/src/raise/temp/speech.mp3" 
OUTPUT_FILE = "/home/luca-garello/Downloads/test.mp3"
DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

HAPPY = 2
VOICE_INTENSITY = 10
INTENSITY_LEVELS = 3

class EmbodiedTextToSpeech:
    """
    Class for the speech synthesis with mouth movement synchronization
    """

    def __init__(self, model_type="web", language="italian", robot_name='robot_alterego3'):

        self.model_type = model_type

        # Set the language for the speech synthesis
        self.set_language(language)

        self.mouth_mode = VOICE_INTENSITY
        self.is_robot_speaking = False

        # Inizializza il nodo ROS
        rospy.init_node('text2speech', anonymous=False)

        self.subscriber               = rospy.Subscriber(robot_name + '/text2speech', String, self.callback)
        self.language_subscriber      = rospy.Subscriber(robot_name + '/set_language', String, self.set_language_callback)
        self.intensity_publisher      = rospy.Publisher(robot_name + '/voice_intensity', Float64, queue_size=10)
        self.emoticons_publisher      = rospy.Publisher(robot_name + '/emoticons', Float64, queue_size=10)        
        self.robot_speaking_publisher = rospy.Publisher(robot_name + '/is_robot_speaking', Bool, queue_size=10)

        self.robot_speaking_thread = threading.Thread(target=self.publish_robot_speaking)
        self.robot_speaking_thread.start()

        rospy.loginfo("Text to Speech node initialized")
        #DEBUG ONLY TO DELETE
        self.speak("Do you want me to pick the apple from the Table in the KIT kitchen? I can do it for you")
        rospy.spin()

    def set_language(self, language):
        """ Set the language for the speech """
        if self.model_type == "web":
            if language == "english":
                self.voice = "en-US-GuyNeural"
            elif language == "italian":
                self.voice = "it-IT-GiuseppeNeural"
            else:
                raise ValueError("Invalid language")
        elif self.model_type == "local":
            if language == "english":
                self.voice = "en_US-lessac-low" # BEST 'en_US-lessac-low' 
            elif language == "italian":
                self.voice = "it_IT-riccardo-x_low"
            else:
                raise ValueError("Invalid language")
        else:
            raise ValueError("Invalid model name, choose between 'edge' and 'web'")
        

    def set_language_callback(self, msg):
        """
        Callback function to set the language for speech synthesis
        """
        lang = msg.data
        print(f"Set language: {lang}")
        self.set_language(lang)
    
    def publish_robot_speaking(self):
        """
        Publish the status of the robot speaking
        """
        while not rospy.is_shutdown():
            self.robot_speaking_publisher.publish(self.is_robot_speaking)
            time.sleep(0.1)

    def callback(self, data):
        """
        Callback function for the subscriber
        """
        text2vocalize = data.data
        rospy.loginfo(f"Received text: {text2vocalize}")
        self.speak(text2vocalize)

    def assign_intensity(self, file_path, frequency):
        """
        evaluate the intensity of the audio file and quatize it to make the robot mouth move accordingly
        voice intensity is quantized in 60 chunks in the range [0, 1] with 1 step
        """
        try:
            # Load the audio file
            if self.model_type == "web":
                audio = pydub.AudioSegment.from_mp3(file_path)
            elif self.model_type == "local":
                audio = pydub.AudioSegment.from_wav(file_path)
            else:
                raise ValueError("Invalid model name, choose between 'local' and 'web'")

            # Convert the audio to a numpy array
            audio_data = np.array(audio.get_array_of_samples())

            # Calculate the number of chunks based on the frequency
            chunk_size = int(audio.frame_rate / frequency)
            num_chunks = np.ceil(audio_data.shape[0] / chunk_size).astype(int)
            #print("Il segnale ha un totale di " + str(audio_data.shape[0]) + " campioni" + " e verrà diviso in " + str(num_chunks) + " chunk di " + str(chunk_size) + " campioni ciascuno")

            # Initialize the intensity array
            intensity = np.zeros(num_chunks)

            # Calculate the intensity for each chunk, assign an intensity value to each chunk in the range [0, 4] with 1 step
            for i in range(num_chunks):
                start = i * chunk_size
                end = (i + 1) * chunk_size
                chunk = audio_data[start:end]      
                #calculate the average intensity of the chunk
                intensity[i] = np.average(np.abs(chunk))

            # normalize the vector of average intensities
            intensity = intensity / np.max(np.abs(intensity))

            # quantize the intensity in the range [0, 4] with 1 step
            intensity = np.round(intensity * INTENSITY_LEVELS)

            return intensity
        
        except Exception as e:
            print(f"Error while assigning intensity: {e}")
            return None

    def move_mouth(self, array, frequency=20):
        """
        Sychronized print of the intensity of the audio
        """
        interval = 1 / frequency  # secondi
        next_time = time.time() + interval
        for value in array:
            # DEBUG: PRINT A BAR ON THE TERMINAL TO VISUALIZE THE INTENSITY
            # print("\033[H\033[J")
            # print("|" + int(value) * "██")
            # print("intensity: " + str(int(value)) + "/4\n")
            time.sleep(max(0, next_time - time.time()))
            next_time += interval

            self.intensity_publisher.publish(value)
            self.emoticons_publisher.publish(self.mouth_mode)

    def play_audio(self, file_path=OUTPUT_FILE):
        """
        Jusk play the audio file
        """
        # play the audio file
        try:
            subprocess.run(['mpv', file_path, '--really-quiet'])
        except Exception as e:
            print(f"Error while playing the file: {e}")


    def speak(self, text):
        """
        Synthesize the text and play it
        """
        self.is_robot_speaking = True
        
        start_time = time.time()
        if self.model_type == "web":
            communicate = edge_tts.Communicate(text, self.voice)
            communicate.save_sync(OUTPUT_FILE)

            end = time.time()
            print("Time elapsed: " + str(end - start_time))
            
        elif self.model_type == "local":
            start_time = time.time()
            subprocess.run([
                'piper', 
                '--model', self.voice,
                '--download-dir', DOWNLOAD_DIR,
                '--data-dir', DOWNLOAD_DIR,
                '--output_file', OUTPUT_FILE,
            ], input=text, text=True)
        end = time.time()
        print("Time elapsed: " + str(end - start_time))

        ##
        intensity = self.assign_intensity(OUTPUT_FILE, self.mouth_mode)

        # Thread per la riproduzione del suono
        audio_thread = threading.Thread(target=self.play_audio)
        # Thread per la stampa dell'array
        mouth_thread = threading.Thread(target=self.move_mouth, args=(intensity, self.mouth_mode))

        # Avvia  i thread
        audio_thread.start()
        mouth_thread.start()

        # Attendi i thread siano completati
        audio_thread.join()
        mouth_thread.join()
        self.is_robot_speaking = False

if __name__ == "__main__":
    # Filter arguments that are not meant for the parser
    filtered_args = [arg for arg in sys.argv if not arg.startswith('__')]

    parser = argparse.ArgumentParser(description="Speech to Text ROS Node")
    parser.add_argument('--model', type=str, default='local',
                    help='model for text2speech, either "web" (edge web API) or "local" (local pyttsx3) (defaul: web)')
    parser.add_argument('--language', type=str, default='english',
                    help='Language for speech recognition, either "english" or "italian" (default: italian)')
    parser.add_argument('--robot_name', type=str, default=os.getenv('ROBOT_NAME', 'robot_alterego3'),
                    help='Robot name for topic naming (default: robot_alterego3)')
    args = parser.parse_args(filtered_args[1:])

    embodied_tts = EmbodiedTextToSpeech(args.model, args.language,  args.robot_name)
