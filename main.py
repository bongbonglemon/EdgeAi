"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import time

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.1,
                        help="Probability threshold for detections filtering"
                        "(0.1 by default)")
    parser.add_argument("-mode", "--mode", type=str, default="single-image-mode",
                        help="Toggle single-image or video input")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(host=MQTT_HOST, port=MQTT_PORT, keepalive = MQTT_KEEPALIVE_INTERVAL)
    return client


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return cv2.filter2D(image, -1, kernel)

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    ### TODO: Handle the input stream ### #Code to take the pictures out of the video
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)  
    cap.open(args.input)
    
    # Grab the shape of the input 
    original_width = int(cap.get(3))
    original_height = int(cap.get(4))
    
    total_count = 0
    is_first = True
    average_duration = 0
    frame_count=0
    frames_wo_boxes = 0
    just_start = True
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        current_count = 0
        ### TODO: Read from the video capture ###
        ret, frame = cap.read()
        # Check if there's a frame
        if not ret:
            print("Input not supported.")
            break
        key_pressed = cv2.waitKey(60)
        
        ### TODO: Pre-process the image as needed ###
        width, height = 300, 300
        image = cv2.resize(frame, (width, height))
        cv2.imwrite("what_model_sees.png", image)
        image = image.transpose((2,0,1))
        image = image.reshape(1, 3, height, width)

        #fps is 10
        '''
        #CHECK FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) :", fps)
        ''' 
        
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(image)
        
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            ### TODO: Extract any desired stats from the results ###
            
            frame_w_box = frame
            for detection in result[0,0,:,:]:        
                # get the class index
                class_index = int(detection[1])
                # get the score
                score = float(detection[2])
                # threshold
                if score > prob_threshold: # dk if human
                    current_count+=1
                    left   = int(detection[3] * original_width)
                    top    = int(detection[4] * original_height)
                    right  = int(detection[5] * original_width)
                    bottom = int(detection[6] * original_height)
            
                    width  = right - left
                    height = bottom - top

                    color = (255, 0, 0)
                    thickness = 2
                    frame_w_box = cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
                
            if current_count>0 and is_first == True:
                
                
                is_first = False
                
                
                ### Topic "person/duration": key of "duration" ###
                if frames_wo_boxes>40:
                    frame_count+=frames_wo_boxes
                    if just_start == True:
                        frame_count = 0
                        just_start = False
                    
                    client.publish("person/duration", payload=json.dumps({"duration":frame_count/10}), qos=0, retain=False)
                    frame_count=0
                frames_wo_boxes=0
                total_count +=1
                
                frame_count+=1
                
            elif current_count>0:
                frame_count+=1
                
            if current_count == 0:
                is_first = True
                frames_wo_boxes+=1
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            
            ### Topic "person": keys of "count" and "total" ###
            client.publish("person", payload=json.dumps({"count":current_count, "total":total_count}), qos=0, retain=False)
            
            
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame_w_box)  
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if args.mode == "single-image-mode":
            cv2.imwrite("output.png", frame_w_box)   

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()