from genericpath import exists
import cv2
from flask import json
import numpy as np
import os
from kami_no_me.stream_utils import *
import itertools
import math
import time
from os.path import isfile, join
import requests
import codecs, json 
#from io import StringIO




if not os.path.exists('./stream'):
    os.mkdir('./stream')

def stream(mySrc, addData_callbackFunc):
    mySrc.data_signal.connect(addData_callbackFunc)

    save_path = os.path.join('./stream','stream.mp4')

    # Create an object to read
    # from camera
    video = cv2.VideoCapture(0)
    fps = 40
    width = video.get(3)
    height = video.get(4)
    size = (int(width),int(height))
    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)


    # We need to check if camera
    # is opened previously or not
    if (video.isOpened() == False):
        print("Error reading video file")

    count = 0
    interval = 0
    frames = []
    start = time.time()

    while(True):
        ret, frame = video.read()
        if ret:
            frames.append(frame.tolist())
            if count%16 == 0 and count != 0:
                #file_path = "./frames.json"  # your path variable
                #json.dump(frames, codecs.open(file_path, 'w', encoding='utf-8'), 
                                    #separators=(',', ':'), sort_keys=True, indent=4) 


                response = requests.post('http://10.77.173.21:5000/stream', json=json.dumps(frames))
                
            
                # get response object
                res = response.json()
                predictions = np.array(res)

                print(predictions[0])
                mySrc.data_signal.emit(predictions[0])

                # list of tuples containing indexes of predictions lower than treshold
                frame_index = list(np.where(predictions < 0.4))

                # convert list of tuples to list
                frame_index = list(itertools.chain(*frame_index))
                for index in sorted(frame_index, reverse=True):
                    del frames[index]

                # delete predictions less than treshold value
                predictions = np.delete(predictions, np.where(predictions < 0.4))    
                num_frames = len(frames)

                #write frame
                save_video(out,frames)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break

                frames = []
                count = 0

            count += 1
            end = time.time()
            # interval = end - start
            print(interval)
        else:
            break
        
        
    # release video capture and video
    video.release()
        
    # Closes all the frames
    cv2.destroyAllWindows()

    print("The video was successfully saved")

