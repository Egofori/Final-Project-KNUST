from genericpath import exists
import cv2
import numpy as np
import os
from stream_c3d import *
from stream_classifier import *
from stream_utils import *
import itertools
import math
import time
from os.path import isfile, join

if not os.path.exists('./stream'):
    os.mkdir('./stream')

def stream(mySrc, addData_callbackFunc):
    mySrc.data_signal.connect(addData_callbackFunc)

    # build models
    feature_extractor = c3d_feature_extractor()
    classifier_model = create_classifier_model()

    print("Models initialized")

    save_path = os.path.join('./stream','stream.avi')

    # Create an object to read
    # from camera
    video = cv2.VideoCapture(0)
    fps = 40
    width = video.get(3)
    height = video.get(4)
    size = (int(width),int(height))
    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'XVID'), fps, size)


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
            frames.append(frame)
            if count%16 == 0 and count != 0:
                video_clips, num_frames = get_video_clips(frames)
                # print("Number of clips in the video : ", len(video_clips))
                rgb_features = []
                for i, clip in enumerate(video_clips):
                    clip = np.array(clip)
                    if len(clip) < frame_count:
                        continue

                    clip = c3d_preprocess_input(clip)
                    rgb_feature = feature_extractor.predict(clip)[0]
                    # print("rgb_feature=",rgb_feature.shape)
                    rgb_features.append(rgb_feature)

                    value = math.ceil(i*100/len(video_clips))

                    # print("Processed clip : ", i, "Percentage: ", value)

                rgb_features = np.array(rgb_features)
                # print("rgb_features =",rgb_features.shape)
                rgb_feature_bag = interpolate(rgb_features, features_per_bag)
                # print("rgb_feature_bag =",rgb_feature_bag.shape)
            
                # classify using the trained classifier model
                predictions = classifier_model.predict(rgb_feature_bag)

                predictions = np.array(predictions).squeeze()
                predictions = extrapolate(predictions, num_frames)
                
                mySrc.data_signal.emit(predictions[0])
                print(predictions[0])

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
                # convert_frames_to_video(save_path, frames, 40)
                save_video(out,frames)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break

                frames = []
                count = 0

            count += 1
            end = time.time()
            interval = end - start
            # print(interval)
        else:
            break
        
        
    # release video capture and video
    video.release()
        
    # Closes all the frames
    cv2.destroyAllWindows()

    print("The video was successfully saved")

