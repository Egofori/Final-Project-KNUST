from flask import Flask, request,jsonify, send_from_directory, send_file, abort
import os
import shutil
import cv2
import numpy as np
from c3d import *
from classifier import *
from visual_utils import *
from utils import *
from stream_utils import *
import sklearn.preprocessing
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import itertools
import math
import codecs, json
import zipfile





# Initialize Flask application
app = Flask(__name__)

value = -1



@app.route('/', methods=['GET','POST'])
def get_anomalies():
    global value
    save_zip_path = os.path.join('./output_zip', 'output.zip')

    if not os.path.exists('./uploads'):
        os.mkdir('./uploads')

    if not os.path.exists('./output'):
        os.mkdir('./output')

    if not os.path.exists('./output_preds'):
        os.mkdir('./output_preds')
    
    if not os.path.exists('./output_zip'):
        os.mkdir('./output_zip')
    
    if os.path.isfile(save_zip_path):
        os.remove(save_zip_path)


    if request.method == 'POST':
        # Get the file from post request
        f = request.files['File']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        sample_video_path = os.path.join(
            basepath, './uploads', secure_filename(f.filename))
        f.save(sample_video_path)

    # PARAMETERS
    frame_height = 240
    frame_width = 320
    channels = 3
    frame_count = 16
    features_per_bag = 32

    # build models
    feature_extractor = c3d_feature_extractor()
    classifier_model = create_classifier_model()
    print("Models initialized")


    video_name = os.path.basename(sample_video_path).split('.')[0]
    print(video_name)

    # read video
    video_clips, frames = get_video_clips(sample_video_path)
    num_frames = len(frames)

    cap = cv2.VideoCapture(sample_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Number of clips in the video : ", len(video_clips))

    # extract features
    rgb_features = []
    for i, clip in enumerate(video_clips):
        clip = np.array(clip)
        if len(clip) < frame_count:
            continue

        clip = c3d_preprocess_input(clip)
        rgb_feature = feature_extractor.predict(clip)[0]
        print("rgb_feature=",rgb_feature.shape)
        rgb_features.append(rgb_feature)

        value = math.ceil(i*100/len(video_clips))

        print("Processed clip : ", i, "Percentage: ", value)

    rgb_features = np.array(rgb_features)
    rgb_feature_bag = interpolate(rgb_features, features_per_bag)
    
    
    # classify using the trained classifier model
    predictions = classifier_model.predict(rgb_feature_bag)

    predictions = np.array(predictions).squeeze()
    predictions = extrapolate(predictions, num_frames)

    # list of tuples containing indexes of predictions lower than treshold
    frame_index = list(np.where(predictions < 0.4))

    # convert list of tuples to list
    frame_index = list(itertools.chain(*frame_index))
    save_preds_path = os.path.join("./output_preds", video_name + "_preds.mp4")
    print(save_preds_path)

    # visualize plot for predictions
    visualize_predictions(frames, predictions, save_preds_path)
    
    # delete frames where predictions are lower than treshold
    for index in sorted(frame_index, reverse=True):
        del frames[index]

    # delete predictions less than treshold value
    predictions = np.delete(predictions, np.where(predictions < 0.4))    
    num_frames = len(frames)
    
    print("num_frames =", num_frames)
    print("predictions = ", predictions.shape)
    save_path = os.path.join("./output", video_name + ".mp4")
  
    convert_frames_to_video(save_path,frames,fps)
    print('Executed Successfully - '+video_name + '.mp4')
     
    zipfolder = zipfile.ZipFile(save_zip_path,'w', compression = zipfile.ZIP_STORED) # Compression type 

    # zip all the file which are inside in the folders
    zipfolder.write(save_preds_path)
    zipfolder.write(save_path)
    zipfolder.close()

    value = 100

    try:
        return send_file(save_zip_path,
                mimetype = 'zip',
                download_name= 'output.zip',
                as_attachment = True)
  
    except FileNotFoundError:
        abort(404)



@app.route('/stream', methods=['GET','POST'])
def get_predictions():
    global value

    if request.method == 'POST':
        # Get frames from post request
        frame = request.get_json()

    # PARAMETERS
    frame_height = 240
    frame_width = 320
    channels = 3
    frame_count = 16
    features_per_bag = 32

    # build models
    feature_extractor = c3d_feature_extractor()
    classifier_model = create_classifier_model()
    print("Models initialized")


    # read frames
    frames = json.loads(frame)
    video_clips, num_frames = stream_get_video_clips(frames)


    fps = 40

    print("Number of clips in the video : ", len(video_clips))

    # extract features
    rgb_features = []
    for i, clip in enumerate(video_clips):
        clip = np.array(clip, dtype='float32')
        if len(clip) < frame_count:
            continue

        clip = c3d_preprocess_input(clip)
        rgb_feature = feature_extractor.predict(clip)[0]
        print("rgb_feature=",rgb_feature.shape)
        rgb_features.append(rgb_feature)

        value = math.ceil(i*100/len(video_clips))

        print("Processed clip : ", i, "Percentage: ", value)

    rgb_features = np.array(rgb_features)
    rgb_feature_bag = interpolate(rgb_features, features_per_bag)
   
    
    # classify using the trained classifier model
    predictions = classifier_model.predict(rgb_feature_bag)

    predictions = np.array(predictions).squeeze()
    predictions = extrapolate(predictions, num_frames)
    predictions = predictions.tolist()

    try:
        return jsonify(predictions)
    except FileNotFoundError:
        abort(404)




@app.route('/progress', methods=['GET'])
def send_progress():
    return str(value)
       
if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)




