import numpy as np
import cv2

#PARAMETERS
frame_height = 240
frame_width = 320
channels = 3
frame_count = 16
features_per_bag = 32


def sliding_window(arr, size, stride):
    """Apply sliding window to an array, getting chunks of
    of specified size using the specified stride
    :param arr: Array to be divided
    :param size: Size of the chunks
    :param stride: Number of frames to skip for the next chunk
    :returns: Tensor with the resulting chunks
    :rtype: np.ndarray
    """
    num_chunks = int((len(arr) - size) / stride) + 2
    result = []
    for i in range(0,  num_chunks * stride, stride):
        if len(arr[i:i + size]) > 0:
            result.append(arr[i:i + size])
    return np.array(result, dtype='object')


def interpolate(features, features_per_bag):
    """Transform a bag with an arbitrary number of features into a bag
    with a fixed amount, using interpolation of consecutive features
    :param features: Bag of features to pad
    :param features_per_bag: Number of features to obtain
    :returns: Interpolated features
    :rtype: np.ndarray
    """
    feature_size = np.array(features).shape[1]
    interpolated_features = np.zeros((features_per_bag, feature_size))
    interpolation_indices = np.round(np.linspace(0, len(features) - 1, num=features_per_bag + 1))
    count = 0
    for index in range(0, len(interpolation_indices)-1):
        start = int(interpolation_indices[index])
        end = int(interpolation_indices[index + 1])

        assert end >= start

        if start == end:
            temp_vect = features[start, :]
        else:
            temp_vect = np.mean(features[start:end+1, :], axis=0)

        temp_vect = temp_vect / np.linalg.norm(temp_vect)

        if np.linalg.norm(temp_vect) == 0:
            print("Error")

        interpolated_features[count,:]=temp_vect
        count = count + 1

    return np.array(interpolated_features)


def extrapolate(outputs, num_frames):
    """Expand output to match the video length
    :param outputs: Array of predicted outputs
    :param num_frames: Expected size of the output array
    :returns: Array of output size
    :rtype: np.ndarray
    """

    extrapolated_outputs = []
    extrapolation_indices = np.round(np.linspace(0, len(outputs) - 1, num=num_frames))
    for index in extrapolation_indices:
        extrapolated_outputs.append(outputs[int(index)])
    return np.array(extrapolated_outputs)

def get_video_frames(video_path):
    """Reads the video given a file path
    :param video_path: Path to the video
    :returns: Video as an array of frames
    :rtype: np.ndarray
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break
    cap.release()
    return frames

def get_video_clips(video_path):
    """Divides the input video into non-overlapping clips
    :param video_path: Path to the video
    :returns: Array with the fragments of video
    :rtype: np.ndarray
    """
    frames = get_video_frames(video_path)
    clips = sliding_window(frames, frame_count, frame_count)
    return clips, frames