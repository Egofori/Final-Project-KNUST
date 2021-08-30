import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter 
from utils import *


def visualize_predictions(frames, predictions, save_path):
    assert len(frames) == len(predictions)

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.set_tight_layout(True)

    line = matplotlib.lines.Line2D([], [])

    fig_frame = plt.subplot(2, 1, 1)
    img = fig_frame.imshow(frames[0])
    fig_prediction = plt.subplot(2, 1, 2)
    fig_prediction.set_xlim(0, len(frames))
    fig_prediction.set_ylim(0, 1.15)
    fig_prediction.add_line(line)

    def update(i):
        frame = frames[i]
        x = range(0, i)
        y = predictions[0:i]
        line.set_data(x, y)
        img.set_data(frame)
        return plt

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 20ms between frames.
    writervideo = FFMpegWriter(fps=30)
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(frames), 10), interval=1, repeat=False)

    if save_path:
    #    anim.save(save_path, dpi=200, writer='imagemagick')
        anim.save(save_path, writer=writervideo)

    else:
        return None


def convert_frames_to_video(save_path,frames,fps):
    """ transforms frames into video and returns a video file with specified fps"""
    
    
    height = 240
    width = 320
    size = (width,height)
    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
    if len(frames) == 0:
        print('No Anomaly found')
        out.release()
        return 'None'
    else :
        for i in range(len(frames)):
        # writing to a image array
            out.write(cv2.cvtColor(frames[i],cv2.COLOR_RGB2BGR))
    out.release()
