import numpy as np
import os
import skvideo
from utils import *
from metrics import *


if __name__ == '__main__':
    target_frames = load_video('lossless.mp4')
    frames = load_video('video.mp4')
    size = os.stat('video.mp4').st_size
    print(size)
    print(target_frames.shape, frames.shape)

    metrics = [PSNR(), SSIM()]

    n_frames = len(frames)
    n_metrics = len(metrics)
    
    values = np.zeros((n_frames, n_metrics))
    for i in range(n_frames):
        for j in range(n_metrics):
            values[i, j] = metrics[j](frames[i:i+1], target_frames[i:i+1])

    print(values.mean(0))
    print(8 * size / (len(frames) * frames.shape[-2] * frames.shape[-1]))

