# NRFF
Neural Residual Flow Fields for Efficient Video Representations


You can compress or convert a video using this repository.
As an input, our codes only accept an image folder, ".avi" or ".mp4" encoded video file.

There are three main files.
One is "compress_siren.py", which compresses a video using color-based INR.
Another is "compress_single_nrff.py", which uses NRFF with a single reference frame to compress a video.
Lastly, "compress_multi_nrff.py" compresses a video using two reference frames.


### 1. Prepare for video files
You can download video files from the following urls: [MPI SINTEL](http://sintel.is.tue.mpg.de/), [UVG](http://ultravideo.fi/#testsequences).

Since we do not support ".yuv" videos, we strongly recommend you convert ".yuv" videos using lossless video compression, such as x264 with a crf of 0.


### 2. How to compress a video
1. Common options for the three main files

    "--tag": The results will be stored in a folder named "tag".
    
    "--use_amp": (recommended) reduce the bit precision from 32 to 16.

    "--video_path": the path of the input video. This can be an image folder or a video file.

    "--n_frames": the size of a group of pictures (GOP). The input video will be split so that the size of each GOP will be similar to the predefined "n_frames".

    "--lr": the initial learning rate.

    "--epochs": the number of epochs.
   
   

   
2. Other options ("compress_siren.py")

    "--bpp": It controls bits per pixel (bpp). The network size will be automatically set to match a predefined bpp.


3. Other options ("compress_single_nrff.py")

    "--ratio": the ratio of the network size to the key frame size. If it is set to 1/4, the network size will be automatically set to one quarter of the key frame size (assuming 16 bit precision).

    "--codec": key frame compression codec. It currently supports only "jpeg", "avif", and "h264".

    "--quality": the quality factor for key frame compression. For example, if "codec" is "jpeg" and "quality" is set to 85, it will compress the key frame with "jpeg" with a quality factor of 85. Likewise, if "codec" is "h264" and "quality" is set to 15, it will compress the key frame with "h264" with a crf of 15.

    "--flow-warmup-step": the number of epochs for pretraining (with pseudo-ground truth optical flows and residuals extracted by the optical flow estimator).

    "--image-warmup-step": the number of image-warmup epochs.



4. Other options ("compress_multi_nrff.py")

    "--ratio", "--codec", "--quality": same as those in "compress_single_nrff.py".


### 3. Examples

```bash
# Compressing "alley_1" video from SINTEL using SIREN.
python3 compress_siren.py --use_amp --n_frames=5 --bpp=0.2 --tag=SIREN_ALLEY_1 --video_path=training/final/alley_1 --lr=1e-5

# Compressing "ReadySteadyGo" video from UVG using multiple reference NRFF.
python3 compress_multi_nrff.py --use_amp --n_frames=15 --ratio=1 --tag=mNRFF_RSG --video_path=ReadySteadyGo_1920x1080_120fps_420_8bit_YUV.mp4 --lr=1e-3
```
