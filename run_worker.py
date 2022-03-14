import os
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--tag', required=True, type=str)
parser.add_argument('--gpu', required=True, type=str)
parser.add_argument('--type', type=str, default='ours',
                    choices=['ours', 'siren', 'first'])
parser.add_argument('--large_gop', action='store_true')
parser.add_argument('--chunk', type=int, default=0)


if __name__ == '__main__':
    args = parser.parse_args()
    print(str(vars(args)))

    videos = sorted(glob.glob('../uvg1k/*.mp4')) # 'training/final/*'))
    videos = videos # [args.chunk::2]

    scope = {
        '../uvg1k/Beauty_1920x1080_120fps_420_8bit_YUV.mp4': [0.14, 0.2],
        # '../uvg1k/Jockey_1920x1080_120fps_420_8bit_YUV.mp4': [10, 10, 10],
        # '../uvg1k/HoneyBee_1920x1080_120fps_420_8bit_YUV.mp4': [0.14],
        # '../uvg1k/YachtRide_1920x1080_120fps_420_8bit_YUV.mp4': [10]
        # '../uvg1k/ShakeNDry_1920x1080_120fps_420_8bit_YUV.mp4': [15, 15, 15],
    }

    # run
    for video in scope: # videos:
        if args.type == 'ours':
            if args.large_gop:
                n_frames = 15
                ratio = 1
                qualities = [10, 15, 20]
            else:
                n_frames = 5
                ratio = 0.25
                qualities = [9, 14, 19]

            qualities = scope[video] # temp

            for quality in qualities:
                os.system(f'rm -r {args.tag}')
                os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python3 compress.py'
                          f' --use_amp --n_frames={n_frames} --ratio={ratio} '
                          f'--quality={quality} --tag={args.tag} '
                          f'--video_path={video}')
        elif args.type == 'first':
            n_frames = 15
            ratio = 1
            qualities = [22, 17, 12] # [12, 17, 22]

            for quality in qualities:
                os.system(f'rm -r {args.tag}')
                os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python3 compressv3.py'
                          f' --use_amp --n_frames={n_frames} --width={ratio} '
                          f'--quality={quality} --tag={args.tag} '
                          f'--video_path={video}')
        elif args.type == 'siren':
            if args.large_gop:
                n_frames = 500
                lr = 1e-5
            else:
                n_frames = 5
                lr = 1e-3
            bpps = scope[video] # [0.1, 0.2, 0.3]

            for bpp in bpps:
                os.system(f'rm -r {args.tag}')
                os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python3 '
                          f'compress_siren.py --use_amp --n_frames={n_frames} '
                          f'--bpp={bpp} --tag={args.tag} --video_path={video} '
                          f'--lr={lr}')

