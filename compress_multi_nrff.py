import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torchvision.utils import save_image
from sympy.solvers import solve
from sympy import Symbol

from embedding import *
from flow_utils import *
from metrics import PSNR, SSIM, MSE
from network import *
from utils import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='temp', help='tag (name)')
parser.add_argument('--save_path', type=str, default='./')

# model
parser.add_argument('--ratio', type=float, default=1,
                    help='the ratio between network and keyframe')
parser.add_argument('--hidden_layers', type=int, default=1,
                    help='the number of layers (default: 1)')
parser.add_argument('--reuse', type=int, default=2)
parser.add_argument('--use_amp', action='store_true',
                    help='use automatic mixed precision (32 to 16 bits)')

# video
parser.add_argument('--video_path', type=str,
                    default='./training/final/alley_1',
                    help='video path (for images use folder path, '
                         'for a video use video path)')
parser.add_argument('--video_scale', type=float, default=1,
                    help='the height and width of a output video will be '
                         'multiplied by video_scale')
parser.add_argument('--n_frames', type=int, default=15,
                    help='the number of frames')
parser.add_argument('--quality', type=int, default=25)
parser.add_argument('--codec', type=str, default='h264',
                    choices=['jpeg', 'avif', 'h264'])

# training
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 0.0005)')
parser.add_argument('--epochs', type=int, default=5000,
                    help='the number of training epochs')

parser.add_argument('--eval_interval', type=int, default=5000)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--visualize', action='store_true')


class EmptyContext:
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def main(args, target_frames, keyframes, kf_size, kf_idx, metrics,
         save_path, name='', verbose=False, visualize=True):
    keyframes = keyframes.cuda()
    target_frames = target_frames.cuda()

    # grids
    T, _, H, W = target_frames.size()
    input_grid = make_input_grid(T, H, W)
    flow_grid = make_flow_grid(H, W).unsqueeze(0)

    """ PREPARING A NETWORK """
    encoding = PosEncoding(3, np.floor(np.log2([T, H, W])).astype('int'), True,
                           trainable=True)
    x = Symbol('x')
    eq = x**2 + x*(11+encoding.get_output_size()) - args.ratio*kf_size/2
    width = int(np.round(float(max(solve(eq)))/2)*2)

    net = NeuralFieldsNetwork(3, 9, width, args.hidden_layers,
                              encoding, lambda : Swish(bias=True),
                              reuse=args.reuse)
    '''
    x = Symbol('x')
    eq = 3*x**2 + 12*x - args.ratio*kf_size/2
    width = int(np.round(float(max(solve(eq)))/2)*2)

    net = Siren(in_features=3, hidden_features=width, hidden_layers=3,
                out_features=9, outermost_linear=True)
    '''
    net = nn.DataParallel(net.cuda())

    optimizer = optim.Adam(net.parameters(),
                           betas=(0.9, 0.99), eps=1e-15, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    model_size = sum(p.numel() for p in net.parameters()) * (4 - 2*args.use_amp)
    total_size = model_size + keyframe_size

    """ MISC """
    n_metrics = len(metrics)

    header = ', '.join([str(m) for m in metrics] \
                       + ['MSE(flows)', 'MSE(residuals)'])
    perf_logs = []
    default_perfs = np.array(
        [m(keyframes[1:2], target_frames[kf_idx].unsqueeze(0)).cpu() / T
         for m in metrics])

    print(f'approx. total bytes ({total_size}) = '
          f'model size ({model_size}({width})) + keyframe ({keyframe_size})')
    print('keyframe quality:', default_perfs * T)
    if verbose:
        print(net)

    if args.use_amp:
        context = lambda : torch.cuda.amp.autocast(enabled=True)
    else:
        context = EmptyContext

    """ START TRAINING """
    flows_min = torch.tensor([-W/2, -H/2])
    flows_max = torch.tensor([W/2, H/2])

    net.train()
    # with tqdm.tqdm(range(args.epochs)) as loop:
    #     for epoch in loop:
    with EmptyContext(): # tqdm.tqdm(range(args.epochs)) as loop:
        for epoch in range(args.epochs):
            optimizer.zero_grad()

            is_eval_epoch = (epoch + 1) % args.eval_interval == 0

            if is_eval_epoch:
                perf_logs.append(np.zeros(n_metrics))
                perf_logs[-1][:n_metrics] = default_perfs

            for i in np.random.permutation(range(T-1)):
                dst = i + (i >= kf_idx)

                if i < kf_idx:
                    src_frame = keyframes[0]
                else:
                    src_frame = keyframes[2]

                with context():
                    i_grid = input_grid[dst] # [H, W, 3]
                    f_grid = flow_grid # [1, H, W, 2]
                    target = target_frames[dst].unsqueeze(0)

                    # for efficient training
                    b = torch.randint(4, ())
                    i_grid = i_grid[b::4]
                    f_grid = f_grid[:, b::4]
                    target = target[..., b::4, :]

                    b = torch.randint(4, ())
                    i_grid = i_grid[:, b::4]
                    f_grid = f_grid[:, :, b::4]
                    target = target[..., b::4]

                    outputs = net(i_grid)
                    flows0, flows1 = outputs[..., :2], outputs[..., 2:4]
                    # flows0 = flows0.clamp(min=flows_min)
                    # flows1 = flows1.clamp(max=flows_max)

                    alpha = torch.sigmoid(outputs[..., 4:5]) \
                                 .permute(2, 0, 1).unsqueeze(0)
                    outputs = outputs[..., -4:].permute(2, 0, 1).unsqueeze(0)
                    mu = 1 # torch.sigmoid(outputs[:, -4:-3])
                    outputs = torch.tanh(outputs[:, -3:])

                    warped0 = warp_frames(src_frame, flows0, f_grid)
                    warped1 = warp_frames(keyframes[1], flows1, f_grid)

                    reconstructed_frame = warped0 * alpha \
                                        + warped1 * (1-alpha)

                    reconstructed_frame = reconstructed_frame \
                                        + outputs * mu
                    reconstructed_frame = reconstructed_frame.clamp(0, 1)

                    loss = F.mse_loss(reconstructed_frame, target)

                    if torch.isnan(loss).item():
                        raise ValueError('NaN error')

                    if args.use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # update
                if args.use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            scheduler.step()

            if is_eval_epoch:
                net.eval()

                for i in range(T-1):
                    dst = i + (i >= kf_idx)

                    if i < kf_idx:
                        src_frame = keyframes[0]
                    else:
                        src_frame = keyframes[2]

                    with context():
                        outputs = net(input_grid[dst])
                        flows0, flows1 = outputs[..., :2], outputs[..., 2:4]
                        # flows0 = flows0.clamp(min=flows_min)
                        # flows1 = flows1.clamp(max=flows_max)

                        alpha = torch.sigmoid(outputs[..., 4:5]) \
                                     .permute(2, 0, 1).unsqueeze(0)
                        outputs = outputs[..., -4:].permute(2, 0, 1).unsqueeze(0)
                        mu = 1 # torch.sigmoid(outputs[:, -4:-3])
                        outputs = torch.tanh(outputs[:, -3:])

                        warped0 = warp_frames(src_frame, flows0, flow_grid)
                        warped1 = warp_frames(keyframes[1], flows1, flow_grid)

                        reconstructed_frame = warped0 * alpha \
                                            + warped1 * (1-alpha)

                        reconstructed_frame = reconstructed_frame \
                                            + outputs * mu
                        reconstructed_frame = reconstructed_frame.clamp(0, 1)

                    if visualize:
                        postfix = f'{name}_{epoch:04d}:{dst}.jpg'
                        save_image(warped0,
                                   os.path.join(save_path, f'warped0_{postfix}'))
                        save_image(warped1,
                                   os.path.join(save_path, f'warped1_{postfix}'))
                        save_image(alpha[0],
                                   os.path.join(save_path, f'alpha_{postfix}'))
                        save_image(mu,
                                   os.path.join(save_path, f'mu_{postfix}'))
                        save_image(outputs[0, -3:],
                                   os.path.join(save_path, f'res_{postfix}'))
                        save_image(reconstructed_frame[0],
                                   os.path.join(save_path, f'final_{postfix}'))
                        save_image(target_frames[dst],
                                   os.path.join(save_path,
                                                f'target_{name}:{dst}.jpg'))

                    for j in range(n_metrics):
                        value = metrics[j](
                            reconstructed_frame.float(),
                            target_frames[dst].unsqueeze(0)).item()
                        perf_logs[-1][j] += value / T

                net.train()

                # postfix = {str(metrics[i]): perf_logs[-1][i]
                #            for i in range(n_metrics)} # test performance
                # loop.set_postfix(postfix)

            # save logs
            if (epoch+1) == args.epochs:
                model_path = os.path.join(
                    save_path, f'model_{name}_{epoch+1:05d}.pt')
                torch.save({'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, model_path)
                np.savetxt(os.path.join(save_path, "logs.csv"),
                           perf_logs, fmt='%0.6f',
                           delimiter=", ", header=header, comments='')

    return perf_logs[-1], model_size


if __name__ == '__main__':
    args = parser.parse_args()
    print('configs')
    print(str(vars(args)))

    save_path = os.path.join(args.save_path, args.tag)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    """ PREPARING A VIDEO """
    target_frames = load_video(args.video_path, scale=args.video_scale)
    target_frames = target_frames[:100] # temp
    total_frames = len(target_frames)
    print(target_frames.shape)

    n_gop = int(max(np.round(total_frames / args.n_frames), 1))
    gop_indices = np.round(np.linspace(0, total_frames, n_gop+1)).astype('int')
    keyframe_indices = ((gop_indices[:-1] + gop_indices[1:])/2).astype('int')

    # keyframes
    keyframes = []
    keyframe_sizes = []

    for i, kf_idx in enumerate(keyframe_indices):
        keyframe, keyframe_size = save_keyframe(
            target_frames[kf_idx], args.quality,
            os.path.join(save_path, f'keyframe{i:02d}.jpeg'),
            codec=args.codec)
        keyframes.append(keyframe)
        keyframe_sizes.append(keyframe_size)

    metrics = [PSNR(), SSIM()]

    # compress
    performances = []
    model_sizes = []
    start = time.time()

    for i in range(n_gop): 
        # print(gop_indices[i:i+2], keyframe_indices[i])
        perfs, size = main(
            args,
            target_frames[gop_indices[i]:gop_indices[i+1]],
            torch.stack([keyframes[max(0, i-1)],
                         keyframes[i],
                         keyframes[min(n_gop-1, i+1)]], 0),
            keyframe_sizes[i],
            keyframe_indices[i]-gop_indices[i], metrics,
            save_path, f'{i}', args.verbose, args.visualize)
        performances.append(perfs)
        model_sizes.append(size)

    total_keyframe_size = sum(keyframe_sizes)
    total_model_size = sum(model_sizes)
    total_size = total_keyframe_size + total_model_size

    performances = np.stack(performances, 0)
    weights = ((gop_indices[1:] - gop_indices[:-1])
               / total_frames)[..., None]
    performances = np.sum(performances * weights, 0)

    print(f'total performances: {performances} ({time.time() - start} sec)')
    print(f'total_size: {total_size} '
          f'(kf: {total_keyframe_size}, mdl: {total_model_size})')
    print(f'bpp: {8*total_size/(total_frames * target_frames.shape[-2] * target_frames.shape[-1])}')

