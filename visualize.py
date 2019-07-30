import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

def visualize_LiquidFun(points, outf, boundary=[[-2, 0], [2, 0], [2, 4], [-2, 4]]):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    boundary = np.asarray(boundary)
    x, y = np.min(boundary, axis=0)
    width, height = np.max(boundary, axis=0) - np.asarray([x, y])
    # print('(x, y, width, height)', x, y, width, height)
    box = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(box)
    ax.scatter(pos[:, 0], pos[:, 1], alpha=0.5, s=np.pi*3)
    # plt.show()
    plt.savefig(outf)

    plt.close()


if __name__ == '__main__':
    rollout = 1

    outf = 'out/test_LiquidFun/train/{rollout}'.format(rollout=rollout)
    if not os.path.exists(outf):
        os.makedirs(outf)

    for time_step in tqdm(range(1, 150 + 1), desc=outf):
        data_path = 'data/test_LiquidFun/train/{rollout}/{time_step}.txt'.format(rollout=rollout, time_step=time_step)
        data = np.loadtxt(data_path)
        pos = data[:, :2]
        visualize_LiquidFun(pos, outf='{outf}/{time_step}.jpg'.format(outf=outf, time_step=time_step))

    os.system('ffmpeg -framerate 60 -i {outf}/%d.jpg out/test_LiquidFun/train/{rollout}.mp4'.format(outf=outf, rollout=rollout))
