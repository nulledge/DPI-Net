import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_LiquidFun(points, outf, boundary=[[-2, 0], [2, 0], [2, 4], [-2, 4]]):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    boundary = np.asarray(boundary)
    x, y = np.min(boundary, axis=0)
    width, height = np.max(boundary, axis=0) - np.asarray([x, y])
    # print('(x, y, width, height)', x, y, width, height)
    box = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(box)
    ax.scatter(points[:, 0], points[:, 1], alpha=0.5, s=np.pi*3)
    # plt.show()
    plt.savefig(outf)

    plt.close()

def visualize_LiquidFun_Rigid(points, outf, boundary=[[-2, 0], [2, 0], [2, 4], [-2, 4]]):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    boundary = np.asarray(boundary)
    x, y = np.min(boundary, axis=0)
    width, height = np.max(boundary, axis=0) - np.asarray([x, y])
    # print('(x, y, width, height)', x, y, width, height)
    box = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(box)
    ax.scatter(points[64:, 0], points[64:, 1], alpha=0.5, s=np.pi*3, c='b')
    ax.scatter(points[:64, 0], points[:64, 1], alpha=0.5, s=np.pi * 3, c='g')
    # plt.show()
    plt.savefig(outf)

    plt.close()


if __name__ == '__main__':

    ################################
    #
    # Visualize raw data
    #
    ################################

    phases = ['train', 'eval']

    from environment import configs
    config = {phase: configs['{env}_{phase}'.format(env='LiquidFun', phase=phase)] for phase in phases}

    from liquidfun import LiquidFunDataset
    dataset = {phase: LiquidFunDataset(config[phase]) for phase in phases}

    from torch.utils.data import DataLoader
    from data import collate_fn
    loader = {
        phase: DataLoader(dataset[phase],
                          batch_size=config[phase].batch_size,
                          shuffle=False,
                          num_workers=config[phase].num_workers,
                          collate_fn=collate_fn,
                          ) for phase in phases}


    import os
    rollout = 1
    outf = 'out/test_LiquidFun/test/{rollout}'.format(rollout=rollout)
    if not os.path.exists(outf):
        os.makedirs(outf)

    from tqdm import tqdm
    from data import denormalize
    progress = tqdm(loader['train'], desc='dataloader')
    for data in progress:
        attr, state, rels, n_particles, n_shapes, instance_idx, label = data
        Ra, node_r_idx, node_s_idx, pstep = rels[3], rels[4], rels[5], rels[6]

        pos = state.data.cpu().numpy()[:, :2]
        pos = denormalize([pos], [dataset['train'].stats[0]])[0]

        visualize_LiquidFun(pos, outf='{outf}/{time_step}.jpg'.format(outf=outf, time_step=progress.n + 1))

        progress.update(1)
        if progress.n == config['train'].time_step:
            break

    progress.close()

    os.system('ffmpeg -framerate 60 -i {outf}/%d.jpg out/test_LiquidFun/test/{rollout}_test.mp4'.format(outf=outf, rollout=rollout))


    ################################
    #
    # Visualize raw data
    #
    ################################

    from tqdm import tqdm
    import os
    rollout = 1

    outf = 'out/test_LiquidFun/train/{rollout}'.format(rollout=rollout)
    if not os.path.exists(outf):
        os.makedirs(outf)

    for time_step in tqdm(range(1, 150 + 1), desc=outf):
        data_path = 'data/test_LiquidFun/train/{rollout}/{time_step}.txt'.format(rollout=rollout, time_step=time_step)
        data = np.loadtxt(data_path)
        pos = data[:, :2]
        visualize_LiquidFun(pos, outf='{outf}/{time_step}.jpg'.format(outf=outf, time_step=time_step))

    os.system('ffmpeg -framerate 60 -i {outf}/%d.jpg out/test_LiquidFun/train/{rollout}_train.mp4'.format(outf=outf, rollout=rollout))
