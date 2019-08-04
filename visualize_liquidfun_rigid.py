import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-env', '--environment', default='LiquidFun_Rigid')
parser.add_argument('-e', '--epoch', type=int, default=-1)
parser.add_argument('-iter', '--iteration', type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__':
    phase = 'eval'

    from environment import configs
    config = configs['{env}_{phase}'.format(env=args.environment, phase=phase)]

    from liquidfun import LiquidFunDataset
    dataset = LiquidFunDataset(config)

    from torch.utils.data import DataLoader
    from data import collate_fn
    loader = DataLoader(dataset,
                      batch_size=config.batch_size,
                      shuffle = False,
                      num_workers = config.num_workers,
                      collate_fn = collate_fn,
            )

    import torch
    from models import DPINet
    model = DPINet(config, dataset.stats, None, residual=True, use_gpu=torch.cuda.is_available())

    if args.epoch >= 0 and args.iteration >= 0:
        ckpt_path = '%s/ckpt/net_epoch_%d_iter_%d.pth' % (config.outf, args.epoch, args.iteration)
        print('load model:', ckpt_path)
        model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    if config.verbose_model:
        from utils import count_parameters
        print("Number of parameters: %d" % count_parameters(model))

    from torch import nn
    criterionMSE = nn.MSELoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterionMSE = criterionMSE.cuda()

    import os
    import numpy as np
    from tqdm import tqdm
    from torch.autograd import Variable
    from data import denormalize
    from visualize import visualize_LiquidFun_Rigid

    losses = 0.0
    pred_data = None
    with torch.set_grad_enabled(False):
        with tqdm(total=len(loader)) as progress:
            for idx in range(config.time_step - 1):
                rollout = progress.n // (config.time_step - 1) + 1
                time_step = progress.n % (config.time_step - 1)

                outf = 'out/test_LiquidFun_Rigid/eval/{rollout}'.format(rollout=rollout)
                if not os.path.exists(outf):
                    os.makedirs(outf)

                attr, state, rels, n_particles, n_shapes, instance_idx, label = dataset.__getitem__(idx, data=pred_data)
                Ra, node_r_idx, node_s_idx, pstep = rels[3], rels[4], rels[5], rels[6]

                Rr, Rs = [], []
                for j in range(len(rels[0])):
                    Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]
                    Rr.append(torch.sparse.FloatTensor(
                        Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))
                    Rs.append(torch.sparse.FloatTensor(
                        Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

                data = [attr, state, Rr, Rs, Ra, label]

                if torch.cuda.is_available():
                    for d in range(len(data)):
                        if type(data[d]) == list:
                            for t in range(len(data[d])):
                                data[d][t] = Variable(data[d][t]).cuda()
                        else:
                            data[d] = Variable(data[d]).cuda()
                else:
                    for d in range(len(data)):
                        if type(data[d]) == list:
                            for t in range(len(data[d])):
                                data[d][t] = torch.Variable(data[d][t])
                        else:
                            data[d] = torch.Variable(data[d])

                attr, state, Rr, Rs, Ra, label = data

                predicted = model(
                    attr, state, Rr, Rs, Ra, n_particles,
                    node_r_idx, node_s_idx, pstep,
                    instance_idx, config.material, config.verbose_model)

                loss = criterionMSE(predicted, label)
                losses += np.sqrt(loss.item())

                pos = state.data.cpu().numpy()[:1024+64, :2]
                pos = denormalize([pos], [dataset.stats[0]])[0]

                vel = state.data.cpu().numpy()[:1024 + 64, 2:]
                vel = denormalize([pos], [dataset.stats[1]])[0]

                if time_step == 0:
                    pred_pos = pos.copy()
                    pred_vel = vel.copy()
                else:
                    pred_vel = denormalize([predicted.data.cpu().numpy()], [dataset.stats[1]])[0]
                    pred_pos = pred_pos + pred_vel[:1024 + 64] * config.dt
                pred_data = [pred_pos, pred_vel]

                # visualize_LiquidFun_Rigid(pos, outf='{outf}/gt_{time_step}.jpg'.format(outf=outf, time_step=time_step))
                visualize_LiquidFun_Rigid(pred_pos, outf='{outf}/pred_{time_step}.jpg'.format(outf=outf, time_step=time_step))
                if time_step == config.time_step - 2:
                    # os.system('ffmpeg -framerate 60 -i {outf}/gt_%d.jpg out/test_LiquidFun_Rigid/eval/{rollout}_gt.mp4'.format(outf=outf, rollout=rollout))
                    os.system('ffmpeg -framerate 60 -i {outf}/pred_%d.jpg out/test_LiquidFun_Rigid/eval/{rollout}_pred.mp4'.format(outf=outf, rollout=rollout))

                progress.set_postfix(loss='%.3f' % np.sqrt(loss.item()), agg='%.3f' % (losses / (progress.n + 1)))
                # progress.set_postfix(mean_delta='%.3f' % float(delta / (progress.n + 1)))
                progress.update()

            losses /= len(loader)
            progress.set_postfix(loss=losses)