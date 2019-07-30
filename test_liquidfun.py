import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-env', '--environment', default='LiquidFun')
args = parser.parse_args()

if __name__ == '__main__':
    phases = ['train', 'eval']

    from environment import configs
    config = {phase: configs['{env}_{phase}'.format(env=args.environment, phase=phase)] for phase in phases}

    from liquidfun import LiquidFunDataset
    dataset = {phase: LiquidFunDataset(config[phase]) for phase in phases}

    from torch.utils.data import DataLoader
    from data import collate_fn
    loader = {
        phase: DataLoader(dataset[phase],
                          batch_size=config[phase].batch_size,
                          shuffle = (not config[phase].eval),
                          num_workers = config[phase].num_workers,
                          collate_fn = collate_fn,
        ) for phase in phases}

    import torch
    from models import DPINet
    model = DPINet(config['train'], dataset['train'].stats, None, residual=True, use_gpu=torch.cuda.is_available())

    if config['train'].verbose_model:
        from utils import count_parameters
        print("Number of parameters: %d" % count_parameters(model))

    from torch import nn, optim
    criterionMSE = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train'].lr, betas=(config['train'].beta1, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

    if torch.cuda.is_available():
        model = model.cuda()
        criterionMSE = criterionMSE.cuda()

    import numpy as np
    from tqdm import tqdm
    from torch.autograd import Variable
    best_valid_loss = np.inf
    for epoch in range(config['train'].n_epoch):

        from torch.utils.tensorboard import SummaryWriter
        from datetime import datetime
        writer = SummaryWriter('{dir}/log/{time}.log'.format(
            dir=config['train'].outf,
            time=datetime.now().strftime('%b%d_%H-%M-%S')))

        for phase in phases:

            if phase == 'train':
                model.train()
            elif phase == 'eval':
                model.eval()
            else:
                raise Exception('Not implemented')

            with torch.set_grad_enabled(phase == 'train'):
                losses = 0
                with tqdm(loader[phase]) as progress:
                    for data in progress:
                        attr, state, rels, n_particles, n_shapes, instance_idx, label = data
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

                        # st_time = time.time()
                        predicted = model(
                            attr, state, Rr, Rs, Ra, n_particles,
                            node_r_idx, node_s_idx, pstep,
                            instance_idx, config[phase].material, config[phase].verbose_model)
                        # print('Time forward', time.time() - st_time)

                        # print(predicted)
                        # print(label)

                        loss = criterionMSE(predicted, label)
                        losses += np.sqrt(loss.item())

                        if phase == 'train':
                            if progress.n % config[phase].forward_times == 0:
                                # update parameters every args.forward_times
                                if progress.n != 0:
                                    loss_acc /= config[phase].forward_times
                                    optimizer.zero_grad()
                                    loss_acc.backward()
                                    optimizer.step()
                                loss_acc = loss
                            else:
                                loss_acc += loss

                        # if progress.n % config[phase].log_per_iter == 0:
                        n_relations = 0
                        for j in range(len(Ra)):
                            n_relations += Ra[j].size(0)
                        progress.set_postfix(loss='%.3f' % np.sqrt(loss.item()), agg='%.3f' % (losses / (progress.n + 1)))
                                             # , n_rel=n_relations)
                            # print('%s [%d/%d][%d/%d] n_relations: %d, Loss: %.6f, Agg: %.6f' %
                            #       (phase, epoch, args.n_epoch, i, len(dataloaders[phase]),
                            #        n_relations, , ))

                        if progress.n % 100 == 0:
                            writer.add_scalar('{phase}/loss'.format(phase=phase),
                                               np.sqrt(loss.item()),
                                               epoch * len(dataset[phase]) + progress.n)
                            writer.add_scalar('{phase}/agg'.format(phase=phase),
                                               losses / (progress.n + 1),
                                               epoch * len(dataset[phase]) + progress.n)

                        if phase == 'train' and progress.n > 0 and progress.n % config['train'].ckp_per_iter == 0:
                            torch.save(model.state_dict(), '%s/ckpt/net_epoch_%d_iter_%d.pth' % (config['train'].outf, epoch, progress.n))

                losses /= len(loader[phase])
                progress.set_postfix(loss=losses, best=best_valid_loss)
                writer.add_scalar('{phase}/mean_loss'.format(phase=phase),
                                   losses,
                                  (epoch + 1) * len(dataset[phase]))
                writer.add_scalar('{phase}/best_eval_loss'.format(phase=phase),
                                   best_valid_loss,
                                  (epoch + 1) * len(dataset[phase]))

                # print('%s [%d/%d] Loss: %.4f, Best valid: %.4f' %
                #       (phase, epoch, args.n_epoch, losses, best_valid_loss))

                if phase == 'valid':
                    scheduler.step(losses)
                    if losses < best_valid_loss:
                        best_valid_loss = losses
                        torch.save(model.state_dict(), '%s/ckpt/net_best.pth' % config['train'].outf)

            writer.close()
    print('hello, world!')