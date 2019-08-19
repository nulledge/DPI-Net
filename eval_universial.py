import os
import cv2
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import gzip
import pickle
import h5py

import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from data_universial import load_data, prepare_input, normalize, denormalize
from models import DPINet
from utils import calc_box_init_FluidShake


parser = argparse.ArgumentParser()
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--env', default='')
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--time_step_clip', type=int, default=0)
parser.add_argument('--dt', type=float, default=1./60.)
parser.add_argument('--nf_relation', type=int, default=300)
parser.add_argument('--nf_particle', type=int, default=200)
parser.add_argument('--nf_effect', type=int, default=200)
parser.add_argument('--outf', default='files')
parser.add_argument('--dataf', default='data')
parser.add_argument('--evalf', default='eval')
parser.add_argument('--eval', type=int, default=1)
parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)

parser.add_argument('--debug', type=int, default=0)

parser.add_argument('--n_instances', type=int, default=0)
parser.add_argument('--n_stages', type=int, default=0)
parser.add_argument('--n_his', type=int, default=0)

# shape state:
# [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
parser.add_argument('--shape_state_dim', type=int, default=14)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)
parser.add_argument('--position_dim', type=int, default=0)

# relation attr:
parser.add_argument('--relation_dim', type=int, default=0)

args = parser.parse_args()


phases_dict = dict()
# args.env = env

if args.env == 'BoxBath':
    env_idx = 1
    args.n_rollout = 3000

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # universial [rigid, fluid, wall_0, wall_1, wall_2, wall_3, wall_4, root]
    args.attr_dim = 8

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.time_step = 151
    args.time_step_clip = 0
    args.n_instance = 2
    args.n_stages = 4

    args.neighbor_radius = 0.08

    # ball, fluid
    phases_dict["instance_idx"] = [0, 64, 1024]
    phases_dict["root_num"] = [[8], []]
    phases_dict["root_sib_radius"] = [[0.4], []]
    phases_dict["root_des_radius"] = [[0.08], []]
    phases_dict["root_pstep"] = [[args.pstep], []]
    phases_dict["instance"] = ['cube', 'fluid']
    phases_dict["material"] = ['rigid', 'fluid']

    args.outf = 'dump_mixed/files'

elif args.env == 'FluidShake':
    env_idx = 6

    height = 1.0
    border = 0.025

    args.n_rollout = 2000

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # wall_0: floor
    # wall_1: left
    # wall_2: right
    # wall_3: back
    # wall_4: front
    # universial [rigid, fluid, wall_0, wall_1, wall_2, wall_3, wall_4, root]
    args.attr_dim = 8

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = 1
    args.time_step = 301
    args.time_step_clip = 0
    # args.n_stages = 2
    args.n_stages = 4

    args.neighbor_radius = 0.08

    phases_dict["root_num"] = [[]]
    phases_dict["instance"] = ["fluid"]
    phases_dict["material"] = ["fluid"]

    args.outf = 'dump_mixed/files'

else:
    raise AssertionError("Unsupported env")


args.outf = args.outf + '_' + args.env
args.dataf = 'data/data_' + args.env


# args.outf = args.outf + '_' + args.env
# args.evalf = args.evalf + '_' + args.env
# args.dataf = 'data/' + args.dataf + '_' + args.env

print(args)

if args.env == 'FluidShake':
    data_names = ['positions', 'velocities', 'clusters', 'shape_quats', 'scene_params']
elif args.env == 'BoxBath':
    data_names = ['positions', 'velocities', 'clusters', 'shape_quats', 'scene_params']
else:
    raise Exception()


print("Loading stored stat from %s" % args.dataf)
stat_path = os.path.join('data/data_BoxBath', 'stat.h5')
stat_BoxBath = load_data(data_names[:2], stat_path)
for i in range(len(stat_BoxBath)):
    stat_BoxBath[i] = stat_BoxBath[i][-args.position_dim:, :]
    # print(data_names[i], stat[i].shape)

stat_path = os.path.join('data/data_FluidShake', 'stat.h5')
stat_FluidShake = load_data(data_names[:2], stat_path)
for i in range(len(stat_FluidShake)):
    stat_FluidShake[i] = stat_FluidShake[i][-args.position_dim:, :]

from data_universial import combine_stat
for j in range(len(stat_BoxBath)):
    stat_BoxBath[j] = combine_stat(stat_BoxBath[j], stat_FluidShake[j])
stat = stat_BoxBath

use_gpu = torch.cuda.is_available()

model = DPINet(args, stat, None, residual=True, use_gpu=use_gpu)

# if args.epoch == 0 and args.iter == 0:
#     model_file = os.path.join(args.outf, 'net_best.pth')
# else:
#     model_file = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter))
model_file = 'dump_mixed/files_FluidShake/net_best.pth'

print("Loading network from %s" % model_file)
model.load_state_dict(torch.load(model_file))
model.eval()

criterionMSE = nn.MSELoss()

if use_gpu:
    model.cuda()


infos = np.arange(5)

import pyflex
pyflex.init()

recs = []

for idx in range(len(infos)):

    print("Rollout %d / %d" % (idx, len(infos)))
    des_dir = os.path.join(args.evalf, 'rollout_%d' % idx)
    os.system('mkdir -p ' + des_dir)

    # ground truth
    for step in range(args.time_step - 1):
        data_path = os.path.join(args.dataf, 'valid', str(infos[idx]), str(step) + '.h5')
        data_nxt_path = os.path.join(args.dataf, 'valid', str(infos[idx]), str(step + 1) + '.h5')

        data = load_data(data_names, data_path)
        data_nxt = load_data(data_names, data_nxt_path)
        velocities_nxt = data_nxt[1]

        if step == 0:
            if args.env == 'FluidShake' or args.env == 'BoxBath':
                positions, velocities, clusters, shape_quats, scene_params = data
                n_shapes = shape_quats.shape[0]
            else:
                raise Exception()

            count_nodes = positions.shape[0]
            n_particles = count_nodes - n_shapes
            print("n_particles", n_particles)
            print("n_shapes", n_shapes)

            p_gt = np.zeros((args.time_step - 1, n_particles + n_shapes, args.position_dim))
            s_gt = np.zeros((args.time_step - 1, n_shapes, args.shape_state_dim))
            v_nxt_gt = np.zeros((args.time_step - 1, n_particles + n_shapes, args.position_dim))

            p_pred = np.zeros((args.time_step - 1, n_particles + n_shapes, args.position_dim))

        p_gt[step] = positions[:, -args.position_dim:]
        v_nxt_gt[step] = velocities_nxt[:, -args.position_dim:]

        # print(step, np.sum(np.abs(v_nxt_gt[step, :args.n_particles])))

        if args.env == 'RiceGrip' or args.env == 'FluidShake' or args.env == 'BoxBath':
            s_gt[step, :, :3] = positions[n_particles:, :3]
            s_gt[step, :, 3:6] = p_gt[max(0, step-1), n_particles:, :3]
            s_gt[step, :, 6:10] = data[3]
            s_gt[step, :, 10:] = data[3]

        positions = positions + velocities_nxt * args.dt

    # model rollout
    data_path = os.path.join(args.dataf, 'valid', str(infos[idx]), '0.h5')
    data = load_data(data_names, data_path)

    for step in range(args.time_step - 1):
        if step % 10 == 0:
            print("Step %d / %d" % (step, args.time_step - 1))

        p_pred[step] = data[0]

        if args.env == 'RiceGrip' and step == 0:
            data[0] = p_gt[step + 1].copy()
            data[1] = np.concatenate([v_nxt_gt[step]] * (args.n_his + 1), 1)
            continue

        # st_time = time.time()
        attr, state, rels, n_particles, n_shapes, instance_idx = \
                prepare_input(data, stat, args, phases_dict, args.verbose_data)

        Ra, node_r_idx, node_s_idx, pstep = rels[3], rels[4], rels[5], rels[6]

        Rr, Rs = [], []
        for j in range(len(rels[0])):
            Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]
            Rr.append(torch.sparse.FloatTensor(
                Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))
            Rs.append(torch.sparse.FloatTensor(
                Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

        buf = [attr, state, Rr, Rs, Ra]

        with torch.set_grad_enabled(False):
            if use_gpu:
                for d in range(len(buf)):
                    if type(buf[d]) == list:
                        for t in range(len(buf[d])):
                            buf[d][t] = Variable(buf[d][t].cuda())
                    else:
                        buf[d] = Variable(buf[d].cuda())
            else:
                for d in range(len(buf)):
                    if type(buf[d]) == list:
                        for t in range(len(buf[d])):
                            buf[d][t] = Variable(buf[d][t])
                    else:
                        buf[d] = Variable(buf[d])

            attr, state, Rr, Rs, Ra = buf
            # print('Time prepare input', time.time() - st_time)

            # st_time = time.time()
            vels = model(
                attr, state, Rr, Rs, Ra, n_particles,
                node_r_idx, node_s_idx, pstep,
                instance_idx, phases_dict['material'], args.verbose_model)
            # print('Time forward', time.time() - st_time)

            # print(vels)

            if args.debug:
                data_nxt_path = os.path.join(args.dataf, 'valid', str(infos[idx]), str(step + 1) + '.h5')
                data_nxt = normalize(load_data(data_names, data_nxt_path), stat)
                label = Variable(torch.FloatTensor(data_nxt[1][:n_particles]).cuda())
                # print(label)
                loss = np.sqrt(criterionMSE(vels, label).item())
                print(loss)

        vels = denormalize([vels.data.cpu().numpy()], [stat[1]])[0]

        if args.env == 'RiceGrip' or args.env == 'FluidShake' or args.env == 'BoxBath':
            vels = np.concatenate([vels, v_nxt_gt[step, n_particles:]], 0)
        data[0] = data[0] + vels * args.dt

        if args.env == 'RiceGrip':
            # shifting the history
            # positions, restPositions
            data[1][:, args.position_dim:] = data[1][:, :-args.position_dim]
        data[1][:, :args.position_dim] = vels

        if args.debug:
            data[0] = p_gt[step + 1].copy()
            data[1][:, :args.position_dim] = v_nxt_gt[step]

    ##### render for the ground truth
    pyflex.set_scene(env_idx, scene_params, 0)

    if args.env == 'RiceGrip':
        halfEdge = np.array([0.15, 0.8, 0.15])
        center = np.array([0., 0., 0.])
        quat = np.array([1., 0., 0., 0.])
        pyflex.add_box(halfEdge, center, quat)
        pyflex.add_box(halfEdge, center, quat)
    elif args.env == 'FluidShake':
        x, y, z, dim_x, dim_y, dim_z, box_dis_x, box_dis_z = scene_params
        boxes = calc_box_init_FluidShake(box_dis_x, box_dis_z, height, border)

        x_box = x + (dim_x-1)/2.*0.055

        for box_idx in range(len(boxes) - 1):
            halfEdge = boxes[box_idx][0]
            center = boxes[box_idx][1]
            quat = boxes[box_idx][2]
            pyflex.add_box(halfEdge, center, quat)


    for step in range(args.time_step - 1):
        if args.env == 'RiceGrip':
            pyflex.set_shape_states(s_gt[step])
        elif args.env == 'FluidShake':
            pyflex.set_shape_states(s_gt[step, :-1])

        mass = np.zeros((n_particles, 1))
        if args.env == 'RiceGrip':
            p = np.concatenate([p_gt[step, :n_particles, -3:], mass], 1)
        else:
            p = np.concatenate([p_gt[step, :n_particles], mass], 1)

        pyflex.set_positions(p)
        pyflex.render(capture=1, path=os.path.join(des_dir, 'gt_%d.tga' % step))

    ##### render for the predictions
    pyflex.set_scene(env_idx, scene_params, 0)

    if args.env == 'RiceGrip':
        pyflex.add_box(halfEdge, center, quat)
        pyflex.add_box(halfEdge, center, quat)
    elif args.env == 'FluidShake':
        for box_idx in range(len(boxes) - 1):
            halfEdge = boxes[box_idx][0]
            center = boxes[box_idx][1]
            quat = boxes[box_idx][2]
            pyflex.add_box(halfEdge, center, quat)
    elif args.env == 'BoxBath':
        height = 1.0
        border = 0.025

        x_center = 1.25170865 / 2
        z_center = 0.38907054 / 2

        box_dis_x = 1.26127375 + border
        box_dis_z = 0.41770718 + border

        boxes = calc_box_init_FluidShake(box_dis_x, box_dis_z, height, border)
        for i in range(len(boxes)):
            halfEdge = boxes[i][0]
            center = boxes[i][1]
            quat = boxes[i][2]
            pyflex.add_box(halfEdge, center, quat)
    from tqdm import tqdm
    for step in tqdm(range(args.time_step - 1)):
        if args.env == 'RiceGrip':
            pyflex.set_shape_states(s_gt[step])
        elif args.env == 'FluidShake' or args.env == 'BoxBath':
            pyflex.set_shape_states(s_gt[step, :-1])

        mass = np.zeros((n_particles, 1))
        if args.env == 'RiceGrip':
            p = np.concatenate([p_pred[step, :n_particles, -3:], mass], 1)
        else:
            p = np.concatenate([p_pred[step, :n_particles], mass], 1)

        pyflex.set_positions(p)
        pyflex.render(capture=1, path=os.path.join(des_dir, 'pred_%d.tga' % step))

pyflex.clean()

