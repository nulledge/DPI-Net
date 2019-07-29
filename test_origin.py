import os
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import DPINet
from data import PhysicsFleXDataset, collate_fn

from utils import count_parameters


parser = argparse.ArgumentParser()
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--n_rollout', type=int, default=0)
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--time_step_clip', type=int, default=0)
parser.add_argument('--dt', type=float, default=1./60.)
parser.add_argument('--nf_relation', type=int, default=300)
parser.add_argument('--nf_particle', type=int, default=200)
parser.add_argument('--nf_effect', type=int, default=200)
parser.add_argument('--env', default='')
parser.add_argument('--train_valid_ratio', type=float, default=0.9)
parser.add_argument('--outf', default='files')
parser.add_argument('--dataf', default='data')
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--gen_data', type=int, default=0)
parser.add_argument('--gen_stat', type=int, default=0)
parser.add_argument('--log_per_iter', type=int, default=1000)
parser.add_argument('--ckp_per_iter', type=int, default=10000)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)

parser.add_argument('--n_instance', type=int, default=0)
parser.add_argument('--n_stages', type=int, default=0)
parser.add_argument('--n_his', type=int, default=0)

parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--forward_times', type=int, default=2)

parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--resume_iter', type=int, default=0)

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


if args.env == 'FluidFall':
    args.n_rollout = 3000

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [fluid]
    args.attr_dim = 1

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.time_step = 121
    args.time_step_clip = 5
    args.n_instance = 1
    args.n_stages = 1

    args.neighbor_radius = 0.08

    phases_dict["instance_idx"] = [0, 189]
    phases_dict["root_num"] = [[]]
    phases_dict["instance"] = ['fluid']
    phases_dict["material"] = ['fluid']

    args.outf = 'dump_FluidFall/' + args.outf

elif args.env == 'BoxBath':
    args.n_rollout = 3000

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [rigid, fluid, root_0]
    args.attr_dim = 3

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

    args.outf = 'dump_BoxBath/' + args.outf

elif args.env == 'FluidShake':
    args.n_rollout = 2000

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [fluid, wall_0, wall_1, wall_2, wall_3, wall_4]
    # wall_0: floor
    # wall_1: left
    # wall_2: right
    # wall_3: back
    # wall_4: front
    args.attr_dim = 6

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = 1
    args.time_step = 301
    args.time_step_clip = 0
    args.n_stages = 2

    args.neighbor_radius = 0.08

    phases_dict["root_num"] = [[]]
    phases_dict["instance"] = ["fluid"]
    phases_dict["material"] = ["fluid"]

    args.outf = 'dump_FluidShake/' + args.outf

elif args.env == 'RiceGrip':
    args.n_rollout = 5000
    args.n_his = 3

    # object state:
    # [rest_x, rest_y, rest_z, rest_xdot, rest_ydot, rest_zdot,
    #  x, y, z, xdot, ydot, zdot, quat.x, quat.y, quat.z, quat.w]
    args.state_dim = 16 + 6 * args.n_his
    args.position_dim = 6

    # object attr:
    # [fluid, root, gripper_0, gripper_1,
    #  clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep]
    args.attr_dim = 7

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = 1
    args.time_step = 41
    args.time_step_clip = 0
    args.n_stages = 4
    args.n_roots = 30

    args.neighbor_radius = 0.08

    phases_dict["root_num"] = [[args.n_roots]]
    phases_dict["root_sib_radius"] = [[5.0]]
    phases_dict["root_des_radius"] = [[0.2]]
    phases_dict["root_pstep"] = [[args.pstep]]
    phases_dict["instance"] = ["rice"]
    phases_dict["material"] = ["fluid"]

    args.outf = 'dump_RiceGrip/' + args.outf

else:
    raise AssertionError("Unsupported env")


args.outf = args.outf + '_' + args.env
args.dataf = 'data/' + args.dataf + '_' + args.env

os.system('mkdir -p ' + args.outf)
os.system('mkdir -p ' + args.dataf)

# generate data
datasets = {phase: PhysicsFleXDataset(
    args, phase, phases_dict, args.verbose_data) for phase in ['train', 'valid']}

# for phase in ['train', 'valid']:
#     if args.gen_data:
#         datasets[phase].gen_data(args.env)
#     else:
#         datasets[phase].load_data(args.env)
#
# use_gpu = torch.cuda.is_available()
#
#
# dataloaders = {x: torch.utils.data.DataLoader(
#     datasets[x], batch_size=args.batch_size,
#     shuffle=True if x == 'train' else False,
#     num_workers=args.num_workers,
#     collate_fn=collate_fn)
#     for x in ['train', 'valid']}

datasets['train'].load_data(args.env)

attr, state, rels, n_particles, n_shapes, instance_idx, label = datasets['train'][325]
Ra, node_r_idx, node_s_idx, pstep = rels[3], rels[4], rels[5], rels[6]

Rr, Rs = [], []
for j in range(len(rels[0])):
    Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]
    Rr.append(torch.sparse.FloatTensor(
        Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))
    Rs.append(torch.sparse.FloatTensor(
        Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

from test import *

dataset = PyFlexDataset(config)

attr_copy, state_copy, rels_copy, n_particles_copy, n_shapes_copy, instance_idx_copy, label_copy= dataset[315]
Ra, node_r_idx, node_s_idx, pstep = rels[3], rels[4], rels[5], rels[6]

print('mean diff(attr)', torch.mean(attr - attr_copy), attr.shape, attr_copy.shape)
print('mean diff(state)', torch.mean(state - state_copy), state.shape, state_copy.shape)
print('\tmean diff(pos)', torch.mean(state[:3] - state_copy[:3]))
print('\tmean diff(vel)', torch.mean(state[3:] - state_copy[3:]))
print('diff(n_particles)', n_particles - n_particles_copy)
print('diff(n_shapes)', n_shapes - n_shapes_copy)
print('instance idx', instance_idx, instance_idx_copy)
print('mean diff(label)', torch.mean(label - label_copy), label.shape, label_copy.shape)
print()

# relations = [relation_receivers, relation_senders, relation_values, relation_attrs, node_receivers, node_senders, psteps]
print(rels[0][0].dtype, rels_copy[0][0].dtype)
print('len(rels)', len(rels), len(rels_copy))
print('\trel recv',
        (rels[0][0] == rels_copy[0][0]).all(),
        (rels[0][1] == rels_copy[0][1]).all(),
        (rels[0][2] == rels_copy[0][2]).all(),
        (rels[0][3] == rels_copy[0][3]).all(),
      )
print('\trel send',
        (rels[1][0] == rels_copy[1][0]).all(),
        (rels[1][1] == rels_copy[1][1]).all(),
        (rels[1][2] == rels_copy[1][2]).all(),
        (rels[1][3] == rels_copy[1][3]).all(),
      )
print('\trel values',
        torch.mean(rels[2][0] - rels_copy[2][0]),
        torch.mean(rels[2][1] - rels_copy[2][1]),
        torch.mean(rels[2][2] - rels_copy[2][2]),
        torch.mean(rels[2][3] - rels_copy[2][3]),
      )
print('\trel attrs',
        torch.mean(rels[3][0] - rels_copy[3][0]),
        torch.mean(rels[3][1] - rels_copy[3][1]),
        torch.mean(rels[3][2] - rels_copy[3][2]),
        torch.mean(rels[3][3] - rels_copy[3][3]),
      )
print('\trel recv',
        (rels[4][0] == rels_copy[4][0]).all(),
        (rels[4][1] == rels_copy[4][1]).all(),
        (rels[4][2] == rels_copy[4][2]).all(),
        (rels[4][3] == rels_copy[4][3]).all(),
      )
print('\trel send',
        (rels[5][0] == rels_copy[5][0]).all(),
        (rels[5][1] == rels_copy[5][1]).all(),
        (rels[5][2] == rels_copy[5][2]).all(),
        (rels[5][3] == rels_copy[5][3]).all(),
      )
print('\tpsteps', rels[6], rels_copy[6])