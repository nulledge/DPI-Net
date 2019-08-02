import os
import numpy as np
import scipy.spatial as spatial
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0**2 * n_0 + std_1**2 * n_1 + \
                   (mean_0 - mean)**2 * n_0 + (mean_1 - mean)**2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)

class LiquidFunDataset(Dataset):

    def __init__(self, config):
        self.config = config

        assert self.config.train_ratio + self.config.eval_ratio== 1.0
        assert (self.config.n_rollout * self.config.eval_ratio / self.config.num_workers).is_integer()
        assert (self.config.n_rollout * self.config.eval_ratio / self.config.num_workers).is_integer()

        if self.config.gen_data:
            self._generate()
        self.stats = np.load(os.path.join(self.config.outf, 'train', 'stat.npy'))

    def __len__(self):
        length = self.config.n_rollout \
            * (self.config.eval_ratio if self.config.eval else self.config.train_ratio) \
            * (self.config.time_step - 1 - self.config.time_step_clip)  # last time step should have the supervised data

        assert length.is_integer()

        return int(length)

    def __getitem__(self, idx):
        rollout = idx // (self.config.time_step - 1 - self.config.time_step_clip) + 1
        time_step = idx % (self.config.time_step - 1 - self.config.time_step_clip) + self.config.time_step_clip

        data_path = os.path.join(
            self.config.outf,
            'eval' if self.config.eval else 'train',
            str(rollout),
            '{time_step}.txt'.format(time_step=time_step),
        )
        data = np.loadtxt(data_path)
        data = [data[:, :self.config.position_dim], data[:, self.config.position_dim:]]
        # position.shape = (n_particles, position_dim)
        # velocity.shape = (n_particles, position_dim * (1 + n_his))

        attr, state, relations, n_particles, n_shapes, instance_idx = self._preprocess(data)

        label_path = os.path.join(
            self.config.outf,
            'eval' if self.config.eval else 'train',
            str(rollout),
            '{time_step}.txt'.format(time_step=time_step + 1),
        )
        label_data = np.loadtxt(label_path)
        label_data = [label_data[:, :self.config.position_dim], label_data[:, self.config.position_dim:]]

        data_nxt = self._normalize(label_data)

        label = torch.tensor(data_nxt[1][:n_particles]).float()

        return attr, state, relations, n_particles, n_shapes, instance_idx, label

    def _preprocess(self, data):

        if self.config.env == 'BoxBath':
            particle_pos, particle_vel, particle_cluster = data
        elif self.config.env == 'FluidFall' or self.config.env == 'LiquidFun' or self.config.env == 'LiquidFun_300':
            particle_pos, particle_vel = data
            particle_cluster = None
        elif self.config.env == 'LiquidFun_Rigid':
            particle_pos, particle_vel = data
            particle_cluster = [[
                0, 0, 1, 1, 2, 2, 3, 3,
                0, 0, 1, 1, 2, 2, 3, 3,
                4, 4, 5, 5, 6, 6, 7, 7,
                4, 4, 5, 5, 6, 6, 7, 7,
                8, 8, 9, 9, 10, 10, 11, 11,
                8, 8, 9, 9, 10, 10, 11, 11,
                12, 12, 13, 13, 14, 14, 15, 15,
                12, 12, 13, 13, 14, 14, 15, 15,
            ], None]
        else:
            raise Exception('Not implemented', self.config.env)
        n_particle = particle_pos.shape[0]
        n_node = n_particle

        if self.config.verbose_data:
            print("positions", particle_pos.shape)
            print("velocities", particle_vel.shape)
            print("n_particles", n_particle)

        particle_attr = np.zeros(shape=(n_particle, self.config.attr_dim))
        relations = []

        instance_idx = self.config.instance + [n_particle]
        for idx, mat in enumerate(self.config.material):
            st, ed = instance_idx[idx], instance_idx[idx + 1]

            if self.config.verbose_data:
                print('instance #%d' % idx, st, ed - 1)

            if self.config.env == 'BoxBath' or self.config.env == 'LiquidFun_Rigid':
                if mat == 'rigid':
                    particle_attr[st:ed, 0] = 1
                    queries = np.arange(st, ed)
                    anchors = np.concatenate((np.arange(0, st), np.arange(ed, n_particle)))
                elif mat == 'fluid':
                    particle_attr[st:ed, 1] = 1
                    queries = np.arange(st, ed)
                    anchors = np.arange(0, n_particle)
                else:
                    raise Exception('Not implemented', mat)
            elif self.config.env == 'FluidFall' or self.config.env == 'LiquidFun' or self.config.env == 'LiquidFun_300':
                if mat == 'fluid':
                    particle_attr[st:ed, 0] = 1
                    queries = np.arange(st, ed)
                    anchors = np.arange(0, n_particle)
                else:
                    raise Exception('Not implemented', mat)
            else:
                raise Exception('Not implemented', self.config.env)

            leaf2leaf_relations = self._find_relations_neighbor(particle_pos, queries, anchors, 2)
            relations += leaf2leaf_relations
        relations = np.concatenate(relations, 0)

        assert len(relations) > 0

        relation_receivers = [torch.tensor([relations[:, 0], np.arange(len(relations))], dtype=torch.int64)]
        relation_senders = [torch.tensor([relations[:, 1], np.arange(len(relations))], dtype=torch.int64)]
        relation_attrs = [torch.zeros(size=(len(relations), self.config.relation_dim)).float()]
        # relation_attrs[0][:, 0] = 1
        relation_values = [torch.ones(size=(len(relations), )).float()]

        node_receivers = [np.arange(n_particle)]
        node_senders = [np.arange(n_particle)]
        psteps = [self.config.pstep]

        node_pos = particle_pos
        node_vel = particle_vel
        node_attr = particle_attr

        for idx, n_root in enumerate(self.config.root_num):
            st, ed = instance_idx[idx], instance_idx[idx + 1]

            if n_root:
                node_attr, node_pos, node_vel, n_node, \
                new_relations, new_node_receivers, new_node_senders, new_psteps = \
                    self._make_hierarchy(node_attr, node_pos, node_vel, idx, st, ed, n_node, particle_cluster[idx])

                for relation_type in range(len(new_relations)):  # leaf2root, root2root, root2leaf
                    n_new_relation = len(new_relations[relation_type])
                    new_relation_receivers = new_relations[relation_type][:, 0]
                    new_relation_senders = new_relations[relation_type][:, 1]

                    relation_receivers.append(torch.tensor([new_relation_receivers, np.arange(n_new_relation)], dtype=torch.int64))
                    relation_senders.append(torch.tensor([new_relation_senders, np.arange(n_new_relation)], dtype=torch.int64))

                    new_relation_attr = torch.zeros(size=(n_new_relation, self.config.relation_dim))
                    new_relation_attr[:, 0] = 1
                    relation_attrs.append(new_relation_attr)

                    relation_values.append(torch.ones(size=(n_new_relation, )))

                    node_receivers.append(new_node_receivers[relation_type])
                    node_senders.append(new_node_senders[relation_type])

                    psteps.append(new_psteps[relation_type])

        ### normalize data
        data = [node_pos, node_vel]
        node_pos, node_vel = self._normalize(data)

        state = torch.tensor(np.concatenate([node_pos, node_vel], axis=1)).float()
        attr = torch.tensor(node_attr).float()
        relations = [relation_receivers, relation_senders, relation_values, relation_attrs, node_receivers, node_senders, psteps]

        # print('state', state.shape)
        # print('attr', attr.shape)
        # print('rel recv', len(relation_receivers), relation_receivers[0].shape, relation_receivers[1].shape, relation_receivers[2].shape, relation_receivers[3].shape)
        # print('rel send', len(relation_senders))
        # print('rel val', len(relation_values))
        # print('rel attr', len(relation_attrs))
        # print('node recv', len(node_receivers))
        # print('node send', len(node_senders))
        # print('pstep', psteps)

        return attr, state, relations, n_particle, 0, instance_idx


    def _normalize(self, data):
        stat = self.stats
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.  # if stddev is 0, then set to 1

            stat_dim = stat[i].shape[0]  # position_dim
            n_rep = int(data[i].shape[1] / stat_dim)  # velocity is repeated n_his times
            data[i] = data[i].reshape((-1, n_rep, stat_dim))
            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]  # normalize
            data[i] = data[i].reshape((-1, n_rep * stat_dim))
        return data


    def _make_hierarchy(self, particle_attr, particle_pos, particle_vel, instance_idx, instance_st, instance_ed, n_node, clusters, order=2):

        leaf2root_relations, root2leaf_relations, root2root_relations = [], [], []

        n_root = self.config.root_num[instance_idx]
        assert n_root is not None

        root_sib_radius = self.config.root_sib_radius[instance_idx]
        root_des_radius = self.config.root_des_radius[instance_idx]
        root_pstep = self.config.root_pstep[instance_idx]

        if self.config.verbose_data:
            print('root', n_root, root_sib_radius, root_des_radius, root_pstep)

        ################################################################################################################
        # leaft2root and root2leaf relations
        ################################################################################################################

        leaf2root_receivers = np.arange(n_node, n_node + n_root)
        leaf2root_senders = np.arange(instance_st, instance_ed)

        root2leaf_receivers = np.arange(instance_st, instance_ed)
        root2leaf_senders = np.arange(n_node, n_node + n_root)

        root2leaf_psteps = leaf2root_psteps = 1

        for root_idx in range(n_root):
            descendants = np.nonzero(clusters == root_idx)[0]
            roots = np.full(shape=(descendants.shape[0]), fill_value=root_idx, dtype=np.int64)
            if self.config.verbose_data:
                print(roots, descendants)

            leaf2root_relations += [np.stack([roots, descendants, np.zeros(shape=(len(descendants)))], axis=1)]
            root2leaf_relations += [np.stack([descendants, roots, np.zeros(shape=(len(descendants)))], axis=1)]

        leaf2root_relations = np.concatenate(leaf2root_relations, axis=0)
        root2leaf_relations = np.concatenate(root2leaf_relations, axis=0)

        ################################################################################################################
        # root2root relations
        ################################################################################################################

        root2root_receivers = np.arange(n_node, n_node + n_root)
        root2root_senders = np.arange(n_node, n_node + n_root)

        root2root_psteps = root_pstep

        roots = np.repeat(np.arange(n_root), n_root)    # [0, 0, 0, 1, 1, 1, 2, 2, 2]
        siblings = np.tile(np.arange(n_root), n_root)   # [0, 1, 2, 0, 1, 2, 0, 1, 2]

        root2root_relations += [np.stack([roots, siblings, np.zeros(shape=(n_root * n_root))], axis=1)]
        root2root_relations = np.concatenate(root2root_relations, axis=0)


        ################################################################################################################
        # add to attributes/positions/velocities
        ################################################################################################################
        root_pos = []
        root_vel = []
        root_attr =[]

        for root_idx in range(n_root):
            descendants = np.nonzero(clusters == root_idx)[0]

            root_pos += [np.mean(particle_pos[instance_st:instance_ed, :][descendants], 0, keepdims=True)]
            root_vel += [np.mean(particle_vel[instance_st:instance_ed, :][descendants], 0, keepdims=True)]
            root_attr += [np.mean(particle_attr[instance_st:instance_ed, :][descendants], 0, keepdims=True)]

        root_attr = np.concatenate(root_attr, axis=0)
        root_pos = np.concatenate(root_pos, axis=0)
        root_vel = np.concatenate(root_vel, axis=0)

        if self.config.env == 'BoxBath' or self.config.env == 'LiquidFun_Rigid':
            root_attr[:, 2 + 0] = 1

        node_attr = np.concatenate([particle_attr, root_attr], axis=0)
        node_pos = np.concatenate([particle_pos, root_pos], axis=0)
        node_vel = np.concatenate([particle_vel, root_vel], axis=0)

        if self.config.verbose_data:
            print(particle_attr.shape, root_attr.shape)
            print(particle_pos.shape, root_pos.shape)
            print(particle_vel.shape, root_vel.shape)
            print(node_pos.shape, node_vel.shape, node_attr.shape)

        n_node += n_root

        return node_attr, node_pos, node_vel, n_node, \
            [leaf2root_relations, root2root_relations, root2leaf_relations], \
            [leaf2root_receivers, root2root_receivers, root2leaf_receivers], \
            [leaf2root_senders, root2root_senders, root2leaf_senders], \
            [leaf2root_psteps, root2root_psteps, root2leaf_psteps]


    def _find_relations_neighbor(self, pos, queries, anchors, order=2):
        if np.sum(anchors) == 0:
            return []

        tree = spatial.cKDTree(pos[anchors])
        neighbors = tree.query_ball_point(pos[queries], self.config.neighbor_radius, p=order)

        relations = []
        for query, anchor, neighbor in zip(queries, anchors, neighbors):
            if len(neighbor) == 0:
                continue

            receiver = np.full(shape=(len(neighbor)), dtype=np.int64, fill_value=query)
            sender = np.array(anchors[neighbor])
            relations.append(np.stack([receiver, sender, np.ones(len(neighbor))], axis=1))

        return relations


    def _generate(self):
        stats = [
            np.zeros(shape=(self.config.position_dim, 3)),  # mean, stddev, number of particles for position
            np.zeros(shape=(self.config.position_dim, 3)),  # mean, stddev, number of particles for velocity
        ]
        n_rollout = self.config.n_rollout * (self.config.eval_ratio if self.config.eval else self.config.train_ratio)
        assert n_rollout.is_integer()
        for rollout in tqdm(range(int(n_rollout))):
            if self.config.env == 'LiquidFun':
                n_particle = 1024
            elif self.config.env == 'LiquidFun_Rigid':
                n_particle = 1024 + 64

            pos = np.zeros(shape=(self.config.time_step, n_particle, self.config.position_dim))
            vel = np.zeros(shape=(self.config.time_step, n_particle, self.config.position_dim))
            for time_step in range(self.config.time_step):
                data_path = os.path.join(
                    self.config.outf,
                    'eval' if self.config.eval else 'train',
                    str(rollout + 1),
                    '{time_step}.txt'.format(time_step=time_step)
                )
                data = np.loadtxt(data_path)
                pos[time_step, :, :] = data[:, :self.config.position_dim]
                vel[time_step, :, :] = data[:, self.config.position_dim:]

            for idx, attr in enumerate([pos, vel]):
                stat = np.zeros(shape=(self.config.position_dim, 3))
                stat[:, 0] = np.mean(attr, axis=(0, 1))  # mean over time and particles
                stat[:, 1] = np.std(attr, axis=(0, 1))  # stddev over time and particles
                stat[:, 2] = attr.shape[0] * attr.shape[1]
                stats[idx] = combine_stat(stats[idx], stat)
        stat_path = os.path.join(
            self.config.outf,
            'eval' if self.config.eval else 'train',
            'stat'
        )
        np.save(stat_path, stats)
