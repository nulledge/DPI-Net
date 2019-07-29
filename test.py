import os
import numpy as np
import time
import scipy.spatial as spatial
import torch
import h5py
from torch.utils.data import Dataset
from sklearn.cluster import MiniBatchKMeans

class Config(object):
    __slots__ = [
        'env',                  # simulation environment ['BoxBath', 'FluidFall', 'FluidShake', 'RiceGrip']
        'pstep',                # propagation step
        'n_rollout',            # number of data
        'time_step',            # number of time steps
        'time_step_clip',       # number of time steps to be skipped initially
        'dt',                   # interval of single time step
        'nf_relation',          # number of features for relation
        'nf_particle',          # number of features for particle
        'nf_effect',            # number of features for propagation effect

        'train_ratio',          #
        'valid_ratio',          #
        'outf',                 #
        'dataf',                #
        'num_workers',          # number of processes to generate data
        'gen_data',             # generate data
        'gen_stat',             # generate statistics
        'log_per_iter',         # log interval
        'ckp_per_iter',         # check point interval
        'eval',                 # evaluation
        'verbose_data',         # log during data generation
        'verbose_model',        # log during model generation

        'n_instance',           #
        'n_stages',             #
        'n_his',                # number of previous time steps to be inputted

        'n_epoch',              # number of training epochs
        'beta1',                # optimizer parameter
        'lr',                   # learning rate
        'batch_size',           # batch size (should be 1)
        'forward_times',        #

        'resume_epoch',         # epoch to be resumed
        'resume_iter',          # iteration to be resumed

        'shape_state_dim',      # shape states [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
        'attr_dim',             # DoF for object attributes
        'state_dim',            # DoF for object states
        'position_dim',         # DoF for position in object states
        'relation_dim',         # DoF for relation

        'neighbor_radius',      # neighbor threshold
        'root_sib_radius',      #
        'root_des_radius',      #
        'root_pstep',           #

        'instance',             # index of instances
        'material',             # material of each instance
        'root_num',             # sampling for rigid body
    ]

    def __init__(
            self,
            env                 = None,
            pstep               = 2,
            n_rollout           = 0,
            time_step           = 0,
            time_step_clip      = 0,
            dt                  = 1/60,
            nf_relation         = 300,
            nf_particle         = 200,
            nf_effect           = 200,

            train_ratio         = 0.9,
            valid_ratio         = 0.1,
            outf                = 'files',
            dataf               = 'data',
            num_workers         = 10,
            gen_data            = False,
            gen_stat            = False,
            log_per_iter        = 1000,
            ckp_per_iter        = 10000,
            eval                = False,
            verbose_data        = True,
            verbose_model       = True,

            n_instance          = 0,
            n_stages            = 0,
            n_his               = 0,

            n_epoch             = 1000,
            beta1               = 0.9,
            lr                  = 0.0001,
            batch_size          = 1,
            forward_times       = 2,

            resume_epoch        = 0,
            resume_iter         = 0,

            shape_state_dim     = 14,
            attr_dim            = 0,
            state_dim           = 0,
            position_dim        = 0,
            relation_dim        = 0,

            neighbor_radius     = 0.0,
            root_sib_radius     = [],
            root_des_radius     = [],
            root_pstep          = [],

            instance            = [],
            material            = [],
            root_num            = [],
    ):
        self.env = env
        self.pstep = pstep
        self.n_rollout = n_rollout
        self.time_step = time_step
        self.time_step_clip = time_step_clip
        self.dt = dt
        self.nf_relation = nf_relation
        self.nf_particle = nf_particle
        self.nf_effect = nf_effect

        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.outf = outf
        self.dataf = dataf
        self.num_workers = num_workers
        self.gen_data = gen_data
        self.gen_stat = gen_stat
        self.log_per_iter = log_per_iter
        self.ckp_per_iter = ckp_per_iter
        self.eval = eval
        self.verbose_data = verbose_data
        self.verbose_model = verbose_model

        self.n_instance = n_instance
        self.n_stages = n_stages
        self.n_his = n_his

        self.n_epoch = n_epoch
        self.beta1 = beta1
        self.lr = lr
        self.batch_size = batch_size
        self.forward_times = forward_times

        self.resume_epoch = resume_epoch
        self.resume_iter = resume_iter

        self.shape_state_dim = shape_state_dim
        self.attr_dim = attr_dim
        self.state_dim = state_dim
        self.position_dim = position_dim
        self.relation_dim = relation_dim

        self.neighbor_radius = neighbor_radius
        self.root_sib_radius = root_sib_radius
        self.root_des_radius = root_des_radius
        self.root_pstep = root_pstep

        self.instance = instance
        self.material = material
        self.root_num = root_num


configs = {
    'FluidFall' : Config(
        env = 'FluidFall',
        n_rollout = 3000,
        state_dim = 6, position_dim = 3,    # [pos(xyz), vel(xyz)]
        attr_dim = 1,                       # [fluid]
        relation_dim = 1,                   # [None]
        time_step = 121, time_step_clip= 5,
        n_instance = 1, n_stages = 1,
        neighbor_radius = 0.08,
        instance = [0, ],
        root_num = [None, ],
        material = ['fluid', ],
        outf = 'test_FluidFall',
    ),
    'BoxBath' : Config(
        env = 'BoxBath',
        n_rollout = 3000,
        state_dim = 6, position_dim = 3,    # [pos(xyz), vel(xyz)]
        attr_dim = 3,                       # [rigid, fluid, root_0]
        relation_dim = 1,                   # [None]
        time_step = 151,
        n_instance = 2, n_stages = 4,
        neighbor_radius = 0.08,
        root_sib_radius = [0.04, None, ],
        root_des_radius = [0.08, None, ],
        root_pstep = [2, [], ],
        instance = [0, 64, ],
        root_num = [8, None, ],
        material = ['rigid', 'fluid', ],
        outf = 'test_BoxBath',
    ),
}

def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0**2 * n_0 + std_1**2 * n_1 + \
                   (mean_0 - mean)**2 * n_0 + (mean_1 - mean)**2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)

def generate(descriptor):
    config, process_idx = descriptor
    env_index = {
        'BoxBath' : 1,
        'FluidFall' : 4,
        'RiceGrip' : 5,
        'FluidShake' : 6,
    }[config.env]

    # 이거 이유를 모르겠음
    np.random.seed(round(time.time() * 1000 + process_idx) % 2**32)

    stats = [
        np.zeros(shape=(config.position_dim, 3)),  # mean, stddev, number of particles for position
        np.zeros(shape=(config.position_dim, 3)),  # mean, stddev, number of particles for velocity
    ]

    import pyflex
    pyflex.init()

    n_rollout = int(config.n_rollout * (config.valid_ratio if config.eval else config.train_ratio))
    for local_rollout_idx in range(n_rollout // config.num_workers):
        global_rollout_idx = (n_rollout // config.num_workers) * process_idx + local_rollout_idx
        rollout_dir = os.path.join(config.outf, 'eval' if config.eval else 'train', str(global_rollout_idx))
        os.system('mkdir -p {rollout_dir}'.format(rollout_dir=rollout_dir))

        if config.env == 'BoxBath' or config.env == 'FluidFall':
            pyflex.set_scene(
                env_index,
                np.zeros(shape=(1, )),  # scene params
                process_idx,
            )
            n_particles = pyflex.get_n_particles()

            pos = np.zeros(shape=(config.time_step, n_particles, config.position_dim), dtype=np.float32)
            vel = np.zeros(shape=(config.time_step, n_particles, config.position_dim), dtype=np.float32)

            for _ in range(config.time_step_clip):
                pos[-1, :, :] = pyflex.get_positions().reshape((-1, 4))[:, :3]
                pyflex.step()

            # root sampling for each rigid body
            instance_idx = config.instance + [n_particles]
            initial_body = []
            clusters = []
            for idx, mat in enumerate(config.material):
                st, ed = instance_idx[idx], instance_idx[idx+1]
                if mat == 'rigid':
                    initial_body.append(pyflex.get_positions().reshape((-1, 4))[st:ed, :3])
                    kmeans = MiniBatchKMeans(n_clusters=config.root_num[idx], random_state=0).fit(initial_body[idx])
                    clusters.append(kmeans.labels_)
                else:
                    initial_body.append(None)
                    clusters.append(None)

            for time_step in range(config.time_step):
                pos[time_step, :, :] = pyflex.get_positions().reshape((-1, 4))[:, :3]

                for idx, mat in enumerate(config.material):
                    st, ed = instance_idx[idx], instance_idx[idx + 1]

                    if mat == 'rigid':
                        # apply rigid projection to ground truth
                        XX = initial_body[idx].copy()
                        YY = pos[time_step, st:ed, :].copy()
                        # print("MSE init", np.mean(np.square(XX - YY)))

                        # Rigid body를 구성하는 particles 사이 거리가 흐뜨러져서
                        # fitting으로 강제로 맞춰주는 듯함
                        X = XX.copy().T
                        Y = YY.copy().T
                        mean_X = np.mean(X, 1, keepdims=True)
                        mean_Y = np.mean(Y, 1, keepdims=True)
                        X = X - mean_X
                        Y = Y - mean_Y
                        C = np.dot(X, Y.T)
                        U, S, Vt = np.linalg.svd(C)
                        D = np.eye(3)
                        D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
                        R = np.dot(Vt.T, np.dot(D, U.T))
                        t = mean_Y - np.dot(R, mean_X)

                        YY_fitted = (np.dot(R, XX.T) + t).T
                        # print("MSE fit", np.mean(np.square(YY_fitted - YY)))

                        pos[time_step, st:ed, :] = YY_fitted

                    else:
                        pass

                vel[time_step, :, :] = (pos[time_step, :, :] - pos[time_step - 1, : :]) / config.dt

                data = [pos[time_step], vel[time_step], clusters]
                np.save(os.path.join(rollout_dir, '{time_step}'.format(time_step=time_step)), data)

                pyflex.step()
        else:
            raise Exception('Not implemented')

        for idx, attr in enumerate([pos, vel]):
            stat = np.zeros(shape=(config.position_dim, 3))
            stat[:, 0] = np.mean(attr, axis=(0, 1))  # mean over time and particles
            stat[:, 1] = np.std(attr, axis=(0, 1))  # stddev over time and particles
            stat[:, 2] = attr.shape[0] * attr.shape[1]
            stats[idx] = combine_stat(stats[idx], stat)
            # stats[pos/vel, x/y/z, mean/stddev/n]

    pyflex.clean()

    return stats


config = configs['BoxBath']
data = None

config.n_rollout = 200
config.num_workers = 10
config.gen_data = False
config.verbose_data = False

assert config.train_ratio + config.valid_ratio == 1.0
assert (config.n_rollout * config.train_ratio / config.num_workers).is_integer()
assert (config.n_rollout * config.valid_ratio / config.num_workers).is_integer()

if config.gen_data:
    descriptors = []
    for process_idx in range(config.num_workers):
        descriptors.append((config, process_idx))

    import multiprocessing as mp
    process_pool = mp.Pool(processes=config.num_workers)
    data = process_pool.map(generate, descriptors)

    stats = [
        np.zeros(shape=(config.position_dim, 3)),  # mean, stddev, number of particles for position
        np.zeros(shape=(config.position_dim, 3)),  # mean, stddev, number of particles for velocity
    ]

    for worker in range(len(data)):
        for stat in range(len(stats)):
            stats[stat] = combine_stat(stats[stat], data[worker][stat])

    stat_path = os.path.join(config.outf, 'eval' if config.eval else 'train', 'stat')
    np.save(stat_path, stats)

# stats = np.load('{path}.npy'.format(path=stat_path))

# import h5py
# hf = h5py.File('data/data_BoxBath/stat.h5', 'r')
#
# print('replica', 'position')
# print(stats[0])
# print('origin', 'position')
# print(np.array(hf.get('positions')))
#
# print('replica', 'velocity')
# print(stats[1])
# print('origin', 'velocity')
# print(np.array(hf.get('velocities')))

# replica position
# [[5.55087973e-01 3.77842526e-01 2.78323200e+07]
#  [1.04661417e-01 1.21022816e-01 2.78323200e+07]
#  [1.92690892e-01 1.28401237e-01 2.78323200e+07]]
# origin position
# [[5.54782074e-01 3.77912550e-01 4.17484800e+08]
#  [1.04673462e-01 1.20935143e-01 4.17484800e+08]
#  [1.92932816e-01 1.28433721e-01 4.17484800e+08]]
# replica velocity
# [[2.24442301e-01 1.61336689e+00 2.78323200e+07]
#  [2.82442680e-02 2.19080533e+00 2.78323200e+07]
#  [7.64457848e-02 1.12076446e+00 2.78323200e+07]]
# origin velocity
# [[ 1.31977487e-01  6.23665377e-01  4.17484800e+08]
#  [-1.17241629e-01  4.45523826e-01  4.17484800e+08]
#  [-9.35901018e-05  1.27604036e-01  4.17484800e+08]]

class PyFlexDataset(Dataset):

    def __init__(self, config):
        self.config = config

        assert self.config.train_ratio + self.config.valid_ratio == 1.0
        assert (self.config.n_rollout * self.config.train_ratio / self.config.num_workers).is_integer()
        assert (self.config.n_rollout * self.config.valid_ratio / self.config.num_workers).is_integer()

        if self.config.gen_data:
            self._generate()
        self.stats = np.load(os.path.join(self.config.outf, 'train', 'stat.npy'))
        # hf = h5py.File('data/data_BoxBath/stat.h5', 'r')
        # self.stats = [np.array(hf.get('positions')), np.array(hf.get('velocities'))]
        # hf.close()

    def __len__(self):
        length = self.config.n_rollout \
            * (self.config.valid_ratio if self.config.eval else self.config.train_ratio) \
            * (self.config.time_step - 1)  # last time step should have the supervised data

        assert length.is_integer()

        return int(length)

    def __getitem__(self, idx):
        # time step is 0-based
        rollout = idx // (self.config.time_step - 1)
        time_step = idx % (self.config.time_step - 1) - 1

        # data_path = 'data/data_BoxBath/train/2/25.h5'
        # hf = h5py.File(data_path, 'r')
        # data = [np.array(hf.get('positions')), np.array(hf.get('velocities')), np.array(hf.get('clusters'))]
        # hf.close()
        #
        # data[2] = data[2][0][0]

        data = np.load(os.path.join(
            self.config.outf,
            'eval' if config.eval else 'train',
            str(rollout),
            '{time_step}.npy'.format(time_step=time_step)), allow_pickle=True)

        data[1] = [data[1]]

        # for prev in range(1, self.config.n_his + 1):
        #     data_path = os.path.join(self.config.outf, 'eval' if config.eval else 'train', str(rollout), '{time_step}.npy'.format(time_step=time_step - prev))
        #     vel = np.load(data_path, allow_pickle=True)[1]
        #     data[1].append(vel)
        data[1] = np.concatenate(data[1], 1)
        # position.shape = (n_particles, position_dim)
        # velocity.shape = (n_particles, position_dim * (1 + n_his))

        attr, state, relations, n_particles, n_shapes, instance_idx = self._preprocess(data)

        label_data = np.load(os.path.join(
            self.config.outf,
            'eval' if config.eval else 'train',
            str(rollout),
            '{time_step}.npy'.format(time_step=time_step + 1)), allow_pickle=True)

        # data_path = 'data/data_BoxBath/train/2/26.h5'
        # hf = h5py.File(data_path, 'r')
        # label_data = [np.array(hf.get('positions')), np.array(hf.get('velocities')), np.array(hf.get('clusters'))]
        # hf.close()

        data_nxt = self._normalize(label_data)

        label = torch.tensor(data_nxt[1][:n_particles]).float()

        return attr, state, relations, n_particles, n_shapes, instance_idx, label

    def _preprocess(self, data):

        if self.config.env == 'BoxBath':
            particle_pos, particle_vel, particle_cluster = data
        elif self.config.env == 'FluidFall':
            particle_pos, particle_vel = data
            particle_cluster = None
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

            if self.config.env == 'BoxBath':
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
            elif self.config.env == 'FluidFall':
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
        data = [node_pos.copy(), node_vel.copy()]
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
        stat = self.stats.copy()
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

        if self.config.env == 'BoxBath':
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
        descriptors = []
        for process_idx in range(config.num_workers):
            descriptors.append((config, process_idx))

        import multiprocessing as mp
        process_pool = mp.Pool(processes=config.num_workers)
        data = process_pool.map(generate, descriptors)

        stats = [
            np.zeros(shape=(config.position_dim, 3)),  # mean, stddev, number of particles for position
            np.zeros(shape=(config.position_dim, 3)),  # mean, stddev, number of particles for velocity
        ]

        for worker in range(len(data)):
            for stat in range(len(stats)):
                stats[stat] = combine_stat(stats[stat], data[worker][stat])

        stat_path = os.path.join(config.outf, 'eval' if config.eval else 'train', 'stat')
        np.save(stat_path, stats)

dataset = PyFlexDataset(config)

attr, state, rels, n_particles, n_shapes, instance_idx, label = dataset[315]
Ra, node_r_idx, node_s_idx, pstep = rels[3], rels[4], rels[5], rels[6]

relation_type = 3
print(len(node_r_idx), node_r_idx[relation_type].shape, Ra[relation_type].size(0))

Rr, Rs = [], []
for j in range(len(rels[0])):
    Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]
    Rr.append(torch.sparse.FloatTensor(
        Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))  # n(receivers) x n(relations)
    Rs.append(torch.sparse.FloatTensor(
        Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

data = [attr, state, Rr, Rs, Ra, label]