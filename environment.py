import os

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
        'eval_ratio',           #
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
            eval_ratio          = 0.1,
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
        self.eval_ratio = eval_ratio
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
    'LiquidFun_train' : Config(
        env = 'LiquidFun',
        n_rollout = 3000,
        batch_size = 1,
        n_epoch = 5,
        gen_data = False,  # generate mean and stddev
        verbose_data = False,
        verbose_model = False,
        train_ratio = 0.9, eval_ratio = 0.1,
        num_workers = 1,
        state_dim = 4, position_dim = 2,  # [pos(xy) and vel(xy)]
        attr_dim = 1,
        relation_dim = 1,
        time_step = 151, time_step_clip = 0,
        n_instance = 1, n_stages = 1,
        neighbor_radius = 0.08 / 0.05 * 0.025,
        instance = [0, ],
        root_num = [None, ],
        material = ['fluid', ],
        outf = os.path.join('data', 'test_LiquidFun')
    ),
    'LiquidFun_eval' : Config(
        env = 'LiquidFun',
        n_rollout = 3000,
        batch_size = 1,
        eval = True, # evaluation on
        gen_data = False,  # generate mean and stddev
        verbose_data = False,
        verbose_model = False,
        train_ratio = 0.9, eval_ratio = 0.1,
        num_workers = 1,
        state_dim = 4, position_dim = 2,  # [pos(xy) and vel(xy)]
        attr_dim = 1,
        relation_dim = 1,
        time_step = 151, time_step_clip = 0,
        n_instance = 1, n_stages = 1,
        neighbor_radius = 0.08 / 0.05 * 0.025,
        instance = [0, ],
        root_num = [None, ],
        material = ['fluid', ],
        outf = os.path.join('data', 'test_LiquidFun')
    ),
}