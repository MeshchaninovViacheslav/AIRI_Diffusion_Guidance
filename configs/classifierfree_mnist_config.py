import ml_collections


def create_default_mnist_config():
    config = ml_collections.ConfigDict()

    # data
    data = config.data = ml_collections.ConfigDict()
    data.image_size = 32
    data.num_channels = 1
    data.centered = True
    data.batch_size = 1024
    data.norm_mean = (0.5)
    data.norm_std = (0.5)

    # model
    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.nf = 32
    model.ch_mult = (1, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.dropout = 0.1
    model.resamp_with_conv = True
    model.conditional = True
    model.nonlinearity = 'swish'
    model.num_classes = 10
    model.class_embed_size = 16

    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.0
    optim.linear_warmup = 1000
    optim.lr = 2e-4
    optim.min_lr = 2e-4
    optim.warmup_lr = 0
    optim.weight_decay = 0
    optim.betas = (0.9, 0.999)
    optim.eps = 1e-8

    # training
    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 50_000
    training.checkpoint_freq = 5_000
    training.eval_freq = 2500
    training.snapshot_freq = 1000
    training.snapshot_batch_size = 10
    training.batch_size = 256
    training.ode_sampling = False
    training.logging_freq = 10
    training.p_uncond = 0.1
    training.exp_name = 'ddpm_classifierfree'

    training.checkpoints_folder = '../ddpm_checkpoints/'

    # sde
    sde = config.sde = ml_collections.ConfigDict()
    sde.N = 1000
    sde.beta_min = 0.1
    sde.beta_max = 20

    # 2 assignment - train noisy classifier
    classifier = config.classifier = ml_collections.ConfigDict()
    classifier.type = 'free'
    classifier.training_iters = 20_000
    classifier.eval_freq = 5_000
    classifier.snapshot_freq = 5_000
    classifier.checkpoint_freq = 5_000
    classifier.checkpoint_path = '../ddpm_checkpoints/classifier.pth'
    
    config.checkpoints_prefix = 'ddpm_classifierfree'
    config.predict = 'noise'
    config.device = 'cuda'
    config.chkp_name = 'ddpm_classifierfree-10000.pth'
    return config
