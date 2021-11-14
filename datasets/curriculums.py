"""
To easily reproduce experiments, and avoid passing several command line arguments, we implemented
a curriculum utility. Parameters can be set in a curriculum dictionary.

Curriculum Schema:

    Numerical keys in the curriculum specify an upsample step. When the current step matches the upsample step,
    the values in the corresponding dict be updated in the curriculum. Common curriculum values specified at upsamples:
        batch_size: Batch Size.
        num_steps: Number of samples along ray.
        img_size: Generated image resolution.
        batch_split: Integer number over which to divide batches and aggregate sequentially. (Used due to memory constraints)
        gen_lr: Generator learnig rate.
        disc_lr: Discriminator learning rate.

    fov: Camera field of view
    ray_start: Near clipping for camera rays.
    ray_end: Far clipping for camera rays.
    fade_steps: Number of steps to fade in new layer on discriminator after upsample.
    h_stddev: Stddev of camera yaw in radians.
    v_stddev: Stddev of camera pitch in radians.
    h_mean:  Mean of camera yaw in radians.
    v_mean: Mean of camera yaw in radians.
    sample_dist: Type of camera pose distribution. (gaussian | spherical_uniform | uniform)
    topk_interval: Interval over which to fade the top k ratio.
    topk_v: Minimum fraction of a batch to keep during top k training.
    betas: Beta parameters for Adam.
    unique_lr: Whether to use reduced LRs for mapping network.
    weight_decay: Weight decay parameter.
    r1_lambda: R1 regularization parameter.
    latent_dim: Latent dim for Siren network  in generator.
    grad_clip: Grad clipping parameter.
    model: Siren architecture used in generator. (SPATIALSIRENBASELINE | TALLSIREN)
    generator: Generator class. (ImplicitGenerator3d)
    discriminator: Discriminator class. (ProgressiveEncoderDiscriminator | ProgressiveDiscriminator)
    dataset: Training dataset. (CelebA | Carla | Cats)
    clamp_mode: Clamping function for Siren density output. (relu | softplus)
    z_dist: Latent vector distributiion. (gaussian | uniform)
    hierarchical_sample: Flag to enable hierarchical_sampling from NeRF algorithm. (Doubles the number of sampled points)
    z_labmda: Weight for experimental latent code positional consistency loss.
    pos_lambda: Weight parameter for experimental positional consistency loss.
    last_back: Flag to fill in background color with last sampled color on ray.
"""

Plane = {
    'progress': {0: {'batch_size': 8,
                     'num_steps': 40,
                     'sampled_pixels': 1024,
                     'img_size': 128,
                     'gen_lr': 2e-5},
                 20: {'batch_size': 8,
                      'num_steps': 40,
                      'sampled_pixels': 1024,
                      'img_size': 128,
                      'gen_lr': 0.5e-5},
                 60: {'batch_size': 2,
                      'num_steps': 40,
                      'sampled_pixels': 4096,
                      'img_size': 256,
                      'gen_lr': 0.25e-5}},

    'dataset': 'Plane',
    'img_path': '/data2/ShapeNetPlanes/train/rgb/*.png',
    'mask_path': '/data2/ShapeNetPlanes/train/mask/*.png',
    'depth_path': '/data2/ShapeNetPlanes/train/depth/*.npy',
    'fov': 39.5340878640268,
    'ray_start': 0.2,
    'ray_end': 2.0,
    'betas': (0, 0.9),
    'weight_decay': 0,
    'gamma': 1,
    'latent_dim': 512,
    'grad_clip': 1,
    'clamp_mode': 'relu',
    'hierarchical_sample': True,
}
