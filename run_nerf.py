import os, sys
from pickle import encode_long
import numpy as np
import imageio
import json
import random
import time
from numpy.core.fromnumeric import shape
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data, poses_avg
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data, load_blender_r2n2_plane
from load_LINEMOD import load_LINEMOD_data

from datasets.datasets import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False
device_ids = []

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs, shape_codes):
        return torch.cat([fn(inputs[i:i+chunk], shape_codes[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, shape_codes, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
   
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    # todo : add shape_code
    outputs_flat = batchify(fn, netchunk)(embedded, shape_codes)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, shape_codes, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], shape_codes[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, shape_codes,chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., 
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model. # todo : ??
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w) # img_sizeximg_sizex3; img_sizeximg_sizex3
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs: # # set in lego.txt
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    if c2w is not None:
        code_dim = shape_codes.shape[1]
        shape_codes = shape_codes.repeat(1, rays_o.shape[0]).reshape(-1, code_dim)

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, shape_codes, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, shape_code, chunk, render_kwargs, gt_imgs=None, gt_deps=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    depths = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        if i >= shape_code.shape[0]:
            rgb, disp, acc, depth, _ = render(H, W, K, shape_code, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        else:
            rgb, disp, acc, depth, _ = render(H, W, K, shape_code[i:i+1, :], chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        # todo : concat g.t. rgb to the predicted rgb
        if gt_imgs is not None:
            output_vs_gt = torch.cat((rgb, gt_imgs[i].to(device)), dim=1)
            rgbs.append(output_vs_gt.cpu().numpy())
        else:
            rgbs.append(rgb.cpu().numpy())
        
        if gt_deps is not None: # gt_depth's dim: Bx1xHxW
            output_dep_vs_gt = torch.cat((depth, gt_deps[i][0].to(device)), dim=1)
            depths.append(output_dep_vs_gt.cpu().numpy())
        else:
            depths.append(depth.cpu().numpy())
            
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape, depth.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1]) 
            filename = os.path.join(savedir, 'rgb_{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            print("max depth:", np.nanmax(depths[-1]))
            plt.imsave(os.path.join(savedir, 'depth_{:03d}.png'.format(i)), depths[-1])
            # todo : save depth img
            # dose it make sense? // how to use it
            # np.savez(os.path.join(savedir, '{:03d}.npz'.format(i)), rgb=rgb.cpu().numpy(), disp=disp.cpu().numpy(), acc=acc.cpu().numpy(), depth=depth)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    depths = np.stack(depths, 0)
    return rgbs, disps, depths


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    # output_ch = 5 if args.N_importance > 0 else 4
    output_ch = 4 # https://github.com/yenchenlin/nerf-pytorch/issues/22
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth, input_code_ch=args.code_dim, 
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    if len(device_ids)>1:
        model=torch.nn.DataParallel(model, device_ids=device_ids)#前提是model已经.cuda() 了

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine, input_code_ch=args.code_dim,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())
        if len(device_ids)>1:
            model_fine=torch.nn.DataParallel(model_fine, device_ids=device_ids)#前提是model已经.cuda() 了
    # # remove shape_code change
    # encoder_shape = None
    encoder_shape = ResnetEnc(z_dim=args.code_dim, c_dim=3, img_hw=args.img_size)
    grad_vars += list(encoder_shape.parameters())
    encoder_appearence = ResnetEnc(z_dim=args.code_dim, c_dim=3, img_hw=args.img_size)
    grad_vars += list(encoder_appearence.parameters())

    if len(device_ids)>1:
        encoder_shape=torch.nn.DataParallel(encoder_shape, device_ids=device_ids)#前提是model已经.cuda() 了

    network_query_fn = lambda inputs, shape_codes, viewdirs, network_fn : run_network(inputs, shape_codes, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        encoder_shape.load_state_dict(ckpt['encoder_shape_state_dict'])
        encoder_appearence.load_state_dict(ckpt['encoder_appearence_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        # 'encoder_shape': encoder_shape,
    }

    # NDC only good for LLFF-style forward facing data
    # todo: NDC ??
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, encoder_shape, encoder_appearence


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                shape_codes,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(ray_batch.device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    # todo : WHAT HERE???
    code_dim = shape_codes.shape[-1]
    input_shape_codes = shape_codes.repeat(1, N_samples).reshape(-1, code_dim)
    raw = network_query_fn(pts, input_shape_codes, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0 = rgb_map, disp_map, acc_map, depth_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, shape_codes.repeat(1, N_importance + N_samples).reshape(-1, code_dim), viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map': depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['dep0'] = depth_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='configs/planes.txt',  # change
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--code_dim", type=int, default=512, # change
                        help='num of total epoches')
    parser.add_argument("--img_size", type=int, default=128, # change
                        help='num of total epoches')
    parser.add_argument("--batch_size", type=int, default=2, # change
                        help='num of total epoches')
    parser.add_argument("--nepoch", type=int, default=401,  # change
                        help='num of total epoches')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_false', # change
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", type=bool, default=False,  
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", type=bool, default=False, 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    parser.add_argument("--depth_loss_lambda", type=float, default=0.0, # change DSNERF: 0.1!!
                        help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_trainset", type=int, default=2000, # change
                        help='frequency of testset saving')
    parser.add_argument("--i_testset", type=int, default=2000, # change
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=2000, # change
                        help='frequency of render_poses video saving')

    return parser


def train():
    global device_ids
    parser = config_parser()
    args = parser.parse_args()

    args.gpu_id="2,3,4" ; #指定gpu id
    args.cuda = torch.cuda.is_available() #作为是否使用cpu的判定
    #配置环境  也可以在运行时临时指定 CUDA_VISIBLE_DEVICES='2,7' Python train.py
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id #这里的赋值必须是字符串，list会报错
    # device_ids=range(torch.cuda.device_count())  #torch.cuda.device_count()=2
    #device_ids=[0,1] 这里的0 就是上述指定 2，是主gpu,  1就是7,模型和数据由主gpu分发
    device_ids = []
    

    # Load data
    K = None
    # if args.dataset_type == 'llff':
    #     images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
    #                                                               recenter=True, bd_factor=.75,
    #                                                               spherify=args.spherify)
    #     hwf = poses[0,:3,-1]
    #     poses = poses[:,:3,:4]
    #     print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
    #     if not isinstance(i_test, list):
    #         i_test = [i_test]

    #     if args.llffhold > 0:
    #         print('Auto LLFF holdout,', args.llffhold)
    #         i_test = np.arange(images.shape[0])[::args.llffhold]

    #     i_val = i_test
    #     i_train = np.array([i for i in np.arange(int(images.shape[0])) if
    #                     (i not in i_test and i not in i_val)])

    #     print('DEFINING BOUNDS')
    #     if args.no_ndc:
    #         near = np.ndarray.min(bds) * .9
    #         far = np.ndarray.max(bds) * 1.
            
    #     else:
    #         near = 0.
    #         far = 1.
    #     print('NEAR FAR', near, far) 
    # elif args.dataset_type == 'blender':
    #     # # render_poses: novel-view poses
    #     # change
    #     # images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
    #     images, poses, render_poses, hwf, i_split = load_blender_r2n2_plane(args.datadir)
    #     print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    #     i_train, i_val, i_test = i_split

    #     # near = 2.
    #     # far = 6.
    #     near, far = 0.8 , 2.

    #     if args.white_bkgd:
    #         images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    #     else:
    #         images = images[...,:3]

    # elif args.dataset_type == 'LINEMOD':
    #     images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
    #     print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
    #     print(f'[CHECK HERE] near: {near}, far: {far}.')
    #     i_train, i_val, i_test = i_split

    #     if args.white_bkgd:
    #         images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    #     else:
    #         images = images[...,:3]

    # elif args.dataset_type == 'deepvoxels':

    #     images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
    #                                                              basedir=args.datadir,
    #                                                              testskip=args.testskip)

    #     print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
    #     i_train, i_val, i_test = i_split

    #     hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
    #     near = hemi_R-1.
    #     far = hemi_R+1.

    # else:
    #     print('Unknown dataset type', args.dataset_type, 'exiting')
    #     return
    
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    from utils.logger import Logger
    from datetime import datetime
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    logger = Logger(
        log_dir=os.path.join(basedir, expname),
        img_dir=os.path.join(basedir, expname, 'imgs'),
        monitoring='tensorboard',
        monitoring_dir=os.path.join(basedir, expname, 'events', TIMESTAMP),
        rank=0, is_master=True, multi_process_logging=(False))
    logger.load_stats('stats.p')    # this will be used for plotting

    os.makedirs(os.path.join(basedir, expname, 'events', TIMESTAMP), exist_ok=True)
    f = os.path.join(basedir, expname, 'events', TIMESTAMP, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'events', TIMESTAMP, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, encoder_shape , encoder_appearence= create_nerf(args)
    global_step = start


    # todo : why have to add this line
    torch.multiprocessing.set_start_method('spawn') # https://blog.csdn.net/qazwsxrx/article/details/116806358

    train_data = get_dataset(name='Plane',
                            img_path= '/data2/ShapeNetPlanes/train/rgb/*.png',
                            mask_path= '/data2/ShapeNetPlanes/train/mask/*.png',
                            depth_path= '/data2/ShapeNetPlanes/train/depth/*.npy',
                            camera_path = '/data2/ShapeNetPlanes/train/rendering_metadata.txt', 
                            img_size=args.img_size,
                            batch_size=args.batch_size)

    # # change 
    test_data = get_dataset(name='Plane',
                            img_path= '/data2/ShapeNetPlanes/test/rgb/*.png',
                            mask_path= '/data2/ShapeNetPlanes/test/mask/*.png',
                            depth_path= '/data2/ShapeNetPlanes/test/depth/*.npy',
                            camera_path = '/data2/ShapeNetPlanes/test/rendering_metadata.txt', 
                            img_size=args.img_size,
                            batch_size=4)

    pbar = tqdm(total=args.nepoch * len(train_data))
    test_data_iter = iter(test_data)
    for epoch in range(args.nepoch):
        for num_iter, (rgbs, masks, depths, poses, hwf) in enumerate(train_data):
            # if args.white_bkgd:
                
            # else :
            #     print('black bg')
            rgbs, depths, poses = rgbs.to(device), depths.to(device), poses.to(device)
            pbar.update(1)
            i = epoch * len(train_data) + num_iter

            near = 0.7
            far = 1.7
            # Cast intrinsics to right types
            H, W, focal = hwf[0][0], hwf[1][0], hwf[2][0]
            H, W, focal = int(H), int(W), float(focal)
            hwf = [H, W, focal]

            if K is None:
                K = np.array([
                    [focal, 0, 0.5*W],
                    [0, focal, 0.5*H],
                    [0, 0, 1]
                ])

            # if args.render_test:
            #     render_poses = np.array(poses[i_test])

            bds_dict = {
                'near' : near,
                'far' : far,
            }
            render_kwargs_train.update(bds_dict)
            render_kwargs_test.update(bds_dict)

            # Move testing data to GPU (novel views: 40x4x4)
            # render_poses = torch.Tensor(render_poses).to(device)

            # Short circuit if only rendering out from trained model
            # # these to arguments behaves abnormally when dubugging
            args.render_only = False
            args.render_test = False
            if args.render_only:
                print('RENDER ONLY')
                with torch.no_grad():
                    if args.render_test:
                        # render_test switches to test poses
                        # images = images[i_test]
                        # to change to test_Data
                        test_rgb, test_mask, test_depth, test_pose, test_hwf = next(iter(test_data))
                    else:
                        # Default is smoother render_poses path
                        test_pose = torch.stack([pose_spherical(angle, 0.8, 1.2) for angle in np.linspace(0,2 * np.pi,50+1)[:-1]], 0)
                    
                    render_poses = test_pose.to(device)
                    testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
                    os.makedirs(testsavedir, exist_ok=True)
                    print('test poses shape', render_poses.shape)

                    shape_code = torch.zeros(4, 4)
                    test_depth_cpu = test_depth.squeeze().cpu().numpy()
                    # test_depth_cpu /= np.max(test_depth_cpu)
                    # np.savetxt('test_depth_cpu.txt', test_depth_cpu[0][60:69,60:69])
                    rgbs, disps, depths = render_path(render_poses, hwf, K, shape_code, args.chunk, render_kwargs_test, gt_imgs=None, savedir=testsavedir, render_factor=args.render_factor)
                    # depths /= np.max(depths)
                    # np.savetxt('depths.txt', depths[0][60:69,60:69])
                    print('Done rendering', testsavedir)
                    # delta = test_depth_cpu - depths
                    if args.render_test == False:
                        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=20, quality=8)

                    return

            # Prepare raybatch tensor if batching random rays
            N_rand = args.N_rand
            # use_batching = not args.no_batching
            # if use_batching:
            #     # For random ray batching
            #     # print('get rays')
            #     lsts = [get_rays(H, W, K, p) for p in poses[:,:3,:4]]
            #     rays = torch.stack(lsts, 0) # [N, ro+rd, H, W, 3]
            #     # print('done, concats')
            #     rays_rgb = torch.cat([rays, rgbs[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
            #     rays_rgb = rays_rgb.permute(0,2,3,1,4) # [N, H, W, ro+rd+rgb, 3]
            #     # rays_rgb = torch.stack([rays_rgb[i] for i in i_train], 0) # train images only
            #     rays_rgb = torch.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
            #     rays_rgb = rays_rgb.type(torch.float32)
            #     # rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
            #     # rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
            #     # rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
            #     # rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
            #     # rays_rgb = rays_rgb.astype(np.float32)
            #     # print('shuffle rays')
            #     # np.random.shuffle(rays_rgb)
            #     select_inds = np.random.choice(rays_rgb.shape[0], size=[N_rand * args.batch_size], replace=False)  # (N_rand,)
                
            #     # shuffle = torch.rand(rays_rgb.shape[0]).le(1. / 260.) # todo : shuffle depth && latent code
            #     rays_rgb = rays_rgb[select_inds]

            #     # print('done')
            #     i_batch = 0

            # # Move training data to GPU
            # if use_batching:
            #     rgbs = rgbs.to(device)
            # poses = poses.to(device)
            # if use_batching:
            #     rays_rgb = rays_rgb.to(device)


            # start = start + 1

            time0 = time.time()

            batch_rays_lst, target_s_lst, target_depth_lst = [], [], []
            new_gt_rgb = rgbs.clone().detach()
            new_gt_rgb = new_gt_rgb.permute(0, 3, 1, 2)
            shape_code = encoder_shape(new_gt_rgb) # Bx128
            appearence_code = encoder_appearence(new_gt_rgb)

            for batch in range(args.batch_size):
                c2w, gt_rgb, gt_dep = poses[batch], rgbs[batch], depths[batch]
                batch_rays, target_s, target_d = get_rays_sample(H, W, K, c2w, gt_rgb=gt_rgb, gt_depth=gt_dep[0],i= i,
                                                            precrop_iters=args.precrop_iters, precrop_frac=args.precrop_frac,
                                                            N_rand=args.N_rand)                
                batch_rays_lst.append(batch_rays)
                target_s_lst.append(target_s)
                target_depth_lst.append(target_d)
            
            batch_rays = torch.cat(batch_rays_lst, dim=1)
            target_s = torch.cat(target_s_lst, dim=0) # (Bx1024) x 3
            target_d = torch.cat(target_depth_lst, dim=0)
            # print(shape_code)
            # change i.e. remove this line when code_dim=0
            latentcode = torch.cat((shape_code, appearence_code), dim=1)
            latentcodes = latentcode.repeat(1, args.N_rand).reshape(-1, args.code_dim + args.code_dim)
            # shape_codes = torch.zeros(args.batch_size, 4)

            #####  Core optimization loop  #####

            rgb, disp, acc, depth, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                     retraw=True, shape_codes=latentcodes,
                                                    **render_kwargs_train)

            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][...,-1]
            losses_dct = {'img_loss': img_loss}
            loss = img_loss
            psnr = mse2psnr(img_loss)
            # todo : rgb0:pixel / rgb ?
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                losses_dct['img_loss0'] = img_loss0
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)
            
            depth_loss = 0
            if args.depth_loss_lambda > 0:
                # depth_loss = torch.mean((((depth_col - target_depth) / max_depth) ** 2) * ray_weights)
                depth_loss = img2mse(depth, target_d)
                # depth_loss = torch.mean(((depth - target_d) / target_d)**2)
                loss += args.depth_loss_lambda * depth_loss
                losses_dct['depth_loss'] = depth_loss
                # DSNeRF did't add this term of loss
                # if 'dep0' in extras:
                #     depth_loss0 = img2mse(extras['dep0'], target_d)
                #     loss += args.depth_loss_lambda * depth_loss0
                #     losses_dct['depth_loss0'] = depth_loss0

            loss.backward()
            optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

            dt = time.time()-time0
            # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
            #####           end            #####
            

            # Rest is logging
            if i%args.i_weights==0 and i > 0 :
                path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'encoder_shape_state_dict': encoder_shape.state_dict(),
                    'encoder_appearence_state_dict': encoder_appearence.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
                logger.save_stats('stats.p')
                print("Saved logger")
            
            encoder_shape.eval()
            encoder_appearence.eval()
            render_kwargs_test['network_fn'].eval()
            render_kwargs_test['network_fine'].eval()
            if i%args.i_trainset==0 and i > 0 or i == 5: # change
                trainsavedir = os.path.join(basedir, expname, 'trainset_{:06d}'.format(i))
                os.makedirs(trainsavedir, exist_ok=True)
                print('train poses shape', poses[:4].shape) # 最多render 4张看
                with torch.no_grad():
                    # shape_code = torch.zeros(4, 4)
                    shape_code = encoder_shape(rgbs[:4].permute(0, 3, 1, 2).cuda())
                    appearence_code = encoder_appearence(rgbs[:4].permute(0, 3, 1, 2).cuda())
                    latentcode = torch.cat((shape_code, appearence_code), dim=1)
                    rgbs, disps, depths = render_path(poses[:4].to(device), [H, W, focal], K, latentcode, args.chunk, render_kwargs_test, gt_imgs=rgbs[:4], gt_deps=depths[:4], savedir=trainsavedir)
                print('Saved train set')
                tensor_rgbs = [torch.from_numpy(rgb) for rgb in rgbs[:4]]
                tensor_depths = [torch.from_numpy(depth) for depth in depths[:4]]
                logger.add_imgs(torch.stack(tensor_rgbs, dim=0).permute(0, 3, 1, 2), 'train/rgb', i)
                logger.add_imgs(torch.stack(tensor_depths, dim=0).unsqueeze(1), 'train/depth', i)


            if i%args.i_testset==0 and i > 0 or i == 3: # change
                try:
                    rgbs, masks, depths, poses, _ = next(test_data_iter)
                except StopIteration:
                    test_data_iter = iter(test_data)
                    rgbs, masks, depths, poses, _ = next(test_data_iter)

                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', poses.shape)
                with torch.no_grad():
                    # shape_code = torch.zeros(4, 4)
                    shape_code = encoder_shape(rgbs.permute(0, 3, 1, 2).cuda())
                    appearence_code = encoder_appearence(rgbs[:4].permute(0, 3, 1, 2).cuda())
                    latentcode = torch.cat((shape_code, appearence_code), dim=1)
                    rgbs, disps, depths = render_path(poses.to(device), hwf, K, latentcode, args.chunk, render_kwargs_test, gt_imgs=rgbs, gt_deps=depths, savedir=testsavedir)
                print('Saved test set')
                tensor_rgbs = [torch.from_numpy(rgb) for rgb in rgbs[:4]]
                tensor_depths = [torch.from_numpy(depth) for depth in depths[:4]]
                logger.add_imgs(torch.stack(tensor_rgbs, dim=0).permute(0, 3, 1, 2), 'test/rgb', i)
                logger.add_imgs(torch.stack(tensor_depths, dim=0).unsqueeze(1), 'test/depth', i)


            if i%args.i_video==0 and i > 0 or i == 10:
                
                try:
                    rgbs, masks, depths, poses, _ = next(test_data_iter)
                except StopIteration:
                    test_data_iter = iter(test_data)
                    rgbs, masks, depths, poses, _ = next(test_data_iter)
                
                rgb8_ = to8b((rgbs[:1,...]).numpy())[0]
                filename = os.path.join(os.path.join(basedir, expname, '{:06d}_rgb.png'.format(i)))
                imageio.imwrite(filename, rgb8_)

                render_poses = torch.stack([pose_spherical(angle, 0.8, 1.2) for angle in np.linspace(0,2 * np.pi,60+1)[:-1]], 0)
                render_poses = render_poses.to(device)
                # Turn on testing mode
                with torch.no_grad():
                    # shape_code = torch.zeros(4, 4)
                    shape_code = encoder_shape(rgbs[:1,...].permute(0,3, 1,2).cuda())
                    appearence_code = encoder_appearence(rgbs[:1,...].permute(0, 3, 1, 2).cuda())
                    latentcode = torch.cat((shape_code, appearence_code), dim=1)
                    rgbs, disps, depths = render_path(render_poses, hwf, K, latentcode, args.chunk, render_kwargs_test)
                print('Done, saving', rgbs.shape, disps.shape, depths.shape)
                moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
                imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=20, quality=8)
                # todo  save depth video .mp4
                # imageio.mimwrite(moviebase + 'depth.mp4', to8b(depths), fps=20, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=20, quality=8)

                # if args.use_viewdirs:
                #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
                #     with torch.no_grad():
                #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
                #     render_kwargs_test['c2w_staticcam'] = None
                #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

            encoder_shape.train()
            encoder_appearence.train()
            render_kwargs_test['network_fn'].train()
            render_kwargs_test['network_fine'].train()
            
            # log learning rate
            logger.add('learning rates', 'whole', optimizer.param_groups[0]['lr'], i)
            # log losses
            for k, v in losses_dct.items():
                logger.add('losses', k, v.item(), i)

            logger.add('losses', 'total_loss', loss.item(), i)

            logger.add('metirc', 'psnr', psnr.item(), i)

            if i%args.i_print==0:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
                
            """
                print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
                print('iter time {:.05f}'.format(dt))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                    tf.contrib.summary.scalar('loss', loss)
                    tf.contrib.summary.scalar('psnr', psnr)
                    tf.contrib.summary.histogram('tran', trans)
                    if args.N_importance > 0:
                        tf.contrib.summary.scalar('psnr0', psnr0)


                if i%args.i_img==0:

                    # Log a rendered validation view to Tensorboard
                    img_i=np.random.choice(i_val)
                    target = images[img_i]
                    pose = poses[img_i, :3,:4]
                    with torch.no_grad():
                        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                            **render_kwargs_test)

                    psnr = mse2psnr(img2mse(rgb, target))

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                        tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                        tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                        tf.contrib.summary.scalar('psnr_holdout', psnr)
                        tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                    if args.N_importance > 0:

                        with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                            tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                            tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                            tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
            """

            global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
