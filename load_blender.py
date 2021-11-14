import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

from datasets.shapenetr2r2 import getBlenderProj

blender_T = np.array([
    [0, 1., 0],
    [-1, 0, 0],
    [0, 0, 1.],
])

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


def load_blender_r2n2_plane(basedir='/data2/ShapeNetRendering'):
    path = os.path.join(basedir, '02691156')
    instance = sorted(os.listdir(path))[0]
    print('instance id:',instance)
    image_list = []
    camera_list = []

    sub_flist = os.path.join(path, instance, 'rendering')
    camera_path = os.path.join(sub_flist, 'rendering_metadata.txt')
    if os.path.isfile(camera_path):
        with open(camera_path, 'r') as f:
            camera = f.read().split('\n')[:-1]
            camera_list.extend(camera) 
    else:
        print("can't find rendering_metadata.txt")
        exit(1)

    for i in range(24):
        image_list.append(os.path.join(sub_flist, '%02d.png' % (i)))
    
    assert len(image_list) == len(camera_list) 

    focal = 149.84
    imgs, poses = [], []
    for i, instance_path in enumerate(image_list):
        imgs.append(imageio.imread(instance_path))
        camera = camera_list[i].split(' ')
        az = float(camera[0])
        el = float(camera[1])
        dist = float(camera[3])
        c2w = pose_spherical(az - 180., el, dist * 1.75)
        # K, R, T = getBlenderProj(az, el, dist)
        # R = R @ blender_T
        # K, R, T = np.array(K), np.array(R), np.array(T)
        # pose = np.eye(4) # # should be c2w
        # pose[:3, :3] = R.transpose(1, 0)
        # pose[:3, 3] = -T.reshape(-1)
        pose = c2w.cpu().numpy()
        poses.append(pose)


    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)

    H, W = imgs[0].shape[:2]

    render_poses = torch.stack([pose_spherical(angle, 27.0, 1.6) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    counts = [0, 20, 20, 24] # train , val, test
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    

    return imgs, poses, render_poses, [H, W, focal], i_split

if __name__ == '__main__':
    load_blender_r2n2_plane()
