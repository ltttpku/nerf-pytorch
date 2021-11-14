"""Datasets"""
from imageio.core.functions import imwrite
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import glob
import PIL
import numpy as np
import pandas as pd
import os
import math

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
    az, el, rho = theta, phi, radius
    trans = rho * np.array([np.cos(el) * np.cos(az), np.sin(el), -np.cos(el) * np.sin(az)])
    z_cam = trans / np.linalg.norm(trans)
    y = np.array([0., 1., 0.])
    x_cam = np.cross(y, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    y_cam /= np.linalg.norm(y_cam)

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.stack((x_cam, y_cam, z_cam)).T
    camera_pose[:3, -1] = trans

    c2w = torch.from_numpy(camera_pose)
    # c2w = trans_t(radius)
    # c2w = rot_phi(phi/180.*np.pi) @ c2w
    # c2w = rot_theta(theta/180.*np.pi) @ c2w
    # c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


class PlaneTEST(Dataset):
    def __init__(self, img_path, mask_path, depth_path, camera_path, img_size):
        super().__init__()
        self.imgs = glob.glob(img_path)
        self.imgs.sort()
        self.masks = glob.glob(mask_path)
        self.masks.sort()
        self.depths = glob.glob(depth_path)
        self.depths.sort()
        
        assert len(self.imgs) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((img_size, img_size))
        self.img_size = img_size

        self.camera_list = []
        if os.path.isfile(camera_path):
            with open(camera_path, 'r') as f:
                camera = f.read().split('\n')[:-1]
                self.camera_list.extend(camera) 
        else:
            print("can't find rendering_metadata.txt")
            exit(1)

        self.imgs , self.masks, self.depths, self.camera_list = self.imgs[40:50], self.masks[40:50], self.depths[40:50], self.camera_list[40:50] # change
        self.focal = 1113.16 # # size=800
        self.focal = self.focal * (float(self.img_size) / 800.) 


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = PIL.Image.open(self.imgs[index]).convert('RGB')
        img = self.resize(self.to_tensor(img))
        mask = PIL.Image.open(self.masks[index])
        mask = self.resize(self.to_tensor(mask))
        depth = torch.from_numpy(np.load(self.depths[index])[np.newaxis, ...])
        depth = self.resize(depth)

        camera = self.camera_list[index].split(' ')
        az = float(camera[0]) 
        el = float(camera[1]) 
        rho = float(camera[3])

        c2w = pose_spherical(az, el, rho)
        # print(c2w)
        rgb = img.permute(1, 2, 0)
        mask = mask.permute(1, 2, 0)
        rgb = rgb*mask + (1.-mask) # white bg b default
        return rgb, mask, depth, c2w, [self.img_size, self.img_size, self.focal]



class Plane(Dataset):
    def __init__(self, img_path, mask_path, depth_path, camera_path, img_size):
        super().__init__()
        self.imgs = glob.glob(img_path)
        self.imgs.sort()
        self.masks = glob.glob(mask_path)
        self.masks.sort()
        self.depths = glob.glob(depth_path)
        self.depths.sort()
        
        assert len(self.imgs) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((img_size, img_size))
        self.img_size = img_size

        self.camera_list = []
        if os.path.isfile(camera_path):
            with open(camera_path, 'r') as f:
                camera = f.read().split('\n')[:-1]
                self.camera_list.extend(camera) 
        else:
            print("can't find rendering_metadata.txt")
            exit(1)

        self.imgs , self.masks, self.depths, self.camera_list = self.imgs[:40], self.masks[:40], self.depths[:40], self.camera_list[:40] # change
        self.focal = 1113.16 # # size=800
        self.focal = self.focal * (float(self.img_size) / 800.) 


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = PIL.Image.open(self.imgs[index]).convert('RGB')
        img = self.resize(self.to_tensor(img))
        mask = PIL.Image.open(self.masks[index])
        mask = self.resize(self.to_tensor(mask))
        depth = torch.from_numpy(np.load(self.depths[index])[np.newaxis, ...])
        depth = self.resize(depth)

        camera = self.camera_list[index].split(' ')
        az = float(camera[0]) 
        el = float(camera[1]) 
        rho = float(camera[3])

        c2w = pose_spherical(az, el, rho)
        # print(c2w)
        rgb = img.permute(1, 2, 0)
        mask = mask.permute(1, 2, 0)
        rgb = rgb*mask + (1.-mask) # white bg b default
        return rgb, mask, depth, c2w, [self.img_size, self.img_size, self.focal]


class Cars(Dataset):
    def __init__(self, img_path, img_size, use_camera_gt=False, camera_gt_path=None):
        super().__init__()
        self.imgs = glob.glob(img_path)
        self.imgs.sort()
        self.use_camera_gt = use_camera_gt
        if use_camera_gt:
            df = pd.read_csv(camera_gt_path, sep=' ', header=None)
            camera_gt = df.to_numpy()
            camera_gt = torch.from_numpy(camera_gt)
            self.az, self.el, rho = camera_gt[:, :1] * np.pi / 180, camera_gt[:, 1:2] * np.pi / 180, camera_gt[:, 3:4]
            self.trans = torch.cat([-rho * torch.cos(self.el) * torch.cos(self.az),
                                    -rho * torch.cos(self.el) * torch.sin(self.az),
                                    -rho * torch.sin(self.el)], dim=1)
        assert len(self.imgs) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor()])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = PIL.Image.open(self.imgs[index]).convert('RGB')
        img = self.transform(img)
        az, el, trans = None, None, None
        if self.use_camera_gt:
            az = self.az[index]
            el = self.el[index]
            trans = self.trans[index]
        return img, az, el, trans


class Lego(Dataset):
    def __init__(self, img_path, mask_path, img_size, use_camera_gt=False, camera_gt_path=None):
        super().__init__()
        self.imgs = glob.glob(img_path)
        self.imgs.sort()
        self.masks = glob.glob(mask_path)
        self.masks.sort()
        self.use_camera_gt = use_camera_gt
        if use_camera_gt:
            df = pd.read_csv(camera_gt_path, sep=' ', header=None)
            camera_gt = df.to_numpy()
            camera_gt = torch.from_numpy(camera_gt)
            self.az, self.el, rho = camera_gt[:, :1] * np.pi / 180, camera_gt[:, 1:2] * np.pi / 180, camera_gt[:, 3:4]
            self.trans = torch.cat([-rho * torch.cos(self.el) * torch.cos(self.az),
                                    -rho * torch.cos(self.el) * torch.sin(self.az),
                                    -rho * torch.sin(self.el)], dim=1)
        assert len(self.imgs) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor()])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = PIL.Image.open(self.imgs[index]).convert('RGB')
        img = self.transform(img)
        mask = PIL.Image.open(self.masks[index])
        mask = self.transform(mask)
        #img *= mask
        az, el, trans = None, None, None
        if self.use_camera_gt:
            az = self.az[index]
            el = self.el[index]
            trans = self.trans[index]
        return img, mask, az, el, trans


def get_dataset(name, img_path, mask_path, depth_path, camera_path, img_size, batch_size=1, shuffle=True):
    dataset = globals()[name](img_path, mask_path, depth_path, camera_path, img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=False,
        num_workers=1
    )
    return dataloader

if __name__ == '__main__':
    train_data = get_dataset(name='Plane',
                            img_path= '/data2/ShapeNetPlanes/train/rgb/*.png',
                            mask_path= '/data2/ShapeNetPlanes/train/mask/*.png',
                            depth_path= '/data2/ShapeNetPlanes/train/depth/*.npy',
                            camera_path = '/data2/ShapeNetPlanes/train/rendering_metadata.txt', 
                            img_size=400,
                            batch_size=8)
    print(len(train_data))

    for i, (rgb, mask, depth, pose, hwf) in enumerate(train_data):
        print(rgb.shape)
        print(rgb[0][1][200])
        d = depth[0].numpy()
        from matplotlib import pyplot as plt
        # plt.imshow(d[0])
        plt.imsave('1.png', d[0])

        depth = plt.imread('1.png')
        print(depth.shape)
        # imageio.imwrite('1.png', np.uint8(d[0] * 100))