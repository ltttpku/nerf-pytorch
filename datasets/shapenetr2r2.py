import os, sys
from posixpath import dirname
import cv2
import pickle
from numpy.core.fromnumeric import sort
from numpy.lib.twodim_base import mask_indices
from pyparsing import lineno
import torch
from torch.functional import split
import torch.utils.data
import json
import glob
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('..')
# sys.path.append('datasets/')
# print(sys.path)
from utils import rend_util


# from synthesis import *
# from common.geometry import *
# from common.utils.io_utils import read_exr
import numpy as np

F_MM = 35.  # Focal length
SENSOR_SIZE_MM = 32.
PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
RESOLUTION_PCT = 100
SKEW = 0.
CAM_MAX_DIST = 1.75
IMG_W = 127 + 10  # Rendering image size. Network input size + cropping margin.
IMG_H = 127 + 10

CAM_ROT = np.matrix(((1.910685676922942e-15, 4.371138828673793e-08, 1.0),
                     (1.0, -4.371138828673793e-08, -0.0),
                     (4.371138828673793e-08, 1.0, -4.371138828673793e-08)))

blender_T = np.array([
    [1, 0., 0],
    [0, 0, -1],
    [0, 1, 0.],
])


def getBlenderProj(az, el, distance_ratio, img_w=IMG_W, img_h=IMG_H):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""

    # Calculate intrinsic matrix.
    scale = RESOLUTION_PCT / 100
    f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
    f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    u_0 = img_w * scale / 2
    v_0 = img_h * scale / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
                                          (sa * ce, ca, sa * se),
                                          (-se, 0, ce))))

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    R_world2cam = R_obj2cam * R_world2obj
    cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST,
                                           0,
                                           0)))
    T_world2cam = -1 * R_obj2cam * cam_location

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_world2cam = R_camfix * R_world2cam
    T_world2cam = R_camfix * T_world2cam

    return K, R_world2cam, T_world2cam

blender_T = np.array([
    [0, 1., 0],
    [-1, 0, 0],
    [0, 0, 1.],
])

class r2n2ValLoader(torch.utils.data.Dataset):
    def __init__(self,train_cameras, data_dir, json_file=None, img_res=[137, 137], num_worker=8, mode='single-view', split_file=None):
        self.train_cameras = train_cameras
        self.data_dir = data_dir
        self.json_file = json_file
        self.img_res = img_res
        self.total_pixels = img_res[0] * img_res[1]
        self.mode = mode
        self.sampling_idx = None
        self.instance_list = None
        self.split_file = split_file
        if self.json_file is not None:
            with open(self.json_file) as f:
                data = json.load(f)
            data = data[list(data.keys())[0]] # # category id
            data = data[list(data.keys())[0]]
            self.category_path = data
            # dir_paths = glob.glob(os.path.join(data_dir, data, '*'))
            dir_names = sorted(os.listdir(os.path.join(data_dir, data)))
            if split_file != None:
                self.instance_list = self.get_val_instances_list(self.split_file)
            else:
                print("no split file")
                exit(1)

        self.num_worker = num_worker
        self.image_list, self.camera_list, self.mask_list = self.get_image_file_list()

    def __len__(self):
        return len(self.image_list)

    def check_if_in_instance_list(self, instance_name):
        if self.instance_list is None:
            return True
        else:
            return (instance_name in self.instance_list)

    def get_image_file_list(self):
        '''
        get list of image files.
        '''
        image_list = []
        camera_list = []
        mask_list = []
    
        print("length of instance_list:", len(self.instance_list))
        for fname in self.instance_list:
            sub_flist = os.path.join(self.data_dir, self.category_path, fname, 'rendering')
            camera_path = os.path.join(sub_flist, 'rendering_metadata.txt')
            if os.path.isfile(camera_path):
                with open(camera_path, 'r') as f:
                    camera = f.read().split('\n')[:-1]
                    if self.mode == 'single-view':
                        camera_list.extend(camera[0:1])
                    elif self.mode == 'multi-view':
                        camera_list.extend(camera) 
                    else:
                        print("mode not implemented")
                        exit(1)
            else:
                print("can't find rendering_metadata.txt")
                exit(1)
            if self.mode == 'single-view':
                for i in range(1):
                    image_list.append(os.path.join(sub_flist, '%02d.png' % (i)))
                    mask_list.append(os.path.join(sub_flist, "masks", '%02d_m.png' % (i)))
            elif self.mode == 'multi-view':
                for i in range(24):
                    image_list.append(os.path.join(sub_flist, '%02d.png' % (i)))
                    mask_list.append(os.path.join(sub_flist, "masks", '%02d_m.png' % (i)))
            else:
                print("mode not implemented")
                exit(1)
        
        print("mode: {0}".format(self.mode))
        print('{0} training images in total.'.format(len(image_list)))
        assert len(image_list) == len(camera_list) 
        assert len(image_list) == len(mask_list)
        return image_list, camera_list, mask_list

    def get_instance_name_from_fname_png(self, fname_png):
        return fname_png.split('/')[-3]

    def get_instance_name(self, idx):
        fname_png = self.image_list[idx]
        return self.get_instance_name_from_fname_png(fname_png)
    
    def get_val_instances_list(self, split_file):
        f = open(split_file, 'r')
        res = []
        flag = False
        for line in f.readlines():
            line = line.strip('\n')
            if 'test' in line:
                break
            elif 'val' in line:
                flag = True
                continue

            if flag:
                res.append(line)
                continue
        return res

    def __getitem__(self, idx):
        fname_png = self.image_list[idx]
        mask_fname_png = self.mask_list[idx]
        # print(fname_png)
        img = cv2.imread(fname_png, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (IMG_W, IMG_H))# # fake resize on this dataset
        # cv2.imwrite(str(idx) + '.png', img) # # 貌似要程序正常退出才能写入
        
        mask = img[:, :, 3] # # get mask from alpha channel

        img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.
        img -= 0.5
        img *= 2.
        # img = rend_util.load_rgb(fname_png)
        img = torch.from_numpy(img).float()

        sampled_pixels = img.clone().detach()
        sampled_pixels = sampled_pixels.reshape(3, -1).transpose(1, 0)

        # # load mask
        # mask = cv2.imread(mask_fname_png, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_W, IMG_H))
        mask = mask.reshape(-1)
        mask = torch.from_numpy(mask).bool()
        # torch.set_printoptions(profile='full')
        # print(mask)

        # # params of camera 
        camera = self.camera_list[idx].split(' ')
        az = float(camera[0])
        el = float(camera[1])
        dist = float(camera[3])
        # print(az,el,dist)
        K, R, T = getBlenderProj(az, el, dist)
        R = R @ blender_T
        K, R, T = np.array(K), np.array(R), np.array(T)
        # K, R, T = torch.from_numpy(K).float(), torch.from_numpy(R).float(), torch.from_numpy(T).float()
        # pose = torch.eye(4)
        # pose[:3, :3] = R.transpose(1, 0)
        # pose[:3, 3] = -T.reshape(-1)
        # intrinsics = torch.eye(4)
        # intrinsics[:3, :3] = K
        world_mat = K @ np.concatenate((R, T), axis=1)
        world_mat = np.concatenate((world_mat, np.array([[0,0,0,1.0]])), axis=0)
        scale_mat = self.get_scale_mat()
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        intrinsics = torch.from_numpy(intrinsics).float()
        pose = torch.from_numpy(pose).float()

        # # uv grid of 2D image
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        if self.sampling_idx is not None:
            sampled_pixels = sampled_pixels[self.sampling_idx, :]
            mask = mask[self.sampling_idx]
            uv = uv[self.sampling_idx, :]
        
        # if not self.train_cameras: # # always true when using g.t. camera params
        #     sample["pose"] = self.pose_all[idx]
      
        return idx, (uv, intrinsics, pose, sampled_pixels, img, mask)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.eye(4) * 1.0

class r2n2TestLoader(torch.utils.data.Dataset):
    def __init__(self,train_cameras, data_dir, json_file=None, img_res=[137, 137], num_worker=8, mode='single-view', split_file=None):
        self.train_cameras = train_cameras
        self.data_dir = data_dir
        self.json_file = json_file
        self.img_res = img_res
        self.total_pixels = img_res[0] * img_res[1]
        self.mode = mode
        self.sampling_idx = None
        self.instance_list = None
        self.split_file = split_file
        if self.json_file is not None:
            with open(self.json_file) as f:
                data = json.load(f)
            data = data[list(data.keys())[0]] # # category id
            data = data[list(data.keys())[0]]
            self.category_path = data
            # dir_paths = glob.glob(os.path.join(data_dir, data, '*'))
            dir_names = sorted(os.listdir(os.path.join(data_dir, data)))
            if split_file != None:
                self.instance_list = self.get_test_instances_list(self.split_file)
            else:
                print("not inplemented")
                exit(1)

        self.num_worker = num_worker
        self.image_list, self.camera_list, self.mask_list = self.get_image_file_list()

    def __len__(self):
        return len(self.image_list)

    def check_if_in_instance_list(self, instance_name):
        if self.instance_list is None:
            return True
        else:
            return (instance_name in self.instance_list)

    def get_image_file_list(self):
        '''
        get list of image files.
        '''
        image_list = []
        camera_list = []
        mask_list = []
    
        print("length of instance_list:", len(self.instance_list))
        for fname in self.instance_list:
            sub_flist = os.path.join(self.data_dir, self.category_path, fname, 'rendering')
            camera_path = os.path.join(sub_flist, 'rendering_metadata.txt')
            if os.path.isfile(camera_path):
                with open(camera_path, 'r') as f:
                    camera = f.read().split('\n')[:-1]
                    if self.mode == 'single-view':
                        camera_list.extend(camera[0:1])
                    elif self.mode == 'multi-view':
                        camera_list.extend(camera) 
                    else:
                        print("mode not implemented")
                        exit(1)
            else:
                print("can't find rendering_metadata.txt")
                exit(1)
            if self.mode == 'single-view':
                for i in range(1):
                    image_list.append(os.path.join(sub_flist, '%02d.png' % (i)))
                    mask_list.append(os.path.join(sub_flist, "masks", '%02d_m.png' % (i)))
            elif self.mode == 'multi-view':
                for i in range(24):
                    image_list.append(os.path.join(sub_flist, '%02d.png' % (i)))
                    mask_list.append(os.path.join(sub_flist, "masks", '%02d_m.png' % (i)))
            else:
                print("mode not implemented")
                exit(1)
        
        print("mode: {0}".format(self.mode))
        print('{0} testing images in total.'.format(len(image_list)))
        assert len(image_list) == len(camera_list) 
        assert len(image_list) == len(mask_list)
        return image_list, camera_list, mask_list

    def get_instance_name_from_fname_png(self, fname_png):
        return fname_png.split('/')[-3]

    def get_instance_name(self, idx):
        fname_png = self.image_list[idx]
        return self.get_instance_name_from_fname_png(fname_png)
    
    def get_test_instances_list(self, split_file):
        f = open(split_file, 'r')
        res = []
        flag = False
        for line in f.readlines():
            line = line.strip('\n')
            if flag:
                res.append(line)
                continue
            if 'test' in line:
                flag = True
        return res

    def __getitem__(self, idx):
        fname_png = self.image_list[idx]
        mask_fname_png = self.mask_list[idx]
        # print(fname_png)
        img = cv2.imread(fname_png, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (IMG_W, IMG_H))# # fake resize on this dataset
        # cv2.imwrite(str(idx) + '.png', img) # # 貌似要程序正常退出才能写入
        
        mask = img[:, :, 3] # # get mask from alpha channel

        img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.
        img -= 0.5
        img *= 2.
        # img = rend_util.load_rgb(fname_png)
        img = torch.from_numpy(img).float()

        sampled_pixels = img.clone().detach()
        sampled_pixels = sampled_pixels.reshape(3, -1).transpose(1, 0)

        # # load mask
        # mask = cv2.imread(mask_fname_png, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_W, IMG_H))
        mask = mask.reshape(-1)
        mask = torch.from_numpy(mask).bool()
        # torch.set_printoptions(profile='full')
        # print(mask)

        # # params of camera 
        camera = self.camera_list[idx].split(' ')
        az = float(camera[0])
        el = float(camera[1])
        dist = float(camera[3])
        # print(az,el,dist)
        K, R, T = getBlenderProj(az, el, dist)
        R = R @ blender_T
        K, R, T = np.array(K), np.array(R), np.array(T)
        # K, R, T = torch.from_numpy(K).float(), torch.from_numpy(R).float(), torch.from_numpy(T).float()
        # pose = torch.eye(4)
        # pose[:3, :3] = R.transpose(1, 0)
        # pose[:3, 3] = -T.reshape(-1)
        # intrinsics = torch.eye(4)
        # intrinsics[:3, :3] = K
        world_mat = K @ np.concatenate((R, T), axis=1)
        world_mat = np.concatenate((world_mat, np.array([[0,0,0,1.0]])), axis=0)
        scale_mat = self.get_scale_mat()
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        intrinsics = torch.from_numpy(intrinsics).float()
        pose = torch.from_numpy(pose).float()

        # # uv grid of 2D image
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        
        assert self.sampling_idx == None

        if self.sampling_idx is not None:
            sampled_pixels = sampled_pixels[self.sampling_idx, :]
            mask = mask[self.sampling_idx]
            uv = uv[self.sampling_idx, :]
        
        # if not self.train_cameras: # # always true when using g.t. camera params
        #     sample["pose"] = self.pose_all[idx]

        return idx, (uv, intrinsics, pose, sampled_pixels, img, mask)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.eye(4) * 1.0


class r2n2TrainLoader(torch.utils.data.Dataset):
    def __init__(self,train_cameras, data_dir, json_file=None, img_res=[137, 137], num_worker=8, mode='single-view', split_file=None):
        self.train_cameras = train_cameras
        self.data_dir = data_dir
        self.json_file = json_file
        self.img_res = img_res
        self.total_pixels = img_res[0] * img_res[1]
        self.mode = mode
        self.sampling_idx = None
        self.instance_list = None
        self.split_file = split_file
        if self.json_file is not None:
            with open(self.json_file) as f:
                data = json.load(f)
            data = data[list(data.keys())[0]] # # category id
            data = data[list(data.keys())[0]]
            self.category_path = data
            # dir_paths = glob.glob(os.path.join(data_dir, data, '*'))
            dir_names = sorted(os.listdir(os.path.join(data_dir, data)))
            if self.split_file != None:
                self.instance_list = self.get_train_instances_list(self.split_file)
            else:
                print("no split file")
                exit(1)

        self.num_worker = num_worker
        self.image_list, self.camera_list, self.mask_list = self.get_image_file_list()

    def __len__(self):
        return len(self.image_list)

    def check_if_in_instance_list(self, instance_name):
        if self.instance_list is None:
            return True
        else:
            return (instance_name in self.instance_list)

    def get_image_file_list(self):
        '''
        get list of image files.
        '''
        image_list = []
        camera_list = []
        mask_list = []
    
        print("length of instance_list:", len(self.instance_list))
        for fname in self.instance_list:
            sub_flist = os.path.join(self.data_dir, self.category_path, fname, 'rendering')
            camera_path = os.path.join(sub_flist, 'rendering_metadata.txt')
            if os.path.isfile(camera_path):
                with open(camera_path, 'r') as f:
                    camera = f.read().split('\n')[:-1]
                    if self.mode == 'single-view':
                        camera_list.extend(camera[0:1])
                    elif self.mode == 'multi-view':
                        camera_list.extend(camera) 
                    else:
                        print("mode not implemented")
                        exit(1)
            else:
                print("can't find rendering_metadata.txt")
                exit(1)
            if self.mode == 'single-view':
                for i in range(1):
                    image_list.append(os.path.join(sub_flist, '%02d.png' % (i)))
                    mask_list.append(os.path.join(sub_flist, "masks", '%02d_m.png' % (i)))
            elif self.mode == 'multi-view':
                for i in range(24):
                    image_list.append(os.path.join(sub_flist, '%02d.png' % (i)))
                    mask_list.append(os.path.join(sub_flist, "masks", '%02d_m.png' % (i)))
            else:
                print("mode not implemented")
                exit(1)
        
        print("mode: {0}".format(self.mode))
        print('{0} training images in total.'.format(len(image_list)))
        assert len(image_list) == len(camera_list) 
        assert len(image_list) == len(mask_list)
        return image_list, camera_list, mask_list

    def get_instance_name_from_fname_png(self, fname_png):
        return fname_png.split('/')[-3]

    def get_instance_name(self, idx):
        fname_png = self.image_list[idx]
        return self.get_instance_name_from_fname_png(fname_png)

    def get_train_instances_list(self, split_file):
        f = open(split_file, 'r')
        res = []
        for line in f.readlines():
            line = line.strip('\n')
            if 'train' in line:
                continue
            elif 'test' in line or 'val' in line:
                break
            res.append(line)
        return res

    def __getitem__(self, idx):
        fname_png = self.image_list[idx]
        mask_fname_png = self.mask_list[idx]
        # print(fname_png)
        img = cv2.imread(fname_png, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (IMG_W, IMG_H))# # fake resize on this dataset
        # cv2.imwrite(str(idx) + '.png', img) # # 貌似要程序正常退出才能写入
        
        mask = img[:, :, 3] # # get mask from alpha channel

        img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.
        img -= 0.5
        img *= 2.
        # img = rend_util.load_rgb(fname_png)
        img = torch.from_numpy(img).float()

        sampled_pixels = img.clone().detach()
        sampled_pixels = sampled_pixels.reshape(3, -1).transpose(1, 0)

        # # load mask
        # mask = cv2.imread(mask_fname_png, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_W, IMG_H))
        mask = mask.reshape(-1)
        mask = torch.from_numpy(mask).bool()
        # torch.set_printoptions(profile='full')
        # print(mask)

        # # params of camera 
        camera = self.camera_list[idx].split(' ')
        az = float(camera[0])
        el = float(camera[1])
        dist = float(camera[3])
        # print(az,el,dist)
        K, R, T = getBlenderProj(az, el, dist)
        R = R @ blender_T
        K, R, T = np.array(K), np.array(R), np.array(T)
        # K, R, T = torch.from_numpy(K).float(), torch.from_numpy(R).float(), torch.from_numpy(T).float()
        # pose = torch.eye(4)
        # pose[:3, :3] = R.transpose(1, 0)
        # pose[:3, 3] = -T.reshape(-1)
        # intrinsics = torch.eye(4)
        # intrinsics[:3, :3] = K
        world_mat = K @ np.concatenate((R, T), axis=1)
        world_mat = np.concatenate((world_mat, np.array([[0,0,0,1.0]])), axis=0)
        scale_mat = self.get_scale_mat()
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        intrinsics = torch.from_numpy(intrinsics).float()
        pose = torch.from_numpy(pose).float()

        # # uv grid of 2D image
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        if self.sampling_idx is not None:
            sampled_pixels = sampled_pixels[self.sampling_idx, :]
            mask = mask[self.sampling_idx]
            uv = uv[self.sampling_idx, :]
        
        # if not self.train_cameras: # # always true when using g.t. camera params
        #     sample["pose"] = self.pose_all[idx]
        # if self.test_mode == 'test':
        #     return idx, (uv, intrinsics, pose, sampled_pixels, img, mask, fname_png)

        return idx, (uv, intrinsics, pose, sampled_pixels, img, mask)
        return idx, sample, ground_truth
        return (fname_png, img, K, R, T)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.eye(4) * 1.0

if __name__ == "__main__":
    train_dataset = r2n2TrainLoader(train_cameras=False,
                            data_dir='/data2/ShapeNetRendering',
                            json_file='/home/tinglei/3D_Recon/datasets/shapenet.json',
                            img_res=[IMG_H, IMG_W],
                            num_worker=8,
                            mode='single-view',
                            split_file='/home/tinglei/3D_Recon/datasets/split.txt')

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False
    )
    print(len(dataloader))
    for index, (_, (uv, intrinsics, pose,sampled_pixels, img, mask)) in enumerate(dataloader):
        print(uv.size())
        print(intrinsics, pose, sep='\n')
        print(img.size(), sampled_pixels.size(), mask.size())
        break
      
        img = img.squeeze(0)
        mask = mask.reshape(137, 137)
        masked_img = img.numpy()
        masked_img = (masked_img / 2 + 0.5) * 255
        masked_img = masked_img * mask.numpy()
        masked_img = np.transpose(masked_img, (1, 2, 0))
        masked_img = masked_img.astype(np.uint8)
        
        cv2.imwrite(os.path.join('datasets','glimpse',str(index) + '.png'), img=masked_img)

        print(mask.sum())
        mask = (mask.numpy().astype(np.uint8) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join('datasets','glimpse',str(index) + '_m.png'), img=mask)
        

    # for index, (fname_png, img, K, R, T) in enumerate(dataloader):
    #     print(fname_png, K, R, T, sep='\n')
    #     break
